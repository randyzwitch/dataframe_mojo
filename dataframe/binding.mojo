"""Schema-only binding and expression shape analysis before execution."""
from .expr import (
    Expr,
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_STRING,
    ADD,
    SUB,
    MUL,
    GT,
    EQ,
    SUM,
    COUNT,
    NE,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    CLIP_LOW,
    CLIP_HIGH,
    NEG,
    ABS,
    SQRT,
    EXP,
    LOG,
    FLOOR,
    CEIL,
    ROUND,
    LIT_NULL,
    AND,
    OR,
    XOR,
    FILL_NULL,
    FILL_NAN,
    KEEP_NULLS,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    ANY,
    ALL,
    NULL_COUNT,
    is_logical,
    is_binary,
    is_unary,
    is_reduction,
    is_comparison,
)
from .series import Series

comptime SCALAR = 0
comptime ROWS = 1
comptime AGGREGATE = 2


@fieldwise_init
struct BoundExpr(Copyable):
    var expr: Expr
    var dtypes: List[String]
    var shapes: List[Int]
    var aggregated: List[Bool]
    var sources: List[Int]

    def shape(self) -> Int:
        return self.shapes[len(self.shapes) - 1]


def op_name(op: Int) -> String:
    if op == ADD:
        return "+"
    if op == SUB:
        return "-"
    if op == MUL:
        return "*"
    if op == DIV:
        return "/"
    if op == FLOORDIV:
        return "//"
    if op == MOD:
        return "%"
    if op == POW:
        return "pow"
    if op == GT:
        return ">"
    if op == EQ:
        return "eq"
    if op == NE:
        return "ne"
    if op == CLIP_LOW or op == CLIP_HIGH:
        return "clip"
    if op == NEG:
        return "negation"
    if op == ABS:
        return "abs"
    if op == SQRT:
        return "sqrt"
    if op == EXP:
        return "exp"
    if op == LOG:
        return "log"
    if op == FLOOR:
        return "floor"
    if op == CEIL:
        return "ceil"
    if op == ROUND:
        return "round"
    if op == AND:
        return "and"
    if op == OR:
        return "or"
    if op == XOR:
        return "xor"
    if op == NOT:
        return "not"
    if op == FILL_NULL:
        return "fill_null"
    if op == FILL_NAN:
        return "fill_nan"
    if op == IS_NAN:
        return "is_nan"
    if op == IS_NOT_NAN:
        return "is_not_nan"
    if op == IS_FINITE:
        return "is_finite"
    if op == IS_INFINITE:
        return "is_infinite"
    if op == ANY:
        return "any"
    if op == ALL:
        return "all"
    if op == NULL_COUNT:
        return "null_count"
    if op == SUM:
        return "sum"
    if op == COUNT:
        return "count"
    if is_comparison(op):
        return "comparison"
    return "operation " + String(op)


def _numeric(dtype: String) -> Bool:
    return dtype == "int64" or dtype == "float64"


def _binary_dtype(op: Int, left: String, right: String) raises -> String:
    if op == KEEP_NULLS:
        return right
    if left != right:
        raise Error(
            op_name(op)
            + " requires matching dtypes, found "
            + left
            + " and "
            + right
            + "; use typed literals"
        )
    if op == EQ or op == NE:
        return "bool"
    if is_comparison(op):
        return "bool"
    if is_logical(op):
        if left != "bool":
            raise Error(op_name(op) + " requires bool operands, found " + left)
        return "bool"
    if op == FILL_NULL:
        return left
    if op == FILL_NAN:
        if left != "float64":
            raise Error("fill_nan requires float64 operands, found " + left)
        return left
    if not _numeric(left):
        raise Error(op_name(op) + " requires numeric operands, found " + left)
    if op == DIV:
        return "float64"
    return left


def _unary_dtype(op: Int, input: String) raises -> String:
    if op == IS_NULL or op == IS_NOT_NULL:
        return "bool"
    if op == NOT:
        if input != "bool":
            raise Error("not requires a bool operand, found " + input)
        return "bool"
    if op >= IS_NAN and op <= IS_INFINITE:
        if input != "float64":
            raise Error(
                op_name(op) + " requires a float64 operand, found " + input
            )
        return "bool"
    if not _numeric(input):
        raise Error(op_name(op) + " requires a numeric operand, found " + input)
    if op == SQRT or op == EXP or op == LOG:
        return "float64"
    return input


def _reduction_dtype(op: Int, input: String) raises -> String:
    if op == SUM:
        if not _numeric(input):
            raise Error("sum requires a numeric expression, found " + input)
        return input
    if op == COUNT or op == NULL_COUNT:
        return "int64"
    if op == ANY or op == ALL:
        if input != "bool":
            raise Error(
                op_name(op) + " requires a bool expression, found " + input
            )
        return "bool"
    raise Error("Unsupported reduction: " + op_name(op))


def bind(expr: Expr, columns: List[Series]) raises -> BoundExpr:
    if len(expr._nodes) == 0:
        raise Error("An expression must contain at least one node")
    var types = List[String]()
    var shapes = List[Int]()
    var aggregated = List[Bool]()
    var sources = List[Int]()
    for i in range(len(expr._nodes)):
        var node = expr._nodes[i].copy()
        var dtype: String
        var shape = SCALAR
        var has_aggregate = False
        var source = -1
        if node.op == COL:
            for j in range(len(columns)):
                if columns[j].name() == node.text:
                    source = j
                    break
            if source < 0:
                raise Error("Unknown expression column: " + node.text)
            dtype = columns[source].dtype()
            shape = ROWS
        elif node.op == LIT_INT:
            dtype = "int64"
        elif node.op == LIT_FLOAT:
            dtype = "float64"
        elif node.op == LIT_BOOL:
            dtype = "bool"
        elif node.op == LIT_STRING:
            dtype = "string"
        elif node.op == LIT_NULL:
            if not (
                _numeric(node.text)
                or node.text == "bool"
                or node.text == "string"
            ):
                raise Error("Unknown null literal dtype: " + node.text)
            dtype = node.text
        elif is_reduction(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid aggregate input")
            if shapes[node.left] != ROWS or aggregated[node.left]:
                raise Error(
                    "Aggregate input must be row-valued without nested aggregates"
                )
            if node.min_count < 0:
                raise Error("min_count must be nonnegative")
            dtype = _reduction_dtype(node.op, types[node.left])
            shape = AGGREGATE
            has_aggregate = True
        elif is_unary(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid unary expression input")
            dtype = _unary_dtype(node.op, types[node.left])
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
        elif is_binary(node.op):
            if (
                node.left < 0
                or node.right < 0
                or node.left >= i
                or node.right >= i
            ):
                raise Error("Invalid binary expression inputs")
            dtype = _binary_dtype(node.op, types[node.left], types[node.right])
            has_aggregate = aggregated[node.left] or aggregated[node.right]
            if shapes[node.left] == ROWS or shapes[node.right] == ROWS:
                shape = ROWS
            elif has_aggregate:
                shape = AGGREGATE
        else:
            raise Error("Unknown expression operation")
        types.append(dtype)
        shapes.append(shape)
        aggregated.append(has_aggregate)
        sources.append(source)
    return BoundExpr(expr.copy(), types^, shapes^, aggregated^, sources^)
