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
    MIN,
    MAX,
    MEAN,
    FIRST,
    LAST,
    N_UNIQUE,
    STD,
    VAR,
    MEDIAN,
    QUANTILE,
    LEN,
    Node,
    WHEN,
    SELECTOR,
    SEP,
    CUM_SUM,
    CUM_COUNT,
    RANK,
    ROLLING_SUM,
    ROLLING_MEAN,
    ROLLING_MIN,
    ROLLING_MAX,
    FORWARD_FILL,
    BACKWARD_FILL,
    OVER,
    is_window,
    STR_CONCAT,
    CAST,
    STR_LEN_CHARS,
    STR_LEN_BYTES,
    STR_STARTS_WITH,
    STR_ENDS_WITH,
    STR_CONTAINS,
    STR_SLICE,
    STR_PAD,
    is_string_op,
    is_conditional,
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
    if op == MIN:
        return "min"
    if op == MAX:
        return "max"
    if op == MEAN:
        return "mean"
    if op == FIRST:
        return "first"
    if op == LAST:
        return "last"
    if op == N_UNIQUE:
        return "n_unique"
    if op == STD:
        return "std"
    if op == VAR:
        return "var"
    if op == MEDIAN:
        return "median"
    if op == QUANTILE:
        return "quantile"
    if op == LEN:
        return "len"
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
    if op == STR_CONCAT:
        if left != "string" or right != "string":
            raise Error(
                "concat_str requires string operands, found "
                + left
                + " and "
                + right
            )
        return "string"
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


def _reduction_dtype(node: Node, input: String) raises -> String:
    var op = node.op
    if op == SUM:
        if not _numeric(input):
            raise Error("sum requires a numeric expression, found " + input)
        return input
    if op == COUNT or op == NULL_COUNT or op == N_UNIQUE or op == LEN:
        return "int64"
    if op == ANY or op == ALL:
        if input != "bool":
            raise Error(
                op_name(op) + " requires a bool expression, found " + input
            )
        return "bool"
    if op == MIN or op == MAX or op == FIRST or op == LAST:
        return input
    if op == MEAN or op == STD or op == VAR or op == MEDIAN or op == QUANTILE:
        if not _numeric(input):
            raise Error(
                op_name(op) + " requires a numeric expression, found " + input
            )
        if (op == STD or op == VAR) and node.integer < 0:
            raise Error("ddof must be nonnegative")
        if op == QUANTILE:
            if not (node.floating >= 0 and node.floating <= 1):
                raise Error("quantile must be between 0 and 1")
            var method = node.text
            if (
                method != "linear"
                and method != "nearest"
                and method != "lower"
                and method != "higher"
                and method != "midpoint"
            ):
                raise Error(
                    "interpolation must be nearest, lower, higher, midpoint,"
                    " or linear"
                )
        return "float64"
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
        elif node.op == SELECTOR:
            raise Error(
                "Selectors must be expanded before binding; use select,"
                " with_columns, agg, or filter"
            )
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
            dtype = _reduction_dtype(node, types[node.left])
            shape = AGGREGATE
            has_aggregate = True
        elif is_conditional(node.op):
            if (
                node.left < 0
                or node.right < 0
                or node.left >= i
                or node.right >= i
                or node.extra >= i
            ):
                raise Error("Invalid conditional expression inputs")
            if types[node.left] != "bool":
                raise Error(
                    "when requires a bool predicate, found " + types[node.left]
                )
            dtype = types[node.right]
            var children = List[Int]()
            children.append(node.left)
            children.append(node.right)
            if node.extra >= 0:
                if types[node.extra] != dtype:
                    raise Error(
                        "when/then/otherwise branches require matching dtypes,"
                        " found " + dtype + " and " + types[node.extra]
                    )
                children.append(node.extra)
            for child in children:
                has_aggregate = has_aggregate or aggregated[child]
                if shapes[child] == ROWS:
                    shape = ROWS
            if shape != ROWS and has_aggregate:
                shape = AGGREGATE
        elif is_window(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid window input")
            if shapes[node.left] != ROWS:
                raise Error("Window operations require a row-valued input")
            var input = types[node.left]
            dtype = input
            if node.op == CUM_SUM or node.op == ROLLING_SUM:
                if not _numeric(input):
                    raise Error(
                        "cum_sum and rolling_sum require a numeric expression,"
                        " found " + input
                    )
            elif node.op == ROLLING_MEAN:
                if not _numeric(input):
                    raise Error(
                        "rolling_mean requires a numeric expression, found "
                        + input
                    )
                dtype = "float64"
            elif node.op == CUM_COUNT:
                dtype = "int64"
            elif node.op == RANK:
                var method = node.text
                if (
                    method != "average"
                    and method != "min"
                    and method != "max"
                    and method != "dense"
                    and method != "ordinal"
                ):
                    raise Error(
                        "rank method must be average, min, max, dense, or"
                        " ordinal"
                    )
                dtype = "float64" if method == "average" else "int64"
            if node.op >= ROLLING_SUM and node.op <= ROLLING_MAX:
                if node.integer < 1:
                    raise Error("window_size must be at least 1")
                if node.floating < -1:
                    raise Error("min_samples must be nonnegative")
            if (
                node.op == FORWARD_FILL or node.op == BACKWARD_FILL
            ) and node.integer < -1:
                raise Error("fill limit must be nonnegative")
            shape = ROWS
            has_aggregate = aggregated[node.left]
        elif node.op == OVER:
            if node.left < 0 or node.left >= i:
                raise Error("Invalid over input")
            if node.text2.byte_length() == 0:
                raise Error("over requires at least one partition column")
            for part in node.text2.split(SEP):
                var found = False
                for column in columns:
                    found = found or column.name() == String(part)
                if not found:
                    raise Error("Unknown partition column: " + String(part))
            dtype = types[node.left]
            shape = ROWS
        elif node.op == CAST:
            if node.left < 0 or node.left >= i:
                raise Error("Invalid cast input")
            var target = node.text
            if not (_numeric(target) or target == "bool" or target == "string"):
                raise Error("Unknown cast dtype: " + target)
            dtype = target
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
        elif is_string_op(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid string expression input")
            if types[node.left] != "string":
                raise Error(
                    "str operations require a string expression, found "
                    + types[node.left]
                )
            if node.op == STR_SLICE and node.min_count < -1:
                raise Error("str.slice length must be nonnegative")
            if node.op == STR_PAD:
                if node.min_count < 0:
                    raise Error("pad width must be nonnegative")
                if len(node.text.codepoints()) != 1:
                    raise Error("pad fill_char must be one character")
            if node.op == STR_LEN_CHARS or node.op == STR_LEN_BYTES:
                dtype = "int64"
            elif (
                node.op == STR_STARTS_WITH
                or node.op == STR_ENDS_WITH
                or node.op == STR_CONTAINS
            ):
                dtype = "bool"
            else:
                dtype = "string"
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
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
