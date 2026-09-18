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
        elif node.op == SUM or node.op == COUNT:
            if node.left < 0 or node.left >= i:
                raise Error("Invalid aggregate input")
            if shapes[node.left] != ROWS or aggregated[node.left]:
                raise Error(
                    "Aggregate input must be row-valued without nested aggregates"
                )
            if node.min_count < 0:
                raise Error("min_count must be nonnegative")
            dtype = types[node.left]
            if node.op == SUM and dtype != "int64" and dtype != "float64":
                raise Error("sum requires a numeric expression")
            if node.op == COUNT:
                dtype = "int64"
            shape = AGGREGATE
            has_aggregate = True
        elif node.op >= ADD and node.op <= EQ:
            if (
                node.left < 0
                or node.right < 0
                or node.left >= i
                or node.right >= i
            ):
                raise Error("Invalid binary expression inputs")
            dtype = types[node.left]
            if dtype != types[node.right]:
                raise Error(
                    "Expression operands require matching dtypes; use typed literals"
                )
            if node.op != EQ and dtype != "int64" and dtype != "float64":
                raise Error(
                    "Arithmetic and greater-than require numeric operands"
                )
            if node.op == GT or node.op == EQ:
                dtype = "bool"
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
