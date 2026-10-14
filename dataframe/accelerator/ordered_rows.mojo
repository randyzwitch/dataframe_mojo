"""CPU metadata for ordered direct terminal ROWS expression occurrences."""
from dataframe.binding import BoundExpr, ROWS
from dataframe.dtype import DataType
from dataframe.expr import (
    Expr,
    Node,
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_NULL,
    CAST,
    WHEN,
    ADD,
    SUB,
    MUL,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    NEG,
    ABS,
    SQRT,
    EXP,
    LOG,
    FLOOR,
    CEIL,
    ROUND,
    GT,
    LT,
    GE,
    LE,
    EQ,
    NE,
    AND,
    OR,
    XOR,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    FILL_NULL,
    FILL_NAN,
    KEEP_NULLS,
    CLIP_LOW,
    CLIP_HIGH,
    INT_IS_IN,
)
from . import temporal_expressions
from .integer_ranges import proven_integer_nodes


def eligible(bound: BoundExpr) -> Bool:
    """Exclude state phases and structural/dictionary operations, not data."""
    if bound.shape() != ROWS:
        return False
    var supported: List[Int] = [
        COL,
        LIT_INT,
        LIT_FLOAT,
        LIT_BOOL,
        LIT_NULL,
        CAST,
        WHEN,
        ADD,
        SUB,
        MUL,
        DIV,
        FLOORDIV,
        MOD,
        POW,
        NEG,
        ABS,
        SQRT,
        EXP,
        LOG,
        FLOOR,
        CEIL,
        ROUND,
        GT,
        LT,
        GE,
        LE,
        EQ,
        NE,
        AND,
        OR,
        XOR,
        NOT,
        IS_NULL,
        IS_NOT_NULL,
        IS_NAN,
        IS_NOT_NAN,
        IS_FINITE,
        IS_INFINITE,
        FILL_NULL,
        FILL_NAN,
        KEEP_NULLS,
        CLIP_LOW,
        CLIP_HIGH,
        INT_IS_IN,
    ]
    for index in range(len(bound.expr._nodes)):
        if bound.expr._nodes[index].op not in supported:
            return False
        var dtype = bound.dtypes[index]
        if not (
            dtype.is_numeric() or dtype == DataType.BOOL or dtype.is_temporal()
        ):
            return False
    return True


def needed(bound: BoundExpr) -> Bool:
    """Retain the ordinary route for infallible or single-fault expressions."""
    if temporal_expressions.contains(bound):
        return True
    var checked = 0
    var proven = proven_integer_nodes(bound)
    for index in range(len(bound.expr._nodes)):
        ref node = bound.expr._nodes[index]
        var dtype = bound.dtypes[index]
        if node.op == CAST and node.integer and not proven[index]:
            checked += 1
        elif (
            not proven[index]
            and (dtype.is_integer() or dtype.is_decimal())
            and (
                node.op == ADD
                or node.op == SUB
                or node.op == MUL
                or node.op == DIV
                or node.op == FLOORDIV
                or node.op == MOD
                or node.op == POW
                or node.op == NEG
                or node.op == ABS
            )
        ):
            checked += 1
    return checked > 1


def occurrences(
    bound: BoundExpr, max_nodes: Int
) raises -> Tuple[BoundExpr, List[UInt32]]:
    """Clone each AST visit while retaining the exact CPU-bound metadata.

    No second bind changes the operation order. Shared checked nodes have one
    rank per occurrence; their original sources/dtypes/shapes are copied.
    """
    var pending = List[Tuple[Int, Bool]]()
    pending.append((len(bound.expr._nodes) - 1, False))
    var roots = List[Int]()
    var nodes = List[Node]()
    var dtypes = List[DataType]()
    var shapes = List[Int]()
    var aggregated = List[Bool]()
    var sources = List[Int]()
    var fusible = List[Bool]()
    var ranks = List[UInt32]()
    var next_rank = UInt32(1)
    while len(pending):
        var visit = pending.pop()
        var index = visit[0]
        var node = bound.expr._nodes[index].copy()
        if not visit[1]:
            pending.append((index, True))
            if node.extra >= 0:
                pending.append((node.extra, False))
            if node.right >= 0:
                pending.append((node.right, False))
            if node.left >= 0:
                pending.append((node.left, False))
            continue
        if len(nodes) >= max_nodes:
            raise Error(
                "Metal unsupported [ordered rows]: occurrence program exceeds device node capacity"
            )
        if node.extra >= 0:
            node.extra = roots.pop()
        if node.right >= 0:
            node.right = roots.pop()
        if node.left >= 0:
            node.left = roots.pop()
        ranks.append(next_rank)
        next_rank += 1
        dtypes.append(bound.dtypes[index])
        shapes.append(bound.shapes[index])
        aggregated.append(bound.aggregated[index])
        sources.append(bound.sources[index])
        fusible.append(bound.fusible[index])
        nodes.append(node^)
        roots.append(len(nodes) - 1)
    return (
        BoundExpr(
            Expr(nodes^, bound.expr._name),
            dtypes^,
            shapes^,
            aggregated^,
            sources^,
            fusible^,
        ),
        ranks^,
    )
