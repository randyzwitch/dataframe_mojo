"""Capability-checked lowering of shared logical plans, with no GPU imports.

The first region is one in-memory Float32/64 column, an optional comparison
with a scalar, and sum/count projections of that column or one scalar
arithmetic operation. Binding owns literal coercion and aggregate semantics.
"""
from .binding import BoundExpr, bind, op_name
from .dtype import DataType
from .expr import (
    COL,
    LIT_FLOAT,
    ADD,
    SUB,
    MUL,
    GT,
    LT,
    GE,
    LE,
    EQ,
    NE,
    SUM,
    COUNT,
)
from .lazy import LazyFrame, SCAN_FRAME, FILTER, SELECT, SLICE
from .series import Series


@fieldwise_init
struct AccelScalar(Copyable):
    var source: Int
    var op: Int
    var literal: Float64
    var literal_left: Bool


@fieldwise_init
struct AccelReduction(Copyable):
    var input: AccelScalar
    var op: Int
    var min_count: Int
    var name: String


@fieldwise_init
struct AccelPlan(Movable):
    var source: Series
    var predicate: AccelScalar
    var reductions: List[AccelReduction]
    var root: Int
    var limit: Int


def _unsupported(reason: String) raises:
    raise Error("NVIDIA unsupported: " + reason)


def _scalar(bound: BoundExpr, root: Int, predicate: Bool) raises -> AccelScalar:
    ref nodes = bound.expr._nodes
    ref node = nodes[root]
    if node.op == COL and not predicate:
        return AccelScalar(bound.sources[root], -1, 0, False)
    if predicate:
        if not (
            node.op == GT
            or node.op == LT
            or node.op == GE
            or node.op == LE
            or node.op == EQ
            or node.op == NE
        ):
            _unsupported("filter requires one scalar comparison")
    elif not (node.op == ADD or node.op == SUB or node.op == MUL):
        _unsupported(
            "reduction input supports a column or one scalar +, -, * operation"
        )
    if node.left < 0 or node.right < 0:
        _unsupported("binary expression operands are required")
    var column = node.left
    var literal = node.right
    var literal_left = nodes[node.left].op == LIT_FLOAT
    if literal_left:
        column = node.right
        literal = node.left
    if nodes[column].op != COL or nodes[literal].op != LIT_FLOAT:
        _unsupported("operation requires a column and a bound floating scalar")
    if bound.dtypes[column] != bound.dtypes[literal]:
        _unsupported("mixed operand dtypes")
    return AccelScalar(
        bound.sources[column], node.op, nodes[literal].floating, literal_left
    )


def lower_accel(plan: LazyFrame) raises -> AccelPlan:
    """Validate the whole region before allocating or submitting device work."""
    var root = len(plan._nodes) - 1
    if root < 0:
        _unsupported("empty plan")
    var limit = -1
    var cursor = root
    if plan._nodes[cursor].kind == SLICE:
        if plan._nodes[cursor].offset != 0:
            _unsupported("only head after reduction is supported")
        limit = plan._nodes[cursor].length
        cursor = plan._nodes[cursor].left
    if cursor < 0 or plan._nodes[cursor].kind != SELECT:
        _unsupported("plan must end with sum/count projections")
    var select = cursor
    cursor = plan._nodes[cursor].left
    var filter = -1
    if cursor >= 0 and plan._nodes[cursor].kind == FILTER:
        filter = cursor
        cursor = plan._nodes[cursor].left
    if cursor < 0 or plan._nodes[cursor].kind != SCAN_FRAME:
        _unsupported("requires one in-memory scan and at most one filter")
    ref frame = plan._frames[plan._nodes[cursor].offset]
    var reductions = List[AccelReduction]()
    var source = -1
    for expr in plan._nodes[select].exprs:
        var bound = bind(expr, frame._columns)
        ref node = bound.expr._nodes[len(bound.expr._nodes) - 1]
        if node.op != SUM and node.op != COUNT:
            _unsupported(
                "projection requires sum or count; got " + op_name(node.op)
            )
        var input = _scalar(bound, node.left, False)
        if source < 0:
            source = input.source
        if source != input.source:
            _unsupported("all expressions must read the same column")
        for previous in reductions:
            if previous.name == expr._name:
                _unsupported("duplicate output name: " + expr._name)
        reductions.append(
            AccelReduction(input^, node.op, node.min_count, expr._name)
        )
    if source < 0:
        _unsupported("at least one reduction is required")
    var column = frame._columns[source].copy()
    if (
        column.dtype() != DataType.FLOAT32
        and column.dtype() != DataType.FLOAT64
    ):
        _unsupported("input dtype must be Float32 or Float64")
    if column.is_chunked():
        _unsupported("chunked input; rechunk before accelerator execution")
    var predicate = AccelScalar(source, -1, 0, False)
    if filter >= 0:
        if len(plan._nodes[filter].exprs) != 1:
            _unsupported("requires one filter expression")
        var bound = bind(plan._nodes[filter].exprs[0], frame._columns)
        predicate = _scalar(bound, len(bound.expr._nodes) - 1, True)
        if predicate.source != source:
            _unsupported("filter and reductions must read the same column")
    return AccelPlan(column^, predicate^, reductions^, root, limit)
