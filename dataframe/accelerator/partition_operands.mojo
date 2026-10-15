"""Exact private numeric OVER operands without an automatic fault boundary.

The public normalizer captures original binding first. Scalar literal leaves
are broadcast through an infallible row anchor before generic folding/lowering;
original checked AST occurrence order remains unchanged. A private RankedRows
consumer must observe tokens using the CPU OVER run/state/operand schedule.
"""
from std.memory import ArcPointer
from dataframe import DataFrame, DataType, Expr, Series, col, when
from dataframe.binding import bind, ROWS, BoundExpr
from dataframe.expr import Node, LIT_INT, LIT_FLOAT, LIT_BOOL, LIT_NULL
from . import ordered_rows
from .implode_ordered import signature
from .rows import RowPlan
from .gpu.runtime import _preflight


def certificate(expression: Expr, source: DataFrame) raises -> String:
    """Capture before scalar lifting, dictionary recoding or value folding."""
    var original = bind(expression, source._columns)
    if not ordered_rows.eligible(original):
        raise Error(
            "Partition CUT/QCUT requires a pure fixed-width ROWS operand"
        )
    return signature(original)


def _append(mut nodes: List[Node], expression: Expr) -> Int:
    var offset = len(nodes)
    for original in expression._nodes:
        var node = original.copy()
        if node.left >= 0:
            node.left += offset
        if node.right >= 0:
            node.right += offset
        if node.extra >= 0:
            node.extra += offset
        nodes.append(node^)
    return len(nodes) - 1


def lift_literals(bound: BoundExpr, source: DataFrame) raises -> BoundExpr:
    """Preserve literal payloads, masks and original checked-operation order.

    Both branches contain the same original literal, including typed nulls;
    anchor nullness therefore never changes its value. The anchor is a fixed
    source column already referenced by this ROWS operand, not a new text/key
    input. Checked scalar ancestors become ROWS and cannot be eagerly folded.
    """
    if not ordered_rows.eligible(bound):
        raise Error("Partition operand AST is not a pure numeric ROWS program")
    var anchor = -1
    for input in bound.sources:
        if input >= 0:
            anchor = input
            break
    if anchor < 0 or anchor >= source.width():
        raise Error("Partition operand has no original row source anchor")
    var positions = List[Int](length=len(bound.expr._nodes), fill=-1)
    var nodes = List[Node]()
    for at in range(len(bound.expr._nodes)):
        var node = bound.expr._nodes[at].copy()
        if (
            node.op == LIT_INT
            or node.op == LIT_FLOAT
            or node.op == LIT_BOOL
            or node.op == LIT_NULL
        ):
            var leaf = Expr([node.copy()], "")
            var lifted = (
                when(col(source._columns[anchor].name()).is_not_null())
                .then(leaf.copy())
                .otherwise(leaf.copy())
            )
            positions[at] = _append(nodes, lifted)
        else:
            if node.left >= 0:
                node.left = positions[node.left]
            if node.right >= 0:
                node.right = positions[node.right]
            if node.extra >= 0:
                node.extra = positions[node.extra]
            positions[at] = len(nodes)
            nodes.append(node^)
    var lifted = bind(Expr(nodes^, bound.expr._name), source._columns)
    for at in range(len(positions)):
        var mapped = positions[at]
        if (
            mapped < 0
            or mapped >= len(lifted.dtypes)
            or lifted.dtypes[mapped] != bound.dtypes[at]
            or lifted.aggregated[mapped] != bound.aggregated[at]
            or lifted.shapes[mapped] != ROWS
        ):
            raise Error(
                "Partition literal lift changed original bound semantics"
            )
        if (
            bound.expr._nodes[at].op
            not in [LIT_INT, LIT_FLOAT, LIT_BOOL, LIT_NULL]
            and lifted.expr._nodes[mapped].op != bound.expr._nodes[at].op
        ):
            raise Error("Partition literal lift changed an original operation")
    return lifted^


def prepare(
    expression: Expr,
    source: DataFrame,
    rows: Int,
    expected_signature: String,
    *,
    source_proved: Bool = False,
    numeric_state: Bool = False,
) raises -> RowPlan:
    """Return one exact ranked Float64 operand; never evaluate input values."""
    if not source_proved:
        raise Error("Partition operand original source certificate is missing")
    if rows < 0 or rows > Int(Int32.MAX):
        raise Error("Partition operand source cardinality exceeds device range")
    var original = bind(expression, source._columns)
    if not expected_signature or signature(original) != expected_signature:
        raise Error(
            "Partition operand original AST or logical metadata changed"
        )
    var lifted = lift_literals(original, source)
    # CPU _bins performs this infallible numeric conversion after _full. It
    # receives a positive occurrence rank too; no checked rank0 CAST is needed.
    var converted = lifted.copy() if numeric_state else bind(lifted.expr.cast(DataType.FLOAT64), source._columns)
    var output_dtype = original.dtypes[len(original.dtypes) - 1] if numeric_state else DataType.FLOAT64
    var cloned = ordered_rows.occurrences(converted, 128)
    var rebound = bind(cloned[0].expr, source._columns)
    if signature(rebound) != signature(cloned[0]):
        raise Error("Partition operand binder changed occurrence geometry")
    var plan = _preflight(
        source.clear().lazy().select(cloned[0].expr), source_rows=rows
    )
    if (
        len(plan.steps) != 1
        or len(plan.outputs) != 1
        or plan.reductions
        or plan.limit >= 0
        or plan.steps[0].filter
        or plan.steps[0].order
        or plan.steps[0].grouped
        or plan.steps[0].start != 0
        or plan.steps[0].nodes != len(cloned[1])
        or plan.outputs[0].dtype != output_dtype
    ):
        raise Error("Partition operand private program geometry changed")
    var ranks = cloned[1].copy()
    while len(ranks) < len(plan.literals):
        ranks.append(UInt32(0))
    if len(ranks) != len(plan.literals):
        raise Error("Partition operand auxiliary rank geometry changed")
    plan.ordered_ranks = ArcPointer(ranks^)
    return plan^
