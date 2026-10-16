"""Bound checked scalar operands for the exact serial grouped feed route."""
from std.memory import ArcPointer
from dataframe import DataFrame, DataType, Expr, Series, when
from dataframe.binding import bind, ROWS, BoundExpr
from dataframe.expr import (
    CAST,
    ADD,
    SUB,
    MUL,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    NEG,
    ABS,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    DT_OFFSET_BY,
    DT_TRUNCATE,
    DT_REPLACE_TZ,
    ANY,
    ALL,
    is_reduction,
    subtree,
)
from dataframe.lazy import LazyFrame, SCAN_FRAME, AGG, SELECT, SLICE
from .rows import RowPlan
from .integer_ranges import proven_integer_nodes
from . import ordered_rows, temporal_expressions
from .gpu.runtime import _preflight
from .gpu.program import kind
from .gpu.decimal_descriptors import reduction_op, reduction_parameter
from .gpu.grouped_feed import admitted


@fieldwise_init
struct FeedPlan(Copyable):
    var operand: ArcPointer[RowPlan]
    var prototype: Series
    var op: Int64
    var input_kind: Int64
    var output_kind: Int64
    var parameter: Int64
    var rows: Int
    var batch_size: Int


def checked_operand(bound: BoundExpr, top: Int) -> Int:
    var checked = 0
    var proven = proven_integer_nodes(bound)
    for at in range(top):
        ref child = bound.expr._nodes[at]
        if child.op == CAST and child.integer and not proven[at]:
            checked += 1
        elif (
            not proven[at]
            and (
                bound.dtypes[at].is_integer()
                or bound.dtypes[at].is_decimal()
                or bound.dtypes[at].is_temporal()
            )
            and child.op
            in List[Int]([ADD, SUB, MUL, DIV, FLOORDIV, MOD, POW, NEG, ABS])
        ):
            checked += 1
        elif child.op in List[Int]([DT_OFFSET_BY, DT_TRUNCATE, DT_REPLACE_TZ]):
            checked += 1
        elif child.op in List[Int]([EQ, NE, LT, LE, GT, GE]) and (
            (child.left >= 0 and bound.dtypes[child.left].is_decimal())
            or (child.right >= 0 and bound.dtypes[child.right].is_decimal())
        ):
            checked += 1
    return checked


def prepare_feed(
    expression: Expr,
    source: DataFrame,
    prototype: Series,
    rows: Int,
    *,
    ordered_source: Bool = False,
) raises -> FeedPlan:
    if not ordered_source:
        raise Error(
            "Metal unsupported [grouped feed]: original raw source provenance is not proved"
        )
    if rows < 0 or rows >= 32768:
        raise Error(
            "Metal unsupported [grouped feed]: exact serial grouped feed route required"
        )
    var bound = bind(expression, source._columns)
    var top = len(bound.expr._nodes) - 1
    ref node = bound.expr._nodes[top]
    if (
        not is_reduction(node.op)
        or node.left < 0
        or bound.shapes[node.left] != ROWS
    ):
        raise Error(
            "Metal unsupported [grouped feed]: one row-valued reduction operand required"
        )
    var input_dtype = bound.dtypes[node.left]
    var operation = reduction_op(Int64(node.op), input_dtype)
    if not admitted(operation, kind(input_dtype), kind(prototype.dtype())):
        raise Error(
            "Metal unsupported [grouped feed]: reduction state association is not certified"
        )
    var operand = bind(
        subtree(bound.expr, node.left).alias(expression._name), source._columns
    )
    if not ordered_rows.eligible(operand) or temporal_expressions.contains(
        operand
    ):
        raise Error(
            "Metal unsupported [grouped feed]: pure numeric checked operand required"
        )
    var cloned = ordered_rows.occurrences(operand, 128)
    var plan = _preflight(
        source.clear().lazy().select(cloned[0].expr), source_rows=rows
    )
    if not len(plan.steps):
        # A plain COL is a zero-step projection. Force an infallible typed
        # identity program so the private feed still owns a distinct matrix.
        # Nulls choose the same original value, including hidden payload; no
        # arithmetic/cast is introduced. Source errors are proved zero before
        # any rank interpretation, and output metadata remains the prototype.
        var identity = (
            when(operand.expr.is_null())
            .then(operand.expr.copy())
            .otherwise(operand.expr.copy())
            .alias(expression._name)
        )
        operand = bind(identity, source._columns)
        cloned = ordered_rows.occurrences(operand, 128)
        plan = _preflight(
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
        or plan.steps[0].nodes != len(cloned[1])
    ):
        raise Error(
            "Metal unsupported [grouped feed]: private operand geometry changed"
        )
    var ranks = cloned[1].copy()
    while len(ranks) < len(plan.literals):
        ranks.append(UInt32(0))
    if len(ranks) != len(plan.literals):
        raise Error(
            "Metal unsupported [grouped feed]: private operand rank geometry changed"
        )
    plan.ordered_ranks = ArcPointer(ranks^)
    return FeedPlan(
        ArcPointer(plan^),
        prototype.slice(0, 0),
        operation,
        kind(input_dtype),
        kind(prototype.dtype()),
        reduction_parameter(
            input_dtype, node.min_count
        ) if input_dtype.is_decimal() else Int64(node.min_count) if node.op
        != ANY
        and node.op != ALL else node.integer,
        rows,
        1024,
    )


def has_checked_grouped_feeds(
    query: LazyFrame, *, eager_feed: Bool = False
) raises -> Bool:
    """Select only the new small direct basic-state family, without data reads."""
    if not eager_feed:
        return False
    var original = query.copy()
    original._reorder(len(original._nodes) - 1)
    original._compact_scan_slots()
    var root = len(original._nodes) - 1
    while root >= 0 and original._nodes[root].kind == SLICE:
        root = original._nodes[root].left
    if root >= 0 and original._nodes[root].kind == SELECT:
        root = original._nodes[root].left
    if (
        root < 0
        or original._nodes[root].kind != AGG
        or original._nodes[root].left != 0
        or original._nodes[0].kind != SCAN_FRAME
    ):
        return False
    ref frame = original._frames[original._nodes[0].offset]
    if frame.height() >= 32768:
        return False
    # The precision-only original GROUP observer already covers this family.
    # Resolve source-independent typed text constants before choosing a feed;
    # otherwise their STRING leaves look like transported text operands.
    # The exact issuer proof includes source-prefix and key restrictions.
    from .constants import fold_literal_text_parsing
    from .decimal_group_feed import producer as decimal_group_producer

    if not original._nodes[0].source_failure:
        var canonical = original.copy()
        var canonicalized = True
        try:
            canonical = fold_literal_text_parsing(original)
        except:
            # A failed metadata bridge proves no route preference. Leave the
            # original query and the legacy adapter's diagnostics untouched.
            canonicalized = False
        if canonicalized and decimal_group_producer(canonical, root):
            return False
    var needed = False
    for expression in original._nodes[root].exprs:
        var bound = bind(expression, frame._columns)
        var top = len(bound.expr._nodes) - 1
        ref node = bound.expr._nodes[top]
        if not is_reduction(node.op) or node.left < 0:
            return False
        var dtype = bound.dtypes[node.left]
        var target = bound.dtypes[top]
        if not (
            dtype.is_numeric() or dtype == DataType.BOOL or dtype.is_temporal()
        ) or not (
            target.is_numeric()
            or target == DataType.BOOL
            or target.is_temporal()
        ):
            return False
        if not admitted(
            reduction_op(Int64(node.op), dtype),
            kind(dtype),
            kind(bound.dtypes[top]),
        ):
            return False
        needed |= checked_operand(bound, top) != 0
    return needed
