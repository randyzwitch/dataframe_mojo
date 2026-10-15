"""Captured metadata for the single-worker plain floating pipeline route."""
from dataframe import DataFrame, DataType, Expr, Series
from dataframe.binding import bind
from dataframe.expr import COL, SUM, MEAN, MIN, MAX
from dataframe.pipeline import _morsel_ranges, _flush_groups
from .gpu.program import kind


@fieldwise_init
struct FloatStreamPlan(Copyable):
    var ranges: List[UInt32]
    var flush_groups: UInt32
    var rows: Int
    var key_column: Int
    var value_column: Int
    var dtype: Int64
    var op: Int64
    var minimum: UInt32
    var prototype: Series


def prepare(
    source: DataFrame,
    expression: Expr,
    key: String,
    prototype: Series,
    workers: Int,
    morsel_rows: Int,
    *,
    original_source: Bool = False,
    skip_ranges: Bool = False,
) raises -> FloatStreamPlan:
    if not original_source or workers != 1:
        raise Error(
            "Metal unsupported [float stream]: original source and captured single worker required"
        )
    if source.height() <= 0 or source.height() >= (1 << 30):
        raise Error(
            "Metal unsupported [float stream]: nonempty bounded source required"
        )
    var key_column = -1
    for index in range(source.width()):
        if source._columns[index].name() == key:
            key_column = index
    if key_column < 0 or source._columns[key_column].dtype() != DataType.INT32:
        raise Error(
            "Metal unsupported [float stream]: one physical Int32 key required"
        )
    var bound = bind(expression, source._columns)
    if len(bound.expr._nodes) != 2:
        raise Error(
            "Metal unsupported [float stream]: direct column reduction required"
        )
    ref node = bound.expr._nodes[1]
    if (
        node.left != 0
        or bound.expr._nodes[0].op != COL
        or node.op not in [SUM, MEAN, MIN, MAX]
    ):
        raise Error(
            "Metal unsupported [float stream]: plain floating sum/mean/min/max required"
        )
    var dtype = bound.dtypes[0]
    if dtype != DataType.FLOAT32 and dtype != DataType.FLOAT64:
        raise Error("Metal unsupported [float stream]: floating input required")
    if prototype.dtype() != bound.dtypes[1]:
        raise Error(
            "Metal unsupported [float stream]: result prototype differs from CPU binding"
        )
    var ranges = List[UInt32]()
    if not skip_ranges:
        for value in _morsel_ranges(source, max(1, morsel_rows), 1):
            ranges.append(UInt32(value))
    return FloatStreamPlan(
        ranges^,
        UInt32(min(_flush_groups(), Int(UInt32.MAX))),
        source.height(),
        key_column,
        bound.sources[0],
        kind(dtype),
        Int64(node.op),
        UInt32(min(max(0, node.min_count), Int(UInt32.MAX))),
        prototype.slice(0, 0),
    )


# Terminal preparation consumes the original source geometry certificate;
# device projection cannot recompute the CPU morsel boundaries.
from dataframe.lazy import LazyFrame, PlanNode, SCAN_FRAME, AGG
from dataframe.expr import col
from dataframe.accelerator.metadata import schema_frame
from dataframe.accelerator.rows import RowPlan
from dataframe.accelerator.gpu.runtime import _preflight


@fieldwise_init
struct FloatStreamQueryPlan(Movable):
    var source: RowPlan
    var keys: RowPlan
    var feeds: List[FloatStreamPlan]
    var schema: DataFrame
    var names: List[String]
    var scan: PlanNode
    var rows: Int


def prepare_query(
    query: LazyFrame,
    workers: Int,
    batch_size: Int,
    *,
    original_source: Bool = False,
    streaming: Bool = True,
    captured_ranges: List[UInt32] = List[UInt32](),
    captured_flush_groups: Int = -1,
    defer_ranges: Bool = False,
) raises -> FloatStreamQueryPlan:
    var original = query.copy()
    original._reorder(len(original._nodes) - 1)
    original._compact_scan_slots()
    if (
        not original_source
        or not streaming
        or workers != 1
        or len(original._nodes) != 2
        or original._nodes[0].kind != SCAN_FRAME
        or original._nodes[1].kind != AGG
        or original._nodes[1].left != 0
        or not original._nodes[1].maintain_order
        or len(original._nodes[1].names) != 1
        or not len(original._nodes[1].exprs)
    ):
        raise Error(
            "Metal unsupported [float stream]: direct ordered terminal aggregate with captured single-worker streaming required"
        )
    ref scan = original._nodes[0]
    if scan.source_failure:
        raise Error(scan.source_failure)
    if len(scan.names2):
        raise Error(
            "Metal unsupported [float stream]: rewritten source provenance is not certified"
        )
    var frame = original._frames[scan.offset].copy()
    if len(scan.names):
        frame = frame.select(scan.names)
    var expected = schema_frame(original, 1).clear()
    ref operation = original._nodes[1]
    var feeds = List[FloatStreamPlan]()
    var needed = operation.names.copy()
    for at in range(len(operation.exprs)):
        var feed = prepare(
            frame,
            operation.exprs[at],
            operation.names[0],
            expected._columns[1 + at],
            workers,
            batch_size,
            original_source=True,
            skip_ranges=defer_ranges or len(captured_ranges) > 0,
        )
        if len(captured_ranges) and not defer_ranges:
            feed.ranges = captured_ranges.copy()
        if captured_flush_groups >= 0:
            if captured_flush_groups < 1:
                raise Error(
                    "Grouped float stream captured flush threshold must be positive"
                )
            feed.flush_groups = UInt32(
                min(captured_flush_groups, Int(UInt32.MAX))
            )
        var name = frame._columns[feed.value_column].name()
        if name not in needed:
            needed.append(name)
        feeds.append(feed^)
    # Input projection uses original columns; preserve the source's captured
    # morsel ranges above even if unused columns have different chunking.
    var selected = frame.select(needed)
    var columns = List[Expr]()
    for field in selected._columns:
        columns.append(col(field.name()))
    for at in range(len(feeds)):
        var name = frame._columns[feeds[at].value_column].name()
        feeds[at].key_column = 0
        for column in range(selected.width()):
            if selected._columns[column].name() == name:
                feeds[at].value_column = column
    var source = _preflight(selected.lazy().select_exprs(columns))
    var keys = _preflight(
        selected.clear()
        .lazy()
        .group_by(operation.names, maintain_order=True)
        .agg(List[Expr]()),
        source_rows=frame.height(),
    )
    return FloatStreamQueryPlan(
        source^,
        keys^,
        feeds^,
        expected^,
        operation.names.copy(),
        scan.copy(),
        frame.height(),
    )
