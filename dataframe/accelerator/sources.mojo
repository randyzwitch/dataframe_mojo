"""Decode file sources once, leaving all query operators for the accelerator."""
from .memory import checked_add, checked_mul
from .metadata import prune_projections
from .source_compaction import compact_source
from dataframe.csv import CsvSchema
from dataframe.bool_column import BoolColumn
from dataframe.dtype import DataType
from dataframe.frame import DataFrame
from dataframe.lazy import (
    LazyFrame,
    SCAN_CSV,
    SCAN_PARQUET,
    SCAN_FRAME,
    _plan_node,
)


def file_source_nodes(query: LazyFrame) raises -> List[Int]:
    """Find reachable file scans in CPU left-before-right observation order."""
    var sources = List[Int]()
    if len(query._nodes) == 0:
        return sources^
    var seen = List[Bool](length=len(query._nodes), fill=False)
    var pending = List[Int]([len(query._nodes) - 1])
    while len(pending):
        var index = pending.pop()
        if index < 0 or index >= len(query._nodes):
            raise Error("Metal unsupported [plan]: invalid source child index")
        if seen[index]:
            continue
        seen[index] = True
        ref node = query._nodes[index]
        if node.kind == SCAN_CSV or node.kind == SCAN_PARQUET:
            sources.append(index)
            continue
        for child in [node.right, node.left]:
            if child < 0:
                continue
            if child >= index:
                raise Error(
                    "Metal unsupported [plan]: sources require a topological DAG"
                )
            pending.append(child)
    return sources^


def source_prefixes(
    query: LazyFrame, failed_source: Int
) raises -> List[LazyFrame]:
    """Closed subqueries before a source in left-before-right execution order.

    This structural discovery performs no schema probes or value evaluation.
    The caller must first preserve any earlier CPU planning-time metadata
    failure; these prefixes describe eager execution, not optimizer timing.
    Previously completed shared inputs remain before later consumers, so an
    error in one cannot move behind an unrelated later operand.
    """
    if failed_source < 0 or failed_source >= len(query._nodes):
        raise Error("Metal source failure index is outside the execution DAG")
    if query._nodes[failed_source].kind not in [SCAN_CSV, SCAN_PARQUET]:
        raise Error("Metal source failure must name an original file scan")
    _ = file_source_nodes(query)
    var seen = List[Bool](length=len(query._nodes), fill=False)
    var pending = List[Tuple[Int, Int]]()
    pending.append((len(query._nodes) - 1, -1))
    var frontier = List[Int]()
    var found = False
    while len(pending):
        var item = pending.pop()
        var index = item[0]
        var first = item[1]
        if first >= 0:
            # Collapse only work completed after this node was entered.
            # A shared child visited earlier belongs to its earlier boundary.
            while len(frontier) > first:
                _ = frontier.pop()
            frontier.append(index)
            continue
        if index == failed_source:
            found = True
            break
        if seen[index]:
            continue
        seen[index] = True
        pending.append((index, len(frontier)))
        for child in [query._nodes[index].right, query._nodes[index].left]:
            if child >= 0:
                pending.append((child, -1))
    if not found:
        raise Error("Metal source failure is not reachable from the query root")
    var prefixes = List[LazyFrame](capacity=len(frontier))
    for root in frontier:
        var prefix = query.copy()
        prefix._report = None
        prefix._reorder(root)
        # _reorder trims nodes only. Future/unreachable frames must not enter
        # this prefix's dictionary discovery or consume its staging budget.
        for node_index in range(len(prefix._nodes)):
            ref node = prefix._nodes[node_index]
            if node.kind not in [SCAN_FRAME, SCAN_CSV, SCAN_PARQUET]:
                continue
            var source = node.offset
            if (
                source < 0
                or source >= len(prefix._frames)
                or source >= len(prefix._schemas)
            ):
                raise Error("Metal source prefix has an invalid scan slot")
        # The shared helper also remaps measured source cardinalities.
        prefix._compact_scan_slots()
        prefixes.append(prefix^)
    return prefixes^


def source_payload_bytes(frame: DataFrame) raises -> Int:
    """Compacted decoded buffers retained while device operators execute."""
    var total = 0
    for column in frame._columns:
        var rows = len(column)
        var dtype = column.dtype()
        if column._field_metadata:
            total = checked_add(total, len(column._field_metadata.value()[]))
        if dtype == DataType.BOOL:
            # Compact Bool values and validity are both packed bitmaps; an
            # omitted all-valid bitmap retains no payload bytes at all.
            ref booleans = column._data[BoolColumn]
            total = checked_add(total, len(booleans._data[]))
            total = checked_add(total, len(booleans._bits[]))
            continue
        total = checked_add(total, checked_add(rows, 7) // 8)
        if dtype == DataType.STRING or dtype._is_bytes():
            total = checked_add(total, column._text_bytes())
            total = checked_add(total, checked_mul(checked_add(rows, 1), 8))
        else:
            total = checked_add(
                total,
                checked_mul(rows, max(1, dtype.physical().bit_width() // 8)),
            )
            if dtype.is_categorical():
                total = checked_add(total, len(dtype.dictionary()[].bytes))
                total = checked_add(
                    total, checked_mul(len(dtype.dictionary()[].offsets), 8)
                )
    return total


@fieldwise_init
struct PreparedFileSources(Movable):
    """Staged scans, or an unavailable schema with earlier closed prefixes."""

    var query: LazyFrame
    var staging_bytes: Int
    var scan_count: Int
    var failed_source: Int
    var failure: String
    var prefixes: List[LazyFrame]


def prepare_file_sources(
    query: LazyFrame,
    *,
    empty: Bool = False,
    memory_limit_bytes: Int = -1,
    optimize: Bool = True,
) raises -> PreparedFileSources:
    """Read scan nodes only; preserve every expression, filter, join and window.

    With optimization enabled, projection analysis narrows decoder columns.
    Disabling optimization preserves eager source decode and prefix errors
    before any later optimizer schema probe. Metadata mode uses the CPU
    readers' schema paths, not an eager zero-row decode. The retained source
    payload is charged before GPU submission; reader scratch is independent.
    """
    var staged = query.copy()
    staged._report = None
    _ = file_source_nodes(staged)
    # CPU predicate/projection planning can discover a later schema before
    # evaluating an earlier branch. Preserve those failures before arranging
    # any observation-only GPU prefix execution. Trim dead nodes first: no
    # unreachable source is part of the query's metadata observation order.
    var planning = staged.copy()
    if optimize and len(planning._nodes):
        planning._reorder(len(planning._nodes) - 1)
        _ = planning._optimized()
    if optimize:
        staged._push_slices()
        prune_projections(staged)
    var sources = file_source_nodes(staged)
    var total = 0
    for index in sources:
        var frame: DataFrame
        var failure = String()
        if empty:
            frame = staged._execute(index, True)
        else:
            try:
                frame = staged._execute(index, False)
            except error:
                # A readable schema permits a typed failed-scan marker.
                # Otherwise return only closed earlier queries; no invented
                # dtype may reach a later binder or device kernel.
                failure = String(error)
                try:
                    frame = staged._execute(index, True)
                except:
                    # The failed scan remains untouched. Earlier closed
                    # branches can bind real schemas, observe their device
                    # errors, then raise this source's original failure.
                    var prefixes = source_prefixes(staged, index)
                    return PreparedFileSources(
                        staged^,
                        total,
                        len(sources),
                        index,
                        failure,
                        prefixes^,
                    )
        frame = compact_source(frame)
        total = checked_add(total, source_payload_bytes(frame))
        if memory_limit_bytes >= 0 and total > memory_limit_bytes:
            raise Error("Metal file source staging exceeds payload budget")
        staged._frames.append(frame^)
        staged._schemas.append(Optional[CsvSchema]())
        staged._nodes[index] = _plan_node(
            SCAN_FRAME,
            offset=len(staged._frames) - 1,
            names2=["metal_staged_file_source"],
            source_failure=failure,
        )
    return PreparedFileSources(
        staged^, total, len(sources), -1, "", List[LazyFrame]()
    )


def stage_file_sources(
    query: LazyFrame,
    *,
    empty: Bool = False,
    memory_limit_bytes: Int = -1,
    optimize: Bool = True,
) raises -> Tuple[LazyFrame, Int, Int]:
    """Compatibility wrapper; unavailable schemas raise without GPU observers."""
    var prepared = prepare_file_sources(
        query,
        empty=empty,
        memory_limit_bytes=memory_limit_bytes,
        optimize=optimize,
    )
    if prepared.failure:
        raise Error(prepared.failure)
    var bytes = prepared.staging_bytes
    var count = prepared.scan_count
    return (prepared.query.copy(), bytes, count)
