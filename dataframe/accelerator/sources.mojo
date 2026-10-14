"""Decode file sources once, leaving all query operators for the accelerator."""
from .memory import checked_add, checked_mul
from .metadata import prune_projections
from .source_compaction import compact_source
from dataframe.csv import CsvSchema
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


def source_payload_bytes(frame: DataFrame) raises -> Int:
    """Compacted decoded buffers retained while device operators execute."""
    var total = 0
    for column in frame._columns:
        var rows = len(column)
        var dtype = column.dtype()
        if column._field_metadata:
            total = checked_add(total, len(column._field_metadata.value()[]))
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


def stage_file_sources(
    query: LazyFrame, *, empty: Bool = False, memory_limit_bytes: Int = -1
) raises -> Tuple[LazyFrame, Int, Int]:
    """Read scan nodes only; preserve every expression, filter, join and window.

    Projection analysis narrows decoder columns. Metadata mode uses the CPU
    readers' schema paths, not an eager zero-row decode. The retained source
    payload is charged before GPU submission; reader scratch is independent.
    """
    var staged = query.copy()
    staged._report = None
    _ = file_source_nodes(staged)
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
                # Only defer a value-decode error whose source still has a
                # readable schema. Metadata errors propagate normally; no
                # invented dtype may reach a later binder or device kernel.
                frame = staged._execute(index, True)
                failure = String(error)
        frame = compact_source(frame)
        total = checked_add(total, source_payload_bytes(frame))
        if memory_limit_bytes >= 0 and total > memory_limit_bytes:
            raise Error("Metal file source staging exceeds payload budget")
        staged._frames.append(frame^)
        staged._schemas.append(Optional[CsvSchema]())
        staged._nodes[index] = _plan_node(
            SCAN_FRAME,
            offset=len(staged._frames) - 1,
            source_failure=failure,
        )
    return (staged^, total, len(sources))
