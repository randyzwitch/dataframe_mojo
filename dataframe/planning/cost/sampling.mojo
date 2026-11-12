"""Shared strategy observations, sampling rules, and memory budgets."""
from dataframe.aggregation.expressions import (
    _row_local,
)
from dataframe.expr import (
    Expr,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.kernels.hashing import (
    encode_rows,
)
from dataframe.planning.expression_inputs import (
    _references,
)
from dataframe.planning.model import (
    FILTER,
    PlanNode,
)
from dataframe.storage.retained import RetainedBuffers
from dataframe.series import (
    Series,
)
from std.os import (
    getenv,
)


def _stream_split_groups() -> Int:
    """Groups in the first batch at which streaming aggregation splits its
    state into hash parts. DATAFRAME_STREAM_SPLIT_GROUPS overrides it, so
    tests can exercise the split path on small inputs."""
    var setting = getenv("DATAFRAME_STREAM_SPLIT_GROUPS")
    if setting.byte_length() > 0:
        try:
            return max(1, Int(setting))
        except:
            pass
    return 4096


def _stream_collect_bytes() -> Int:
    """Retained-buffer threshold for reducing collected aggregate batches.

    Default 4 GiB; DATAFRAME_STREAM_COLLECT_BYTES overrides it. Checked
    between worker waves. Input owners, hash/aggregate state, temporary
    allocations and final output can exceed it: this is not a query limit.
    """
    var setting = getenv("DATAFRAME_STREAM_COLLECT_BYTES")
    if setting.byte_length() > 0:
        try:
            return max(1, Int(setting))
        except:
            pass
    return 4 << 30


def _frame_bytes(frame: DataFrame) -> Int:
    """Known buffer bytes retained by a frame, including backing allocations."""
    var usage = RetainedBuffers()
    _retain_frame(frame, usage)
    return usage.bytes


def _retain_frame(frame: DataFrame, mut usage: RetainedBuffers):
    """Add a still-live frame to a shared retained-allocation count."""
    for column in frame._columns:
        column._retained_buffers(usage)


def _keys_repeat(
    input: DataFrame, keys: List[String], operations: List[PlanNode]
) raises -> Bool:
    """Whether the plan's grouping keys repeat: among the rows of eight
    evenly spaced runs of 512 rows of `input` that pass the plan's leading
    row-local filters, at most half as many distinct keys as rows. Runs,
    not single rows, see the repeats a morsel does. False when too few
    rows pass to say (under 64) or a filter reads what `input` lacks."""
    var height = input.height()
    if height == 0:
        return False
    var names = keys.copy()
    var filters = List[Expr]()
    for operation in operations:
        if operation.kind != FILTER or not _row_local(operation.exprs):
            break
        filters.append(operation.exprs[0].copy())
        var reads = _references(operation.exprs[0])
        if not reads:
            return False
        for name in reads.value():
            if name not in names:
                names.append(name)
    for name in names:
        if name not in input.columns():
            return False
    var narrowed = input.select(names)
    var runs = 8
    var length = 512
    var rows = List[Int]()
    if height <= runs * length:
        for r in range(height):
            rows.append(r)
    else:
        for k in range(runs):
            var start = k * (height - length) // (runs - 1)
            for r in range(start, start + length):
                rows.append(r)
    var picked = narrowed.take(rows)
    for f in filters:
        picked = picked.filter(f)
    if picked.height() < 64:
        return False
    var columns = List[Series](capacity=len(keys))
    for name in keys:
        columns.append(picked.column(name))
    var distinct = encode_rows(columns, nulls_equal=True).count()
    return 2 * distinct <= picked.height()
