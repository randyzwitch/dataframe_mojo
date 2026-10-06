"""Grouped reductions that read their values through row numbers.

The partitioned group-by (`group_by.partitioned`) orders row numbers by key
hash into buckets and, before #426, gathered every aggregated column into
that order so each bucket could reduce a contiguous slice. For a plain
reduction of a column that copy is not needed: the bucket reads each value
at its source row (`order[lo + p]`) and updates its group's state in place,
as Polars' and DuckDB's hash aggregates update states from the rows they
are handed. The states and `finish` are `Reducer`'s, so results are those
of the gathered path.

Plain numeric reductions update shared Reducer states directly, including
first/last, moments and distinct counts. A median or quantile lays the
bucket's values out in group order in one buffer and selects within each
group's segment. Decimal and computed inputs use bounded batches through
the ordinary evaluator.
"""
from std.math import isnan
from std.memory import Pointer, bitcast

from .aggregate import Reducer, float_key, quantile_in
from .binding import BoundExpr, bind
from .column import Column
from .execution import _new_reducer, _batch, evaluate
from .expr import (
    COUNT,
    LEN,
    MAX,
    MEAN,
    MEDIAN,
    MIN,
    QUANTILE,
    SUM,
    COL,
    FIRST,
    LAST,
    STD,
    VAR,
    N_UNIQUE,
    NULL_COUNT,
    col,
    is_reduction,
    is_pair_reduction,
    subtree,
)
from .dtype import DataType, NUMERIC_DTYPES
from .series import Series


def _supported(bound: BoundExpr, columns: List[Series]) -> Bool:
    ref nodes = bound.expr._nodes
    if len(nodes) != 2:
        return False
    ref node = nodes[1]
    if node.op not in [
        SUM,
        MEAN,
        MIN,
        MAX,
        COUNT,
        LEN,
        FIRST,
        LAST,
        STD,
        VAR,
        N_UNIQUE,
        NULL_COUNT,
        MEDIAN,
        QUANTILE,
    ]:
        return False
    if nodes[node.left].op != COL:
        return False
    ref source = columns[bound.sources[node.left]]
    if source.dtype().is_decimal() or source.dtype().is_categorical():
        return False
    if (
        node.op == MEDIAN or node.op == QUANTILE
    ) and source.dtype() == DataType.of(DType.uint64):
        # The batch route re-encodes UInt64 before taking positions.
        return False
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if source._data.isa[Column[Scalar[D]]]():
            return True
    return False


def indexed_reductions(bound: List[BoundExpr], columns: List[Series]) -> Bool:
    """Whether every expression is a reduction `reduce_indexed` serves."""
    if len(bound) == 0:
        return False
    for expression in bound:
        if not _supported(expression, columns):
            return False
    return True


def reduce_indexed(
    bound: BoundExpr,
    columns: List[Series],
    rows: Pointer[Int, _],
    ids: List[Int],
    group_count: Int,
) raises -> Series:
    """Reduce rows `rows[0..len(ids))` of unchunked `columns` into
    `group_count` groups, row p into group `ids[p]`."""
    ref node = bound.expr._nodes[1]
    var reducer = _new_reducer(bound, node, group_count)
    var n = len(ids)
    var g = ids.unsafe_ptr()
    var op = node.op
    if op == LEN:
        var counts = reducer.counts.unsafe_ptr()
        for p in range(n):
            counts[unsafe_offset=g[unsafe_offset=p]] += 1
        return reducer.finish()
    ref source = columns[bound.sources[node.left]]
    if op == N_UNIQUE:
        comptime for t in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[t]
            if source._data.isa[Column[Scalar[D]]]():
                return _distinct_counts[D](
                    source._data[Column[Scalar[D]]], rows, g, n, group_count
                )
    if op == MEDIAN or op == QUANTILE:
        comptime for t in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[t]
            if source._data.isa[Column[Scalar[D]]]():
                return _quantiles[D](
                    source._data[Column[Scalar[D]]],
                    rows,
                    g,
                    n,
                    group_count,
                    node.floating,
                    node.text,
                ).with_dtype(bound.dtypes[len(bound.dtypes) - 1])
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if source._data.isa[Column[Scalar[D]]]():
            comptime if D.is_floating_point():
                _reduce_floats[D](
                    reducer, source._data[Column[Scalar[D]]], rows, g, n, op
                )
            else:
                _reduce_ints[D](
                    reducer, source._data[Column[Scalar[D]]], rows, g, n, op
                )
    return reducer.finish().with_dtype(bound.dtypes[len(bound.dtypes) - 1])


@always_inline
def _mix_pair(group: Int, key: UInt64) -> UInt64:
    """splitmix64's finalizer over a group and value together."""
    var z = key ^ (UInt64(group) * 0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _distinct_counts[
    D: DType
](
    column: Column[Scalar[D]],
    rows: Pointer[Int, _],
    g: Pointer[Int, _],
    n: Int,
    group_count: Int,
) raises -> Series:
    """Distinct values per group through one open-addressing set of
    (group, value) pairs for the whole bucket, where a set per group cost
    an allocation for every group with a second value: PDS-H q21 counts
    suppliers for 1.5M orders, and those sets were a fifth of its time.
    A null counts once per group, as `n_unique` counts it."""
    var values = column._ptr()
    var nulls = column.null_count() > 0
    var counts = List[Int64](length=group_count, fill=0)
    var saw_null = List[Bool](length=group_count, fill=False)
    # At most n distinct pairs; keep the table at most half full.
    var size = 16
    while size < 2 * n:
        size *= 2
    var mask = size - 1
    var keys = List[UInt64](unsafe_uninit_length=size)
    var groups = List[Int](length=size, fill=-1)
    var slot_keys = keys.unsafe_ptr()
    var slot_groups = groups.unsafe_ptr()
    var tally = counts.unsafe_ptr()
    for p in range(n):
        var row = rows[unsafe_offset=p]
        var group = g[unsafe_offset=p]
        if nulls and not column._valid(row):
            saw_null[group] = True
            continue
        var key: UInt64
        comptime if D.is_floating_point():
            key = float_key(Float64(values[unsafe_offset=row]))
        else:
            key = bitcast[DType.uint64](
                _canonical_int[D](values[unsafe_offset=row])
            )
        var at = Int(_mix_pair(group, key)) & mask
        while True:
            var held = slot_groups[unsafe_offset=at]
            if held < 0:
                slot_groups[unsafe_offset=at] = group
                slot_keys[unsafe_offset=at] = key
                tally[unsafe_offset=group] += 1
                break
            if held == group and slot_keys[unsafe_offset=at] == key:
                break
            at = (at + 1) & mask
    for group in range(group_count):
        if saw_null[group]:
            counts[group] += 1
    return Series("", Column[Int64](counts^))


def _quantiles[
    D: DType
](
    column: Column[Scalar[D]],
    rows: Pointer[Int, _],
    g: Pointer[Int, _],
    n: Int,
    group_count: Int,
    q: Float64,
    method: String,
) raises -> Series:
    """Each group's quantile of its non-null values, as `quantile_in` takes
    it. The values go into one buffer in group order (a count per group,
    then a cursor per group), where a list per group grew by appends and
    was copied again for selection: H2O q6 takes medians over 10,000
    groups of 1,000 rows."""
    var nulls = column.null_count() > 0
    var values = column._ptr()
    var starts = List[Int](length=group_count + 1, fill=0)
    var cursor = starts.unsafe_ptr()
    for p in range(n):
        if nulls and not column._valid(rows[unsafe_offset=p]):
            continue
        cursor[unsafe_offset=g[unsafe_offset=p] + 1] += 1
    for group in range(group_count):
        cursor[unsafe_offset=group + 1] += cursor[unsafe_offset=group]
    var total = cursor[unsafe_offset=group_count]
    var buffer = List[Float64](unsafe_uninit_length=total)
    var out = buffer.unsafe_ptr()
    # Fill from each group's start; afterwards cursor[group] is the end of
    # group `group`, the start of the next.
    for p in range(n):
        var row = rows[unsafe_offset=p]
        if nulls and not column._valid(row):
            continue
        var group = g[unsafe_offset=p]
        out[unsafe_offset=cursor[unsafe_offset=group]] = values[
            unsafe_offset=row
        ].cast[DType.float64]()
        cursor[unsafe_offset=group] += 1
    var output = List[Float64](length=group_count, fill=0)
    var valid = List[Bool](length=group_count, fill=False)
    var start = 0
    for group in range(group_count):
        var end = cursor[unsafe_offset=group]
        var result = quantile_in(Span(buffer)[start:end], q, method)
        if result:
            output[group] = result.value()
            valid[group] = True
        start = end
    return Series("", Column[Float64](output^, valid^))


def _canonical_int[D: DType](value: Scalar[D]) -> Int64:
    comptime if D == DType.uint64:
        return (
            value.cast[DType.uint64]() ^ UInt64(0x8000_0000_0000_0000)
        ).cast[DType.int64]()
    return value.cast[DType.int64]()


def _reduce_ints[
    D: DType
](
    mut reducer: Reducer,
    column: Column[Scalar[D]],
    rows: Pointer[Int, _],
    g: Pointer[Int, _],
    n: Int,
    op: Int,
):
    var values = column._ptr()
    var nulls = column.null_count() > 0
    if op == COUNT or op == NULL_COUNT:
        var counts = reducer.counts.unsafe_ptr()
        for p in range(n):
            var valid = not nulls or column._valid(rows[unsafe_offset=p])
            if valid != (op == NULL_COUNT):
                counts[unsafe_offset=g[unsafe_offset=p]] += 1
    elif op == SUM or op == MEAN:
        var sums = reducer.int_sums.unsafe_ptr()
        for p in range(n):
            var row = rows[unsafe_offset=p]
            if nulls and not column._valid(row):
                continue
            ref state = sums[unsafe_offset=g[unsafe_offset=p]]
            state.total += values[unsafe_offset=row].cast[DType.int128]()
            state.count += 1
    elif op == FIRST or op == LAST:
        for p in range(n):
            var group = g[unsafe_offset=p]
            if op == LAST or not reducer.seen[group]:
                var row = rows[unsafe_offset=p]
                reducer.seen[group] = True
                reducer.picked_valid[group] = column._valid(row)
                reducer.ints[group] = _canonical_int[D](
                    values[unsafe_offset=row]
                )
    elif op == STD or op == VAR:
        for p in range(n):
            var row = rows[unsafe_offset=p]
            if not nulls or column._valid(row):
                reducer.moments[g[unsafe_offset=p]].add(
                    Float64(_canonical_int[D](values[unsafe_offset=row]))
                )
    elif op == N_UNIQUE:
        for p in range(n):
            var row = rows[unsafe_offset=p]
            var group = g[unsafe_offset=p]
            if not nulls or column._valid(row):
                reducer._add_distinct(
                    group,
                    bitcast[DType.uint64](
                        _canonical_int[D](values[unsafe_offset=row])
                    ),
                )
            else:
                reducer.picked_valid[group] = True
    else:
        var is_max = op == MAX
        var seen = reducer.seen.unsafe_ptr()
        var best = reducer.ints.unsafe_ptr()
        for p in range(n):
            var row = rows[unsafe_offset=p]
            if nulls and not column._valid(row):
                continue
            var group = g[unsafe_offset=p]
            var value = _canonical_int[D](values[unsafe_offset=row])
            # Ties keep the earlier value, as `_extreme` does.
            if (
                not seen[unsafe_offset=group]
                or (is_max and value > best[unsafe_offset=group])
                or (not is_max and value < best[unsafe_offset=group])
            ):
                best[unsafe_offset=group] = value
                seen[unsafe_offset=group] = True


def _reduce_floats[
    D: DType
](
    mut reducer: Reducer,
    column: Column[Scalar[D]],
    rows: Pointer[Int, _],
    g: Pointer[Int, _],
    n: Int,
    op: Int,
):
    var values = column._ptr()
    var nulls = column.null_count() > 0
    if op == COUNT or op == NULL_COUNT:
        var counts = reducer.counts.unsafe_ptr()
        for p in range(n):
            var valid = not nulls or column._valid(rows[unsafe_offset=p])
            if valid != (op == NULL_COUNT):
                counts[unsafe_offset=g[unsafe_offset=p]] += 1
    elif op == SUM or op == MEAN:
        var sums = reducer.float_sums.unsafe_ptr()
        for p in range(n):
            var row = rows[unsafe_offset=p]
            if nulls and not column._valid(row):
                continue
            ref state = sums[unsafe_offset=g[unsafe_offset=p]]
            state.total += Float64(values[unsafe_offset=row])
            state.count += 1
    elif op == FIRST or op == LAST:
        for p in range(n):
            var group = g[unsafe_offset=p]
            if op == LAST or not reducer.seen[group]:
                var row = rows[unsafe_offset=p]
                reducer.seen[group] = True
                reducer.picked_valid[group] = column._valid(row)
                reducer.floats[group] = Float64(values[unsafe_offset=row])
    elif op == STD or op == VAR:
        for p in range(n):
            var row = rows[unsafe_offset=p]
            if not nulls or column._valid(row):
                reducer.moments[g[unsafe_offset=p]].add(
                    Float64(Float64(values[unsafe_offset=row]))
                )
    elif op == N_UNIQUE:
        for p in range(n):
            var row = rows[unsafe_offset=p]
            var group = g[unsafe_offset=p]
            if not nulls or column._valid(row):
                reducer._add_distinct(
                    group, float_key(Float64(values[unsafe_offset=row]))
                )
            else:
                reducer.picked_valid[group] = True
    else:
        var is_max = op == MAX
        var seen = reducer.seen.unsafe_ptr()
        var nan_seen = reducer.nan_seen.unsafe_ptr()
        var best = reducer.floats.unsafe_ptr()
        for p in range(n):
            var row = rows[unsafe_offset=p]
            if nulls and not column._valid(row):
                continue
            var group = g[unsafe_offset=p]
            var value = Float64(values[unsafe_offset=row])
            if isnan(value):
                nan_seen[unsafe_offset=group] = True
            elif (
                not seen[unsafe_offset=group]
                or (is_max and value > best[unsafe_offset=group])
                or (not is_max and value < best[unsafe_offset=group])
            ):
                best[unsafe_offset=group] = value
                seen[unsafe_offset=group] = True


def reduce_indexed_batches(
    bound: BoundExpr,
    columns: List[Series],
    rows: Pointer[Int, _],
    ids: List[Int],
    group_count: Int,
    batch_size: Int,
) raises -> Series:
    """Fold bounded gathers through the ordinary reducers, preserving their
    dtype, overflow, null and order semantics. No full value-column gather.
    The caller supplies the same group IDs to every aggregate in a bucket.
    """
    var sources = List[Series]()
    for c in range(len(columns)):
        var used = False
        for i in range(len(bound.expr._nodes)):
            if bound.expr._nodes[i].op == COL and bound.sources[i] == c:
                used = True
        if used:
            sources.append(columns[c].copy())
    var local = bind(bound.expr, sources)
    var rewritten = bound.expr.copy()
    var results = List[Series]()
    var reducers = List[Reducer]()
    var positions = List[Int]()
    for i in range(len(local.expr._nodes)):
        if is_reduction(local.expr._nodes[i].op):
            positions.append(i)
            reducers.append(
                _new_reducer(local, local.expr._nodes[i], group_count)
            )
    for start in range(0, len(ids), max(1, batch_size)):
        var length = min(max(1, batch_size), len(ids) - start)
        var selected = List[Int](capacity=length)
        for p in range(start, start + length):
            selected.append(rows[unsafe_offset=p])
        var batch = List[Series](capacity=len(sources))
        for source in sources:
            batch.append(source.take(selected))
        for r in range(len(reducers)):
            ref node = local.expr._nodes[positions[r]]
            var chunk = _batch[8](
                local, batch, List[Series](), node.left, 0, length, False
            )
            if is_pair_reduction(node.op):
                var other = _batch[8](
                    local, batch, List[Series](), node.right, 0, length, False
                )
                reducers[r].update_pair(chunk, other, start, True, ids)
            else:
                reducers[r].update(chunk, start, True, ids)
    for r in range(len(reducers)):
        var i = positions[r]
        var name = "__indexed_reduction_" + String(i)
        results.append(
            reducers[r].finish().with_dtype(local.dtypes[i]).renamed(name)
        )
        rewritten._nodes[i] = col(name)._nodes[0].copy()
    var output = subtree(rewritten, len(rewritten._nodes) - 1)
    return evaluate(bind(output, results), results, group_count)
