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
first/last, moments and distinct counts. Decimal and computed inputs use
bounded batches through the ordinary evaluator. Median/quantile retain
the gathered route selected per expression by the partitioned caller.
"""
from std.math import isnan
from std.memory import Pointer, bitcast

from .aggregate import Reducer, float_key
from .binding import BoundExpr, bind
from .column import Column
from .execution import _new_reducer, _batch, evaluate
from .expr import (
    COUNT,
    LEN,
    MAX,
    MEAN,
    MIN,
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
    ]:
        return False
    if nodes[node.left].op != COL:
        return False
    ref source = columns[bound.sources[node.left]]
    if source.dtype().is_decimal() or source.dtype().is_categorical():
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
