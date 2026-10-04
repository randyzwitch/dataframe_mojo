"""Grouped reductions that read their values through row numbers.

The partitioned group-by (`group_by.partitioned`) orders row numbers by key
hash into buckets and, before #426, gathered every aggregated column into
that order so each bucket could reduce a contiguous slice. For a plain
reduction of a column that copy is not needed: the bucket reads each value
at its source row (`order[lo + p]`) and updates its group's state in place,
as Polars' and DuckDB's hash aggregates update states from the rows they
are handed. The states and `finish` are `Reducer`'s, so results are those
of the gathered path.

This serves SUM, MEAN, MIN, MAX, COUNT and LEN of one Int64 or Float64
column, the set `hash_agg.mojo` merges; other aggregations keep the gather.
"""
from std.math import isnan
from std.memory import Pointer

from .aggregate import Reducer
from .binding import BoundExpr
from .column import Column
from .execution import _new_reducer
from .expr import COUNT, LEN, MAX, MEAN, MIN, SUM
from .hash_agg import _supported
from .series import Series


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
    if source._data.isa[Column[Int64]]():
        _reduce_ints(reducer, source._data[Column[Int64]], rows, g, n, op)
    else:
        _reduce_floats(reducer, source._data[Column[Float64]], rows, g, n, op)
    return reducer.finish()


def _reduce_ints(
    mut reducer: Reducer,
    column: Column[Int64],
    rows: Pointer[Int, _],
    g: Pointer[Int, _],
    n: Int,
    op: Int,
):
    var values = column._ptr()
    var nulls = column.null_count() > 0
    if op == COUNT:
        var counts = reducer.counts.unsafe_ptr()
        for p in range(n):
            if nulls and not column._valid(rows[unsafe_offset=p]):
                continue
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
    else:
        var is_max = op == MAX
        var seen = reducer.seen.unsafe_ptr()
        var best = reducer.ints.unsafe_ptr()
        for p in range(n):
            var row = rows[unsafe_offset=p]
            if nulls and not column._valid(row):
                continue
            var group = g[unsafe_offset=p]
            var value = values[unsafe_offset=row]
            # Ties keep the earlier value, as `_extreme` does.
            if (
                not seen[unsafe_offset=group]
                or (is_max and value > best[unsafe_offset=group])
                or (not is_max and value < best[unsafe_offset=group])
            ):
                best[unsafe_offset=group] = value
                seen[unsafe_offset=group] = True


def _reduce_floats(
    mut reducer: Reducer,
    column: Column[Float64],
    rows: Pointer[Int, _],
    g: Pointer[Int, _],
    n: Int,
    op: Int,
):
    var values = column._ptr()
    var nulls = column.null_count() > 0
    if op == COUNT:
        var counts = reducer.counts.unsafe_ptr()
        for p in range(n):
            if nulls and not column._valid(rows[unsafe_offset=p]):
                continue
            counts[unsafe_offset=g[unsafe_offset=p]] += 1
    elif op == SUM or op == MEAN:
        var sums = reducer.float_sums.unsafe_ptr()
        for p in range(n):
            var row = rows[unsafe_offset=p]
            if nulls and not column._valid(row):
                continue
            ref state = sums[unsafe_offset=g[unsafe_offset=p]]
            state.total += values[unsafe_offset=row]
            state.count += 1
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
            var value = values[unsafe_offset=row]
            if isnan(value):
                nan_seen[unsafe_offset=group] = True
            elif (
                not seen[unsafe_offset=group]
                or (is_max and value > best[unsafe_offset=group])
                or (not is_max and value < best[unsafe_offset=group])
            ):
                best[unsafe_offset=group] = value
                seen[unsafe_offset=group] = True
