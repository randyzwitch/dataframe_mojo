"""Per-group reduction state, updated one evaluated batch at a time.

A Reducer owns one state per group for one reduction node. Batches arrive in
row order with their offset, so group ids come from the shared mapping.
Mergeable states (sums, counts, min/max, moments, logic, distinct sets) could
be kept per worker and merged; first/last need partition order, and
median/quantile keep every valid value of a group.
"""
from std.collections import Dict, Optional
from std.math import ceil, floor, isnan, sqrt
from std.memory import bitcast
from .column import Column
from .series import Series
from .expr import (
    SUM,
    COUNT,
    ANY,
    ALL,
    NULL_COUNT,
    MIN,
    MAX,
    MEAN,
    FIRST,
    LAST,
    N_UNIQUE,
    STD,
    VAR,
    MEDIAN,
    QUANTILE,
    LEN,
)
from .reductions import IntSumState, FloatSumState, LogicState, VarState


def _group(grouped: Bool, groups: List[Int], row: Int) -> Int:
    return groups[row] if grouped else 0


def _count_valid[
    T: Copyable & Deinitable
](
    column: Column[T],
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    nulls: Bool,
    mut counts: List[Int64],
):
    for i in range(len(column)):
        if column._valid(i) != nulls:
            counts[_group(grouped, groups, offset + i)] += 1


def _extreme[
    T: Copyable & Deinitable & Comparable
](
    column: Column[T],
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    is_max: Bool,
    mut seen: List[Bool],
    mut best: List[T],
):
    """Track the extreme valid value; ties keep the earlier value."""
    for i in range(len(column)):
        if not column._valid(i):
            continue
        var g = _group(grouped, groups, offset + i)
        ref value = column._values[i]
        if (
            not seen[g]
            or (is_max and value > best[g])
            or (not is_max and value < best[g])
        ):
            best[g] = value.copy()
            seen[g] = True


def _pick[
    T: Copyable & Deinitable
](
    column: Column[T],
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    last: Bool,
    mut seen: List[Bool],
    mut valid: List[Bool],
    mut best: List[T],
):
    for i in range(len(column)):
        var g = _group(grouped, groups, offset + i)
        if last or not seen[g]:
            seen[g] = True
            valid[g] = column._valid(i)
            best[g] = column._values[i].copy()


def _distinct[
    T: Copyable & Deinitable & Hashable & Equatable
](
    column: Column[T],
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    mut sets: List[Dict[T, Bool]],
    mut nulls: List[Bool],
):
    for i in range(len(column)):
        var g = _group(grouped, groups, offset + i)
        if column._valid(i):
            sets[g][column._values[i].copy()] = True
        else:
            nulls[g] = True


def float_key(value: Float64) -> UInt64:
    """Equality key where every NaN is one value and -0.0 equals 0.0."""
    if isnan(value):
        return 0x7FF8000000000000
    if value == 0:
        return 0
    return bitcast[DType.uint64](value)


def quantile_of(
    var values: List[Float64], q: Float64, method: String
) -> Optional[Float64]:
    """Order statistic with NaN sorted above every number."""
    var n = len(values)
    if n == 0:
        return None
    var ordered = List[Float64](capacity=n)
    var nans = 0
    for value in values:
        if isnan(value):
            nans += 1
        else:
            ordered.append(value)
    sort(ordered)
    for _ in range(nans):
        ordered.append(Float64(0) / Float64(0))
    var position = q * Float64(n - 1)
    var lower = Int(floor(position))
    var upper = min(Int(ceil(position)), n - 1)
    var fraction = position - Float64(lower)
    if method == "lower":
        return ordered[lower]
    if method == "higher":
        return ordered[upper]
    if method == "nearest":
        return ordered[min(Int(floor(position + 0.5)), n - 1)]
    if lower == upper or fraction == 0:
        return ordered[lower]
    if method == "midpoint":
        return (ordered[lower] + ordered[upper]) / 2
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


struct Reducer(Movable):
    var op: Int
    var dtype: String
    var group_count: Int
    var min_count: Int
    var integer: Int64
    var floating: Float64
    var text: String
    var counts: List[Int64]
    var int_sums: List[IntSumState]
    var float_sums: List[FloatSumState]
    var logic: List[LogicState]
    var moments: List[VarState]
    var samples: List[List[Float64]]
    var seen: List[Bool]
    var nan_seen: List[Bool]
    var picked_valid: List[Bool]
    var ints: List[Int64]
    var floats: List[Float64]
    var bools: List[Bool]
    var strings: List[String]
    var int_sets: List[Dict[Int64, Bool]]
    var float_sets: List[Dict[UInt64, Bool]]
    var string_sets: List[Dict[String, Bool]]

    def __init__(
        out self,
        op: Int,
        input_dtype: String,
        group_count: Int,
        min_count: Int,
        integer: Int64,
        floating: Float64 = 0,
        text: String = "",
    ):
        self.op = op
        self.dtype = input_dtype
        self.group_count = group_count
        self.min_count = min_count
        self.integer = integer
        self.floating = floating
        self.text = text
        var n = group_count
        var is_int = input_dtype == "int64"
        var is_float = input_dtype == "float64"
        var summing = op == SUM or op == MEAN
        var picking = op == MIN or op == MAX or op == FIRST or op == LAST
        var distinct = op == N_UNIQUE
        self.counts = List[Int64](length=n, fill=0)
        self.int_sums = List[IntSumState](
            length=n if summing and is_int else 0, fill=IntSumState()
        )
        self.float_sums = List[FloatSumState](
            length=n if summing and is_float else 0, fill=FloatSumState()
        )
        self.logic = List[LogicState](
            length=n if op == ANY
            or op == ALL
            or (distinct and input_dtype == "bool") else 0,
            fill=LogicState(),
        )
        self.moments = List[VarState](
            length=n if op == STD or op == VAR else 0, fill=VarState()
        )
        self.samples = List[List[Float64]](
            length=n if op == MEDIAN or op == QUANTILE else 0,
            fill=List[Float64](),
        )
        self.seen = List[Bool](length=n if picking else 0, fill=False)
        self.nan_seen = List[Bool](length=n if picking else 0, fill=False)
        self.picked_valid = List[Bool](
            length=n if picking or distinct else 0, fill=False
        )
        self.ints = List[Int64](length=n if picking and is_int else 0, fill=0)
        self.floats = List[Float64](
            length=n if picking and is_float else 0, fill=0
        )
        self.bools = List[Bool](
            length=n if picking and input_dtype == "bool" else 0, fill=False
        )
        self.strings = List[String](
            length=n if picking and input_dtype == "string" else 0, fill=""
        )
        self.int_sets = List[Dict[Int64, Bool]](
            length=n if distinct and is_int else 0, fill=Dict[Int64, Bool]()
        )
        self.float_sets = List[Dict[UInt64, Bool]](
            length=n if distinct and is_float else 0,
            fill=Dict[UInt64, Bool](),
        )
        self.string_sets = List[Dict[String, Bool]](
            length=n if distinct and input_dtype == "string" else 0,
            fill=Dict[String, Bool](),
        )

    def update(
        mut self, chunk: Series, offset: Int, grouped: Bool, groups: List[Int]
    ) raises:
        var op = self.op
        if op == COUNT or op == NULL_COUNT:
            var nulls = op == NULL_COUNT
            if chunk._data.isa[Column[Int64]]():
                _count_valid(
                    chunk._data[Column[Int64]],
                    offset,
                    grouped,
                    groups,
                    nulls,
                    self.counts,
                )
            elif chunk._data.isa[Column[Float64]]():
                _count_valid(
                    chunk._data[Column[Float64]],
                    offset,
                    grouped,
                    groups,
                    nulls,
                    self.counts,
                )
            elif chunk._data.isa[Column[Bool]]():
                _count_valid(
                    chunk._data[Column[Bool]],
                    offset,
                    grouped,
                    groups,
                    nulls,
                    self.counts,
                )
            else:
                _count_valid(
                    chunk._data[Column[String]],
                    offset,
                    grouped,
                    groups,
                    nulls,
                    self.counts,
                )
        elif op == LEN:
            for i in range(len(chunk)):
                self.counts[_group(grouped, groups, offset + i)] += 1
        elif (op == SUM or op == MEAN) and self.dtype == "int64":
            ref column = chunk._data[Column[Int64]]
            for i in range(len(column)):
                if column._valid(i):
                    self.int_sums[_group(grouped, groups, offset + i)].add(
                        column._values[i]
                    )
        elif op == SUM or op == MEAN:
            ref column = chunk._data[Column[Float64]]
            for i in range(len(column)):
                if column._valid(i):
                    self.float_sums[_group(grouped, groups, offset + i)].add(
                        column._values[i]
                    )
        elif op == ANY or op == ALL:
            ref column = chunk._data[Column[Bool]]
            for i in range(len(column)):
                self.logic[_group(grouped, groups, offset + i)].add(
                    column._valid(i), column._values[i]
                )
        elif op == STD or op == VAR or op == MEDIAN or op == QUANTILE:
            for i in range(len(chunk)):
                var value: Float64
                if chunk._data.isa[Column[Int64]]():
                    if not chunk._data[Column[Int64]]._valid(i):
                        continue
                    value = Float64(chunk._data[Column[Int64]]._values[i])
                else:
                    if not chunk._data[Column[Float64]]._valid(i):
                        continue
                    value = chunk._data[Column[Float64]]._values[i]
                var g = _group(grouped, groups, offset + i)
                if op == STD or op == VAR:
                    self.moments[g].add(value)
                else:
                    self.samples[g].append(value)
        elif op == MIN or op == MAX:
            var is_max = op == MAX
            if chunk._data.isa[Column[Int64]]():
                _extreme(
                    chunk._data[Column[Int64]],
                    offset,
                    grouped,
                    groups,
                    is_max,
                    self.seen,
                    self.ints,
                )
            elif chunk._data.isa[Column[Float64]]():
                ref column = chunk._data[Column[Float64]]
                for i in range(len(column)):
                    if not column._valid(i):
                        continue
                    var g = _group(grouped, groups, offset + i)
                    var value = column._values[i]
                    if isnan(value):
                        self.nan_seen[g] = True
                    elif (
                        not self.seen[g]
                        or (is_max and value > self.floats[g])
                        or (not is_max and value < self.floats[g])
                    ):
                        self.floats[g] = value
                        self.seen[g] = True
            elif chunk._data.isa[Column[Bool]]():
                _extreme(
                    chunk._data[Column[Bool]],
                    offset,
                    grouped,
                    groups,
                    is_max,
                    self.seen,
                    self.bools,
                )
            else:
                _extreme(
                    chunk._data[Column[String]],
                    offset,
                    grouped,
                    groups,
                    is_max,
                    self.seen,
                    self.strings,
                )
        elif op == FIRST or op == LAST:
            var last = op == LAST
            if chunk._data.isa[Column[Int64]]():
                _pick(
                    chunk._data[Column[Int64]],
                    offset,
                    grouped,
                    groups,
                    last,
                    self.seen,
                    self.picked_valid,
                    self.ints,
                )
            elif chunk._data.isa[Column[Float64]]():
                _pick(
                    chunk._data[Column[Float64]],
                    offset,
                    grouped,
                    groups,
                    last,
                    self.seen,
                    self.picked_valid,
                    self.floats,
                )
            elif chunk._data.isa[Column[Bool]]():
                _pick(
                    chunk._data[Column[Bool]],
                    offset,
                    grouped,
                    groups,
                    last,
                    self.seen,
                    self.picked_valid,
                    self.bools,
                )
            else:
                _pick(
                    chunk._data[Column[String]],
                    offset,
                    grouped,
                    groups,
                    last,
                    self.seen,
                    self.picked_valid,
                    self.strings,
                )
        elif op == N_UNIQUE:
            if chunk._data.isa[Column[Int64]]():
                _distinct(
                    chunk._data[Column[Int64]],
                    offset,
                    grouped,
                    groups,
                    self.int_sets,
                    self.picked_valid,
                )
            elif chunk._data.isa[Column[Float64]]():
                ref column = chunk._data[Column[Float64]]
                for i in range(len(column)):
                    var g = _group(grouped, groups, offset + i)
                    if column._valid(i):
                        self.float_sets[g][float_key(column._values[i])] = True
                    else:
                        self.picked_valid[g] = True
            elif chunk._data.isa[Column[Bool]]():
                ref column = chunk._data[Column[Bool]]
                for i in range(len(column)):
                    self.logic[_group(grouped, groups, offset + i)].add(
                        column._valid(i), column._values[i]
                    )
            else:
                _distinct(
                    chunk._data[Column[String]],
                    offset,
                    grouped,
                    groups,
                    self.string_sets,
                    self.picked_valid,
                )
        else:
            raise Error("Unsupported reduction")

    def finish(self) raises -> Series:
        var op = self.op
        var n = self.group_count
        var valid = List[Bool](length=n, fill=True)
        if op == COUNT or op == NULL_COUNT or op == LEN:
            return Series("", Column[Int64](self.counts.copy()))
        if op == SUM and self.dtype == "int64":
            var output = List[Int64](length=n, fill=0)
            for g in range(n):
                valid[g] = self.int_sums[g].count >= Int64(self.min_count)
                if valid[g]:
                    output[g] = self.int_sums[g].value()
            return Series("", Column[Int64](output^, valid))
        if op == SUM:
            var output = List[Float64](length=n, fill=0)
            for g in range(n):
                valid[g] = self.float_sums[g].count >= Int64(self.min_count)
                if valid[g]:
                    output[g] = self.float_sums[g].total
            return Series("", Column[Float64](output^, valid))
        if op == MEAN:
            var output = List[Float64](length=n, fill=0)
            for g in range(n):
                if self.dtype == "int64":
                    ref state = self.int_sums[g]
                    valid[g] = state.count > 0
                    if valid[g]:
                        output[g] = state.total.cast[DType.float64]() / Float64(
                            state.count
                        )
                else:
                    ref state = self.float_sums[g]
                    valid[g] = state.count > 0
                    if valid[g]:
                        output[g] = state.total / Float64(state.count)
            return Series("", Column[Float64](output^, valid))
        if op == STD or op == VAR:
            var output = List[Float64](length=n, fill=0)
            for g in range(n):
                var variance = self.moments[g].variance(Int(self.integer))
                valid[g] = Bool(variance)
                if variance:
                    output[g] = (
                        sqrt(variance.value()) if op
                        == STD else variance.value()
                    )
            return Series("", Column[Float64](output^, valid))
        if op == MEDIAN or op == QUANTILE:
            var output = List[Float64](length=n, fill=0)
            for g in range(n):
                var result = quantile_of(
                    self.samples[g].copy(), self.floating, self.text
                )
                valid[g] = Bool(result)
                if result:
                    output[g] = result.value()
            return Series("", Column[Float64](output^, valid))
        if op == N_UNIQUE:
            var output = List[Int64](length=n, fill=0)
            for g in range(n):
                var count: Int
                if self.dtype == "int64":
                    count = len(self.int_sets[g]) + Int(self.picked_valid[g])
                elif self.dtype == "float64":
                    count = len(self.float_sets[g]) + Int(self.picked_valid[g])
                elif self.dtype == "bool":
                    ref state = self.logic[g]
                    count = (
                        Int(state.saw_true)
                        + Int(state.saw_false)
                        + Int(state.saw_null)
                    )
                else:
                    count = len(self.string_sets[g]) + Int(self.picked_valid[g])
                output[g] = Int64(count)
            return Series("", Column[Int64](output^))
        if op == MIN or op == MAX or op == FIRST or op == LAST:
            var picked = op == FIRST or op == LAST
            for g in range(n):
                valid[g] = self.seen[g] and (not picked or self.picked_valid[g])
            if self.dtype == "int64":
                return Series("", Column[Int64](self.ints.copy(), valid))
            if self.dtype == "float64":
                var output = self.floats.copy()
                if not picked:
                    for g in range(n):
                        var nan_wins = self.nan_seen[g] and (
                            op == MAX or not self.seen[g]
                        )
                        if nan_wins:
                            output[g] = Float64(0) / Float64(0)
                            valid[g] = True
                return Series("", Column[Float64](output^, valid))
            if self.dtype == "bool":
                return Series("", Column[Bool](self.bools.copy(), valid))
            return Series("", Column[String](self.strings.copy(), valid))
        var output = List[Bool](length=n, fill=False)
        var ignore_nulls = self.integer != 0
        for g in range(n):
            var result = self.logic[g].any(
                ignore_nulls
            ) if op == ANY else self.logic[g].all(ignore_nulls)
            valid[g] = Bool(result)
            if result:
                output[g] = result.value()
        return Series("", Column[Bool](output^, valid))
