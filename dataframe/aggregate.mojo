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
from .bool_column import BoolColumn
from .column import Column
from .string_column import StringColumn, StringBuilder
from .dtype import DataType, NUMERIC_DTYPES
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
        ref value = column._get(i)
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
            best[g] = column._get(i).copy()


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
            sets[g][column._get(i).copy()] = True
        else:
            nulls[g] = True


# BoolColumn overloads (bit-packed values).


def _count_valid(
    column: BoolColumn,
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    nulls: Bool,
    mut counts: List[Int64],
):
    for i in range(len(column)):
        if column._valid(i) != nulls:
            counts[_group(grouped, groups, offset + i)] += 1


def _extreme(
    column: BoolColumn,
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    is_max: Bool,
    mut seen: List[Bool],
    mut best: List[Bool],
):
    for i in range(len(column)):
        if not column._valid(i):
            continue
        var g = _group(grouped, groups, offset + i)
        var value = column._get(i)
        if (
            not seen[g]
            or (is_max and value and not best[g])
            or (not is_max and not value and best[g])
        ):
            best[g] = value
            seen[g] = True


def _pick(
    column: BoolColumn,
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    last: Bool,
    mut seen: List[Bool],
    mut valid: List[Bool],
    mut best: List[Bool],
):
    for i in range(len(column)):
        var g = _group(grouped, groups, offset + i)
        if last or not seen[g]:
            seen[g] = True
            valid[g] = column._valid(i)
            best[g] = column._get(i)


# StringColumn overloads: rows are borrowed slices; per-group state owns
# Strings, so copies happen only when a group's value changes.


def _count_valid(
    column: StringColumn,
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    nulls: Bool,
    mut counts: List[Int64],
):
    for i in range(len(column)):
        if column._valid(i) != nulls:
            counts[_group(grouped, groups, offset + i)] += 1


def _extreme(
    column: StringColumn,
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    is_max: Bool,
    mut seen: List[Bool],
    mut best: List[String],
):
    """Track the extreme valid value; ties keep the earlier value."""
    for i in range(len(column)):
        if not column._valid(i):
            continue
        var g = _group(grouped, groups, offset + i)
        var value = column._get(i)
        if (
            not seen[g]
            or (is_max and value > best[g])
            or (not is_max and value < best[g])
        ):
            best[g] = String(value)
            seen[g] = True


def _pick(
    column: StringColumn,
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    last: Bool,
    mut seen: List[Bool],
    mut valid: List[Bool],
    mut best: List[String],
):
    # Scan backwards for `last` so each group copies at most once per chunk.
    var n = len(column)
    var taken = Dict[Int, Bool]()
    for k in range(n):
        var i = n - 1 - k if last else k
        var g = _group(grouped, groups, offset + i)
        if last:
            if g in taken:
                continue
            taken[g] = True
        elif seen[g]:
            continue
        seen[g] = True
        valid[g] = column._valid(i)
        best[g] = String(column._get(i))


def _distinct(
    column: StringColumn,
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    mut sets: List[Dict[String, Bool]],
    mut nulls: List[Bool],
):
    for i in range(len(column)):
        var g = _group(grouped, groups, offset + i)
        if column._valid(i):
            var value = column._get(i)
            if String(value) not in sets[g]:
                sets[g][String(value)] = True
        else:
            nulls[g] = True


comptime _UINT64_BIAS = UInt64(1) << 63


def _state_type(dtype: DataType) -> DataType:
    if dtype.is_integer():
        return DataType.INT64
    if dtype.is_float():
        return DataType.FLOAT64
    return dtype


def _canonical(chunk: Series) raises -> Series:
    """Narrow integers as Int64 (UInt64 biased to keep order), Float32 as
    Float64; exact in every case."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if D != DType.int64 and D != DType.float64:
            if chunk._data.isa[Column[Scalar[D]]]():
                ref column = chunk._data[Column[Scalar[D]]]
                var valid = List[Bool](capacity=len(column))
                comptime if D.is_floating_point():
                    var values = List[Float64](capacity=len(column))
                    for i in range(len(column)):
                        values.append(column._get(i).cast[DType.float64]())
                        valid.append(column._valid(i))
                    return Series("", Column[Float64](values^, valid))
                else:
                    var values = List[Int64](capacity=len(column))
                    for i in range(len(column)):
                        comptime if D == DType.uint64:
                            values.append(
                                (
                                    column._get(i).cast[DType.uint64]()
                                    ^ _UINT64_BIAS
                                ).cast[DType.int64]()
                            )
                        else:
                            values.append(column._get(i).cast[DType.int64]())
                        valid.append(column._valid(i))
                    return Series("", Column[Int64](values^, valid))
    return chunk.copy()


def _from_canonical(result: Series, dtype: DataType) raises -> Series:
    """Undo _canonical for min/max/first/last/sum results."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if dtype == DataType.of(D):
            var valid = validity_of(result)
            var values = List[Scalar[D]](capacity=len(result))
            comptime if D.is_floating_point():
                ref column = result._data[Column[Float64]]
                for i in range(len(column)):
                    values.append(column._get(i).cast[D]())
            else:
                ref column = result._data[Column[Int64]]
                for i in range(len(column)):
                    comptime if D == DType.uint64:
                        values.append(
                            (
                                column._get(i).cast[DType.uint64]()
                                ^ _UINT64_BIAS
                            ).cast[D]()
                        )
                    else:
                        values.append(column._get(i).cast[D]())
            return Series("", Column[Scalar[D]](values^, valid))
    return result.copy()


def validity_of(series: Series) -> List[Bool]:
    var valid = List[Bool](capacity=len(series))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            ref column = series._data[Column[Scalar[D]]]
            for i in range(len(column)):
                valid.append(column._valid(i))
    return valid^


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
    """Per-group reduction state for one expression.

    State is kept in a canonical type: every integer width as Int64 (UInt64
    re-encoded order-preservingly, or summed exactly in 128 bits) and Float32
    as Float64. `update` converts input batches and `finish` converts the
    result back to the output type.
    """

    var op: Int
    var input: DataType  # the physical input type
    var dtype: DataType  # the canonical state type
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
        input_dtype: DataType,
        group_count: Int,
        min_count: Int,
        integer: Int64,
        floating: Float64 = 0,
        text: String = "",
    ):
        self.op = op
        self.input = input_dtype
        self.dtype = _state_type(input_dtype)
        self.group_count = group_count
        self.min_count = min_count
        self.integer = integer
        self.floating = floating
        self.text = text
        var n = group_count
        var is_int = self.dtype == DataType.INT64
        var is_float = self.dtype == DataType.FLOAT64
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
            or (distinct and input_dtype == DataType.BOOL) else 0,
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
            length=n if picking and input_dtype == DataType.BOOL else 0,
            fill=False,
        )
        self.strings = List[String](
            length=n if picking and input_dtype == DataType.STRING else 0,
            fill="",
        )
        self.int_sets = List[Dict[Int64, Bool]](
            length=n if distinct and is_int else 0, fill=Dict[Int64, Bool]()
        )
        self.float_sets = List[Dict[UInt64, Bool]](
            length=n if distinct and is_float else 0,
            fill=Dict[UInt64, Bool](),
        )
        self.string_sets = List[Dict[String, Bool]](
            length=n if distinct and input_dtype == DataType.STRING else 0,
            fill=Dict[String, Bool](),
        )

    def update(
        mut self, chunk: Series, offset: Int, grouped: Bool, groups: List[Int]
    ) raises:
        # Batch slicing can span physical arrays.  State kernels operate on
        # one contiguous typed buffer, so advance the global group offset for
        # each array instead of silently reading chunk zero.
        if chunk.is_chunked():
            var part_offset = offset
            for part in chunk.chunks():
                self.update(part, part_offset, grouped, groups)
                part_offset += len(part)
            return
        if self.input == self.dtype:
            self._update(chunk, offset, grouped, groups)
            return
        if self.input == DataType.UINT64 and (
            self.op == SUM or self.op == MEAN
        ):
            ref column = chunk._data[Column[UInt64]]
            for i in range(len(column)):
                if column._valid(i):
                    self.int_sums[_group(grouped, groups, offset + i)].add_wide(
                        column._get(i).cast[DType.int128]()
                    )
            return
        self._update(_canonical(chunk), offset, grouped, groups)

    def _update(
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
            elif chunk._data.isa[BoolColumn]():
                _count_valid(
                    chunk._data[BoolColumn],
                    offset,
                    grouped,
                    groups,
                    nulls,
                    self.counts,
                )
            else:
                _count_valid(
                    chunk._data[StringColumn],
                    offset,
                    grouped,
                    groups,
                    nulls,
                    self.counts,
                )
        elif op == LEN:
            for i in range(len(chunk)):
                self.counts[_group(grouped, groups, offset + i)] += 1
        elif (op == SUM or op == MEAN) and self.dtype == DataType.INT64:
            ref column = chunk._data[Column[Int64]]
            for i in range(len(column)):
                if column._valid(i):
                    self.int_sums[_group(grouped, groups, offset + i)].add(
                        column._get(i)
                    )
        elif op == SUM or op == MEAN:
            ref column = chunk._data[Column[Float64]]
            for i in range(len(column)):
                if column._valid(i):
                    self.float_sums[_group(grouped, groups, offset + i)].add(
                        column._get(i)
                    )
        elif op == ANY or op == ALL:
            ref column = chunk._data[BoolColumn]
            for i in range(len(column)):
                self.logic[_group(grouped, groups, offset + i)].add(
                    column._valid(i), column._get(i)
                )
        elif op == STD or op == VAR or op == MEDIAN or op == QUANTILE:
            for i in range(len(chunk)):
                var value: Float64
                if chunk._data.isa[Column[Int64]]():
                    if not chunk._data[Column[Int64]]._valid(i):
                        continue
                    value = Float64(chunk._data[Column[Int64]]._get(i))
                else:
                    if not chunk._data[Column[Float64]]._valid(i):
                        continue
                    value = chunk._data[Column[Float64]]._get(i)
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
                    var value = column._get(i)
                    if isnan(value):
                        self.nan_seen[g] = True
                    elif (
                        not self.seen[g]
                        or (is_max and value > self.floats[g])
                        or (not is_max and value < self.floats[g])
                    ):
                        self.floats[g] = value
                        self.seen[g] = True
            elif chunk._data.isa[BoolColumn]():
                _extreme(
                    chunk._data[BoolColumn],
                    offset,
                    grouped,
                    groups,
                    is_max,
                    self.seen,
                    self.bools,
                )
            else:
                _extreme(
                    chunk._data[StringColumn],
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
            elif chunk._data.isa[BoolColumn]():
                _pick(
                    chunk._data[BoolColumn],
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
                    chunk._data[StringColumn],
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
                        self.float_sets[g][float_key(column._get(i))] = True
                    else:
                        self.picked_valid[g] = True
            elif chunk._data.isa[BoolColumn]():
                ref column = chunk._data[BoolColumn]
                for i in range(len(column)):
                    self.logic[_group(grouped, groups, offset + i)].add(
                        column._valid(i), column._get(i)
                    )
            else:
                _distinct(
                    chunk._data[StringColumn],
                    offset,
                    grouped,
                    groups,
                    self.string_sets,
                    self.picked_valid,
                )
        else:
            raise Error("Unsupported reduction")

    def merge(mut self, other: Self):
        """Fold in the state of the next, disjoint row partition.

        Partitions are merged in row order, so first/last and tie-breaking
        (ties keep the earlier value) match a single-threaded pass.
        """
        var op = self.op
        var is_max = op == MAX
        for g in range(self.group_count):
            if op == COUNT or op == NULL_COUNT or op == LEN:
                self.counts[g] += other.counts[g]
            elif op == SUM or op == MEAN:
                if self.dtype == DataType.INT64:
                    self.int_sums[g].merge(other.int_sums[g])
                else:
                    self.float_sums[g].merge(other.float_sums[g])
            elif op == STD or op == VAR:
                self.moments[g].merge(other.moments[g])
            elif op == MEDIAN or op == QUANTILE:
                for value in other.samples[g]:
                    self.samples[g].append(value)
            elif op == ANY or op == ALL:
                self.logic[g].merge(other.logic[g])
            elif op == N_UNIQUE:
                if self.dtype == DataType.BOOL:
                    self.logic[g].merge(other.logic[g])
                    continue
                self.picked_valid[g] = (
                    self.picked_valid[g] or other.picked_valid[g]
                )
                if self.dtype == DataType.INT64:
                    for key in other.int_sets[g].keys():
                        self.int_sets[g][key] = True
                elif self.dtype == DataType.FLOAT64:
                    for key in other.float_sets[g].keys():
                        self.float_sets[g][key] = True
                else:
                    for key in other.string_sets[g].keys():
                        self.string_sets[g][key] = True
            elif op == MIN or op == MAX:
                if self.dtype == DataType.FLOAT64:
                    self.nan_seen[g] = self.nan_seen[g] or other.nan_seen[g]
                if not other.seen[g]:
                    continue
                var take = not self.seen[g]
                if not take:
                    if self.dtype == DataType.INT64:
                        take = (
                            other.ints[g]
                            > self.ints[g] if is_max else other.ints[g]
                            < self.ints[g]
                        )
                    elif self.dtype == DataType.FLOAT64:
                        take = (
                            other.floats[g]
                            > self.floats[g] if is_max else other.floats[g]
                            < self.floats[g]
                        )
                    elif self.dtype == DataType.BOOL:
                        take = (
                            other.bools[g]
                            > self.bools[g] if is_max else other.bools[g]
                            < self.bools[g]
                        )
                    else:
                        take = (
                            other.strings[g]
                            > self.strings[g] if is_max else other.strings[g]
                            < self.strings[g]
                        )
                if take:
                    self._take_value(other, g)
            elif op == FIRST or op == LAST:
                if other.seen[g] and (op == LAST or not self.seen[g]):
                    self._take_value(other, g)
                    self.picked_valid[g] = other.picked_valid[g]

    def _take_value(mut self, other: Self, g: Int):
        self.seen[g] = True
        if self.dtype == DataType.INT64:
            self.ints[g] = other.ints[g]
        elif self.dtype == DataType.FLOAT64:
            self.floats[g] = other.floats[g]
        elif self.dtype == DataType.BOOL:
            self.bools[g] = other.bools[g]
        else:
            self.strings[g] = other.strings[g]

    def finish(self) raises -> Series:
        if self.input == self.dtype:
            return self._finish()
        var op = self.op
        if op == SUM and self.input.is_integer():
            return self._integer_sum()
        var result = self._finish()
        if op == SUM or op == MIN or op == MAX or op == FIRST or op == LAST:
            return _from_canonical(result, self.input)
        return result^

    def _integer_sum(self) raises -> Series:
        """Exact sums of narrow or unsigned integers in their sum type."""
        var n = self.group_count
        var target = self.input.sum_type()
        var valid = List[Bool](length=n, fill=False)
        for g in range(n):
            valid[g] = self.int_sums[g].count >= Int64(self.min_count)
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            comptime if D.is_integral():
                if target == DataType.of(D):
                    var output = List[Scalar[D]](length=n, fill=0)
                    for g in range(n):
                        if not valid[g]:
                            continue
                        var total = self.int_sums[g].total
                        if (
                            total > Scalar[D].MAX.cast[DType.int128]()
                            or total < Scalar[D].MIN.cast[DType.int128]()
                        ):
                            raise Error(
                                target.name() + " expression sum overflow"
                            )
                        output[g] = total.cast[D]()
                    return Series("", Column[Scalar[D]](output^, valid))
        raise Error("Unsupported sum type " + target.name())

    def _finish(self) raises -> Series:
        var op = self.op
        var n = self.group_count
        var valid = List[Bool](length=n, fill=True)
        if op == COUNT or op == NULL_COUNT or op == LEN:
            return Series("", Column[Int64](self.counts.copy()))
        if op == SUM and self.dtype == DataType.INT64:
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
                if self.dtype == DataType.INT64:
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
                if self.dtype == DataType.INT64:
                    count = len(self.int_sets[g]) + Int(self.picked_valid[g])
                elif self.dtype == DataType.FLOAT64:
                    count = len(self.float_sets[g]) + Int(self.picked_valid[g])
                elif self.dtype == DataType.BOOL:
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
            if self.dtype == DataType.INT64:
                return Series("", Column[Int64](self.ints.copy(), valid))
            if self.dtype == DataType.FLOAT64:
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
            if self.dtype == DataType.BOOL:
                return Series("", BoolColumn(self.bools.copy(), valid))
            return Series("", StringColumn(self.strings, valid))
        var output = List[Bool](length=n, fill=False)
        var ignore_nulls = self.integer != 0
        for g in range(n):
            var result = self.logic[g].any(
                ignore_nulls
            ) if op == ANY else self.logic[g].all(ignore_nulls)
            valid[g] = Bool(result)
            if result:
                output[g] = result.value()
        return Series("", BoolColumn(output^, valid))
