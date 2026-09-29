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
from .decimal import check_precision
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
    ARG_MIN,
    ARG_MAX,
    MODE,
    SKEW,
    KURTOSIS,
    CORR,
    COV,
    VALUE_COUNTS,
    SEP,
)
from .reductions import (
    CoMomentState,
    FloatSumState,
    IntSumState,
    LogicState,
    MomentState,
    VarState,
)
from .nested_column import ListColumn, StructColumn


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


def _float_values(chunk: Series) raises -> Tuple[List[Float64], List[Bool]]:
    """Any numeric chunk as Float64 values and validity (UInt64 unbiased)."""
    var values = List[Float64](capacity=len(chunk))
    var valid = List[Bool](capacity=len(chunk))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if chunk._data.isa[Column[Scalar[D]]]():
            ref column = chunk._data[Column[Scalar[D]]]
            for i in range(len(column)):
                values.append(column._get(i).cast[DType.float64]())
                valid.append(column._valid(i))
            return (values^, valid^)
    raise Error("expected a numeric column")


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


def _bisect(sorted: List[Float64], value: Float64, right: Bool) -> Int:
    """First index whose value is > (right) or >= (left) `value`."""
    var lo = 0
    var hi = len(sorted)
    while lo < hi:
        var mid = (lo + hi) // 2
        if sorted[mid] < value or (right and sorted[mid] == value):
            lo = mid + 1
        else:
            hi = mid
    return lo


def _average_ranks(values: List[Float64]) -> List[Float64]:
    """1-based ranks; ties share the average of the ranks they span."""
    var ordered = values.copy()
    sort(ordered)
    var ranks = List[Float64](capacity=len(values))
    for value in values:
        var first = _bisect(ordered, value, False)
        var last = _bisect(ordered, value, True)
        ranks.append(Float64(first + last + 1) / 2)
    return ranks^


def _spearman(xs: List[Float64], ys: List[Float64]) -> Float64:
    """Pearson's r on average ranks; NaN below two pairs or with a NaN."""
    for i in range(len(xs)):
        if isnan(xs[i]) or isnan(ys[i]):
            return Float64(0) / Float64(0)
    var rx = _average_ranks(xs)
    var ry = _average_ranks(ys)
    var state = CoMomentState()
    for i in range(len(rx)):
        state.add(rx[i], ry[i])
    return state.correlation()


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
    var decimals: List[Int128]
    var bools: List[Bool]
    var strings: List[String]
    var int_sets: List[Dict[Int64, Bool]]
    var float_sets: List[Dict[UInt64, Bool]]
    var string_sets: List[Dict[String, Bool]]
    # arg_min/arg_max: index of the best value and of the first NaN, within
    # each group; `counts` holds the rows seen per group.
    var positions: List[Int64]
    var nan_positions: List[Int64]
    var moments4: List[MomentState]
    var comoments: List[CoMomentState]
    # Spearman keeps the pairs, since ranks need every value of a group.
    var pair_x: List[List[Float64]]
    var pair_y: List[List[Float64]]
    # mode: occurrences per canonical value (bools as 0/1) and nulls.
    var mode_ints: List[Dict[Int64, Int64]]
    var mode_floats: List[Dict[UInt64, Int64]]
    var mode_float_values: List[Dict[UInt64, Float64]]
    var mode_strings: List[Dict[String, Int64]]
    var mode_nulls: List[Int64]
    # The logical input type, for results (mode) that carry input values.
    var logical: DataType

    def __init__(
        out self,
        op: Int,
        input_dtype: DataType,
        group_count: Int,
        min_count: Int,
        integer: Int64,
        floating: Float64 = 0,
        text: String = "",
        logical: Optional[DataType] = None,
    ):
        self.op = op
        self.input = input_dtype
        self.logical = logical.value() if logical else input_dtype
        self.dtype = self.logical if self.logical.is_decimal() else _state_type(
            input_dtype
        )
        self.group_count = group_count
        self.min_count = min_count
        self.integer = integer
        self.floating = floating
        self.text = text
        var n = group_count
        var is_int = self.dtype == DataType.INT64
        var is_float = self.dtype == DataType.FLOAT64
        var summing = op == SUM or op == MEAN
        var arg = op == ARG_MIN or op == ARG_MAX
        var picking = op == MIN or op == MAX or op == FIRST or op == LAST or arg
        var mode = op == MODE or op == VALUE_COUNTS
        var spearman = op == CORR and text == "spearman"
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
        self.decimals = List[Int128](
            length=n if (picking or summing) and self.dtype.is_decimal() else 0,
            fill=0,
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
        self.positions = List[Int64](length=n if arg else 0, fill=0)
        self.nan_positions = List[Int64](length=n if arg else 0, fill=0)
        self.moments4 = List[MomentState](
            length=n if op == SKEW or op == KURTOSIS else 0, fill=MomentState()
        )
        self.comoments = List[CoMomentState](
            length=n if (op == CORR and not spearman) or op == COV else 0,
            fill=CoMomentState(),
        )
        self.pair_x = List[List[Float64]](
            length=n if spearman else 0, fill=List[Float64]()
        )
        self.pair_y = List[List[Float64]](
            length=n if spearman else 0, fill=List[Float64]()
        )
        var mode_ints = mode and (is_int or input_dtype == DataType.BOOL)
        self.mode_ints = List[Dict[Int64, Int64]](
            length=n if mode_ints else 0, fill=Dict[Int64, Int64]()
        )
        self.mode_floats = List[Dict[UInt64, Int64]](
            length=n if mode and is_float else 0, fill=Dict[UInt64, Int64]()
        )
        self.mode_float_values = List[Dict[UInt64, Float64]](
            length=n if mode and is_float else 0, fill=Dict[UInt64, Float64]()
        )
        self.mode_strings = List[Dict[String, Int64]](
            length=n if mode and input_dtype == DataType.STRING else 0,
            fill=Dict[String, Int64](),
        )
        self.mode_nulls = List[Int64](length=n if mode else 0, fill=0)

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
        if self.op == SKEW or self.op == KURTOSIS:
            var floats = _float_values(chunk)
            for i in range(len(floats[0])):
                if floats[1][i]:
                    self.moments4[_group(grouped, groups, offset + i)].add(
                        floats[0][i]
                    )
            return
        if self.logical.is_decimal():
            self._update(chunk, offset, grouped, groups)
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

    def update_pair(
        mut self,
        left: Series,
        right: Series,
        offset: Int,
        grouped: Bool,
        groups: List[Int],
    ) raises:
        """Feed (left, right) rows to corr/cov; a null on either side drops
        the pair."""
        var xs = _float_values(left.rechunk())
        var ys = _float_values(right.rechunk())
        if len(xs[0]) != len(ys[0]):
            if len(ys[0]) == 1:
                var value = ys[0][0]
                var valid = ys[1][0]
                ys = (
                    List[Float64](length=len(xs[0]), fill=value),
                    List[Bool](length=len(xs[0]), fill=valid),
                )
            elif len(xs[0]) == 1:
                var value = xs[0][0]
                var valid = xs[1][0]
                xs = (
                    List[Float64](length=len(ys[0]), fill=value),
                    List[Bool](length=len(ys[0]), fill=valid),
                )
            else:
                raise Error("corr and cov inputs differ in length")
        var spearman = len(self.pair_x) > 0
        for i in range(len(xs[0])):
            if not (xs[1][i] and ys[1][i]):
                continue
            var g = _group(grouped, groups, offset + i)
            if spearman:
                self.pair_x[g].append(xs[0][i])
                self.pair_y[g].append(ys[0][i])
            else:
                self.comoments[g].add(xs[0][i], ys[0][i])

    def _update(
        mut self, chunk: Series, offset: Int, grouped: Bool, groups: List[Int]
    ) raises:
        var op = self.op
        if op == ARG_MIN or op == ARG_MAX:
            self._update_arg(chunk, offset, grouped, groups)
            return
        if op == MODE or op == VALUE_COUNTS:
            self._update_mode(chunk, offset, grouped, groups)
            return
        if op == COUNT or op == NULL_COUNT:
            var nulls = op == NULL_COUNT
            if chunk._data.isa[Column[Int128]]():
                _count_valid(
                    chunk._data[Column[Int128]],
                    offset,
                    grouped,
                    groups,
                    nulls,
                    self.counts,
                )
                return
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
        elif (op == SUM or op == MEAN) and self.dtype.is_decimal():
            ref column = chunk._data[Column[Int128]]
            for i in range(len(column)):
                if column._valid(i):
                    var g = _group(grouped, groups, offset + i)
                    self.decimals[g] = check_precision(
                        self.decimals[g] + column._get(i), self.dtype
                    )
                    self.counts[g] += 1
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
            if chunk._data.isa[Column[Int128]]():
                ref column = chunk._data[Column[Int128]]
                for i in range(len(column)):
                    if not column._valid(i):
                        continue
                    var g = _group(grouped, groups, offset + i)
                    var value = column._get(i)
                    if (
                        not self.seen[g]
                        or (is_max and value > self.decimals[g])
                        or (not is_max and value < self.decimals[g])
                    ):
                        self.decimals[g] = value
                        self.seen[g] = True
                return
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
            if chunk._data.isa[Column[Int128]]():
                _pick(
                    chunk._data[Column[Int128]],
                    offset,
                    grouped,
                    groups,
                    last,
                    self.seen,
                    self.picked_valid,
                    self.decimals,
                )
                return
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

    def _update_arg(
        mut self, chunk: Series, offset: Int, grouped: Bool, groups: List[Int]
    ) raises:
        """Track the first best value and its index within each group, and
        the first NaN, which counts only when a group has nothing else."""
        var is_max = self.op == ARG_MAX
        if chunk._data.isa[Column[Int64]]():
            ref column = chunk._data[Column[Int64]]
            for i in range(len(column)):
                var g = _group(grouped, groups, offset + i)
                var at = self.counts[g]
                self.counts[g] += 1
                if not column._valid(i):
                    continue
                var value = column._get(i)
                if (
                    not self.seen[g]
                    or (is_max and value > self.ints[g])
                    or (not is_max and value < self.ints[g])
                ):
                    self.ints[g] = value
                    self.positions[g] = at
                    self.seen[g] = True
        elif chunk._data.isa[Column[Float64]]():
            ref column = chunk._data[Column[Float64]]
            for i in range(len(column)):
                var g = _group(grouped, groups, offset + i)
                var at = self.counts[g]
                self.counts[g] += 1
                if not column._valid(i):
                    continue
                var value = column._get(i)
                if isnan(value):
                    if not self.nan_seen[g]:
                        self.nan_seen[g] = True
                        self.nan_positions[g] = at
                elif (
                    not self.seen[g]
                    or (is_max and value > self.floats[g])
                    or (not is_max and value < self.floats[g])
                ):
                    self.floats[g] = value
                    self.positions[g] = at
                    self.seen[g] = True
        elif chunk._data.isa[BoolColumn]():
            ref column = chunk._data[BoolColumn]
            for i in range(len(column)):
                var g = _group(grouped, groups, offset + i)
                var at = self.counts[g]
                self.counts[g] += 1
                if not column._valid(i):
                    continue
                var value = column._get(i)
                if (
                    not self.seen[g]
                    or (is_max and value and not self.bools[g])
                    or (not is_max and not value and self.bools[g])
                ):
                    self.bools[g] = value
                    self.positions[g] = at
                    self.seen[g] = True
        else:
            ref column = chunk._data[StringColumn]
            for i in range(len(column)):
                var g = _group(grouped, groups, offset + i)
                var at = self.counts[g]
                self.counts[g] += 1
                if not column._valid(i):
                    continue
                var value = column._get(i)
                if (
                    not self.seen[g]
                    or (is_max and value > self.strings[g])
                    or (not is_max and value < self.strings[g])
                ):
                    self.strings[g] = String(value)
                    self.positions[g] = at
                    self.seen[g] = True

    def _update_mode(
        mut self, chunk: Series, offset: Int, grouped: Bool, groups: List[Int]
    ) raises:
        if chunk._data.isa[Column[Int64]]():
            ref column = chunk._data[Column[Int64]]
            for i in range(len(column)):
                var g = _group(grouped, groups, offset + i)
                if not column._valid(i):
                    self.mode_nulls[g] += 1
                    continue
                var key = column._get(i)
                self.mode_ints[g][key] = self.mode_ints[g].get(key, 0) + 1
        elif chunk._data.isa[Column[Float64]]():
            ref column = chunk._data[Column[Float64]]
            for i in range(len(column)):
                var g = _group(grouped, groups, offset + i)
                if not column._valid(i):
                    self.mode_nulls[g] += 1
                    continue
                var value = column._get(i)
                var key = float_key(value)
                var seen = self.mode_floats[g].get(key, 0)
                if seen == 0:
                    self.mode_float_values[g][key] = value
                self.mode_floats[g][key] = seen + 1
        elif chunk._data.isa[BoolColumn]():
            ref column = chunk._data[BoolColumn]
            for i in range(len(column)):
                var g = _group(grouped, groups, offset + i)
                if not column._valid(i):
                    self.mode_nulls[g] += 1
                    continue
                var key = Int64(column._get(i))
                self.mode_ints[g][key] = self.mode_ints[g].get(key, 0) + 1
        else:
            ref column = chunk._data[StringColumn]
            for i in range(len(column)):
                var g = _group(grouped, groups, offset + i)
                if not column._valid(i):
                    self.mode_nulls[g] += 1
                    continue
                var key = String(column._get(i))
                self.mode_strings[g][key] = self.mode_strings[g].get(key, 0) + 1

    def _better(self, other: Self, g: Int, source: Int, is_max: Bool) -> Bool:
        """Whether other's value for `source` strictly beats ours for `g`."""
        if self.dtype.is_decimal():
            return (
                other.decimals[source]
                > self.decimals[g] if is_max else other.decimals[source]
                < self.decimals[g]
            )
        if self.dtype == DataType.INT64:
            return (
                other.ints[source]
                > self.ints[g] if is_max else other.ints[source]
                < self.ints[g]
            )
        if self.dtype == DataType.FLOAT64:
            return (
                other.floats[source]
                > self.floats[g] if is_max else other.floats[source]
                < self.floats[g]
            )
        if self.dtype == DataType.BOOL:
            return (
                other.bools[source]
                > self.bools[g] if is_max else other.bools[source]
                < self.bools[g]
            )
        return (
            other.strings[source]
            > self.strings[g] if is_max else other.strings[source]
            < self.strings[g]
        )

    def merge(
        mut self,
        other: Self,
        groups: List[Int] = List[Int](),
        sources: List[Int] = List[Int](),
    ) raises:
        """Fold in the state of the next, disjoint row partition.

        Partitions are merged in row order, so first/last and tie-breaking
        (ties keep the earlier value) match a single-threaded pass. `groups`
        maps other's group ids to this reducer's. `sources` instead selects
        other's groups in order, into this reducer's groups 0, 1, ...; a
        streaming state uses it to copy one hash part's groups (#326).
        """
        var op = self.op
        var is_max = op == MAX
        var steps = len(sources) if len(sources) else other.group_count
        for step in range(steps):
            var source = sources[step] if len(sources) else step
            var g = step if len(sources) else (
                groups[source] if len(groups) else source
            )
            if op == COUNT or op == NULL_COUNT or op == LEN:
                self.counts[g] += other.counts[source]
            elif op == SUM or op == MEAN:
                if self.dtype.is_decimal():
                    self.decimals[g] = check_precision(
                        self.decimals[g] + other.decimals[source], self.dtype
                    )
                    self.counts[g] += other.counts[source]
                    continue
                if self.dtype == DataType.INT64:
                    self.int_sums[g].merge(other.int_sums[source])
                else:
                    self.float_sums[g].merge(other.float_sums[source])
            elif op == STD or op == VAR:
                self.moments[g].merge(other.moments[source])
            elif op == ARG_MIN or op == ARG_MAX:
                # Other's indices count from the start of its partition,
                # which follows every row this state has seen for g.
                var shift = self.counts[g]
                if other.nan_seen[source] and not self.nan_seen[g]:
                    self.nan_seen[g] = True
                    self.nan_positions[g] = shift + other.nan_positions[source]
                if other.seen[source] and (
                    not self.seen[g]
                    or self._better(other, g, source, op == ARG_MAX)
                ):
                    self._take_value(other, g, source)
                    self.positions[g] = shift + other.positions[source]
                self.counts[g] += other.counts[source]
            elif op == SKEW or op == KURTOSIS:
                self.moments4[g].merge(other.moments4[source])
            elif op == CORR or op == COV:
                if len(self.pair_x) > 0:
                    for value in other.pair_x[source]:
                        self.pair_x[g].append(value)
                    for value in other.pair_y[source]:
                        self.pair_y[g].append(value)
                else:
                    self.comoments[g].merge(other.comoments[source])
            elif op == MODE or op == VALUE_COUNTS:
                self.mode_nulls[g] += other.mode_nulls[source]
                if len(self.mode_ints) > 0:
                    for item in other.mode_ints[source].items():
                        self.mode_ints[g][item.key] = (
                            self.mode_ints[g].get(item.key, 0) + item.value
                        )
                elif len(self.mode_floats) > 0:
                    for item in other.mode_floats[source].items():
                        var seen = self.mode_floats[g].get(item.key, 0)
                        if seen == 0:
                            self.mode_float_values[g][
                                item.key
                            ] = other.mode_float_values[source].get(item.key, 0)
                        self.mode_floats[g][item.key] = seen + item.value
                else:
                    for item in other.mode_strings[source].items():
                        self.mode_strings[g][item.key] = (
                            self.mode_strings[g].get(item.key, 0) + item.value
                        )
            elif op == MEDIAN or op == QUANTILE:
                for value in other.samples[source]:
                    self.samples[g].append(value)
            elif op == ANY or op == ALL:
                self.logic[g].merge(other.logic[source])
            elif op == N_UNIQUE:
                if self.dtype == DataType.BOOL:
                    self.logic[g].merge(other.logic[source])
                    continue
                self.picked_valid[g] = (
                    self.picked_valid[g] or other.picked_valid[source]
                )
                if self.dtype == DataType.INT64:
                    for key in other.int_sets[source].keys():
                        self.int_sets[g][key] = True
                elif self.dtype == DataType.FLOAT64:
                    for key in other.float_sets[source].keys():
                        self.float_sets[g][key] = True
                else:
                    for key in other.string_sets[source].keys():
                        self.string_sets[g][key] = True
            elif op == MIN or op == MAX:
                if self.dtype == DataType.FLOAT64:
                    self.nan_seen[g] = (
                        self.nan_seen[g] or other.nan_seen[source]
                    )
                if not other.seen[source]:
                    continue
                var take = not self.seen[g]
                if not take:
                    if self.dtype.is_decimal():
                        take = (
                            other.decimals[source]
                            > self.decimals[g] if is_max else other.decimals[
                                source
                            ]
                            < self.decimals[g]
                        )
                    elif self.dtype == DataType.INT64:
                        take = (
                            other.ints[source]
                            > self.ints[g] if is_max else other.ints[source]
                            < self.ints[g]
                        )
                    elif self.dtype == DataType.FLOAT64:
                        take = (
                            other.floats[source]
                            > self.floats[g] if is_max else other.floats[source]
                            < self.floats[g]
                        )
                    elif self.dtype == DataType.BOOL:
                        take = (
                            other.bools[source]
                            > self.bools[g] if is_max else other.bools[source]
                            < self.bools[g]
                        )
                    else:
                        take = (
                            other.strings[source]
                            > self.strings[g] if is_max else other.strings[
                                source
                            ]
                            < self.strings[g]
                        )
                if take:
                    self._take_value(other, g, source)
            elif op == FIRST or op == LAST:
                if other.seen[source] and (op == LAST or not self.seen[g]):
                    self._take_value(other, g, source)
                    self.picked_valid[g] = other.picked_valid[source]

    def _take_value(mut self, other: Self, g: Int, source: Int):
        self.seen[g] = True
        if self.dtype.is_decimal():
            self.decimals[g] = other.decimals[source]
        elif self.dtype == DataType.INT64:
            self.ints[g] = other.ints[source]
        elif self.dtype == DataType.FLOAT64:
            self.floats[g] = other.floats[source]
        elif self.dtype == DataType.BOOL:
            self.bools[g] = other.bools[source]
        else:
            self.strings[g] = other.strings[source]

    def grow(mut self, count: Int):
        """Add empty states for groups [group_count, count), in place.

        Existing states are neither copied nor rebuilt: a reducer for just
        the new groups supplies their empty states, and each per-group list
        is extended with a copy of its counterpart, which holds only the new
        groups. Lists this operation does not use
        are empty in both, so they stay empty. Rebuilding the reducer and
        merging the old state into it (as this used to) re-inserted every
        n_unique set whenever a streaming batch brought a new group (#326).
        """
        if count <= self.group_count:
            return
        var added = Self(
            self.op,
            self.input,
            count - self.group_count,
            self.min_count,
            self.integer,
            self.floating,
            self.text,
            self.logical,
        )
        self.counts.extend(added.counts.copy())
        self.int_sums.extend(added.int_sums.copy())
        self.float_sums.extend(added.float_sums.copy())
        self.logic.extend(added.logic.copy())
        self.moments.extend(added.moments.copy())
        self.samples.extend(added.samples.copy())
        self.seen.extend(added.seen.copy())
        self.nan_seen.extend(added.nan_seen.copy())
        self.picked_valid.extend(added.picked_valid.copy())
        self.ints.extend(added.ints.copy())
        self.floats.extend(added.floats.copy())
        self.decimals.extend(added.decimals.copy())
        self.bools.extend(added.bools.copy())
        self.strings.extend(added.strings.copy())
        self.int_sets.extend(added.int_sets.copy())
        self.float_sets.extend(added.float_sets.copy())
        self.string_sets.extend(added.string_sets.copy())
        self.positions.extend(added.positions.copy())
        self.nan_positions.extend(added.nan_positions.copy())
        self.moments4.extend(added.moments4.copy())
        self.comoments.extend(added.comoments.copy())
        self.pair_x.extend(added.pair_x.copy())
        self.pair_y.extend(added.pair_y.copy())
        self.mode_ints.extend(added.mode_ints.copy())
        self.mode_floats.extend(added.mode_floats.copy())
        self.mode_float_values.extend(added.mode_float_values.copy())
        self.mode_strings.extend(added.mode_strings.copy())
        self.mode_nulls.extend(added.mode_nulls.copy())
        self.group_count = count

    def finish(self) raises -> Series:
        var op = self.op
        if op == MODE:
            return self._finish_mode()
        if op == VALUE_COUNTS:
            return self._finish_value_counts()
        if (
            op == ARG_MIN
            or op == ARG_MAX
            or op == SKEW
            or op == KURTOSIS
            or op == CORR
            or op == COV
        ):
            return self._finish()
        if self.logical.is_decimal():
            return self._finish()
        if self.input == self.dtype:
            return self._finish()
        if op == SUM and self.input.is_integer():
            return self._integer_sum()
        var result = self._finish()
        if op == SUM or op == MIN or op == MAX or op == FIRST or op == LAST:
            return _from_canonical(result, self.input)
        return result^

    def _finish_mode(self) raises -> Series:
        """Each group's most frequent values as a list: ascending, NaN above
        numbers, null last. A group with no rows has an empty list."""
        var n = self.group_count
        var offsets = List[Int64](capacity=n + 1)
        offsets.append(0)
        var total = 0
        var child: Series
        if len(self.mode_ints) > 0:
            var values = List[Int64]()
            var valid = List[Bool]()
            for g in range(n):
                var best = self.mode_nulls[g]
                for item in self.mode_ints[g].items():
                    best = max(best, item.value)
                if best > 0:
                    var chosen = List[Int64]()
                    for item in self.mode_ints[g].items():
                        if item.value == best:
                            chosen.append(item.key)
                    sort(chosen)
                    for value in chosen:
                        values.append(value)
                        valid.append(True)
                    if self.mode_nulls[g] == best:
                        values.append(0)
                        valid.append(False)
                    total = len(values)
                offsets.append(Int64(total))
            if self.input == DataType.BOOL:
                var flags = List[Bool](capacity=len(values))
                for value in values:
                    flags.append(value != 0)
                child = Series("", BoolColumn(flags^, valid^))
            else:
                child = _from_canonical(
                    Series("", Column[Int64](values^, valid^)), self.input
                )
        elif len(self.mode_floats) > 0:
            var values = List[Float64]()
            var valid = List[Bool]()
            for g in range(n):
                var best = self.mode_nulls[g]
                for item in self.mode_floats[g].items():
                    best = max(best, item.value)
                if best > 0:
                    var numbers = List[Float64]()
                    var nans = List[Float64]()
                    for item in self.mode_floats[g].items():
                        if item.value == best:
                            var value = self.mode_float_values[g][item.key]
                            if isnan(value):
                                nans.append(value)
                            else:
                                numbers.append(value)
                    sort(numbers)
                    for value in numbers:
                        values.append(value)
                        valid.append(True)
                    for value in nans:
                        values.append(value)
                        valid.append(True)
                    if self.mode_nulls[g] == best:
                        values.append(0)
                        valid.append(False)
                    total = len(values)
                offsets.append(Int64(total))
            child = _from_canonical(
                Series("", Column[Float64](values^, valid^)), self.input
            )
        else:
            var values = List[String]()
            var valid = List[Bool]()
            for g in range(n):
                var best = self.mode_nulls[g]
                for item in self.mode_strings[g].items():
                    best = max(best, item.value)
                if best > 0:
                    var chosen = List[String]()
                    for item in self.mode_strings[g].items():
                        if item.value == best:
                            chosen.append(item.key)
                    sort(chosen)
                    for value in chosen:
                        values.append(value)
                        valid.append(True)
                    if self.mode_nulls[g] == best:
                        values.append("")
                        valid.append(False)
                    total = len(values)
                offsets.append(Int64(total))
            child = Series("", StringColumn(values, valid))
        if self.logical.is_temporal():
            child = child.with_dtype(self.logical)
        return Series("", ListColumn(offsets^, child.renamed("item")))

    def _finish_value_counts(self) raises -> Series:
        """Per group, a list of {value, count} structs: every distinct value
        (null last), ascending by value, or by count descending when sorted
        (ties by value)."""
        var names = self.text.split(SEP)
        var value_name = String(names[0])
        var count_name = String(names[1])
        var by_count = (self.integer & 1) != 0
        var normalize = (self.integer & 2) != 0
        var n = self.group_count
        var offsets = List[Int64](capacity=n + 1)
        offsets.append(0)
        var counts = List[Int64]()
        # Per group: distinct keys in value order, as (rank, count) pairs;
        # `order` indexes into the flat value buffers built below.
        var values_i = List[Int64]()
        var values_f = List[Float64]()
        var values_s = List[String]()
        var valid = List[Bool]()
        var totals = List[Int64]()
        for g in range(n):
            var keys_i = List[Int64]()
            var keys_f = List[Float64]()
            var keys_s = List[String]()
            var key_counts = List[Int64]()
            if len(self.mode_ints) > 0:
                var ordered = List[Int64]()
                for key in self.mode_ints[g].keys():
                    ordered.append(key)
                sort(ordered)
                for key in ordered:
                    keys_i.append(key)
                    key_counts.append(self.mode_ints[g][key])
            elif len(self.mode_floats) > 0:
                var numbers = List[Float64]()
                var nans = List[Float64]()
                var lookup = Dict[UInt64, Int64]()
                for item in self.mode_floats[g].items():
                    var value = self.mode_float_values[g][item.key]
                    lookup[float_key(value)] = item.value
                    if isnan(value):
                        nans.append(value)
                    else:
                        numbers.append(value)
                sort(numbers)
                for value in numbers:
                    keys_f.append(value)
                    key_counts.append(lookup[float_key(value)])
                for value in nans:
                    keys_f.append(value)
                    key_counts.append(lookup[float_key(value)])
            else:
                var ordered = List[String]()
                for key in self.mode_strings[g].keys():
                    ordered.append(key)
                sort(ordered)
                for key in ordered:
                    keys_s.append(key)
                    key_counts.append(self.mode_strings[g][key])
            var distinct = len(key_counts)
            var slots = List[Int](capacity=distinct + 1)
            for i in range(distinct):
                slots.append(i)
            if self.mode_nulls[g] > 0:
                slots.append(-1)
            if by_count:
                # Stable insertion sort by count descending; value order
                # (null last) breaks ties.
                for i in range(1, len(slots)):
                    var j = i
                    while j > 0:
                        var here = (
                            key_counts[slots[j]] if slots[j]
                            >= 0 else self.mode_nulls[g]
                        )
                        var before = (
                            key_counts[slots[j - 1]] if slots[j - 1]
                            >= 0 else self.mode_nulls[g]
                        )
                        if here <= before:
                            break
                        var swap = slots[j]
                        slots[j] = slots[j - 1]
                        slots[j - 1] = swap
                        j -= 1
            var total = self.mode_nulls[g]
            for c in key_counts:
                total += c
            for slot in slots:
                if slot < 0:
                    counts.append(self.mode_nulls[g])
                    valid.append(False)
                    values_i.append(0)
                    values_f.append(0)
                    values_s.append("")
                else:
                    counts.append(key_counts[slot])
                    valid.append(True)
                    values_i.append(keys_i[slot] if len(keys_i) else 0)
                    values_f.append(keys_f[slot] if len(keys_f) else 0)
                    values_s.append(keys_s[slot] if len(keys_s) else "")
                totals.append(total)
            offsets.append(Int64(len(counts)))
        var values: Series
        if len(self.mode_ints) > 0:
            if self.input == DataType.BOOL:
                var flags = List[Bool](capacity=len(values_i))
                for value in values_i:
                    flags.append(value != 0)
                values = Series("", BoolColumn(flags^, valid.copy()))
            else:
                values = _from_canonical(
                    Series("", Column[Int64](values_i^, valid.copy())),
                    self.input,
                )
        elif len(self.mode_floats) > 0:
            values = _from_canonical(
                Series("", Column[Float64](values_f^, valid.copy())),
                self.input,
            )
        else:
            values = Series("", StringColumn(values_s, valid))
        if self.logical.is_temporal():
            values = values.with_dtype(self.logical)
        var tally: Series
        if normalize:
            var shares = List[Float64](capacity=len(counts))
            for i in range(len(counts)):
                shares.append(Float64(counts[i]) / Float64(totals[i]))
            tally = Series(count_name, Column[Float64](shares^))
        else:
            var narrow = List[UInt32](capacity=len(counts))
            for c in counts:
                narrow.append(UInt32(c))
            tally = Series(count_name, Column[UInt32](narrow^))
        var fields: List[Series] = [values.renamed(value_name), tally^]
        var structs = Series("", StructColumn(fields^))
        return Series("", ListColumn(offsets^, structs.renamed("item")))

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
        if op == ARG_MIN or op == ARG_MAX:
            var output = List[UInt32](length=n, fill=0)
            for g in range(n):
                valid[g] = self.seen[g] or self.nan_seen[g]
                output[g] = UInt32(
                    self.positions[g] if self.seen[g] else self.nan_positions[g]
                )
            return Series("", Column[UInt32](output^, valid))
        if op == SKEW or op == KURTOSIS:
            var output = List[Float64](length=n, fill=0)
            var bias = (self.integer & 1) != 0
            var fisher = (self.integer & 2) != 0
            for g in range(n):
                var value = self.moments4[g].skew(
                    bias
                ) if op == SKEW else self.moments4[g].kurtosis(fisher, bias)
                valid[g] = Bool(value)
                if value:
                    output[g] = value.value()
            return Series("", Column[Float64](output^, valid))
        if op == CORR or op == COV:
            var output = List[Float64](length=n, fill=0)
            for g in range(n):
                if len(self.pair_x) > 0:
                    output[g] = _spearman(self.pair_x[g], self.pair_y[g])
                elif op == CORR:
                    output[g] = self.comoments[g].correlation()
                else:
                    var value = self.comoments[g].covariance(Int(self.integer))
                    valid[g] = Bool(value)
                    if value:
                        output[g] = value.value()
            return Series("", Column[Float64](output^, valid))
        if op == COUNT or op == NULL_COUNT or op == LEN:
            return Series("", Column[Int64](self.counts.copy()))
        if (op == SUM or op == MEAN) and self.dtype.is_decimal():
            var output = self.decimals.copy()
            var decimal_valid = List[Bool](length=n, fill=False)
            for g in range(n):
                decimal_valid[g] = (
                    self.counts[g]
                    > 0 if op
                    == MEAN else self.counts[g]
                    >= Int64(self.min_count)
                )
                if op == MEAN and decimal_valid[g]:
                    output[g] /= Int128(self.counts[g])
            return Series(
                "", Column[Int128](output^, decimal_valid)
            ).with_dtype(self.dtype)
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
            if self.dtype.is_decimal():
                return Series(
                    "", Column[Int128](self.decimals.copy(), valid)
                ).with_dtype(self.dtype)
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
