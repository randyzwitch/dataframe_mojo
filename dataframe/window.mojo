"""Order-dependent kernels over whole columns, restarted per partition.

Most operations choose, for every output row, a source row to gather (or
none). Ordering uses the same dense sort ranks as `sort`, so ties keep the
earlier row. cum_min and cum_max skip NaN: a NaN row gets the running
value so far, and is NaN only before the first number (#816). A
rolling_min or rolling_max window holding a NaN is NaN.
"""
from dataframe.storage.boolean import BoolColumn
from dataframe.kernels.canonical_labels import canonical_labels
from dataframe.kernels.hashing import encode_rows
from dataframe.kernels.partition import encode_partitioned
from .parallel import worker_count
from dataframe.storage.column import Column, _copy_validity
from .dtype import DataType, NUMERIC_DTYPES
from .expr import (
    split_names,
    Node,
    CUM_SUM,
    CUM_MIN,
    CUM_MAX,
    CUM_COUNT,
    CUM_PROD,
    IS_DUPLICATED,
    IS_UNIQUE,
    IS_FIRST_DISTINCT,
    IS_LAST_DISTINCT,
    is_distinct_flag,
    SHIFT,
    SHIFT_IGNORE_NULLS,
    RANK,
    SUM,
    MEAN,
    MIN,
    MAX,
    STD,
    VAR,
    ROLLING_SUM,
    ROLLING_STD,
    ROLLING_VAR,
    ROLLING_SUM_BY,
    ROLLING_MEAN_BY,
    ROLLING_MIN_BY,
    ROLLING_MAX_BY,
    ROLLING_STD_BY,
    ROLLING_VAR_BY,
    ROLLING_MEAN,
    ROLLING_MIN,
    ROLLING_MAX,
    BACKWARD_FILL,
    INTERPOLATE,
    CUT,
    QCUT,
    SEP,
)
from dataframe.kernels.binary import checked_add, validity
from dataframe.storage.string import StringColumn
from dataframe.storage.nested import StructColumn
from dataframe.text.parse import parse_float64
from dataframe.aggregation.reducer import quantile_of, _checked_product
from .decimal import check_limit, pow10, precision_limit
from dataframe.aggregation.function import ReducerSpec
from .series import Series, FixedElements
from .window_segment_tree import frame_states_supported, window_segment_tree
from .window_sort import rank_rows
from std.collections import Optional
from std.math import isnan, floor, sqrt, isfinite


def partitions(n: Int, ids: List[Int]) -> List[List[Int]]:
    """Row indices per partition in row order; no ids means one partition."""
    var result = List[List[Int]]()
    if len(ids) == 0:
        var rows = List[Int](capacity=n)
        for i in range(n):
            rows.append(i)
        result.append(rows^)
        return result^
    var count = 0
    for id in ids:
        count = max(count, id + 1)
    result = List[List[Int]](length=count, fill=List[Int]())
    for i in range(n):
        result[ids[i]].append(i)
    return result^


def _ordered(rows: List[Int], reverse: Bool) -> List[Int]:
    if not reverse:
        return rows.copy()
    var out = List[Int](capacity=len(rows))
    for k in range(len(rows) - 1, -1, -1):
        out.append(rows[k])
    return out^


def _numeric(input: Series, row: Int) -> Float64:
    """Int64 or Float64 input (narrow types are widened first)."""
    if input._data.isa[Column[Int64]]():
        return Float64(input._data[Column[Int64]]._get(row))
    return input._data[Column[Float64]]._get(row)


def _is_narrow(dtype: DataType) -> Bool:
    """A numeric type the window kernels widen to Int64 or Float64 first;
    not a decimal, whose values are scaled integers (decimal sums run in
    128 bits at their scale, other windows over decimals in Float64)."""
    return (
        dtype.is_numeric()
        and not dtype.is_decimal()
        and dtype != DataType.INT64
        and (dtype != DataType.FLOAT64)
    )


def _widen(input: Series) raises -> Series:
    """Narrow numeric input as Int64 or Float64 (exact; UInt64 values above
    Int64 range raise)."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if input._data.isa[Column[Scalar[D]]]():
            ref column = input._data[Column[Scalar[D]]]
            var valid = validity(input)
            comptime if D.is_floating_point():
                var values = List[Float64](capacity=len(column))
                for i in range(len(column)):
                    values.append(column._get(i).cast[DType.float64]())
                return Series("", Column[Float64](values^, valid))
            else:
                var values = List[Int64](capacity=len(column))
                for i in range(len(column)):
                    var x = column._get(i)
                    comptime if D == DType.uint64:
                        if valid[i] and x > Int64.MAX.cast[D]():
                            raise Error("uint64 window sum overflow")
                    values.append(x.cast[DType.int64]())
                return Series("", Column[Int64](values^, valid))
    return input.copy()


def _narrow_sum(result: Series, target: DataType) raises -> Series:
    """An Int64/Float64 window result in its sum type, range-checked."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if D != DType.int64 and D != DType.float64:
            if target == DataType.of(D):
                var valid = validity(result)
                var values = List[Scalar[D]](capacity=len(result))
                comptime if D.is_floating_point():
                    ref column = result._data[Column[Float64]]
                    for i in range(len(column)):
                        values.append(column._get(i).cast[D]())
                else:
                    ref column = result._data[Column[Int64]]
                    for i in range(len(column)):
                        var x = column._get(i)
                        if valid[i] and (
                            x.cast[DType.int128]()
                            > Scalar[D].MAX.cast[DType.int128]()
                            or x.cast[DType.int128]()
                            < Scalar[D].MIN.cast[DType.int128]()
                        ):
                            raise Error(target.name() + " window sum overflow")
                        values.append(x.cast[D]())
                return Series("", Column[Scalar[D]](values^, valid))
    return result.copy()


def _distinct_flags(op: Int, input: Series, ids: List[Int]) raises -> Series:
    var keys = List[Series]()
    if input.dtype().is_struct():
        var structs = input.struct_column()
        var picks = List[Int]()
        if input.null_count() > 0:
            for row in range(len(input)):
                picks.append(row if structs._valid(row) else -1)
        for k in range(structs.field_count()):
            var field = structs.field(k)
            if len(picks) > 0:
                field = field.take_or_null(picks)
            keys.append(canonical_labels(field))
        keys.append(Series("", BoolColumn(structs.validity())))
    else:
        keys.append(canonical_labels(input))
    if len(ids) > 0:
        var partitions = List[Int64](capacity=len(ids))
        for id in ids:
            partitions.append(Int64(id))
        keys.append(Series("", Column[Int64](partitions^)))
    var workers = worker_count(len(input))
    var encoded = encode_partitioned(
        keys, workers, nulls_equal=True
    ) if workers > 1 else encode_rows(keys, nulls_equal=True)
    var values = List[UInt8](length=(len(input) + 7) // 8, fill=0)
    if op == IS_FIRST_DISTINCT:
        for row in encoded.representatives:
            values[row // 8] |= UInt8(1) << UInt8(row % 8)
    elif op == IS_LAST_DISTINCT:
        var last = encoded.representatives.copy()
        for row in range(len(input)):
            last[encoded.ids[row]] = row
        for row in last:
            values[row // 8] |= UInt8(1) << UInt8(row % 8)
    else:
        var counts = List[Int](length=encoded.count(), fill=0)
        for id in encoded.ids:
            counts[id] += 1
        for row in range(len(input)):
            var unique = counts[encoded.ids[row]] == 1
            if unique == (op == IS_UNIQUE):
                values[row // 8] |= UInt8(1) << UInt8(row % 8)
    return Series(
        "", BoolColumn(values=values^, bits=List[UInt8](), length=len(input))
    )


def window_op(node: Node, input: Series, ids: List[Int]) raises -> Series:
    if input.is_chunked():
        return window_op(node, input.rechunk(), ids)
    var op_code = node.op
    if is_distinct_flag(op_code):
        return _distinct_flags(op_code, input, ids)
    if op_code == CUM_PROD:
        return _cum_prod(input, ids, node.min_count == 1)
    if op_code == CUM_SUM and input.dtype() == DataType.BOOL:
        # Cast Booleans to UInt32 zeros and ones, then sum them to count
        # true values cumulatively.
        return window_op(node, input.cast(DataType.UINT32), ids)
    if input.dtype().is_decimal():
        if op_code == CUM_SUM:
            return _decimal_cum_sum(input, ids, node.min_count == 1)
        if op_code == ROLLING_SUM:
            raise Error(
                "rolling_sum over a decimal column is not supported yet;"
                " cast it to Float64 first"
            )
        if (
            op_code == ROLLING_MEAN
            or op_code == ROLLING_STD
            or op_code == ROLLING_VAR
        ):
            return window_op(node, _as_float(input), ids)
    if (op_code == ROLLING_STD or op_code == ROLLING_VAR) and _is_narrow(
        input.dtype()
    ):
        return window_op(node, input.cast(DataType.FLOAT64), ids)
    if (
        (
            op_code == CUM_SUM
            or op_code == ROLLING_SUM
            or op_code == ROLLING_MEAN
            or op_code == ROLLING_STD
            or op_code == ROLLING_VAR
        )
        and _is_narrow(input.dtype())
        and not (
            input.dtype() == DataType.UINT64
            and op_code in [CUM_SUM, ROLLING_SUM]
        )
    ):
        var result = window_op(node, _widen(input), ids)
        if (
            op_code == ROLLING_MEAN
            or op_code == ROLLING_STD
            or op_code == ROLLING_VAR
        ):
            return result^
        return _narrow_sum(result, input.dtype().sum_type())
    if (
        op_code == CUM_COUNT
        or op_code == CUM_SUM
        or op_code == CUM_MIN
        or op_code == CUM_MAX
    ):
        var running = _running(op_code, input, ids, node.min_count == 1)
        if running:
            return running.take()
    if op_code == RANK:
        # rank() OVER (PARTITION BY ids ORDER BY input): the window layout
        # and its peer boundaries.
        return rank_rows(input, ids, node.text, node.min_count == 1)
    var parts = _contiguous(len(input), ids)
    if len(parts.order) == 0:
        return _window_ranges(node, input, parts.starts)
    # Partitions laid out one after another, evaluated, and put back.
    var result = _window_ranges(node, input.take(parts.order), parts.starts)
    var inverse = List[Int](unsafe_uninit_length=len(parts.order))
    for k in range(len(parts.order)):
        inverse[parts.order[k]] = k
    return result.take(inverse)


struct _Contiguous(Movable):
    """Partitions as row ranges: `starts` bounds each partition's rows in
    `order` (rows grouped by partition, input order within one), or in the
    rows themselves when `order` is empty (the ids never decrease, as in a
    window's sorted layout)."""

    var order: List[Int]
    var starts: List[Int]

    def __init__(out self, var order: List[Int], var starts: List[Int]):
        self.order = order^
        self.starts = starts^


def _contiguous(n: Int, ids: List[Int]) -> _Contiguous:
    """Row ranges of the partitions `ids` gives (no ids: one partition)."""
    var starts = List[Int]()
    starts.append(0)
    if len(ids) == 0 or n == 0:
        if n > 0:
            starts.append(n)
        return _Contiguous(List[Int](), starts^)
    var ip = ids.unsafe_ptr()
    var sorted = True
    var count = 0
    for i in range(n):
        count = max(count, ip[unsafe_offset=i] + 1)
        if i > 0 and ip[unsafe_offset=i] < ip[unsafe_offset=i - 1]:
            sorted = False
    if sorted:
        for i in range(1, n):
            if ip[unsafe_offset=i] != ip[unsafe_offset=i - 1]:
                starts.append(i)
        starts.append(n)
        return _Contiguous(List[Int](), starts^)
    # A stable counting sort by id.
    var at = List[Int](length=count + 1, fill=0)
    for i in range(n):
        at[ip[unsafe_offset=i] + 1] += 1
    for g in range(count):
        at[g + 1] += at[g]
    for g in range(1, count + 1):
        if at[g] != at[g - 1]:
            starts.append(at[g])
    var order = List[Int](unsafe_uninit_length=n)
    for i in range(n):
        var g = ip[unsafe_offset=i]
        order[at[g]] = i
        at[g] += 1
    return _Contiguous(order^, starts^)


def _range_lists(starts: List[Int]) -> List[List[Int]]:
    """Each partition's rows, for the kernels that take row lists."""
    var groups = List[List[Int]](capacity=len(starts) - 1)
    for p in range(len(starts) - 1):
        var rows = List[Int](capacity=starts[p + 1] - starts[p])
        for row in range(starts[p], starts[p + 1]):
            rows.append(row)
        groups.append(rows^)
    return groups^


def _window_ranges(
    node: Node, input: Series, starts: List[Int]
) raises -> Series:
    """`window_op` over partitions that are row ranges."""
    var n = len(input)
    var valid = validity(input)
    var op = node.op
    var reverse = node.min_count == 1
    var parts = len(starts) - 1
    if op == CUM_COUNT:
        var counts = List[Int64](length=n, fill=0)
        for p in range(parts):
            var running = Int64(0)
            for j in range(starts[p], starts[p + 1]):
                var row = starts[p + 1] - 1 - (j - starts[p]) if reverse else j
                if valid[row]:
                    running += 1
                counts[row] = running
        return Series("", Column[Int64](counts^))
    if op == CUM_SUM:
        var is_int = input._data.isa[Column[Int64]]()
        var ints = List[Int64](length=n if is_int else 0, fill=0)
        var floats = List[Float64](length=0 if is_int else n, fill=0)
        for p in range(parts):
            var int_total = Int64(0)
            var float_total = Float64(0)
            for j in range(starts[p], starts[p + 1]):
                var row = starts[p + 1] - 1 - (j - starts[p]) if reverse else j
                if not valid[row]:
                    continue
                if is_int:
                    int_total = checked_add(
                        int_total, input._data[Column[Int64]]._get(row)
                    )
                    ints[row] = int_total
                else:
                    float_total += input._data[Column[Float64]]._get(row)
                    floats[row] = float_total
        if is_int:
            return Series("", Column[Int64](ints^, valid))
        return Series("", Column[Float64](floats^, valid))
    if op == INTERPOLATE:
        return _interpolate(input, _range_lists(starts), node.text)
    if op == CUT or op == QCUT:
        return _bins(input, _range_lists(starts), node)
    if (
        op == ROLLING_STD
        or op == ROLLING_VAR
        or op == ROLLING_SUM
        or op == ROLLING_MEAN
        or op == ROLLING_MIN
        or op == ROLLING_MAX
    ):
        return _rolling_frames(input, starts, node)
    # The remaining operations gather a source row per output row.
    var source = List[Int](length=n, fill=-1)
    var ranks = List[Int]()
    if op == CUM_MIN or op == CUM_MAX:
        ranks = input._sort_ranks(False, True)
    var take_max = op == CUM_MAX
    var nan = List[Bool]()
    if len(ranks) > 0 and input.dtype().is_float():
        nan = _nan_rows(input, valid)
    var has_nan = len(nan) > 0
    for p in range(parts):
        var first = starts[p]
        var m = starts[p + 1] - first
        if op == SHIFT:
            var k = Int(node.integer)
            if (node.min_count & SHIFT_IGNORE_NULLS) != 0 and k != 0:
                # Only valid predecessors/successors count toward the offset.
                # Record the current row after selecting its source so a NULL
                # output row still looks strictly before/after itself.
                var history = List[Int](capacity=m)
                for j in range(m):
                    var row = first + (m - 1 - j if k < 0 else j)
                    var seen = len(history)
                    if k > 0:
                        if k <= seen:
                            source[row] = history[seen - k]
                    elif k >= -seen:
                        # Keeping k negative also handles the minimum Int
                        # offset without trying to negate it.
                        source[row] = history[seen + k]
                    if valid[row]:
                        history.append(row)
            else:
                for j in range(m):
                    var from_index = j - k
                    if from_index >= 0 and from_index < m:
                        source[first + j] = first + from_index
        elif op == CUM_MIN or op == CUM_MAX:
            var best = -1
            for j in range(m):
                var row = first + (m - 1 - j if reverse else j)
                if valid[row]:
                    if has_nan and nan[row]:
                        # Skipped, unless nothing but NaN has come.
                        source[row] = best if best >= 0 else row
                        continue
                    if best < 0 or (
                        ranks[row]
                        > ranks[best] if take_max else ranks[row]
                        < ranks[best]
                    ):
                        best = row
                    source[row] = best
        else:
            var limit = Int(node.integer)
            var last = -1
            var distance = 0
            for j in range(m):
                var row = first + (m - 1 - j if op == BACKWARD_FILL else j)
                if valid[row]:
                    last = row
                    distance = 0
                    source[row] = row
                else:
                    distance += 1
                    if last >= 0 and (limit < 0 or distance <= limit):
                        source[row] = last
    return input.take_or_null(source)


def _nan_rows(input: Series, valid: List[Bool]) raises -> List[Bool]:
    """Whether each valid row of a float column holds NaN."""
    var floats = _as_float(input)
    ref column = floats._data[Column[Float64]]
    var out = List[Bool](length=len(input), fill=False)
    for i in range(len(input)):
        out[i] = valid[i] and isnan(column._get(i))
    return out^


def _cum_prod(input: Series, ids: List[Int], reverse: Bool) raises -> Series:
    """A running product within each partition, in one pass over the rows
    (from the last when reversed), each partition's product kept by id.
    Integers multiply exactly and raise on leaving the result type
    (`product_type`: Int64, or UInt64 for UInt64); floats and decimals
    multiply in Float64. Null rows stay null and leave the product as is."""
    var target = input.dtype().product_type()
    var values: Series
    if target.is_float():
        values = input.cast(DataType.FLOAT64)
    else:
        values = input.cast(target)
    var n = len(values)
    var valid = validity(values)
    var partitioned = len(ids) > 0
    var groups = 1
    if partitioned:
        for i in range(n):
            groups = max(groups, ids[i] + 1)
    var result: Series
    if target.is_float():
        ref column = values._data[Column[Float64]]
        var state = List[Float64](length=groups, fill=1)
        var out = List[Float64](length=n, fill=0)
        for step in range(n):
            var i = n - 1 - step if reverse else step
            if not valid[i]:
                continue
            var g = ids[i] if partitioned else 0
            state[g] *= column._get(i)
            out[i] = state[g]
        result = Series("", Column[Float64](out^, valid))
        if target == DataType.FLOAT32:
            result = result.cast(DataType.FLOAT32)
        return result^
    var unsigned = target == DataType.UINT64
    var state = List[Int128](length=groups, fill=1)
    var out = List[Int128](length=n, fill=0)
    for step in range(n):
        var i = n - 1 - step if reverse else step
        if not valid[i]:
            continue
        var g = ids[i] if partitioned else 0
        var value: Int128
        if unsigned:
            value = values._data[Column[UInt64]]._get(i).cast[DType.int128]()
        else:
            value = values._data[Column[Int64]]._get(i).cast[DType.int128]()
        state[g] = _checked_product(state[g], value, unsigned)
        out[i] = state[g]
    if unsigned:
        var products = List[UInt64](capacity=n)
        for value in out:
            products.append(value.cast[DType.uint64]())
        return Series("", Column[UInt64](products^, valid))
    var products = List[Int64](capacity=n)
    for value in out:
        products.append(value.cast[DType.int64]())
    return Series("", Column[Int64](products^, valid))


def _running(
    op: Int, input: Series, ids: List[Int], reverse: Bool
) raises -> Optional[Series]:
    """A cumulative count, sum, min or max in one pass over the rows (from the
    last row when reversed), each partition's running value kept in a flat array
    by partition id. The general path below collects a row list per partition
    and, for min and max, sorts the whole column to compare ranks, O(n log n)
    and several passes where this is one O(n) pass. Int64 and Float64 input
    (others are widened first or keep the general path). None when not
    handled."""
    var n = len(input)
    var groups = 1
    var partitioned = len(ids) > 0
    if partitioned:
        var id_ptr = ids.unsafe_ptr()
        for i in range(n):
            groups = max(groups, id_ptr[unsafe_offset=i] + 1)
    if op == CUM_COUNT:
        # Counts read the validity bitmap as stored; a column without
        # nulls counts every row.
        var bits = List[UInt8]()
        var offset = 0
        comptime for k in range(len(FixedElements.Ts)):
            comptime E: Copyable & Deinitable = FixedElements.Ts[k]
            if input._data.isa[Column[E]]():
                ref column = input._data[Column[E]]
                if column.null_count() > 0:
                    bits = _copy_validity(column._bits[], column._offset, n)
        if len(bits) == 0 and input.null_count() > 0:
            # Booleans, strings and nested values.
            var flags = validity(input)
            bits = List[UInt8](length=(n + 7) // 8, fill=0)
            for i in range(n):
                if flags[i]:
                    bits[i >> 3] |= UInt8(1) << UInt8(i & 7)
        var masked = len(bits) > 0
        var mask = bits.unsafe_ptr()
        var partition = ids.unsafe_ptr()
        var state = List[Int64](length=groups, fill=0)
        var out = List[Int64](length=n, fill=0)
        var counts = state.unsafe_ptr()
        var written = out.unsafe_ptr()
        for step in range(n):
            var i = n - 1 - step if reverse else step
            var g = partition[unsafe_offset=i] if partitioned else 0
            if (
                not masked
                or ((mask[unsafe_offset=i >> 3] >> UInt8(i & 7)) & 1) != 0
            ):
                counts[unsafe_offset=g] += 1
            written[unsafe_offset=i] = counts[unsafe_offset=g]
        return Series("", Column[Int64](out^))
    if input._data.isa[Column[UInt64]]():
        return _running_values[DType.uint64](
            op, input, input._data[Column[UInt64]], ids, groups, reverse
        )
    if input._data.isa[Column[Int64]]():
        var result = _running_values[DType.int64](
            op, input, input._data[Column[Int64]], ids, groups, reverse
        )
        if input.dtype().is_decimal():
            # decimal64: CUM_SUM went to `_decimal_cum_sum` above.
            return result.with_dtype(input.dtype())
        return result^
    if op == CUM_MIN or op == CUM_MAX:
        # A running min or max is one of the input's own values, so it
        # keeps the input's type: Int32 storage (dates, decimal32) and
        # Int128 (decimal128) compare as stored, a decimal's values all
        # sharing its scale.
        if input._data.isa[Column[Int32]]():
            return _running_values[DType.int32](
                op, input, input._data[Column[Int32]], ids, groups, reverse
            ).with_dtype(input.dtype())
        if input._data.isa[Column[Int128]]():
            return _running_values[DType.int128](
                op, input, input._data[Column[Int128]], ids, groups, reverse
            ).with_dtype(input.dtype())
    if input._data.isa[Column[Float64]]():
        ref column = input._data[Column[Float64]]
        return _running_values[DType.float64](
            op, input, column, ids, groups, reverse
        )
    return None


def _running_values[
    D: DType
](
    op: Int,
    input: Series,
    column: Column[Scalar[D]],
    ids: List[Int],
    groups: Int,
    reverse: Bool,
) raises -> Series:
    """`_running` over Int64 or Float64 values, through pointers. A column
    without nulls reads no validity and writes none; with nulls, a null row
    stays null and does not change its partition's running value."""
    var n = len(column)
    var values = column.unsafe_values()
    var partitioned = len(ids) > 0
    var partition = ids.unsafe_ptr()
    var nulls = column.null_count() > 0
    var state = List[Scalar[D]](length=groups, fill=Scalar[D](0))
    var seen = List[Bool](length=groups, fill=False)
    var out = List[Scalar[D]](length=n, fill=Scalar[D](0))
    var running = state.unsafe_ptr()
    var started = seen.unsafe_ptr()
    var written = out.unsafe_ptr()
    for step in range(n):
        var i = n - 1 - step if reverse else step
        if nulls and not column._valid(i):
            continue
        var g = partition[unsafe_offset=i] if partitioned else 0
        var value = values.unsafe_load(i)
        var current = running[unsafe_offset=g]
        comptime if D == DType.float64:
            if op != CUM_SUM and isnan(value):
                # A running min or max skips NaN: the row gets the value
                # so far, or NaN when no number has come yet (#816).
                written[unsafe_offset=i] = current if started[
                    unsafe_offset=g
                ] else value
                continue
        if op == CUM_SUM:
            comptime if D == DType.int64:
                current = rebind[Scalar[D]](
                    checked_add(rebind[Int64](current), rebind[Int64](value))
                )
            elif D == DType.uint64:
                if value > Scalar[D].MAX - current:
                    raise Error("uint64 window sum overflow")
                current += value
            else:
                current += value
        elif not started[unsafe_offset=g]:
            current = value
        elif op == CUM_MAX:
            # Strictly greater, as the ranked path keeps the first of equal
            # values (-0.0 and 0.0).
            if value > current:
                current = value
        elif value < current:
            current = value
        started[unsafe_offset=g] = True
        running[unsafe_offset=g] = current
        written[unsafe_offset=i] = current
    if nulls:
        return Series(
            "",
            Column[Scalar[D]](
                values=out^,
                bits=_copy_validity(column._bits[], column._offset, n),
            ),
        )
    return Series("", Column[Scalar[D]](out^))


def _decimal_cum_sum(
    input: Series, ids: List[Int], reverse: Bool
) raises -> Series:
    """A running sum of a decimal column within each partition, exact in
    128 bits at the input's scale, typed as SQL's SUM of it (decimal(38,
    scale) for a decimal32 or decimal64 input); nulls are skipped and stay
    null. A running total past 38 digits raises. One pass over the rows
    with each partition's total in a flat array, as `_running` does, read
    from the stored width."""
    var n = len(input)
    var groups = 1
    if len(ids) > 0:
        var id_ptr = ids.unsafe_ptr()
        for i in range(n):
            groups = max(groups, id_ptr[unsafe_offset=i] + 1)
    var target = input.dtype().sum_type()
    if input._data.isa[Column[Int32]]():
        return _decimal_running_sum[DType.int32](
            input._data[Column[Int32]], ids, groups, reverse, target
        )
    if input._data.isa[Column[Int64]]():
        return _decimal_running_sum[DType.int64](
            input._data[Column[Int64]], ids, groups, reverse, target
        )
    return _decimal_running_sum[DType.int128](
        input._data[Column[Int128]], ids, groups, reverse, target
    )


def _decimal_running_sum[
    D: DType
](
    column: Column[Scalar[D]],
    ids: List[Int],
    groups: Int,
    reverse: Bool,
    target: DataType,
) raises -> Series:
    """`_decimal_cum_sum` over decimal storage of width D, widening each
    value to 128 bits as it is read."""
    var n = len(column)
    var limit = precision_limit(target.precision())
    var values = column.unsafe_values()
    var partitioned = len(ids) > 0
    var partition = ids.unsafe_ptr()
    var nulls = column.null_count() > 0
    var state = List[Int128](length=groups, fill=0)
    var out = List[Int128](length=n, fill=0)
    var running = state.unsafe_ptr()
    var written = out.unsafe_ptr()
    for step in range(n):
        var i = n - 1 - step if reverse else step
        if nulls and not column._valid(i):
            continue
        var g = partition[unsafe_offset=i] if partitioned else 0
        var total = check_limit(
            running[unsafe_offset=g] + Int128(values.unsafe_load(i)),
            limit,
            target,
        )
        running[unsafe_offset=g] = total
        written[unsafe_offset=i] = total
    var bits = _copy_validity(
        column._bits[], column._offset, n
    ) if nulls else List[UInt8]()
    return Series("", Column[Int128](values=out^, bits=bits^)).with_dtype(
        target
    )


def _as_float(input: Series) raises -> Series:
    """Convert a primitive numeric column to Float64, preserving validity."""
    var contiguous = input.rechunk()
    if contiguous._data.isa[Column[Float64]]():
        return contiguous.copy()
    var valid = validity(contiguous)
    if contiguous.dtype().is_decimal():
        var wide = contiguous._decimal128()
        ref column = wide._data[Column[Int128]]
        var values = List[Float64](capacity=len(column))
        var divisor = pow10(contiguous.dtype().scale()).cast[DType.float64]()
        for i in range(len(column)):
            values.append(column._get(i).cast[DType.float64]() / divisor)
        return Series("", Column[Float64](values^, valid))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if contiguous._data.isa[Column[Scalar[D]]]():
            ref column = contiguous._data[Column[Scalar[D]]]
            var values = List[Float64](capacity=len(column))
            for i in range(len(column)):
                values.append(column._get(i).cast[DType.float64]())
            return Series("", Column[Float64](values^, valid))
    raise Error("numeric window input has unsupported storage")


def _interpolate(
    input: Series, groups: List[List[Int]], method: String
) raises -> Series:
    var floats = _as_float(input)
    var valid = validity(input)
    ref values = floats._data[Column[Float64]]
    var out = List[Float64](length=len(input), fill=0)
    var source = List[Int](length=len(input), fill=-1)
    for i in range(len(input)):
        if valid[i]:
            out[i] = values._get(i)
            source[i] = i
    for rows in groups:
        var previous = -1
        for j in range(len(rows)):
            var row = rows[j]
            if valid[row]:
                previous = j
                continue
            if previous < 0:
                continue
            var following = j + 1
            while following < len(rows) and not valid[rows[following]]:
                following += 1
            if following == len(rows):
                continue
            var before = rows[previous]
            var after = rows[following]
            if method == "nearest":
                source[row] = before if j - previous < following - j else after
            else:
                var fraction = Float64(j - previous) / Float64(
                    following - previous
                )
                out[row] = (
                    values._get(before)
                    + (values._get(after) - values._get(before)) * fraction
                )
                source[row] = row
    if method == "nearest":
        return input.take_or_null(source)
    var out_valid = List[Bool](capacity=len(source))
    for row in source:
        out_valid.append(row >= 0)
    return Series("", Column[Float64](out^, out_valid))


def interpolate_by_op(
    input: Series, by: Series, ids: List[Int]
) raises -> Series:
    var floats = _as_float(input)
    var axis = _as_float(by)
    var valid = validity(input)
    var by_valid = validity(by)
    ref values = floats._data[Column[Float64]]
    ref positions = axis._data[Column[Float64]]
    var out = List[Float64](length=len(input), fill=0)
    var out_valid = valid.copy()
    for i in range(len(input)):
        if valid[i]:
            out[i] = values._get(i)
    for rows in partitions(len(input), ids):
        var previous = -1
        for j in range(len(rows)):
            var row = rows[j]
            if valid[row] and by_valid[row]:
                previous = j
                continue
            if valid[row] or not by_valid[row] or previous < 0:
                continue
            var following = j + 1
            while following < len(rows) and not (
                valid[rows[following]] and by_valid[rows[following]]
            ):
                following += 1
            if following == len(rows):
                continue
            var before = rows[previous]
            var after = rows[following]
            var span = positions._get(after) - positions._get(before)
            if span == 0:
                continue
            var fraction = (positions._get(row) - positions._get(before)) / span
            out[row] = (
                values._get(before)
                + (values._get(after) - values._get(before)) * fraction
            )
            out_valid[row] = True
    return Series("", Column[Float64](out^, out_valid))


def _edge_text(value: Float64) -> String:
    if value == Float64(1) / Float64(0):
        return "inf"
    if value == -Float64(1) / Float64(0):
        return "-inf"
    if (
        value >= Float64(Int64.MIN)
        and value <= Float64(Int64.MAX)
        and value == floor(value)
    ):
        return String(Int64(value))
    return String(value)


def _bin_label(edges: List[Float64], bin: Int, left_closed: Bool) -> String:
    var lower = "-inf" if bin == 0 else _edge_text(edges[bin - 1])
    var upper = "inf" if bin == len(edges) else _edge_text(edges[bin])
    if left_closed:
        return "[" + lower + ", " + upper + ")"
    return "(" + lower + ", " + upper + "]"


def _bins(input: Series, groups: List[List[Int]], node: Node) raises -> Series:
    var floats = _as_float(input)
    ref values = floats._data[Column[Float64]]
    var valid = validity(input)
    var out_valid = valid.copy()
    var categories = List[String](length=len(input), fill="")
    var breakpoints = List[Float64](length=len(input), fill=0)
    var specs = List[Float64]()
    if node.text.byte_length() > 0:
        for part in node.text.split(SEP):
            specs.append(parse_float64(String(part)))
    var custom = List[String]()
    if node.text2.byte_length() > 0:
        for label in split_names(node.text2):
            custom.append(label)
    var left_closed = (node.integer & 1) != 0
    var allow_duplicates = (node.integer & 4) != 0
    for rows in groups:
        var edges = specs.copy()
        var label_bins = List[Int]()
        label_bins.append(0)
        if node.op == QCUT:
            var samples = List[Float64]()
            for row in rows:
                if valid[row] and not isnan(values._get(row)):
                    samples.append(values._get(row))
            edges = List[Float64]()
            for i in range(len(specs)):
                var q = specs[i]
                if q < 0 or q > 1:
                    raise Error("qcut quantiles must be between 0 and 1")
                var edge = quantile_of(samples.copy(), q, "linear")
                if not edge:
                    continue
                var value = edge.value()
                if len(edges) > 0 and not value > edges[len(edges) - 1]:
                    if not allow_duplicates or value != edges[len(edges) - 1]:
                        raise Error(
                            "breaks must be unique and strictly increasing"
                        )
                    label_bins[len(label_bins) - 1] = i + 1
                else:
                    edges.append(value)
                    label_bins.append(i + 1)
        else:
            for i in range(1, len(edges)):
                if not edges[i] > edges[i - 1]:
                    raise Error("breaks must be unique and strictly increasing")
            for i in range(len(edges)):
                label_bins.append(i + 1)
        for row in rows:
            var value = values._get(row)
            if not valid[row] or isnan(value):
                out_valid[row] = False
                continue
            var bin = 0
            while bin < len(edges) and (
                value >= edges[bin] if left_closed else value > edges[bin]
            ):
                bin += 1
            categories[row] = custom[label_bins[bin]] if len(
                custom
            ) > 0 else _bin_label(edges, bin, left_closed)
            breakpoints[row] = edges[bin] if bin < len(edges) else Float64(
                1
            ) / Float64(0)
    var labels = Series("category", StringColumn(categories, out_valid.copy()))
    if (node.integer & 2) == 0:
        return labels.renamed("")
    var breaks = Series(
        "breakpoint", Column[Float64](breakpoints^, out_valid.copy())
    )
    return Series("", StructColumn([breaks^, labels^]))


def _reduction_of(op: Int) -> Int:
    """The reduction a rolling window applies to each frame."""
    if op == ROLLING_SUM or op == ROLLING_SUM_BY:
        return SUM
    if op == ROLLING_MEAN or op == ROLLING_MEAN_BY:
        return MEAN
    if op == ROLLING_MIN or op == ROLLING_MIN_BY:
        return MIN
    if op == ROLLING_MAX or op == ROLLING_MAX_BY:
        return MAX
    if op == ROLLING_VAR or op == ROLLING_VAR_BY:
        return VAR
    return STD


def _frames(
    node: Node,
    input: Series,
    begins: List[Int],
    ends: List[Int],
    starts: List[Int],
    needed: Int,
) raises -> Series:
    """Each row's frame [begins, ends) of `input`, inside the partitions
    at `starts`: the states the
    aggregate table uses, combined over the frame by the segment tree.
    Fewer than `needed` valid values give null, and a frame without any
    gives null unless it is an empty sum. A minimum or maximum over a frame
    holding a NaN is NaN."""
    var reduction = _reduction_of(node.op)
    var moments = reduction == STD or reduction == VAR
    var feed = _origin_shifted(input, starts) if moments else input.copy()
    var spec = ReducerSpec(
        reduction,
        feed.dtype().physical(),
        node.min_count if moments else 0,
        Int64(node.min_count) if moments else 0,
        0,
        "",
        feed.dtype(),
        None,
    )
    if not frame_states_supported(spec, feed):
        return _ranked_frames(spec, input, begins, ends, starts, needed)
    var frames = window_segment_tree(spec, feed, begins, ends, starts, needed)
    var n = len(begins)
    var empty_ok = node.op == ROLLING_SUM_BY
    var source = List[Int](unsafe_uninit_length=n)
    var nulls = False
    for j in range(n):
        var count = Int(frames.counts[j])
        var ok = (
            count >= needed
            and (count > 0 or empty_ok)
            and (not moments or count > node.min_count)
        )
        source[j] = j if ok else -1
        nulls = nulls or not ok
    var values = frames.values.copy()
    if len(frames.nan) > 0:
        values = _with_nan(values, frames.nan)
    return values.take_or_null(source) if nulls else values^


def _origin_shifted(input: Series, starts: List[Int]) raises -> Series:
    """`input` as Float64 with each partition's median finite value
    subtracted, which leaves a variance unchanged: residuals are then not
    lost to a large common offset (Welford's rounding is relative to the
    values it adds). The median, so one huge outlier does not become the
    origin."""
    var floats = input.cast(DataType.FLOAT64)
    ref column = floats._data[Column[Float64]]
    var out = List[Float64](unsafe_uninit_length=len(column))
    var ok = List[Bool](unsafe_uninit_length=len(column))
    for p in range(len(starts) - 1):
        var finite = List[Float64]()
        for j in range(starts[p], starts[p + 1]):
            if column._valid(j) and isfinite(column._get(j)):
                finite.append(column._get(j))
        var origin = Float64(0)
        if len(finite) > 0:
            sort(finite)
            origin = finite[len(finite) // 2]
        for j in range(starts[p], starts[p + 1]):
            ok[j] = column._valid(j)
            out[j] = column._get(j) - origin if ok[j] else Float64(0)
    return Series("", Column[Float64](out^, ok))


def _with_nan(values: Series, nan: List[Bool]) raises -> Series:
    """`values` (a float column) with NaN at the rows of `nan`, valid
    there."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if D.is_floating_point():
            if values._data.isa[Column[Scalar[D]]]():
                ref column = values._data[Column[Scalar[D]]]
                var out = List[Scalar[D]](unsafe_uninit_length=len(column))
                var ok = List[Bool](unsafe_uninit_length=len(column))
                for i in range(len(column)):
                    out[i] = column._get(i)
                    ok[i] = column._valid(i)
                    if nan[i]:
                        out[i] = Scalar[D](FloatLiteral.nan)
                        ok[i] = True
                return Series("", Column[Scalar[D]](out^, ok))
    return values.copy()


def _ranked_frames(
    spec: ReducerSpec,
    input: Series,
    begins: List[Int],
    ends: List[Int],
    starts: List[Int],
    needed: Int,
) raises -> Series:
    """A minimum or maximum of values no aggregate state holds
    (categoricals): the extreme of their dense sort ranks, then the row
    with that rank."""
    var ranks = input._sort_ranks(False, True)
    var n = len(ranks)
    var rank_column = List[Int64](capacity=n)
    var valid = validity(input)
    var row_of = List[Int](length=n + 1, fill=-1)
    for i in range(n):
        rank_column.append(Int64(ranks[i]))
        if valid[i] and row_of[ranks[i]] < 0:
            row_of[ranks[i]] = i
    var as_ranks = Series("", Column[Int64](rank_column^, valid))
    var rank_spec = ReducerSpec(
        spec.op, DataType.INT64, 0, 0, 0, "", DataType.INT64, None
    )
    var frames = window_segment_tree(
        rank_spec, as_ranks, begins, ends, starts, needed
    )
    ref best = frames.values._data[Column[Int64]]
    var source = List[Int](length=n, fill=-1)
    for j in range(n):
        var count = Int(frames.counts[j])
        if count >= needed and count > 0:
            source[j] = row_of[Int(best._get(j))]
    return input.take_or_null(source)


def _rolling_frames(
    input: Series, starts: List[Int], node: Node
) raises -> Series:
    """Fixed-size rolling aggregates: each row's frame is the window_size
    rows ending at it within its partition (ROWS BETWEEN window_size - 1
    PRECEDING AND CURRENT ROW). Fewer than min_samples valid values
    (default window_size) give null."""
    var n = len(input)
    var window = Int(node.integer)
    var begins = List[Int](unsafe_uninit_length=n)
    var ends = List[Int](unsafe_uninit_length=n)
    for p in range(len(starts) - 1):
        for j in range(starts[p], starts[p + 1]):
            begins[j] = max(starts[p], j - window + 1)
            ends[j] = j + 1
    var needed = window if node.floating < 0 else Int(node.floating)
    return _frames(node, input, begins, ends, starts, needed)


def rolling_by_op(
    node: Node, input: Series, index: Series, ids: List[Int]
) raises -> Series:
    from dataframe.time.windows import rolling_bounds, negate_period

    if input.is_chunked() or index.is_chunked():
        return rolling_by_op(node, input.rechunk(), index.rechunk(), ids)
    if input.dtype().is_decimal():
        if node.op != ROLLING_SUM_BY and (
            node.op != ROLLING_MIN_BY and node.op != ROLLING_MAX_BY
        ):
            return rolling_by_op(node, _as_float(input), index, ids)
    var numeric = node.op != ROLLING_MIN_BY and node.op != ROLLING_MAX_BY
    if numeric and node.op != ROLLING_SUM_BY and _is_narrow(input.dtype()):
        return rolling_by_op(node, input.cast(DataType.FLOAT64), index, ids)
    if (
        numeric
        and input.dtype() != DataType.UINT64
        and _is_narrow(input.dtype())
    ):
        var result = rolling_by_op(node, _widen(input), index, ids)
        return (
            _narrow_sum(result, input.dtype().sum_type()) if node.op
            == ROLLING_SUM_BY else result^
        )
    var groups = partitions(len(input), ids)
    var bounds = rolling_bounds(
        index, groups, node.text, negate_period(node.text), node.text2
    )
    # The partitions laid out one after another, frames at their positions
    # there, and the results put back in row order.
    var n = len(input)
    var order = List[Int](capacity=n)
    var starts = List[Int](capacity=len(groups) + 1)
    var begins = List[Int](capacity=n)
    var ends = List[Int](capacity=n)
    for rows in groups:
        var base = len(order)
        starts.append(base)
        for row in rows:
            order.append(row)
            begins.append(base + bounds[0][row])
            ends.append(base + bounds[1][row])
    starts.append(len(order))
    var result = _frames(
        node, input.take(order), begins, ends, starts, Int(node.floating)
    )
    var inverse = List[Int](unsafe_uninit_length=n)
    for k in range(n):
        inverse[order[k]] = k
    return result.take(inverse)
