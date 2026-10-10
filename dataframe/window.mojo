"""Order-dependent kernels over whole columns, restarted per partition.

Most operations choose, for every output row, a source row to gather (or
none). Ordering uses the same dense sort ranks as `sort`, so NaN sorts above
every number and ties keep the earlier row.
"""
from .column import Column
from .dtype import DataType, NUMERIC_DTYPES
from .expr import (
    Node,
    CUM_SUM,
    CUM_MIN,
    CUM_MAX,
    CUM_COUNT,
    SHIFT,
    RANK,
    ROLLING_SUM,
    ROLLING_STD,
    ROLLING_VAR,
    ROLLING_SUM_BY,
    ROLLING_MEAN_BY,
    ROLLING_MIN_BY,
    ROLLING_MAX_BY,
    ROLLING_STD_BY,
    ROLLING_VAR_BY,
    is_rolling_by,
    ROLLING_MEAN,
    ROLLING_MIN,
    ROLLING_MAX,
    FORWARD_FILL,
    BACKWARD_FILL,
    INTERPOLATE,
    INTERPOLATE_BY,
    CUT,
    QCUT,
    SEP,
)
from .expr_kernels import checked_add, validity
from .string_column import StringColumn
from .nested_column import StructColumn
from .parse import parse_float64
from .aggregate import quantile_of
from .decimal import check_limit, pow10, precision_limit
from .reductions import WideInt
from .series import Series, sort_indices
from .rank import rank_numeric
from .packed_sort import packed_arg_sort
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


def window_op(node: Node, input: Series, ids: List[Int]) raises -> Series:
    if input.is_chunked():
        return window_op(node, input.rechunk(), ids)
    var op_code = node.op
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
        op_code == CUM_SUM
        or op_code == ROLLING_SUM
        or op_code == ROLLING_MEAN
        or op_code == ROLLING_STD
        or op_code == ROLLING_VAR
    ) and _is_narrow(input.dtype()):
        var result = window_op(node, _widen(input), ids)
        if (
            op_code == ROLLING_MEAN
            or op_code == ROLLING_STD
            or op_code == ROLLING_VAR
        ):
            return result^
        return _narrow_sum(result, input.dtype().sum_type())
    if op_code == RANK:
        # Numeric columns sort once per partition, in parallel (#330).
        var fast = rank_numeric(input, ids, node.text, node.min_count == 1)
        if fast:
            return fast.value().copy()
    var n = len(input)
    var valid = validity(input)
    var groups = partitions(n, ids)
    var op = node.op
    var reverse = node.min_count == 1
    if op == CUM_COUNT:
        var counts = List[Int64](length=n, fill=0)
        for rows in groups:
            var running = Int64(0)
            for row in _ordered(rows, reverse):
                if valid[row]:
                    running += 1
                counts[row] = running
        return Series("", Column[Int64](counts^))
    if op == CUM_SUM:
        var is_int = input._data.isa[Column[Int64]]()
        var ints = List[Int64](length=n if is_int else 0, fill=0)
        var floats = List[Float64](length=0 if is_int else n, fill=0)
        for rows in groups:
            var int_total = Int64(0)
            var float_total = Float64(0)
            for row in _ordered(rows, reverse):
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
        return _interpolate(input, groups, node.text)
    if op == CUT or op == QCUT:
        return _bins(input, groups, node)
    if op == RANK:
        return _rank(input, groups, node.text, reverse)
    if op == ROLLING_STD or op == ROLLING_VAR:
        return _rolling_moments(input, valid, groups, node)
    if op == ROLLING_SUM or op == ROLLING_MEAN:
        return _rolling_sum(input, valid, groups, node)
    # The remaining operations gather a source row per output row.
    var source = List[Int](length=n, fill=-1)
    var ranks = List[Int]()
    if op == CUM_MIN or op == CUM_MAX or op == ROLLING_MIN or op == ROLLING_MAX:
        ranks = input._sort_ranks(False, True)
    var take_max = op == CUM_MAX or op == ROLLING_MAX
    for rows in groups:
        var m = len(rows)
        if op == SHIFT:
            var k = Int(node.integer)
            for j in range(m):
                var from_index = j - k
                if from_index >= 0 and from_index < m:
                    source[rows[j]] = rows[from_index]
        elif op == CUM_MIN or op == CUM_MAX:
            var best = -1
            for row in _ordered(rows, reverse):
                if valid[row]:
                    if best < 0 or (
                        ranks[row]
                        > ranks[best] if take_max else ranks[row]
                        < ranks[best]
                    ):
                        best = row
                    source[row] = best
        elif op == ROLLING_MIN or op == ROLLING_MAX:
            var window = Int(node.integer)
            var needed = window if node.floating < 0 else Int(node.floating)
            for j in range(m):
                var best = -1
                var count = 0
                for k in range(max(0, j - window + 1), j + 1):
                    var row = rows[k]
                    if not valid[row]:
                        continue
                    count += 1
                    if best < 0 or (
                        ranks[row]
                        > ranks[best] if take_max else ranks[row]
                        < ranks[best]
                    ):
                        best = row
                if count >= needed and count > 0:
                    source[rows[j]] = best
        else:
            var limit = Int(node.integer)
            var last = -1
            var distance = 0
            for row in _ordered(rows, op == BACKWARD_FILL):
                if valid[row]:
                    last = row
                    distance = 0
                    source[row] = row
                else:
                    distance += 1
                    if last >= 0 and (limit < 0 or distance <= limit):
                        source[row] = last
    return input.take_or_null(source)


def _decimal_cum_sum(
    input: Series, ids: List[Int], reverse: Bool
) raises -> Series:
    """A running sum of a decimal column within each partition, exact in
    128 bits at the input's scale, typed as SQL's SUM of it (decimal(38,
    scale) for a decimal32 or decimal64 input); nulls are skipped and stay
    null. A running total past 38 digits raises."""
    var wide = input._decimal128()
    ref column = wide._data[Column[Int128]]
    var n = len(column)
    var valid = validity(input)
    var target = input.dtype().sum_type()
    var limit = precision_limit(target.precision())
    var values = List[Int128](length=n, fill=0)
    for rows in partitions(n, ids):
        var total = Int128(0)
        for row in _ordered(rows, reverse):
            if not valid[row]:
                continue
            total = check_limit(total + column._get(row), limit, target)
            values[row] = total
    return Series("", Column[Int128](values^, valid)).with_dtype(target)


def _as_float(input: Series) raises -> Series:
    """Convert a primitive numeric column to Float64, preserving validity."""
    if input._data.isa[Column[Float64]]():
        return input.copy()
    var valid = validity(input)
    if input.dtype().is_decimal():
        var wide = input._decimal128()
        ref column = wide._data[Column[Int128]]
        var values = List[Float64](capacity=len(column))
        var divisor = pow10(input.dtype().scale()).cast[DType.float64]()
        for i in range(len(column)):
            values.append(column._get(i).cast[DType.float64]() / divisor)
        return Series("", Column[Float64](values^, valid))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if input._data.isa[Column[Scalar[D]]]():
            ref column = input._data[Column[Scalar[D]]]
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
    return Series("", Column[Float64](out^, out_valid^))


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
        for label in node.text2.split(SEP):
            custom.append(String(label))
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
    var labels = Series("category", StringColumn(categories^, out_valid.copy()))
    if (node.integer & 2) == 0:
        return labels.renamed("")
    var breaks = Series(
        "breakpoint", Column[Float64](breakpoints^, out_valid.copy())
    )
    return Series("", StructColumn([breaks^, labels^]))


def _rolling_sum(
    input: Series, valid: List[Bool], groups: List[List[Int]], node: Node
) raises -> Series:
    var n = len(input)
    var window = Int(node.integer)
    var needed = window if node.floating < 0 else Int(node.floating)
    var integer_sum = (
        node.op == ROLLING_SUM and input._data.isa[Column[Int64]]()
    )
    var ints = List[Int64](length=n if integer_sum else 0, fill=0)
    var floats = List[Float64](length=0 if integer_sum else n, fill=0)
    var out_valid = List[Bool](length=n, fill=False)
    for rows in groups:
        for j in range(len(rows)):
            var count = 0
            var wide = WideInt(0)
            var total = Float64(0)
            for k in range(max(0, j - window + 1), j + 1):
                var row = rows[k]
                if not valid[row]:
                    continue
                count += 1
                if integer_sum:
                    wide += (
                        input._data[Column[Int64]]
                        ._get(row)
                        .cast[DType.int128]()
                    )
                else:
                    total += _numeric(input, row)
            if count < needed or count == 0:
                continue
            var row = rows[j]
            out_valid[row] = True
            if integer_sum:
                if wide > WideInt(9223372036854775807) or wide < WideInt(
                    -9223372036854775808
                ):
                    raise Error("Int64 rolling_sum overflow")
                ints[row] = wide.cast[DType.int64]()
            elif node.op == ROLLING_MEAN:
                floats[row] = total / Float64(count)
            else:
                floats[row] = total
    if integer_sum:
        return Series("", Column[Int64](ints^, out_valid))
    return Series("", Column[Float64](floats^, out_valid))


def _rank(
    input: Series, groups: List[List[Int]], method: String, descending: Bool
) raises -> Series:
    if input._data.isa[StringColumn]():
        var fast = _rank_strings(
            input._data[StringColumn], groups, method, descending
        )
        if fast:
            return fast.take()
    var n = len(input)
    var average = method == "average"
    var ints = List[Int64](length=0 if average else n, fill=0)
    var floats = List[Float64](length=n if average else 0, fill=0)
    var out_valid = List[Bool](length=n, fill=False)
    for rows in groups:
        var part = input.take(rows)
        var dense = part._sort_ranks(descending, True)
        var order = sort_indices([dense.copy()])
        var present = validity(part)
        var start = 0
        var dense_rank = 0
        while start < len(order):
            var end = start + 1
            while end < len(order) and dense[order[end]] == dense[order[start]]:
                end += 1
            if present[order[start]]:
                dense_rank += 1
                for position in range(start, end):
                    var row = rows[order[position]]
                    out_valid[row] = True
                    if average:
                        floats[row] = Float64(start + 1 + end) / 2
                    elif method == "min":
                        ints[row] = Int64(start + 1)
                    elif method == "max":
                        ints[row] = Int64(end)
                    elif method == "dense":
                        ints[row] = Int64(dense_rank)
                    else:
                        ints[row] = Int64(position + 1)
            start = end
    if average:
        return Series("", Column[Float64](floats^, out_valid))
    return Series("", Column[Int64](ints^, out_valid))


def _rank_strings(
    column: StringColumn,
    groups: List[List[Int]],
    method: String,
    descending: Bool,
) raises -> Optional[Series]:
    """Reuse the packed sort's stable partition/string order when supported."""
    var ids = List[Int64](length=len(column), fill=0)
    var count = 0
    for group in range(len(groups)):
        for row in groups[group]:
            ids[row] = Int64(group)
            count += 1
    if count != len(column):
        return None
    var partitions = Column[Int64](ids^)
    var order = packed_arg_sort(
        [Series("", partitions.copy()), Series("", column.copy())],
        [False, descending],
        [True, True],
    )
    if not order:
        return None
    return _rank_string_order(column, partitions, order.take(), method)


def _rank_string_order(
    column: StringColumn, ids: Column[Int64], order: List[Int], method: String
) raises -> Series:
    """Assign ranks from a stable order by (partition, string, source row)."""
    var average = method == "average"
    var ints = List[Int64](length=0 if average else len(column), fill=0)
    var floats = List[Float64](length=len(column) if average else 0, fill=0)
    var valid = List[Bool](length=len(column), fill=False)
    var start = 0
    while start < len(order):
        var base = start
        var end_group = start + 1
        while end_group < len(order) and ids._get(order[end_group]) == ids._get(
            order[base]
        ):
            end_group += 1
        var dense = 0
        while start < end_group and column._valid(order[start]):
            var end = start + 1
            while (
                end < end_group
                and column._valid(order[end])
                and column._get(order[end]) == column._get(order[start])
            ):
                end += 1
            dense += 1
            for position in range(start, end):
                var row = order[position]
                valid[row] = True
                if average:
                    floats[row] = Float64(start + 1 + end - 2 * base) / 2
                elif method == "min":
                    ints[row] = Int64(start - base + 1)
                elif method == "max":
                    ints[row] = Int64(end - base)
                elif method == "dense":
                    ints[row] = Int64(dense)
                else:
                    ints[row] = Int64(position - base + 1)
            start = end
        start = end_group
    if average:
        return Series("", Column[Float64](floats^, valid))
    return Series("", Column[Int64](ints^, valid))


def _rolling_moments(
    input: Series, valid: List[Bool], groups: List[List[Int]], node: Node
) raises -> Series:
    var lower = List[Int](length=len(input), fill=0)
    var upper = List[Int](length=len(input), fill=0)
    var window = Int(node.integer)
    for rows in groups:
        for j in range(len(rows)):
            lower[rows[j]] = max(0, j - window + 1)
            upper[rows[j]] = j + 1
    return _moments_bounds(
        input,
        valid,
        groups,
        lower,
        upper,
        window if node.floating < 0 else Int(node.floating),
        node.min_count,
        node.op == ROLLING_STD,
    )


def _moments_bounds(
    input: Series,
    valid: List[Bool],
    groups: List[List[Int]],
    lower: List[Int],
    upper: List[Int],
    needed: Int,
    ddof: Int,
    standard: Bool,
) raises -> Series:
    # Sliding Welford updates with removal. Translate by the first finite
    # value so large common offsets do not spoil small variances. Each value
    # is added/removed once for monotone window bounds; nonfinite values are
    # counted separately so the state recovers when they leave the window.
    var output = List[Float64](length=len(input), fill=0)
    var mask = List[Bool](length=len(input), fill=False)
    for rows in groups:
        var origin = Float64(0)
        for row in rows:
            if valid[row] and isfinite(_numeric(input, row)):
                origin = _numeric(input, row)
                break
        var lo = 0
        var hi = 0
        var count = 0
        var bad = 0
        var mean = Float64(0)
        var m2 = Float64(0)
        for row in rows:
            var start = lower[row]
            var stop = upper[row]
            if start < lo or stop < hi or start >= hi:
                lo = start
                hi = start
                count = 0
                bad = 0
                mean = 0
                m2 = 0
            while lo < start:
                var at = rows[lo]
                lo += 1
                if not valid[at]:
                    continue
                var x = _numeric(input, at)
                if not isfinite(x):
                    bad -= 1
                    continue
                x -= origin
                count -= 1
                if count == 0:
                    mean = 0
                    m2 = 0
                else:
                    var delta = x - mean
                    mean -= delta / Float64(count)
                    m2 -= delta * (x - mean)
                    m2 = max(m2, 0)
            while hi < stop:
                var at = rows[hi]
                hi += 1
                if not valid[at]:
                    continue
                var x = _numeric(input, at)
                if not isfinite(x):
                    bad += 1
                    continue
                x -= origin
                count += 1
                var delta = x - mean
                mean += delta / Float64(count)
                m2 += delta * (x - mean)
            var total = count + bad
            if total >= needed and total > ddof:
                mask[row] = True
                if bad:
                    output[row] = Float64("nan")
                else:
                    var variance = max(m2, 0) / Float64(count - ddof)
                    output[row] = sqrt(variance) if standard else variance
    return Series("", Column[Float64](output^, mask))


def rolling_by_op(
    node: Node, input: Series, index: Series, ids: List[Int]
) raises -> Series:
    from .time_windows import rolling_bounds, negate_period

    if input.is_chunked() or index.is_chunked():
        return rolling_by_op(node, input.rechunk(), index.rechunk(), ids)
    if node.op == ROLLING_SUM_BY and input.dtype() == DataType.UINT64:
        var groups = partitions(len(input), ids)
        var bounds = rolling_bounds(
            index, groups, node.text, negate_period(node.text), node.text2
        )
        return _rolling_u64_sum(
            input, groups, bounds[0], bounds[1], Int(node.floating)
        )
    var numeric = node.op != ROLLING_MIN_BY and node.op != ROLLING_MAX_BY
    if numeric and node.op != ROLLING_SUM_BY and _is_narrow(input.dtype()):
        return rolling_by_op(node, input.cast(DataType.FLOAT64), index, ids)
    if numeric and _is_narrow(input.dtype()):
        var result = rolling_by_op(node, _widen(input), index, ids)
        return (
            _narrow_sum(result, input.dtype().sum_type()) if node.op
            == ROLLING_SUM_BY else result^
        )
    var groups = partitions(len(input), ids)
    var bounds = rolling_bounds(
        index, groups, node.text, negate_period(node.text), node.text2
    )
    var valid = validity(input)
    var needed = Int(node.floating)
    if node.op == ROLLING_STD_BY or node.op == ROLLING_VAR_BY:
        return _moments_bounds(
            input,
            valid,
            groups,
            bounds[0],
            bounds[1],
            needed,
            node.min_count,
            node.op == ROLLING_STD_BY,
        )
    var is_int = node.op == ROLLING_SUM_BY and input.dtype() == DataType.INT64
    var ints = List[Int64](length=len(input) if is_int else 0, fill=0)
    var floats = List[Float64](
        length=len(input) if numeric and not is_int else 0, fill=0
    )
    var mask = List[Bool](length=len(input), fill=False)
    var source = List[Int](length=len(input), fill=-1)
    var ranks = List[Int]()
    if not numeric:
        ranks = input._sort_ranks(False, True)
    for rows in groups:
        for row in rows:
            var count = 0
            var total = Float64(0)
            var wide = WideInt(0)
            var best = -1
            var nan_row = -1
            var begin = bounds[0][row]
            var end = bounds[1][row]
            for k in range(begin, end):
                var at = rows[k]
                if not valid[at]:
                    continue
                count += 1
                if not numeric:
                    if input.dtype().is_float():
                        var x = input.get(at)._float
                        if x != x:
                            nan_row = at
                    if best < 0 or (
                        ranks[at]
                        > ranks[best] if node.op
                        == ROLLING_MAX_BY else ranks[at]
                        < ranks[best]
                    ):
                        best = at
                elif is_int:
                    wide += (
                        input._data[Column[Int64]]._get(at).cast[DType.int128]()
                    )
                else:
                    total += _numeric(input, at)
            if count < needed or (count == 0 and node.op != ROLLING_SUM_BY):
                continue
            mask[row] = True
            if not numeric:
                source[row] = nan_row if nan_row >= 0 else best
            elif is_int:
                if wide > WideInt(Int64.MAX) or wide < WideInt(Int64.MIN):
                    raise Error("Int64 rolling_sum_by overflow")
                ints[row] = wide.cast[DType.int64]()
            else:
                floats[row] = (
                    total / Float64(count) if node.op
                    == ROLLING_MEAN_BY else total
                )
    if not numeric:
        return input.take_or_null(source)
    if is_int:
        return Series("", Column[Int64](ints^, mask))
    return Series("", Column[Float64](floats^, mask))


def _rolling_u64_sum(
    input: Series,
    groups: List[List[Int]],
    lower: List[Int],
    upper: List[Int],
    needed: Int,
) raises -> Series:
    ref column = input._data[Column[UInt64]]
    var values = List[UInt64](length=len(input), fill=0)
    var valid = List[Bool](length=len(input), fill=False)
    for rows in groups:
        for row in rows:
            var total = Int128(0)
            var count = 0
            for k in range(lower[row], upper[row]):
                var at = rows[k]
                if column._valid(at):
                    total += Int128(column._get(at))
                    count += 1
            if count >= needed:
                if total > Int128(UInt64.MAX):
                    raise Error("UInt64 rolling_sum_by overflow")
                valid[row] = True
                values[row] = UInt64(total)
    return Series("", Column[UInt64](values^, valid))
