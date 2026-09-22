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
    ROLLING_MEAN,
    ROLLING_MIN,
    ROLLING_MAX,
    FORWARD_FILL,
    BACKWARD_FILL,
)
from .expr_kernels import checked_add, validity
from .reductions import WideInt
from .series import Series, sort_indices


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
    return (
        dtype.is_numeric()
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
    if (
        op_code == CUM_SUM or op_code == ROLLING_SUM or op_code == ROLLING_MEAN
    ) and _is_narrow(input.dtype()):
        var result = window_op(node, _widen(input), ids)
        if op_code == ROLLING_MEAN:
            return result^
        return _narrow_sum(result, input.dtype().sum_type())
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
    if op == RANK:
        return _rank(input, groups, node.text, reverse)
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
