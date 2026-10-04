"""Native grouped sums match batch evaluation across source intervals."""
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from std.math import isnan
from std.memory import ArcPointer
from dataframe import Column, DataType, Expr, Series, col
from dataframe.binding import bind
from dataframe.execution import _ReduceJob, _feed, _direct_grouped_sum


def compare(source: Series, expression: Expr, start: Int, end: Int) raises:
    var columns = List[Series]()
    columns.append(source.copy())
    var bound = bind(expression, columns)
    var node = expression._nodes[len(expression._nodes) - 1].copy()
    var ids = List[Int](capacity=len(source))
    for i in range(len(source)):
        ids.append((i * 7 + i // 5) % 4)
    var groups = ArcPointer(ids^)
    var direct = _ReduceJob[8](
        bound, columns, List[Series](), node, start, end, 1024, True, groups, 5
    )
    var reference = _ReduceJob[8](
        bound, columns, List[Series](), node, start, end, 1024, True, groups, 5
    )
    direct.run()
    for offset in range(start, end, 17):
        _feed[8](
            reference.reducer,
            bound,
            columns,
            List[Series](),
            node,
            offset,
            min(17, end - offset),
            True,
            groups[],
        )
    for g in range(5):
        if source.dtype() == DataType.INT64:
            assert_equal(
                direct.reducer.int_sums[g].total,
                reference.reducer.int_sums[g].total,
            )
            assert_equal(
                direct.reducer.int_sums[g].count,
                reference.reducer.int_sums[g].count,
            )
        else:
            var a = direct.reducer.float_sums[g].total
            var b = reference.reducer.float_sums[g].total
            assert_true((isnan(a) and isnan(b)) or a == b)
            assert_equal(
                direct.reducer.float_sums[g].count,
                reference.reducer.float_sums[g].count,
            )


def test_int64_chunk_intervals_nulls_and_wide_totals() raises:
    var values = List[Int64]()
    var valid = List[Bool]()
    for i in range(2057):
        values.append(
            Int64(9223372036854775807) if i % 2
            == 0 else Int64(-9223372036854775807)
        )
        valid.append(i % 11 != 0)
    var source = Series("v", Column[Int64](values^, valid^))
    var chunked = Series._from_chunks(
        [source.slice(0, 9), source.slice(9, 1016), source.slice(1025, 1032)]
    )
    for input in [source.copy(), chunked.copy(), chunked.slice(2, 2053)]:
        for expression in [col("v").sum(), col("v").mean()]:
            compare(input, expression, 3, len(input) - 4)
            compare(input, expression, 10, 10)


def test_float64_chunk_intervals_nulls_nan_infinity() raises:
    var values = List[Float64]()
    var valid = List[Bool]()
    for i in range(2057):
        values.append(-0.0 if i % 13 == 0 else Float64(i % 17) / 8)
        valid.append(i % 11 != 0)
    var source = Series("v", Column[Float64](values.copy(), valid.copy()))
    var chunked = Series._from_chunks(
        [source.slice(0, 9), source.slice(9, 1016), source.slice(1025, 1032)]
    )
    for input in [source.copy(), chunked.copy(), chunked.slice(2, 2053)]:
        for expression in [col("v").sum(), col("v").mean()]:
            compare(input, expression, 3, len(input) - 4)
            compare(input, expression, 10, 10)
    values[7] = Float64(0) / Float64(0)
    values[19] = Float64(1) / Float64(0)
    values[31] = -Float64(1) / Float64(0)
    var special = Series("v", Column[Float64](values^, valid^))
    compare(special, col("v").sum(), 3, len(special) - 4)
    compare(special, col("v").mean(), 3, len(special) - 4)


def test_computed_and_narrow_inputs_retain_batch_route() raises:
    var source = Series("v", Column[Int32]([1, 2, 3]))
    var columns = List[Series]()
    columns.append(source.copy())
    var mapping: List[Int] = [0, 1, 0]
    var groups = ArcPointer(mapping^)
    for expression in [col("v").sum(), (col("v") + 1).sum()]:
        var bound = bind(expression, columns)
        var node = expression._nodes[len(expression._nodes) - 1].copy()
        var job = _ReduceJob[8](
            bound, columns, List[Series](), node, 0, 3, 1024, True, groups, 2
        )
        assert_false(
            _direct_grouped_sum(
                job.reducer, bound, columns, node, 0, 3, True, groups[]
            )
        )
        job.run()
        var result = job.reducer.finish().cast(DataType.INT64).int64()
        assert_equal(
            result._get(0), Int64(4 if len(expression._nodes) == 2 else 6)
        )
        assert_equal(
            result._get(1), Int64(2 if len(expression._nodes) == 2 else 3)
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
