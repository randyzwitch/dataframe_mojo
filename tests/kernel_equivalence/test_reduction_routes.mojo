"""Force reduction specializations and check their domains against scalar models."""
from std.math import abs, sqrt, isnan
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from dataframe.aggregate import _sum_ints, _sum_floats
from dataframe.binding import bind, BoundExpr
from dataframe.execution import (
    _column_stats,
    direct_reductions,
    _direct_numeric_part,
    _new_reducer,
)
from dataframe.indexed_reduce import indexed_reductions, reduce_indexed
from dataframe.reductions import IntSumState, FloatSumState
from dataframe.expr import SUM, MEAN, COUNT
from tests.support.numeric_model import (
    INTEGER_DTYPES,
    FLOAT_DTYPES,
    integer_bounds,
)
from tests.support.selected_numeric_reference import SelectedNumbers
from tests.support.coverage_compare import assert_float, assert_value


def _near(actual: Float64, expected: Float64, context: String = "") raises:
    # Bounded inputs <=16, n<=65. Independent centered two-pass statistics
    # versus online state updates receive this absolute+relative budget.
    assert_true(
        abs(actual - expected) <= 2e-12 * (1 + abs(expected)),
        "numeric reduction mismatch "
        + context
        + " observed="
        + String(actual)
        + " expected="
        + String(expected),
    )


def test_selected_numeric_reference_has_literal_order_null_moment_and_quantile_anchors() raises:
    var model = SelectedNumbers(
        [1, 2, 3, 4], [True, False, True, True], [3, 0, 2, 1]
    )
    assert_equal(model.ordered, [4.0, 1.0, 3.0])
    assert_equal(model.sorted, [1.0, 3.0, 4.0])
    assert_equal(model.count, 3)
    assert_true(model.saw_null)
    assert_equal(model.total, 8.0)
    _near(model.centered, 14.0 / 3.0)
    assert_equal(model.quantile(0.25, "linear"), 2.0)
    assert_equal(model.quantile(0.25, "lower"), 1.0)
    assert_equal(model.quantile(0.25, "higher"), 3.0)
    assert_equal(model.quantile(0.25, "nearest"), 3.0)
    assert_equal(model.quantile(0.25, "midpoint"), 2.0)


def _integer_sum[D: DType]() raises:
    var bounds = integer_bounds[D]()
    for n in [0, 1, 7, 8, 9, 63, 64, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var payload = List[Scalar[D]](
                    length=offset, fill=bounds[1].cast[D]()
                )
                var valid = List[Bool](length=offset, fill=False)
                var exact = List[Int128]()
                var flags = List[Bool]()
                for row in range(n):
                    var value = (
                        bounds[1] if row % 4
                        == 0 else bounds[0] if row % 4
                        == 1 else Int128(row % 7)
                    )
                    var present = mode == 0 or mode == 1 and row % 3 != 0
                    payload.append(value.cast[D]())
                    valid.append(present)
                    exact.append(value)
                    flags.append(present)
                var column = Column[Scalar[D]](payload^, valid^).slice(
                    offset, n
                )
                for grouped in [False, True]:
                    var groups = List[Int](length=3, fill=3)
                    for row in range(n):
                        groups.append((row * 7 + 1) % 3)
                    var totals = List[Int128](length=4, fill=0)
                    var counts = List[Int64](length=4, fill=0)
                    for row in range(n):
                        if flags[row]:
                            var group = groups[row + 3] if grouped else 0
                            totals[group] += exact[row]
                            counts[group] += 1
                    var states = List[IntSumState](length=4, fill=IntSumState())
                    _sum_ints[D](column, 3, grouped, groups, states)
                    for group in range(4):
                        assert_equal(states[group].total, totals[group])
                        assert_equal(states[group].count, counts[group])


def test_direct_typed_integer_accumulators_keep_exact_totals_groups_offsets_and_nulls() raises:
    comptime for index in range(len(INTEGER_DTYPES)):
        _integer_sum[INTEGER_DTYPES[index]]()


def _float_sum[D: DType]() raises:
    for n in [0, 1, 7, 8, 9, 63, 64, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var payload = List[Scalar[D]](
                    length=offset, fill=Float64("nan").cast[D]()
                )
                var valid = List[Bool](length=offset, fill=False)
                var values = List[Float64]()
                var flags = List[Bool]()
                for row in range(n):
                    var present = mode == 0 or mode == 1 and row % 3 != 0
                    var value = Float64(row % 17 - 8) / 4
                    payload.append(
                        value.cast[D]() if present else Float64("nan").cast[D]()
                    )
                    valid.append(present)
                    values.append(value)
                    flags.append(present)
                var column = Column[Scalar[D]](payload^, valid^).slice(
                    offset, n
                )
                for grouped in [False, True]:
                    var groups = List[Int](length=3, fill=3)
                    for row in range(n):
                        groups.append((row * 7 + 1) % 3)
                    var totals = List[Float64](length=4, fill=0)
                    var counts = List[Int64](length=4, fill=0)
                    for row in range(n):
                        if flags[row]:
                            var group = groups[row + 3] if grouped else 0
                            totals[group] += values[row]
                            counts[group] += 1
                    var states = List[FloatSumState](
                        length=4, fill=FloatSumState()
                    )
                    _sum_floats[D](column, 3, grouped, groups, states)
                    for group in range(4):
                        assert_equal(states[group].total, totals[group])
                        assert_equal(states[group].count, counts[group])


def test_direct_typed_float_accumulators_ignore_null_nan_payloads_and_keep_all_groups() raises:
    comptime for index in range(len(FLOAT_DTYPES)):
        _float_sum[FLOAT_DTYPES[index]]()


def _stats[D: DType]() raises:
    for n in [0, 1, 7, 8, 9, 63, 64, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var payload = List[Scalar[D]](length=offset, fill=0)
                var valid = List[Bool](length=offset, fill=False)
                var values = List[Int128]()
                var flags = List[Bool]()
                for row in range(n):
                    var value = Int128(row % 17) - Int128(
                        8 if D.is_signed() else 0
                    )
                    var present = mode == 0 or mode == 1 and row % 3 != 0
                    payload.append(value.cast[D]())
                    valid.append(present)
                    values.append(value)
                    flags.append(present)
                var column = Column[Scalar[D]](payload^, valid^).slice(
                    offset, n
                )
                for start in [0, min(n, 1), min(n, 7)]:
                    var total = Int128(0)
                    var count = 0
                    var least = Int64.MAX
                    var most = Int64.MIN
                    for row in range(start, n):
                        if flags[row]:
                            total += values[row]
                            count += 1
                            least = min(least, Int64(values[row]))
                            most = max(most, Int64(values[row]))
                    var actual = _column_stats[D](column, start, n)
                    assert_equal(actual[0], total)
                    assert_equal(actual[1], count)
                    assert_equal(actual[2], least)
                    assert_equal(actual[3], most)


def test_simd_column_statistics_match_wide_scalar_subranges_and_unaligned_validity() raises:
    comptime for index in range(len(INTEGER_DTYPES) - 1):
        _stats[INTEGER_DTYPES[index]]()


def test_simd_column_statistics_do_not_wrap_before_widening_to_int128() raises:
    for n in [8, 9, 16, 17, 65]:
        for same_lane in [False, True]:
            var payload = List[Int64](length=n, fill=0)
            payload[0] = Int64.MAX
            payload[8 if same_lane and n > 8 else 1] = Int64.MAX
            var actual = _column_stats[DType.int64](
                Column[Int64](payload^), 0, n
            )
            assert_true(
                actual[0] == Int128(Int64.MAX) * 2,
                "SIMD column statistics must widen before summing overflowing lanes",
            )


def test_shifted_sum_shortcut_preserves_final_overflow_and_wide_mean() raises:
    var payload = List[Int64](length=8, fill=0)
    payload[0] = Int64.MAX
    payload[1] = Int64.MAX
    var source = DataFrame([Series("x", Column[Int64](payload^))])
    var overflow = False
    try:
        _ = source.select((col("x") + 0).sum())
    except error:
        if "overflow" not in String(error):
            raise error
        overflow = True
    assert_true(
        overflow, "shifted SUM must reject a wide final total outside Int64"
    )
    assert_float(
        source.select((col("x") + 0).mean()).item().float64(),
        Float64(Int128(Int64.MAX) * 2) / 8,
    )


def test_reduction_specialization_eligibility_distinguishes_direct_indexed_and_fallback_domains() raises:
    var types: List[Series] = [
        Series("x", Column[Int64]([1, 2])),
        Series("x", Column[Float32]([1, 2])),
        Series("x", Column[Float64]([1, 2])),
        Series("x", Column[UInt64]([UInt64.MAX, 0])),
        Series("x", Column[Int128]([100, 200])).with_dtype(
            DataType.decimal(8, 2)
        ),
        Series("x", Column[String](["a", "b"])).cast(DataType.CATEGORICAL),
    ]
    for index in range(len(types)):
        var columns: List[Series] = [types[index].copy()]
        var numeric = index < 4
        for expression in [
            col("x").sum(),
            col("x").mean(),
            col("x").min(),
            col("x").max(),
        ]:
            if index == 5:
                continue
            var op = expression._nodes[len(expression._nodes) - 1].op
            if index == 4:
                # The planner's coarse physical-type candidate hint may accept
                # decimal storage. The actual kernel must decline it so the
                # decimal reducer can honor precision and scaling.
                var bound = bind(expression, columns)
                ref node = bound.expr._nodes[len(bound.expr._nodes) - 1]
                var reducer = _new_reducer(bound, node, 1)
                assert_true(
                    not _direct_numeric_part(reducer, types[index], op, 0, 2)
                )
                continue
            var expected = (
                index == 0
                or index == 3
                or index == 2
                and (op == SUM or op == MEAN)
            )
            assert_true(
                direct_reductions([expression.copy()], columns) == expected,
                "direct eligibility mismatch dtype="
                + types[index].dtype().name()
                + " op="
                + String(op),
            )
        for expression in [
            col("x").count(),
            col("x").median(),
            col("x").quantile(0.25),
        ]:
            if index == 5 and expression._nodes[1].op != COUNT:
                continue
            var bound: List[BoundExpr] = [bind(expression, columns)]
            var expected = numeric and not (
                index == 3 and expression._nodes[1].op != COUNT
            )
            assert_equal(indexed_reductions(bound, columns), expected)
        if numeric:
            var computed: List[BoundExpr] = [
                bind((col("x") + 1).sum(), columns)
            ]
            assert_true(not indexed_reductions(computed, columns))
    assert_true(not indexed_reductions(List[BoundExpr](), List[Series]()))


def _indexed[D: DType]() raises:
    for n in [0, 1, 15, 16, 17, 47, 48, 49, 63, 64, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var payload = List[Scalar[D]](length=offset, fill=0)
                var valid = List[Bool](length=offset, fill=False)
                var values = List[Float64]()
                var flags = List[Bool]()
                for row in range(n):
                    var value = Float64(row % 17) / Float64(
                        4 if D.is_floating_point() else 1
                    )
                    var present = mode == 0 or mode == 1 and row % 3 != 0
                    payload.append(value.cast[D]())
                    valid.append(present)
                    values.append(value)
                    flags.append(present)
                var source = Series(
                    "x", Column[Scalar[D]](payload^, valid^).slice(offset, n)
                )
                var columns: List[Series] = [source.copy()]
                var rows = List[Int]()
                var ids = List[Int]()
                var members = List[List[Int]](length=4, fill=List[Int]())
                for position in range(n):
                    var row = n - position - 1
                    rows.append(row)
                    var group = position % 3
                    ids.append(group)
                    members[group].append(row)
                var expressions: List[Expr] = [
                    col("x").sum(),
                    col("x").mean(),
                    col("x").min(),
                    col("x").max(),
                    col("x").count(),
                    col("x").len(),
                    col("x").first(),
                    col("x").last(),
                    col("x").var(1),
                    col("x").std(1),
                    col("x").n_unique(),
                    col("x").null_count(),
                ]
                comptime if D != DType.uint64:
                    expressions.append(col("x").median())
                    for method in [
                        "linear",
                        "lower",
                        "higher",
                        "nearest",
                        "midpoint",
                    ]:
                        expressions.append(col("x").quantile(0.25, method))
                for op in range(len(expressions)):
                    comptime if D == DType.uint64:
                        # Focused regression below isolates the signed-key
                        # recoding error; all other UInt64 reducers stay here.
                        if op == 8 or op == 9:
                            continue
                    var bound = bind(expressions[op], columns)
                    assert_true(indexed_reductions([bound.copy()], columns))
                    var actual = reduce_indexed(
                        bound, columns, rows.unsafe_ptr(), ids, 4
                    )
                    assert_equal(len(actual), 4)
                    for group in range(4):
                        var model = SelectedNumbers(
                            values, flags, members[group]
                        )
                        var item = actual.get(group)
                        if op == 4 or op == 5 or op == 10 or op == 11:
                            assert_true(actual.dtype() == DataType.INT64)
                            var expected = (
                                model.count if op
                                == 4 else len(model.rows) if op
                                == 5 else len(model.unique)
                                + Int(model.saw_null) if op
                                == 10 else len(model.rows) - model.count
                            )
                            assert_equal(item.int64(), Int64(expected))
                        elif op == 0:
                            assert_true(
                                actual.dtype() == source.dtype().sum_type()
                            )
                            comptime if D.is_floating_point():
                                _near(Float64(item.numeric[D]()), model.total)
                            elif D == DType.int8 or D == DType.int16 or D == DType.uint8 or D == DType.uint16:
                                assert_equal(item.int64(), Int64(model.total))
                            else:
                                assert_equal(
                                    item.numeric[D]().cast[DType.int128](),
                                    Int128(model.total),
                                )
                        elif op == 6 or op == 7:
                            assert_true(actual.dtype() == source.dtype())
                            if not model.rows:
                                assert_true(item.is_null())
                            else:
                                var row = model.rows[
                                    0 if op == 6 else len(model.rows) - 1
                                ]
                                assert_value(item, source.get(row))
                        elif op == 2 or op == 3:
                            assert_true(actual.dtype() == source.dtype())
                            assert_equal(item.is_null(), not model.count)
                            if model.count:
                                assert_equal(
                                    Float64(item.numeric[D]()),
                                    model.sorted[
                                        0 if op == 2 else model.count - 1
                                    ],
                                )
                        else:
                            assert_true(actual.dtype() == DataType.FLOAT64)
                            assert_equal(
                                item.is_null(),
                                model.count < (2 if op == 8 or op == 9 else 1),
                            )
                            if not item.is_null():
                                var expected = (
                                    model.mean if op
                                    == 1 else model.centered
                                    / Float64(model.count - 1) if op
                                    == 8 else sqrt(
                                        model.centered
                                        / Float64(model.count - 1)
                                    ) if op
                                    == 9 else model.quantile(
                                        0.5, "linear"
                                    ) if op
                                    == 12 else model.quantile(
                                        0.25,
                                        [
                                            "linear",
                                            "lower",
                                            "higher",
                                            "nearest",
                                            "midpoint",
                                        ][op - 13],
                                    )
                                )
                                _near(
                                    item.float64(),
                                    expected,
                                    "dtype="
                                    + String(D)
                                    + " n="
                                    + String(n)
                                    + " offset="
                                    + String(offset)
                                    + " mode="
                                    + String(mode)
                                    + " op="
                                    + String(op)
                                    + " group="
                                    + String(group),
                                )


def test_indexed_numeric_reducers_match_independent_ordered_groups_including_empty_groups() raises:
    comptime for index in range(len(INTEGER_DTYPES)):
        _indexed[INTEGER_DTYPES[index]]()
    comptime for index in range(len(FLOAT_DTYPES)):
        _indexed[FLOAT_DTYPES[index]]()


def test_uint64_indexed_moments_preserve_small_differences_before_float_conversion() raises:
    var columns: List[Series] = [Series("x", Column[UInt64]([0, 3, 6]))]
    var rows: List[Int] = [2, 0, 1]
    var ids: List[Int] = [0, 0, 0]
    for ddof in [0, 1]:
        for expression in [col("x").var(ddof), col("x").std(ddof)]:
            var bound = bind(expression, columns)
            assert_true(indexed_reductions([bound.copy()], columns))
            var actual = reduce_indexed(
                bound, columns, rows.unsafe_ptr(), ids, 2
            )
            assert_true(actual.dtype() == DataType.FLOAT64)
            assert_true(actual.get(1).is_null())
            var value = actual.get(0).float64()
            var variance = (
                value * value if expression._nodes[1].op
                != col("x").var(ddof)._nodes[1].op else value
            )
            assert_true(
                abs(variance - (6.0 if ddof == 0 else 9.0)) <= 2e-12,
                "UInt64 indexed moments must preserve small unsigned differences",
            )
