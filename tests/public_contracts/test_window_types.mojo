"""Typed order operations, width limits and IEEE state recovery."""
# contract-group: expr.order_windows
from std.sys import size_of
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from tests.support.numeric_model import INTEGER_DTYPES, FLOAT_DTYPES
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_float, assert_value
from tests.support.window_reference import (
    WindowCells,
    cumulative,
    rolling,
    ranked,
)


def numeric_cells[D: DType](actual: Series, cells: WindowCells) raises:
    assert_true(actual.dtype() == DataType.of(D))
    assert_equal(len(actual), len(cells.values))
    for row in range(len(cells.values)):
        assert_equal(actual.get(row).is_null(), not cells.valid[row])
        if cells.valid[row]:
            comptime if D.is_floating_point():
                assert_float(
                    Float64(actual.get(row).numeric[D]()), cells.values[row]
                )
            else:
                assert_equal(
                    actual.get(row).numeric[D](), Scalar[D](cells.values[row])
                )


def numeric_windows[D: DType]() raises:
    var cells = WindowCells(
        [4, 0, 99, 2, 6, 99, 1, 0],
        [True, True, False, True, True, False, True, True],
    )
    var payload = List[Scalar[D]]()
    for value in cells.values:
        payload.append(Scalar[D](value))
    var original = DataFrame(
        [Series("x", Column[Scalar[D]](payload^, cells.valid.copy()))]
    )
    for layout in range(3):
        var frame = (
            original.copy() if layout
            == 0 else offset_layout(original) if layout
            == 1 else chunk_layout(original)
        )
        for reverse in [False, True]:
            var actual = frame.select_exprs(
                [
                    col("x").cum_sum(reverse).alias("sum"),
                    col("x").cum_min(reverse).alias("min"),
                    col("x").cum_max(reverse).alias("max"),
                    col("x").cum_count(reverse).alias("count"),
                ]
            )
            comptime S = DType.int64 if D.is_integral() and size_of[
                Scalar[D]
            ]() <= 2 else D
            numeric_cells[S](
                actual.column("sum"), cumulative(cells, "sum", reverse)
            )
            numeric_cells[DType.int64](
                actual.column("count"), cumulative(cells, "count", reverse)
            )
            numeric_cells[D](
                actual.column("min"), cumulative(cells, "min", reverse)
            )
            numeric_cells[D](
                actual.column("max"), cumulative(cells, "max", reverse)
            )
        for width in [1, 3, 20]:
            var actual = frame.select_exprs(
                [
                    col("x").rolling_sum(width, 1).alias("sum"),
                    col("x").rolling_min(width, 1).alias("min"),
                    col("x").rolling_max(width, 1).alias("max"),
                ]
            )
            comptime S = DType.int64 if D.is_integral() and size_of[
                Scalar[D]
            ]() <= 2 else D
            numeric_cells[S](
                actual.column("sum"), rolling(cells, width, 1, "sum")
            )
            numeric_cells[D](
                actual.column("min"), rolling(cells, width, 1, "min")
            )
            numeric_cells[D](
                actual.column("max"), rolling(cells, width, 1, "max")
            )
        for method in ["average", "min", "max", "dense", "ordinal"]:
            for descending in [False, True]:
                var actual = frame.select(col("x").rank(method, descending))
                if method == "average":
                    numeric_cells[DType.float64](
                        actual.column("x"), ranked(cells, method, descending)
                    )
                else:
                    numeric_cells[DType.int64](
                        actual.column("x"), ranked(cells, method, descending)
                    )
        var empty = frame.clear().select_exprs(
            [
                col("x").cum_sum().alias("sum"),
                col("x").cum_min().alias("min"),
                col("x").rank().alias("rank"),
            ]
        )
        assert_equal(empty.height(), 0)
        comptime S = DType.int64 if D.is_integral() and size_of[
            Scalar[D]
        ]() <= 2 else D
        assert_true(empty.column("sum").dtype() == DataType.of(S))
        assert_true(empty.column("min").dtype() == DataType.of(D))
        assert_true(empty.column("rank").dtype() == DataType.FLOAT64)


def test_all_numeric_widths_keep_order_result_types_values_and_validity() raises:
    comptime for index in range(len(INTEGER_DTYPES)):
        numeric_windows[INTEGER_DTYPES[index]]()
    comptime for index in range(len(FLOAT_DTYPES)):
        numeric_windows[FLOAT_DTYPES[index]]()


def test_window_sums_check_their_declared_width_after_narrow_integer_promotion() raises:
    # Eight- and sixteen-bit sums widen; 32/64-bit sums retain their width.
    comptime small = (DType.int8, DType.uint8, DType.int16, DType.uint16)
    comptime for index in range(len(small)):
        comptime D = small[index]
        var frame = DataFrame(
            [Series("x", Column[Scalar[D]]([Scalar[D].MAX, Scalar[D](1)]))]
        )
        var expected = Scalar[D].MAX.cast[DType.int64]() + 1
        for expression in [col("x").cum_sum(), col("x").rolling_sum(2, 1)]:
            var result = frame.select(expression)
            assert_true(result.column("x").dtype() == DataType.INT64)
            assert_equal(result.item(1, "x").int64(), expected)
    comptime retained = (DType.int32, DType.uint32, DType.int64, DType.uint64)
    comptime for index in range(len(retained)):
        comptime D = retained[index]
        var frame = DataFrame(
            [Series("x", Column[Scalar[D]]([Scalar[D].MAX, Scalar[D](1)]))]
        )
        for expression in [col("x").cum_sum(), col("x").rolling_sum(2)]:
            with assert_raises():
                _ = frame.select(expression)


# expected-failure: UInt64 cumulative sums reject valid values above Int64.MAX
# expected-failure: despite a UInt64 result type.
# expected-diagnostic: uint64 window sum overflow
def test_uint64_cumulative_sums_preserve_values_above_signed_range() raises:
    var frame = DataFrame([Series("x", Column[UInt64]([UInt64.MAX, 0, 0]))])
    var result = frame.select(col("x").cum_sum())
    assert_true(result.column("x").dtype() == DataType.UINT64)
    for row in range(3):
        assert_equal(result.item(row, "x").numeric[DType.uint64](), UInt64.MAX)


# expected-failure: UInt64 rolling sums reject valid values above Int64.MAX
# expected-failure: despite a UInt64 result type.
# expected-diagnostic: uint64 window sum overflow
def test_uint64_rolling_sums_preserve_values_above_signed_range() raises:
    var frame = DataFrame([Series("x", Column[UInt64]([UInt64.MAX, 0, 0]))])
    var result = frame.select(col("x").rolling_sum(2, 1))
    assert_true(result.column("x").dtype() == DataType.UINT64)
    var expected: List[UInt64] = [UInt64.MAX, UInt64.MAX, 0]
    for row in range(3):
        assert_equal(
            result.item(row, "x").numeric[DType.uint64](), expected[row]
        )


def test_float_windows_recover_when_nonfinite_values_leave_the_window() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var cells = WindowCells(
        [1, inf, -inf, 3, 4, 5, nan, 7, 8, 9],
        [True, True, True, True, True, True, True, True, True, True],
    )
    var original = DataFrame(
        [Series("x", Column[Float64](cells.values.copy()))]
    )
    for layout in range(3):
        var frame = (
            original.copy() if layout
            == 0 else offset_layout(original) if layout
            == 1 else chunk_layout(original)
        )
        var actual = frame.select_exprs(
            [
                col("x").rolling_sum(2, 1).alias("sum"),
                col("x").rolling_mean(2, 1).alias("mean"),
                col("x").rolling_min(2, 1).alias("min"),
                col("x").rolling_max(2, 1).alias("max"),
                col("x").rolling_var(2, 1, 0).alias("var"),
                col("x").rolling_std(2, 1, 0).alias("std"),
            ]
        )
        for kind in ["sum", "mean", "min", "max", "var", "std"]:
            numeric_cells[DType.float64](
                actual.column(kind), rolling(cells, 2, 1, kind, 0)
            )
        for reverse in [False, True]:
            var extrema = frame.select_exprs(
                [
                    col("x").cum_min(reverse).alias("min"),
                    col("x").cum_max(reverse).alias("max"),
                ]
            )
            for kind in ["min", "max"]:
                numeric_cells[DType.float64](
                    extrema.column(kind), cumulative(cells, kind, reverse)
                )


def test_shift_and_directional_fill_preserve_string_bool_and_temporal_types() raises:
    var original = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    [
                        "poison",
                        "é",
                        "poison",
                        "poison",
                        "long external string",
                        "poison",
                    ],
                    [False, True, False, False, True, False],
                ),
            ),
            Series(
                "b",
                Column[Bool](
                    [True, False, True, True, True, True],
                    [False, True, False, False, True, False],
                ),
            ),
            Series(
                "t",
                Column[Int64](
                    [99, -1, 99, 99, 3, 99],
                    [False, True, False, False, True, False],
                ),
            ).cast(DataType.DATE),
        ]
    )
    for layout in range(3):
        var frame = (
            original.copy() if layout
            == 0 else offset_layout(original) if layout
            == 1 else chunk_layout(original)
        )
        for name in ["s", "b", "t"]:
            var actual = frame.select_exprs(
                [
                    col(name).shift(-2).alias("shift"),
                    col(name).forward_fill(1).alias("forward"),
                    col(name).backward_fill(1).alias("backward"),
                ]
            )
            var sources: List[List[Int]] = [
                [-1, -1, 4, -1, -1, -1],
                [-1, 1, 1, -1, 4, 4],
                [1, 1, -1, 4, 4, -1],
            ]
            var names: List[String] = ["shift", "forward", "backward"]
            for column in range(3):
                assert_true(
                    actual.column(names[column]).dtype()
                    == frame.column(name).dtype()
                )
                for row in range(6):
                    var source = sources[column][row]
                    assert_equal(
                        actual.item(row, names[column]).is_null(), source < 0
                    )
                    if source >= 0:
                        assert_value(
                            actual.item(row, names[column]),
                            original.item(source, name),
                        )
