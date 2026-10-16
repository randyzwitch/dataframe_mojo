"""Decimal duration windows retain scale, output precision and observed prefixes."""
# contract-group: expr.order_windows
from std.math import abs
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, DataType, Expr, col
from dataframe.decimal import pow10
from tests.support.coverage_generators import offset_layout, chunk_layout


def decimal_frame(dtype: DataType, sign: Int = 1) raises -> DataFrame:
    var values: Series
    if dtype.decimal_width() == 32:
        values = Series(
            "x",
            Column[Int32](
                [Int32(sign * 1000), Int32(sign * 2000), Int32(sign * 3000), 0],
                [True, True, True, False],
            ),
        ).with_dtype(dtype)
    elif dtype.decimal_width() == 64:
        values = Series(
            "x",
            Column[Int64](
                [Int64(sign * 1000), Int64(sign * 2000), Int64(sign * 3000), 0],
                [True, True, True, False],
            ),
        ).with_dtype(dtype)
    else:
        values = Series(
            "x",
            Column[Int128](
                [
                    Int128(sign * 1000),
                    Int128(sign * 2000),
                    Int128(sign * 3000),
                    0,
                ],
                [True, True, True, False],
            ),
        ).with_dtype(dtype)
    return DataFrame(
        [
            values^,
            Series("t", Column[Int64]([0, 1, 2, 3])).with_dtype(DataType.DATE),
        ]
    )


def decimal_expressions() -> List[Expr]:
    return [
        col("x").rolling_sum_by("t", "3d").alias("s"),
        col("x").rolling_mean_by("t", "3d").alias("m"),
        col("x").rolling_var_by("t", "3d").alias("v"),
        col("x").rolling_std_by("t", "3d").alias("sd"),
    ]


def anchors(actual: DataFrame, dtype: DataType, sign: Int) raises:
    assert_true(actual.column("s").dtype() == dtype.sum_type())
    assert_equal(
        actual.column("s").numeric[DType.int128]().value(2), Int128(sign * 6000)
    )
    assert_equal(
        actual.column("s").numeric[DType.int128]().value(3), Int128(sign * 5000)
    )
    var divisor = pow10(dtype.scale()).cast[DType.float64]()
    var expected: List[Float64] = [
        Float64(sign * 2000) / divisor,
        Float64(1000000) / (divisor * divisor),
        Float64(1000) / divisor,
    ]
    var names: List[String] = ["m", "v", "sd"]
    for i in range(3):
        assert_true(actual.column(names[i]).dtype() == DataType.FLOAT64)
        var value = actual.column(names[i]).float64().value(2)
        assert_true(
            abs(value - expected[i])
            <= Float64(2e-14) * abs(expected[i]) + Float64(1e-323)
        )
    assert_true(
        actual.item(0, "v").is_null() and actual.item(0, "sd").is_null()
    )


def test_decimal_duration_arithmetic_has_independent_scale_and_type_anchors() raises:
    for width in [32, 64, 128]:
        for precision in [6, 9, 18, 38]:
            if (width == 32 and precision > 9) or (
                width == 64 and precision > 18
            ):
                continue
            for scale in [0, 3, precision]:
                var dtype = DataType.decimal(precision, scale, width)
                for sign in [-1, 1]:
                    var source = decimal_frame(dtype, sign)
                    for layout in range(3):
                        var frame = (
                            source.copy() if layout
                            == 0 else offset_layout(source) if layout
                            == 1 else chunk_layout(source)
                        )
                        for streaming in [False, True]:
                            var actual = (
                                frame.lazy()
                                .select_exprs(decimal_expressions())
                                .collect(engine="cpu", streaming=streaming)
                            )
                            anchors(actual, dtype, sign)


def test_decimal_duration_sums_check_observed_prefixes_without_wrapping() raises:
    var maximum = pow10(38) - 1
    var source = DataFrame(
        [
            Series(
                "x", Column[Int128]([maximum, maximum, -maximum])
            ).with_dtype(DataType.decimal(38, 0)),
            Series("t", Column[Int64]([0, 0, 0])).with_dtype(DataType.DATE),
        ]
    )
    for streaming in [False, True]:
        with assert_raises(contains="precision"):
            _ = (
                source.lazy()
                .select(col("x").rolling_sum_by("t", "1d"))
                .collect(engine="cpu", streaming=streaming)
            )
        var hidden = (
            source.lazy()
            .select(col("x").rolling_sum_by("t", "1d", "right", 4))
            .collect(engine="cpu", streaming=streaming)
        )
        assert_equal(hidden.column("x").null_count(), 3)
        var ignored = source.with_column(
            Series(
                "x",
                Column[Int128](
                    [maximum, Int128.MIN, -maximum], [True, False, True]
                ),
            ).with_dtype(DataType.decimal(38, 0))
        )
        var result = (
            ignored.lazy()
            .select(col("x").rolling_sum_by("t", "1d"))
            .collect(engine="cpu", streaming=streaming)
        )
        for row in range(3):
            assert_equal(
                result.column("x").numeric[DType.int128]().value(row), Int128(0)
            )
        var nulls = source.with_column(
            Series(
                "x",
                Column[Int128](
                    [Int128.MIN, Int128.MAX, 0], [False, False, False]
                ),
            ).with_dtype(DataType.decimal(38, 0))
        )
        var zero = (
            nulls.lazy()
            .select(col("x").rolling_sum_by("t", "1d", "right", 0))
            .collect(engine="cpu", streaming=streaming)
        )
        for row in range(3):
            assert_equal(
                zero.column("x").numeric[DType.int128]().value(row), Int128(0)
            )
    var narrow = DataFrame(
        [
            Series("x", Column[Int64]([Int64.MAX, Int64.MIN])).with_dtype(
                DataType.decimal(3, 0, 64)
            ),
            Series("t", Column[Int64]([0, 0])).with_dtype(DataType.DATE),
        ]
    )
    var widened = narrow.select(col("x").rolling_sum_by("t", "1d"))
    assert_true(widened.column("x").dtype() == DataType.decimal(38, 0))
    assert_equal(
        widened.column("x").numeric[DType.int128]().value(0), Int128(-1)
    )
