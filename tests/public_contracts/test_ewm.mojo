"""Exponentially weighted means, variances, deviations and sums, by row and
over a time index."""
# contract-group: expr.rolling_quantiles
from std.math import abs, isnan, sqrt
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.coverage_generators import chunk_layout


def _assert_values(
    column: Series, expected: List[Float64], valid: List[Bool], label: String
) raises:
    var floats = column.cast(DataType.FLOAT64)
    assert_equal(len(floats), len(expected), label)
    for row in range(len(expected)):
        var item = floats.get(row)
        assert_equal(
            item.is_null(), not valid[row], label + " row " + String(row)
        )
        if valid[row]:
            var e = expected[row]
            assert_true(
                abs(item.float64() - e) <= 1e-9 * (1 + abs(e)),
                label
                + " row "
                + String(row)
                + ": "
                + String(item.float64())
                + " != "
                + String(e),
            )


def test_ewm_functions_match_polars_with_nulls() raises:
    # Values 1, null, 3, 4, null, 6 with alpha 0.5 (Polars 2.0 answers).
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [1, 0, 3, 4, 0, 6], [True, False, True, True, False, True]
                ),
            )
        ]
    )
    var present: List[Bool] = [True, False, True, True, False, True]
    var exprs: List[Expr] = [
        col("x").ewm_mean(alpha=0.5).alias("mean"),
        col("x").ewm_mean(alpha=0.5, ignore_nulls=True).alias("ignored"),
        col("x").ewm_mean(alpha=0.5, adjust=False).alias("plain"),
        col("x").ewm_std(alpha=0.5).alias("std"),
        col("x").ewm_var(alpha=0.5, bias=True, min_samples=2).alias("var"),
        col("x").ewm_sum(alpha=0.5).alias("sum"),
        col("x").ewm_mean(com=1).alias("com"),
        col("x").ewm_mean(span=3).alias("span"),
    ]
    var eager = chunk_layout(frame).select_exprs(exprs)
    var lazy = frame.lazy().select_exprs(exprs).collect()
    for result in [eager.copy(), lazy^]:
        var mean: List[Float64] = [
            1.0,
            0,
            2.6,
            3.4615384615384617,
            0,
            5.266666666666667,
        ]
        _assert_values(result.column("mean"), mean, present, "mean")
        _assert_values(result.column("com"), mean, present, "com")
        _assert_values(result.column("span"), mean, present, "span")
        _assert_values(
            result.column("ignored"),
            [
                1.0,
                0,
                2.333333333333333,
                3.2857142857142856,
                0,
                4.733333333333333,
            ],
            present,
            "ignored",
        )
        _assert_values(
            result.column("plain"),
            [
                1.0,
                0,
                2.333333333333333,
                3.1666666666666665,
                0,
                5.055555555555555,
            ],
            present,
            "plain",
        )
        _assert_values(
            result.column("std"),
            [
                0,
                0,
                1.4142135623730951,
                1.1677484162422844,
                0,
                1.834452886095565,
            ],
            [False, False, True, True, False, True],
            "std",
        )
        _assert_values(
            result.column("var"),
            [0, 0, 0.64, 0.7100591715976331, 0, 1.5288888888888887],
            [False, False, True, True, False, True],
            "var",
        )
        _assert_values(
            result.column("sum"),
            [1.0, 0, 3.25, 5.625, 0, 7.40625],
            present,
            "sum",
        )


def test_ewm_restarts_per_partition_and_keeps_float32() raises:
    var frame = DataFrame(
        [
            Series("x", Column[Float64]([1, 10, 3, 20, 5, 30])),
            Series("g", Column[Int64]([0, 1, 0, 1, 0, 1])),
            Series("i", Column[Int32]([1, 10, 3, 20, 5, 30])),
        ]
    )
    var all = List[Bool](length=6, fill=True)
    var result = frame.select_exprs(
        [
            col("x").ewm_mean(alpha=0.5).over("g").alias("g"),
            col("x").cast(DataType.FLOAT32).ewm_mean(alpha=0.5).alias("f32"),
            col("i").ewm_mean(alpha=0.5).alias("i32"),
        ]
    )
    assert_true(result.column("f32").dtype() == DataType.FLOAT32)
    assert_true(result.column("i32").dtype() == DataType.FLOAT64)
    # Group 0: 1, 3, 5 -> 1, 7/3, 27/7; group 1: 10, 20, 30.
    _assert_values(
        result.column("g"),
        [1.0, 10.0, 7.0 / 3.0, 50.0 / 3.0, 27.0 / 7.0, 170.0 / 7.0],
        all,
        "grouped",
    )


def test_ewm_by_decays_with_elapsed_time() raises:
    # Days 1, 2, 4, 3, 9 (out of order), half-life 2 days (Polars 2.0).
    var frame = DataFrame(
        [
            Series("t", Column[Int64]([0, 1, 3, 2, 8])).with_dtype(
                DataType.DATE
            ),
            Series(
                "x",
                Column[Float64](
                    [5, 1, 4, 0, 2], [True, True, True, False, True]
                ),
            ),
        ]
    )
    var present: List[Bool] = [True, True, True, False, True]
    var exprs: List[Expr] = [
        col("x").ewm_mean_by("t", "2d").alias("mean"),
        col("x").ewm_sum_by(col("t"), "48h").alias("sum"),
    ]
    var eager = frame.select_exprs(exprs)
    var lazy = frame.lazy().select_exprs(exprs).collect()
    for result in [eager.copy(), lazy^]:
        _assert_values(
            result.column("mean"),
            [5.0, 3.8284271247461903, 3.914213562373095, 0, 2.3383883476483183],
            present,
            "mean",
        )
        _assert_values(
            result.column("sum"),
            [5.0, 4.535533905932738, 6.267766952966369, 0, 3.107995128834866],
            present,
            "sum",
        )


def test_ewm_rejects_bad_options() raises:
    var frame = DataFrame(
        [
            Series("x", Column[Float64]([1, 2, 3])),
            Series("t", Column[Int64]([0, 1, 2])).with_dtype(DataType.DATE),
        ]
    )
    with assert_raises(contains="one of com, span, half_life or alpha"):
        _ = frame.select_exprs([col("x").ewm_mean()])
    with assert_raises(contains="one of com, span, half_life or alpha"):
        _ = frame.select_exprs([col("x").ewm_mean(com=1, alpha=0.5)])
    with assert_raises(contains="alpha"):
        _ = frame.select_exprs([col("x").ewm_mean(alpha=1.5)])
    with assert_raises(contains="span"):
        _ = frame.lazy().select_exprs([col("x").ewm_std(span=0.5)]).collect()
    with assert_raises(contains="half_life"):
        _ = frame.select_exprs([col("x").ewm_mean_by("t", "1mo")])
