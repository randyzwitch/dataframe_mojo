"""Long-lived rolling states retain small variance beside a large exact baseline."""
from std.math import abs
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, col
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_rolling_moments_keep_small_residuals_after_thousands_of_window_evictions() raises:
    for baseline in [Float64(1e9), Float64(-1e9)]:
        var payload = List[Float64]()
        for row in range(10003):
            payload.append(baseline + Float64(row % 2))
        var source = DataFrame([Series("x", Column[Float64](payload^))])
        for layout in range(3):
            var frame = (
                source.copy() if layout
                == 0 else offset_layout(source) if layout
                == 1 else chunk_layout(source)
            )
            for window in [2, 64, 100]:
                for ddof in [0, 1]:
                    var actual = frame.select_exprs(
                        [
                            col("x")
                            .rolling_var(window, min_samples=1, ddof=ddof)
                            .alias("variance"),
                            col("x")
                            .rolling_std(window, min_samples=1, ddof=ddof)
                            .alias("deviation"),
                        ]
                    )
                    assert_equal(actual.height(), 10003)
                    assert_equal(
                        actual.dtypes(), [DataType.FLOAT64, DataType.FLOAT64]
                    )
                    for row in range(10003):
                        var start = max(0, row - window + 1)
                        var count = row - start + 1
                        # Count binary residuals in [start,row] exactly; never
                        # square or subtract numbers near the 1e9 baseline.
                        var ones = (row + 1) // 2 - start // 2
                        for name in ["variance", "deviation"]:
                            var item = actual.item(row, name)
                            assert_equal(item.is_null(), count <= ddof)
                            if count <= ddof:
                                continue
                            var expected = (
                                Float64(ones)
                                - Float64(ones * ones) / Float64(count)
                            ) / Float64(count - ddof)
                            var value = item.float64()
                            assert_true(value >= 0)
                            var compared = (
                                value * value if name == "deviation" else value
                            )
                            assert_true(
                                abs(compared - expected)
                                <= 1e-12 * (1 + expected),
                                "rolling moments must retain residual variance after repeated evictions",
                            )
