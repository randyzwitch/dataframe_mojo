"""Selection cutoffs and large duplicate sets agree with a histogram oracle."""
# contract-group: kernel.reduce
from std.math import isnan
from std.testing import assert_equal, assert_true
from dataframe.aggregate import quantile_in


def _at(counts: List[Int], position: Int) -> Float64:
    var cumulative = 0
    for bucket in range(len(counts)):
        cumulative += counts[bucket]
        if position < cumulative:
            return Float64(bucket - 15) / 4 if bucket < 31 else Float64(
                0
            ) / Float64(0)
    return Float64(0) / Float64(0)


def _expected(counts: List[Int], n: Int, q: Float64, method: String) -> Float64:
    var location = q * Float64(n - 1)
    var low = Int(location)
    var high = min(n - 1, low + 1)
    var fraction = location - Float64(low)
    var a = _at(counts, low)
    var b = _at(counts, high)
    if method == "lower":
        return a
    if method == "higher":
        return a if fraction == 0 else b
    if method == "nearest":
        return a if fraction < 0.5 else b
    if fraction == 0:
        return a
    return (a + b) / 2 if method == "midpoint" else a * (
        1 - fraction
    ) + b * fraction


def test_quantile_histogram_oracle_has_literal_interpolation_and_nan_order_anchors() raises:
    var counts = List[Int](length=32, fill=0)
    counts[15] = 1
    counts[19] = 2
    counts[31] = 1
    assert_equal(_at(counts, 0), 0.0)
    assert_equal(_at(counts, 1), 1.0)
    assert_true(isnan(_at(counts, 3)))
    assert_equal(_expected(counts, 4, 0.25, "linear"), 0.75)
    assert_equal(_expected(counts, 4, 0.25, "lower"), 0.0)
    assert_equal(_expected(counts, 4, 0.25, "higher"), 1.0)
    assert_equal(_expected(counts, 4, 0.25, "nearest"), 1.0)
    assert_equal(_expected(counts, 4, 0.25, "midpoint"), 0.5)


def test_quantile_selection_small_sort_and_large_partition_routes_match_independent_histograms() raises:
    for n in [0, 1, 23, 24, 25, 26, 63, 64, 65, 20000]:
        for mode in range(4):
            var values = List[Float64]()
            var counts = List[Int](length=32, fill=0)
            for row in range(n):
                var bucket = (
                    15 if mode
                    == 0 else (n - row) % 31 if mode
                    == 1 else (row * 7 + row // 19) % 31
                )
                if mode == 3 and row % 11 == 3:
                    bucket = 31
                counts[bucket] += 1
                values.append(
                    Float64(bucket - 15) / 4 if bucket < 31 else Float64("nan")
                )
            for q in [0.0, 0.1, 0.25, 0.5, 0.9, 1.0]:
                for method in [
                    "linear",
                    "lower",
                    "higher",
                    "nearest",
                    "midpoint",
                ]:
                    var input = values.copy()
                    var actual = quantile_in(Span(input), q, method)
                    assert_equal(Bool(actual), n > 0)
                    if n:
                        var expected = _expected(counts, n, q, method)
                        if isnan(expected):
                            assert_true(isnan(actual.value()))
                        else:
                            assert_equal(actual.value(), expected)
