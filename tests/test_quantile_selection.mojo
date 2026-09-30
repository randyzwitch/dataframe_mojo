"""Quantiles by selection (#337) against the full sort they replace.

`reference` is the sort-based implementation selection replaced; every
result must match it bit for bit (NaN matching NaN), for every
interpolation, q at both ends and between, NaN, heavy duplicates, and
sizes from 0 through a size past the insertion-sort cutoff. Grouped and
global medians are checked against it too, with nulls and all-null
groups, on enough rows that groups finish on several workers.
"""
from std.math import ceil, floor, isnan
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Series, col
from dataframe.aggregate import quantile_of


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def reference(
    var values: List[Float64], q: Float64, method: String
) -> Optional[Float64]:
    """The sort-based quantile selection replaced."""
    var n = len(values)
    if n == 0:
        return None
    var ordered = List[Float64](capacity=n)
    var nans = 0
    for value in values:
        if isnan(value):
            nans += 1
        else:
            ordered.append(value)
    sort(ordered)
    for _ in range(nans):
        ordered.append(Float64(0) / Float64(0))
    var position = q * Float64(n - 1)
    var lower = Int(floor(position))
    var upper = min(Int(ceil(position)), n - 1)
    var fraction = position - Float64(lower)
    if method == "lower":
        return ordered[lower]
    if method == "higher":
        return ordered[upper]
    if method == "nearest":
        return ordered[min(Int(floor(position + 0.5)), n - 1)]
    if lower == upper or fraction == 0:
        return ordered[lower]
    if method == "midpoint":
        return (ordered[lower] + ordered[upper]) / 2
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def same(a: Optional[Float64], b: Optional[Float64]) -> Bool:
    if not a or not b:
        return not a and not b
    var x = a.value()
    var y = b.value()
    if isnan(x) or isnan(y):
        return isnan(x) and isnan(y)
    return bitcast[DType.uint64](x) == bitcast[DType.uint64](y)


def sample(
    mut rng: Lcg, n: Int, distinct: Int, nan_every: Int
) -> List[Float64]:
    var nan = Float64(0) / Float64(0)
    var values = List[Float64](capacity=n)
    for _ in range(n):
        if nan_every > 0 and rng.next(nan_every) == 0:
            values.append(nan)
        else:
            # No -0.0: it equals 0.0, and which of the two a sort puts
            # first is unspecified.
            values.append(Float64(rng.next(distinct)) / 8 - 30)
    return values^


def test_every_method_matches_the_sort() raises:
    var rng = Lcg(7)
    var methods: List[String] = [
        "linear",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ]
    var qs: List[Float64] = [0.0, 0.1, 0.25, 0.5, 0.73, 0.9, 1.0]
    for n in [0, 1, 2, 3, 7, 24, 25, 26, 100, 1000, 20_000]:
        for distinct in [1, 3, 1000, 1_000_000]:
            for nan_every in [0, 5, 1]:
                var values = sample(rng, n, distinct, nan_every)
                for method in methods:
                    for q in qs:
                        var got = quantile_of(values.copy(), q, method)
                        var want = reference(values.copy(), q, method)
                        assert_true(
                            same(got, want),
                            String(n) + " " + method + " " + String(q),
                        )


def test_grouped_and_global_medians() raises:
    var rng = Lcg(11)
    var rows = 300_000
    var keys = List[Int64](capacity=rows)
    var values = List[Float64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    var nan = Float64(0) / Float64(0)
    for _ in range(rows):
        # A few large groups and many small ones; group 99 is all null.
        var k = rng.next(4) if rng.next(3) else 4 + rng.next(96)
        keys.append(Int64(k))
        var pick = rng.next(50)
        values.append(nan if pick == 0 else Float64(rng.next(5000)) / 4)
        valid.append(k != 99 and rng.next(10) != 0)
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys.copy())),
            Series("v", Column[Float64](values.copy(), valid.copy())),
        ]
    )
    for q in [0.5, 0.05, 1.0]:
        for method in ["linear", "nearest", "midpoint"]:
            var out = frame.group_by("k").agg(
                [col("v").quantile(q, method).alias("q")]
            )
            for row in range(out.height()):
                var key = out.column("k").get(row).int64()
                var group = List[Float64]()
                for i in range(rows):
                    if keys[i] == key and valid[i]:
                        group.append(values[i])
                var want = reference(group^, q, method)
                var cell = out.column("q").get(row)
                var got = Optional[Float64]() if cell.is_null() else Optional(
                    cell.float64()
                )
                assert_true(same(got, want), String(key) + " " + method)
    var everything = List[Float64]()
    for i in range(rows):
        if valid[i]:
            everything.append(values[i])
    var median = frame.select(col("v").median()).item()
    assert_true(
        same(Optional(median.float64()), reference(everything^, 0.5, "linear"))
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
