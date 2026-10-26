"""Global variance and standard deviation of numeric columns match a two-pass
Float64 reference over long columns, with nulls, offsets and chunks."""
# contract-group: reduce.global
from std.math import sqrt
from std.testing import assert_true
from dataframe import Column, DataFrame, Expr, Series, col
from tests.support.coverage_generators import chunk_layout, offset_layout


def _close(got: Float64, want: Float64) -> Bool:
    return abs(got - want) <= 1e-9 * max(1.0, abs(want))


def _check(n: Int, nulls: Bool, ddof: Int) raises:
    # Float values sit far from zero, so a one-pass sum of squares would
    # lose their spread; UInt64 values are small, which biased ordering keys
    # would lose. Every type holds its values exactly.
    var narrow = List[Float32]()
    var wide = List[Float64]()
    var ints = List[Int32]()
    var big = List[UInt64]()
    var valid = List[Bool]()
    var kept = List[Float64]()
    for row in range(n):
        var step = (row * 7919) % 2001
        narrow.append(Float32(4096 + Float64(step) / 8))
        wide.append(1e6 + Float64(step) / 8)
        ints.append(Int32(100_000 + step))
        big.append(UInt64(step))
        var present = not nulls or row % 7 != 3
        valid.append(present)
        if present:
            kept.append(Float64(step))
    var frame = DataFrame(
        [
            Series("f", Column[Float32](narrow^, valid.copy())),
            Series("d", Column[Float64](wide^, valid.copy())),
            Series("i", Column[Int32](ints^, valid.copy())),
            Series("u", Column[UInt64](big^, valid)),
        ]
    )
    # The reference variance of each column in units of `step`.
    var mean = Float64(0)
    for value in kept:
        mean += value
    mean /= Float64(max(1, len(kept)))
    var squares = Float64(0)
    for value in kept:
        squares += (value - mean) * (value - mean)
    var defined = len(kept) > ddof
    var unit = squares / Float64(len(kept) - ddof) if defined else 0
    var names: List[String] = ["f", "d", "i", "u"]
    var scales: List[Float64] = [1.0 / 64, 1.0 / 64, 1, 1]
    var layouts: List[DataFrame] = [
        frame.copy(),
        offset_layout(frame),
        chunk_layout(frame),
    ]
    for layout in layouts:
        for lazy in [False, True]:
            var exprs = List[Expr]()
            for name in names:
                exprs.append(col(name).var(ddof).alias("v" + name))
                exprs.append(col(name).std(ddof).alias("s" + name))
            var out = (
                layout.lazy()
                .select_exprs(exprs)
                .collect() if lazy else layout.select_exprs(exprs)
            )
            for k in range(len(names)):
                var label = String(names[k], " n=", n, " ddof=", ddof)
                var variance = out.item(0, "v" + names[k])
                var deviation = out.item(0, "s" + names[k])
                if not defined:
                    assert_true(variance.is_null(), label)
                    assert_true(deviation.is_null(), label)
                    continue
                var want = unit * scales[k]
                assert_true(_close(variance.float64(), want), label)
                assert_true(_close(deviation.float64(), sqrt(want)), label)


def test_numeric_variance_matches_a_two_pass_reference() raises:
    # Lengths around the 128-value block and the eight-lane loop inside it.
    for n in [0, 1, 2, 9, 127, 128, 129, 257, 1000, 70_001]:
        for nulls in [False, True]:
            for ddof in [0, 1]:
                _check(n, nulls, ddof)
