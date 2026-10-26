"""Float32 and Float64 sums and means equal an exact Float64 reference."""
# contract-group: reduce.global
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.coverage_generators import chunk_layout, offset_layout


def _frame(n: Int, nulls: Bool) raises -> Tuple[DataFrame, Float64, Int]:
    """Float32 and Float64 copies of the same values, their exact total and
    the valid count. Every value is a multiple of 1/8 below 7 in magnitude,
    so the Float64 total is exact in any order of addition."""
    var narrow = List[Float32]()
    var wide = List[Float64]()
    var valid = List[Bool]()
    var total = Float64(0)
    var count = 0
    for row in range(n):
        var value = Float64((row * 37) % 101) / 8 - 6
        var present = not nulls or row % 9 != 4
        narrow.append(Float32(value))
        wide.append(value)
        valid.append(present)
        if present:
            total += value
            count += 1
    var frame = DataFrame(
        [
            Series("f", Column[Float32](narrow^, valid.copy())),
            Series("d", Column[Float64](wide^, valid)),
        ]
    )
    return (frame^, total, count)


def test_float_sums_and_means_match_an_exact_reference_in_every_layout() raises:
    # Lengths around the eight-lane loop and its scalar tail, nulls with an
    # unaligned bitmap (offset_layout), and chunk boundaries (chunk_layout).
    for n in [0, 1, 7, 8, 9, 65, 1000, 70_001]:
        for nulls in [False, True]:
            var made = _frame(n, nulls)
            var total = made[1]
            var count = made[2]
            var layouts: List[DataFrame] = [
                made[0].copy(),
                offset_layout(made[0]),
                chunk_layout(made[0]),
            ]
            for frame in layouts:
                for lazy in [False, True]:
                    var totals: List[Expr] = [
                        col("f").sum().alias("f"),
                        col("d").sum().alias("d"),
                    ]
                    var sums = (
                        frame.lazy()
                        .select_exprs(totals)
                        .collect() if lazy else frame.select_exprs(totals)
                    )
                    assert_true(sums.column("f").dtype() == DataType.FLOAT32)
                    assert_equal(sums.item(0, "f").float32(), Float32(total))
                    assert_equal(sums.item(0, "d").float64(), total)
                    var averages: List[Expr] = [col("f").mean().alias("f")]
                    var means = (
                        frame.lazy()
                        .select_exprs(averages)
                        .collect() if lazy else frame.select_exprs(averages)
                    )
                    if count == 0:
                        assert_true(means.item(0, "f").is_null())
                        continue
                    var mean = total / Float64(count)
                    var got = means.item(0, "f")
                    var actual = (
                        Float64(got.float32()) if got.dtype()
                        == DataType.FLOAT32 else got.float64()
                    )
                    assert_true(
                        abs(actual - mean) <= 1e-6 * max(1.0, abs(mean)),
                        String(actual, " ", mean),
                    )
