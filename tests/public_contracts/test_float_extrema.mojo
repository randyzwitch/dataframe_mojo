"""Float32 and Float64 global min and max over long columns equal a scalar
reference: NaN and nulls never win, an all-NaN column gives NaN, and one with
no valid value gives null (the rules in test_nan_extrema.mojo)."""
# contract-group: reduce.global
from std.testing import assert_equal, assert_true
from std.memory import bitcast
from std.utils.numerics import inf, nan
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.coverage_generators import chunk_layout, offset_layout


def _value(row: Int, pattern: Int) -> Float64:
    """Pattern 0 mixes numbers and NaN, 1 is all NaN, 2 all -inf, 3 all +inf,
    4 has infinities at the ends among numbers, 5 and 6 are zeros of both
    signs starting with -0.0 and 0.0."""
    if pattern == 1:
        return nan[DType.float64]()
    if pattern == 2:
        return -inf[DType.float64]()
    if pattern == 3:
        return inf[DType.float64]()
    if pattern == 5 or pattern == 6:
        # Zeros of both signs: the earlier row wins the tie.
        var negative = (row // 3) % 2 == (0 if pattern == 5 else 1)
        return -0.0 if negative else 0.0
    if row % 13 == 5:
        return nan[DType.float64]()
    if pattern == 4 and row == 0:
        return inf[DType.float64]()
    if pattern == 4 and row == 1:
        return -inf[DType.float64]()
    return Float64((row * 7919) % 2001) / 8 - 125


def _check[D: DType](n: Int, pattern: Int, nulls: Bool) raises:
    var values = List[Scalar[D]]()
    var valid = List[Bool]()
    var low = inf[DType.float64]()
    var high = -inf[DType.float64]()
    var numbers = 0
    var present = 0
    var first = Float64(0)
    for row in range(n):
        var value = _value(row, pattern)
        var keep = not nulls or row % 7 != 3
        values.append(value.cast[D]())
        valid.append(keep)
        if not keep:
            continue
        present += 1
        if value == value:
            if numbers == 0:
                first = value
            numbers += 1
            low = min(low, value)
            high = max(high, value)
    var frame = DataFrame([Series("x", Column[Scalar[D]](values^, valid))])
    var layouts: List[DataFrame] = [
        frame.copy(),
        offset_layout(frame),
        chunk_layout(frame),
    ]
    for layout in layouts:
        for lazy in [False, True]:
            var extremes: List[Expr] = [
                col("x").min().alias("lo"),
                col("x").max().alias("hi"),
            ]
            var out = (
                layout.lazy()
                .select_exprs(extremes)
                .collect() if lazy else layout.select_exprs(extremes)
            )
            for name in ["lo", "hi"]:
                var cell = out.item(0, name)
                var label = String(D, " n=", n, " pattern=", pattern, " ", name)
                if present == 0:
                    assert_true(cell.is_null(), label)
                    continue
                var got = Float64(cell.float32()) if D == DType.float32 else (
                    cell.float64()
                )
                if numbers == 0:
                    assert_true(got != got, label)
                    continue
                var want = low if name == "lo" else high
                if pattern >= 5:
                    want = first
                want = want.cast[D]().cast[DType.float64]()
                assert_equal(
                    bitcast[DType.uint64](got),
                    bitcast[DType.uint64](want),
                    label,
                )


def test_float_min_max_match_a_scalar_reference_over_long_columns() raises:
    # Lengths around the 8- and 16-lane loops and their scalar tails.
    for n in [0, 1, 7, 8, 15, 16, 17, 33, 1000, 70_001]:
        for pattern in range(7):
            for nulls in [False, True]:
                _check[DType.float32](n, pattern, nulls)
                _check[DType.float64](n, pattern, nulls)
