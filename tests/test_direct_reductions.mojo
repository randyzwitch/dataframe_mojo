"""Bare numeric reductions retain scalar semantics across SIMD/worker tails."""
from std.ffi import external_call
from std.memory import bitcast
from std.math import isnan
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import Column, DataFrame, Series, col


def set_threads(n: Int):
    var name = String("DATAFRAME_THREADS")
    var value = String(n)
    _ = external_call["setenv", Int32](
        name.unsafe_ptr(), value.unsafe_ptr(), Int32(1)
    )


def check_float_window(offset: Int, length: Int, mode: Int) raises:
    var values = List[Float64](capacity=offset + length)
    var valid = List[Bool](capacity=offset + length)
    var expected = Float64(0)
    var count = Int64(0)
    for i in range(offset + length):
        var present = mode == 0 or (mode == 1 and i % 7 != 0)
        # NaN in a null slot must never contaminate the SIMD accumulator.
        var value = Float64((i % 17) - 8) if present else Float64(0) / Float64(
            0
        )
        values.append(value)
        valid.append(present)
        if i >= offset and present:
            expected += value
            count += 1
    var frame = DataFrame(
        [Series("x", Column[Float64](values^, valid^).slice(offset, length))]
    )
    assert_equal(frame.select(col("x").sum()).item().float64(), expected)
    assert_equal(frame.select(col("x").count()).item().int64(), count)
    var mean = frame.select(col("x").mean()).item()
    if count:
        assert_equal(mean.float64(), expected / Float64(count))
    else:
        assert_true(mean.is_null())
    assert_true(
        frame.select(col("x").sum(min_count=Int(count) + 1)).item().is_null()
    )


def test_float_bitmap_windows() raises:
    set_threads(1)
    for offset in range(8):
        for length in range(18):
            for mode in range(3):
                check_float_window(offset, length, mode)


def test_float_worker_tails() raises:
    set_threads(32)
    for mode in range(3):
        check_float_window(3, 1_000_003, mode)


def test_extrema_keep_first_zero_and_nan_order() raises:
    set_threads(32)
    var values = List[Float64](length=200_003, fill=0.0)
    values[0] = -0.0
    var frame = DataFrame([Series("x", Column[Float64](values^))])
    assert_equal(
        bitcast[DType.uint64](frame.select(col("x").min()).item().float64()),
        UInt64(1) << 63,
    )
    assert_equal(
        bitcast[DType.uint64](frame.select(col("x").max()).item().float64()),
        UInt64(1) << 63,
    )
    var special = DataFrame(
        [Series("x", Column[Float64]([Float64(0) / Float64(0), 2.0, -3.0]))]
    )
    assert_equal(special.select(col("x").min()).item().float64(), -3.0)
    assert_true(isnan(special.select(col("x").max()).item().float64()))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
