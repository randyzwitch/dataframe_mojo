"""Float kernels load SIMD vectors straight from shared column buffers.

Every window (offset, length) of a column is a zero-copy view, and the
unfused float kernels read it with contiguous loads plus a scalar tail.
These tests compare them with element-by-element math for empty windows,
nonzero offsets, validity-byte boundaries, scalar broadcasts, and short
tails at several SIMD widths.
"""
from std.math import floor, sqrt
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import Column, DataFrame, Expr, Series, col, lit
from dataframe.binding import bind
from dataframe.execution import evaluate


def source() raises -> DataFrame:
    var x = List[Float64]()
    var y = List[Float32]()
    var valid = List[Bool]()
    for i in range(64):
        x.append(Float64(i) * 1.25 - 20)
        y.append(Float32(i) * 0.5 + 1)
        valid.append(i % 7 != 3)
    return DataFrame(
        [
            Series("x", Column[Float64](x^, valid)),
            Series("y", Column[Float32](y^, valid)),
        ]
    )


def unfused[width: Int](frame: DataFrame, expr: Expr) raises -> Series:
    return evaluate[width](
        bind(expr, frame._columns, fuse=False),
        frame._columns,
        frame.height(),
        batch_size=1 << 20,
    )


def check_window[width: Int](frame: DataFrame) raises:
    var x = frame._columns[0].float64()
    var n = frame.height()
    var sums = unfused[width](frame, (col("x") + col("x")) * lit(Float64(0.5)))
    var mods = unfused[width](frame, col("x") % lit(Float64(3)))
    var roots = unfused[width](frame, col("x").abs().sqrt())
    var less = unfused[width](frame, col("x") < lit(Float64(0)))
    var f32 = unfused[width](frame, col("y") * lit(Float32(2)) - col("y"))
    assert_equal(len(sums), n)
    for i in range(n):
        var null = x.is_null(i)
        assert_equal(sums.get(i).is_null(), null)
        assert_equal(less.get(i).is_null(), null)
        assert_equal(f32.get(i).is_null(), null)
        if null:
            continue
        var v = x.value(i)
        assert_equal(sums.get(i).float64(), v)
        assert_equal(mods.get(i).float64(), v - 3 * floor(v / 3))
        assert_equal(roots.get(i).float64(), sqrt(abs(v)))
        assert_equal(less.get(i).bool(), v < 0)
        var w = frame._columns[1].float32().value(i)
        assert_equal(f32.get(i).float32(), w * 2 - w)


def test_every_window_and_width() raises:
    var frame = source()
    for offset in range(0, 18):
        for length in range(0, 41):
            var window = frame.slice(offset, length)
            check_window[1](window)
            check_window[4](window)
            check_window[8](window)
            check_window[16](window)


def test_windows_share_the_source_buffer() raises:
    var frame = source()
    var window = frame.slice(13, 29)
    ref a = window._columns[0]._data[Column[Float64]]
    ref b = frame._columns[0]._data[Column[Float64]]
    assert_true(a._shares_buffers_with(b))
    # The kernel reads the window's first row at buffer offset 13.
    assert_equal(Int(a._ptr()) - Int(b._ptr()), 13 * 8)


def test_scalar_broadcasts_on_either_side() raises:
    var frame = source().slice(5, 23)
    var left = unfused[8](frame, lit(Float64(10)) - col("x"))
    var right = unfused[8](frame, col("x") - lit(Float64(10)))
    var x = frame._columns[0].float64()
    for i in range(frame.height()):
        if x.is_null(i):
            assert_true(left.get(i).is_null())
            continue
        assert_equal(left.get(i).float64(), 10 - x.value(i))
        assert_equal(right.get(i).float64(), x.value(i) - 10)


def test_null_lanes_are_zero_filled() raises:
    var frame = source()
    var result = unfused[8](frame, col("x").sqrt())
    ref column = result._data[Column[Float64]]
    for i in range(frame.height()):
        if not column._valid(i):
            assert_equal(column._get(i), 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
