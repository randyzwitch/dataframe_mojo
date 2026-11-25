"""Binary64 accumulator bit patterns, checked against native CPU arithmetic."""

# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal
from dataframe_metal.accelerator.gpu.soft_binary64 import (
    add_binary64,
    widen_binary32,
)


def test_metal_binary32_promotion_matches_native_float64() raises:
    var sign = UInt32(1) << 31
    for negative in [False, True]:
        for exponent in range(256):
            for fraction in [0, 1, 2, 0x3FFFFF, 0x400000, 0x7FFFFE, 0x7FFFFF]:
                var raw = (
                    (sign if negative else UInt32(0))
                    | (UInt32(exponent) << 23)
                    | UInt32(fraction)
                )
                if exponent == 255 and fraction:
                    var widened = widen_binary32(raw)
                    assert_equal(widened >> 63, UInt64(negative))
                    assert_equal((widened >> 51) & 4095, UInt64(4095))
                    assert_equal(
                        widened & 0x0007FFFFFFFFFFFF,
                        UInt64(fraction & 0x3FFFFF) << 29,
                    )
                else:
                    var value = Float64(bitcast[DType.float32](raw))
                    assert_equal(
                        widen_binary32(raw), bitcast[DType.uint64](value)
                    )


def test_metal_binary64_add_rounds_ties_and_cancellation() raises:
    var one = UInt64(0x3FF0000000000000)
    var half_ulp = UInt64(0x3CA0000000000000)
    var sign = UInt64(1) << 63
    assert_equal(add_binary64(one, half_ulp - 1), one)
    assert_equal(add_binary64(one, half_ulp), one)
    assert_equal(add_binary64(one, half_ulp + 1), one + 1)
    assert_equal(add_binary64(one + 1, half_ulp), one + 2)
    assert_equal(
        add_binary64(sign | (one + 1), sign | half_ulp), sign | (one + 2)
    )
    assert_equal(add_binary64(one, sign | half_ulp), one - 1)
    assert_equal(add_binary64(one, sign | one), UInt64(0))
    assert_equal(add_binary64(one + 1, sign | one), UInt64(0x3CB0000000000000))
    var min_normal = UInt64(1) << 52
    assert_equal(add_binary64(min_normal - 1, 1), min_normal)
    assert_equal(add_binary64(min_normal, sign | (min_normal - 1)), UInt64(1))
    assert_equal(add_binary64(sign | min_normal, min_normal - 1), sign | 1)
    var infinity = UInt64(0x7FF0000000000000)
    var overflow_half_ulp = UInt64(0x7C90000000000000)
    assert_equal(
        add_binary64(infinity - 1, overflow_half_ulp - 1), infinity - 1
    )
    assert_equal(add_binary64(infinity - 1, overflow_half_ulp), infinity)
    assert_equal(add_binary64(sign, sign), sign)
    assert_equal(add_binary64(0, sign), UInt64(0))
    assert_equal(add_binary64(sign, 0), UInt64(0))
    assert_equal(add_binary64(sign | infinity, one), sign | infinity)
    assert_equal(add_binary64(infinity, infinity), infinity)
    var quiet = UInt64(1) << 51
    assert_equal(add_binary64(infinity, sign | infinity), infinity | quiet)
    assert_equal(add_binary64(infinity + 23, one), (infinity + 23) | quiet)
    assert_equal(
        add_binary64(one, sign | (infinity + 23)),
        sign | (infinity + 23) | quiet,
    )


def test_metal_binary64_add_matches_finite_cpu_operand_pairs() raises:
    var fraction_mask = (UInt64(1) << 52) - 1
    var sign = UInt64(1) << 63
    for exponent in [0, 1, 2, 53, 54, 1023, 2044, 2046]:
        for delta in [0, 1, 2, 3, 51, 52, 53, 54, 63, 64, 65, 1100]:
            for a_fraction in [UInt64(1), fraction_mask >> 1, fraction_mask]:
                var a = (UInt64(exponent) << 52) | a_fraction
                for b_fraction in [UInt64(0), UInt64(1), fraction_mask]:
                    var b = (
                        UInt64(max(0, exponent - delta)) << 52
                    ) | b_fraction
                    for a_sign in [UInt64(0), sign]:
                        for b_sign in [UInt64(0), sign]:
                            var left = a | a_sign
                            var right = b | b_sign
                            var native = bitcast[DType.float64](left) + bitcast[
                                DType.float64
                            ](right)
                            var expected = bitcast[DType.uint64](native)
                            assert_equal(add_binary64(left, right), expected)
                            assert_equal(add_binary64(right, left), expected)
