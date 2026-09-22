"""Structural bit packing and target-selected quote parity."""
from std.testing import TestSuite, assert_equal
from dataframe.csv_bits import _mask64, _prefix_xor_inclusive


def test_mask_lane_order() raises:
    for bit in range(64):
        var lanes = SIMD[DType.uint8, 64](0)
        lanes[bit] = 1
        assert_equal(
            _mask64(lanes.eq(SIMD[DType.uint8, 64](1))),
            UInt64(1) << UInt64(bit),
        )


def test_prefix_parity_against_bitwise_reference() raises:
    var mask = UInt64(0)
    for _ in range(260):
        var expected = UInt64(0)
        var parity = UInt64(0)
        for bit in range(64):
            parity ^= (mask >> UInt64(bit)) & 1
            expected |= parity << UInt64(bit)
        assert_equal(_prefix_xor_inclusive(mask), expected)
        mask = mask * UInt64(6364136223846793005) + UInt64(1442695040888963407)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
