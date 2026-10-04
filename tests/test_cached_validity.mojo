"""Cached bitmap reads agree with independently generated validity."""
from std.testing import TestSuite, assert_equal
from dataframe import Column
from dataframe.column import _validity_at


def test_cached_validity_respects_slice_offsets_and_byte_boundaries() raises:
    var values = List[Int64]()
    var valid = List[Bool]()
    for i in range(72):
        values.append(Int64(i))
        valid.append(i % 5 != 0)
    var source = Column[Int64](values^, valid^)
    for offset in range(8):
        for length in [0, 1, 7, 8, 9, 63, 64, 65]:
            var part = source.slice(offset, length)
            var bits = part._bits[].unsafe_ptr()
            for i in range(length):
                assert_equal(
                    _validity_at(bits, part._offset + i),
                    (offset + i) % 5 != 0,
                )
                assert_equal(
                    _validity_at(bits, part._offset + i),
                    not part.is_null(i),
                )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
