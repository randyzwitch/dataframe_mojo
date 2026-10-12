"""Allocation advice preserves initialized bytes and caller-owned writable storage."""
from std.testing import assert_equal
from dataframe.huge_pages import advise_huge_pages, huge_list, huge_uninit


def test_advised_allocations_preserve_fills_writes_and_unaligned_range_contents() raises:
    # Zero, ordinary-page and huge-page byte boundaries, independent of whether
    # the operating system elects to provide huge pages.
    for n in [0, 1, 511, 512, 513, 262143, 262144, 262145]:
        var filled = huge_list(n, UInt64(0xDA7A5500FEDCBA98))
        var written = huge_uninit[UInt64](n)
        assert_equal(len(filled), n)
        assert_equal(len(written), n)
        for row in range(n):
            assert_equal(filled[row], UInt64(0xDA7A5500FEDCBA98))
            written[row] = UInt64(row * 17 + 3)
        for row in range(n):
            assert_equal(written[row], UInt64(row * 17 + 3))
    var bytes = huge_uninit[UInt8]((4 << 20) + 8193)
    for row in range(len(bytes)):
        bytes[row] = UInt8(row % 251)
    var address = Int(bytes.unsafe_ptr())
    for offset in [0, 1, 4095, 4096, 4097]:
        for length in [0, 1, (2 << 20) - 1, 2 << 20, (2 << 20) + 1]:
            advise_huge_pages(address + offset, length)
    for row in range(len(bytes)):
        assert_equal(bytes[row], UInt8(row % 251))
