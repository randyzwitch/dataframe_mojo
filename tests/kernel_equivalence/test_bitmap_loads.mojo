"""Bounded bitmap loads match individual bits without reading padding."""
# contract-group: kernel.row_ops
from std.sys import size_of
from std.testing import assert_equal
from dataframe.storage.bitmap import _bitmap_word, _bitmap_bytes


def check_words[D: DType]() raises:
    comptime WIDTH = size_of[Scalar[D]]() * 8
    for size in range(0, 20):
        var bytes = List[UInt8]()
        for k in range(size):
            bytes.append(UInt8((k * 73 + 151) & 255))
        for bit in range(size * 8 + 9):
            for count in range(WIDTH + 1):
                var expected = Scalar[D](0)
                for j in range(count):
                    if bit + j < size * 8:
                        expected |= Scalar[D](
                            (bytes[(bit + j) // 8] >> UInt8((bit + j) & 7)) & 1
                        ) << Scalar[D](j)
                assert_equal(_bitmap_word[D](Span(bytes), bit, count), expected)
                if size == 0:
                    var mask = Scalar[D](0)
                    for j in range(count):
                        mask |= Scalar[D](1) << Scalar[D](j)
                    assert_equal(
                        _bitmap_word[D](Span(bytes), bit, count, Scalar[D].MAX),
                        mask,
                    )


def test_bitmap_words_all_widths_alignments_and_exact_physical_tails() raises:
    check_words[DType.uint8]()
    check_words[DType.uint32]()
    check_words[DType.uint64]()


def check_vectors[W: Int]() raises:
    var bytes = List[UInt8]()
    for k in range(W + 1):
        bytes.append(UInt8((k * 73 + 151) & 255))
    for offset in range(8):
        for scalar in [False, True]:
            var got = _bitmap_bytes[W](Span(bytes), offset, scalar, 0)
            for j in range(W):
                var expected = UInt8(0)
                for lane in range(8):
                    var bit = offset + (0 if scalar else j * 8 + lane)
                    expected |= (
                        (bytes[bit // 8] >> UInt8(bit & 7)) & 1
                    ) << UInt8(lane)
                assert_equal(got[j], expected)
    var absent = List[UInt8]()
    for empty in [UInt8(0), UInt8(255)]:
        var got = _bitmap_bytes[W](Span(absent), 7, False, empty)
        for j in range(W):
            assert_equal(got[j], empty)


def test_bitmap_vectors_keep_broadcast_and_absent_validity_policies() raises:
    check_vectors[1]()
    check_vectors[16]()
    check_vectors[32]()
