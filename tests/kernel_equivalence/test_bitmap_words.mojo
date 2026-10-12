"""Packed validity append and Kleene words are checked against scalar truth."""
from std.testing import assert_equal
from dataframe import Series
from dataframe.bool_column import BoolColumn
from dataframe.column import _append_bits
from dataframe.expr import AND, OR, XOR
from dataframe.expr_kernels import _logical_bits, _logical, _not_bits
from tests.support.boolean_word_reference import truth_column, truth_state


def check_logic[
    op: Int
](
    left: BoolColumn,
    right: BoolColumn,
    n: Int,
    lo: Int,
    ro: Int,
    lm: Int,
    rm: Int,
    broadcast: Int,
) raises:
    var packed = _logical_bits[op](left, right)
    var fallback = _logical[op](left, right)
    var and_table: List[Int] = [0, 0, 0, 0, 1, -1, 0, -1, -1]
    var or_table: List[Int] = [0, 1, -1, 1, 1, 1, -1, 1, -1]
    var xor_table: List[Int] = [0, 1, -1, 1, 0, -1, -1, -1, -1]
    for row in range(n):
        var a = truth_state(lo + (0 if broadcast == 1 else row), lm, 0)
        var b = truth_state(ro + (0 if broadcast == 2 else row), rm, 1)
        var position = (2 if a < 0 else a) * 3 + (2 if b < 0 else b)
        var expected = (
            and_table[position] if op
            == AND else or_table[position] if op
            == OR else xor_table[position]
        )
        for series in [packed.copy(), fallback.copy()]:
            ref result = series._data[BoolColumn]
            assert_equal(result._valid(row), expected >= 0)
            if expected >= 0:
                assert_equal(result._get(row), expected == 1)
        # Clearing null value bits is the packed kernel's explicit internal
        # contract; scalar XOR only promises validity and visible values.
        assert_equal(packed._data[BoolColumn]._get(row), expected == 1)


def test_packed_kleene_words_scalar_fallback_broadcast_and_unaligned_tails_match_literal_truth_tables() raises:
    for n in [0, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 257]:
        for lo in range(8):
            for ro in [0, 3, 7]:
                for lm in range(3):
                    for rm in range(3):
                        for broadcast in [0, 1, 2]:
                            if n == 0 and broadcast != 0:
                                continue
                            var left = truth_column(
                                1 if broadcast == 1 else n, lo, lm, 0
                            )
                            var right = truth_column(
                                1 if broadcast == 2 else n, ro, rm, 1
                            )
                            check_logic[AND](
                                left, right, n, lo, ro, lm, rm, broadcast
                            )
                            check_logic[OR](
                                left, right, n, lo, ro, lm, rm, broadcast
                            )
                            check_logic[XOR](
                                left, right, n, lo, ro, lm, rm, broadcast
                            )
                            var inverse = _not_bits(left)
                            ref bits = inverse._data[BoolColumn]
                            for row in range(len(left)):
                                var state = truth_state(lo + row, lm, 0)
                                assert_equal(bits._valid(row), state >= 0)
                                assert_equal(bits._get(row), state == 0)


def append_pattern(row: Int, side: Int) -> Bool:
    return (row * 37 + side * 19) % 11 < 5


def pack_pattern(n: Int, side: Int) -> List[UInt8]:
    var bits = List[UInt8](length=(n + 7) // 8, fill=UInt8(0))
    for row in range(n):
        if append_pattern(row, side):
            bits[row // 8] |= UInt8(1) << UInt8(row % 8)
    return bits^


def test_validity_append_all_bit_alignments_vector_boundaries_and_dense_poison_padding_preserve_each_bit() raises:
    for alignment in range(8):
        var prefix = 16 + alignment
        for offset in range(8):
            for count in [0, 1, 7, 8, 9, 127, 128, 129, 255, 256, 257]:
                var bits = pack_pattern(prefix, 0)
                # Every unused destination/source padding bit is poison true;
                # neither may leak into visible appended values.
                for bit in range(prefix, len(bits) * 8):
                    bits[bit // 8] |= UInt8(1) << UInt8(bit % 8)
                var source = pack_pattern(offset + count, 1)
                for bit in range(offset + count, len(source) * 8):
                    source[bit // 8] |= UInt8(1) << UInt8(bit % 8)
                _append_bits(bits, prefix, source, offset, count)
                for row in range(prefix + count):
                    var actual = (bits[row // 8] >> (UInt8(row % 8))) & UInt8(1)
                    assert_equal(
                        actual == 1,
                        append_pattern(row, 0) if row
                        < prefix else append_pattern(offset + row - prefix, 1),
                    )
