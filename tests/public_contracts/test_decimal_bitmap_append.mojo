"""Decimal-local validity append retains complete legacy bitmap bytes."""
# contract-group: expr.chunk_windows
# contract-group: types.cast
from std.testing import assert_equal
from dataframe.kernels.cast import _append_decimal_validity
from dataframe.storage.buffer import _Buffer
from dataframe.storage.column import _append_validity, _pack_bits


def _same_bytes(actual: List[UInt8], expected: List[UInt8]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(expected)):
        assert_equal(actual[i], expected[i])


def test_decimal_bitmap_offsets_boundaries_and_sentinels_match_legacy_bytes() raises:
    for length in [0, 1, 2, 3, 4, 5, 6, 7, 64, 65, 71]:
        for source_offset in range(8):
            for count in [0, 1, 7, 8, 9, 63, 64, 65]:
                for padding in [0, 9]:
                    # Empty bitmap means all valid; explicit all-valid and
                    # all-null bitmaps remain distinct physical controls.
                    for incoming_kind in range(4):
                        var flags = List[Bool]()
                        for i in range(source_offset + count + padding):
                            flags.append(
                                incoming_kind == 1
                                or (incoming_kind == 2 and i % 5 != 1)
                            )
                        var incoming = _Buffer[UInt8](List[UInt8]())
                        if incoming_kind != 0:
                            incoming = _Buffer[UInt8](_pack_bits(flags))
                        for prefix_kind in range(2):
                            var prefix_flags = List[Bool]()
                            for i in range(length):
                                prefix_flags.append(
                                    prefix_kind == 0 or i % 3 != 0
                                )
                            var original = List[UInt8]()
                            if prefix_kind != 0:
                                original = _pack_bits(prefix_flags)
                                if length % 8 != 0:
                                    # Valid physical tail bits can be stale ones
                                    # after an earlier all-valid append.
                                    original[len(original) - 1] |= UInt8(
                                        255
                                    ) << UInt8(length % 8)
                            var actual = original.copy()
                            var expected = original.copy()
                            _append_decimal_validity(
                                actual,
                                length,
                                incoming[].span(),
                                source_offset,
                                count,
                            )
                            _append_validity(
                                expected,
                                length,
                                incoming[].span(),
                                source_offset,
                                count,
                            )
                            _same_bytes(actual, expected)
                            # A second independent check reads every logical bit
                            # against the declared Boolean input rows.
                            for row in range(length + count):
                                var wanted = True
                                if row < length:
                                    wanted = prefix_flags[row]
                                elif incoming_kind != 0:
                                    wanted = flags[source_offset + row - length]
                                var got = True
                                if len(actual) != 0:
                                    got = (
                                        (actual[row // 8] >> UInt8(row % 8)) & 1
                                    ) != 0
                                assert_equal(got, wanted)


def test_decimal_bitmap_dense_prefix_and_mixed_chunk_tails_have_literal_bytes() raises:
    var bits = List[UInt8]()
    var dense = _Buffer[UInt8](List[UInt8]())
    _append_decimal_validity(bits, 0, dense[].span(), 5, 3)
    assert_equal(len(bits), 0)
    var flags: List[Bool] = [
        False,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    var mixed = _Buffer[UInt8](_pack_bits(flags))
    _append_decimal_validity(bits, 3, mixed[].span(), 1, 9)
    var first: List[UInt8] = [215, 12]
    _same_bytes(bits, first)
    _append_decimal_validity(bits, 12, dense[].span(), 7, 3)
    # The delegated absent-input helper retains unused all-valid tail bits.
    var second: List[UInt8] = [215, 252]
    _same_bytes(bits, second)
    var false_flags: List[Bool] = [True, False]
    var false_tail = _Buffer[UInt8](_pack_bits(false_flags))
    _append_decimal_validity(bits, 15, false_tail[].span(), 1, 1)
    var third: List[UInt8] = [215, 124]
    _same_bytes(bits, third)
    var true_flags: List[Bool] = [False, True]
    var true_tail = _Buffer[UInt8](_pack_bits(true_flags))
    _append_decimal_validity(bits, 16, true_tail[].span(), 1, 1)
    var fourth: List[UInt8] = [215, 124, 1]
    _same_bytes(bits, fourth)
