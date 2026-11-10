"""Packed record widths sort in place without touching adjacent records."""
# contract-group: kernel.order_keys
from std.testing import assert_equal
from dataframe.sorting.algorithms import Rec, SortKeys, sort_keys


def _packed_records[W: Int, VAR: Bool]() raises:
    comptime K = SortKeys[W, VAR]
    comptime N = 257
    comptime SENTINEL = UInt64(0xA55AA55AA55AA55A)
    var words = List[UInt64](length=(N + 2) * K.R, fill=SENTINEL)
    var bytes = List[UInt8](length=N * 10, fill=97)
    var keys = K(Int(words.unsafe_ptr()))
    for i in range(N):
        # 73 permutes 0..256, with modular inverse 169. Every record has
        # a distinct key, so the expected complete permutation is literal.
        var rank = (i * 73) % N
        var rec = Rec(0)
        comptime if VAR:
            bytes[i * 10 + 8] = UInt8(rank >> 8)
            bytes[i * 10 + 9] = UInt8(rank & 255)
            rec[0] = UInt64(0x6161616161616161)
            rec[1] = 10
            rec[2] = UInt64(Int(bytes.unsafe_ptr().unsafe_offset(i * 10)))
        elif W == 1:
            rec[0] = UInt64(rank) + (UInt64(1) << 63)
        elif W == 2:
            rec[0] = UInt64(rank >> 4)
            rec[1] = UInt64(rank & 15)
        else:
            rec[0] = UInt64(rank >> 6)
            rec[1] = UInt64((rank >> 3) & 7)
            rec[2] = UInt64(rank & 7)
        rec[K.R - 1] = UInt64(i)
        keys.put(i + 1, rec)
    sort_keys(keys, 1, N + 1)
    for rank in range(N):
        assert_equal(keys.payload(rank + 1), (rank * 169) % N)
    for word in range(K.R):
        assert_equal(words[word], SENTINEL)
        assert_equal(words[(N + 1) * K.R + word], SENTINEL)
    # Empty and singleton subranges leave their records and guards intact.
    sort_keys(keys, 1, 1)
    sort_keys(keys, N, N + 1)
    assert_equal(keys.payload(N), ((N - 1) * 169) % N)
    _ = bytes^


def test_standard_sort_keeps_packed_record_strides_payloads_and_guards() raises:
    _packed_records[1, False]()
    _packed_records[2, False]()
    _packed_records[3, False]()
    _packed_records[0, True]()
