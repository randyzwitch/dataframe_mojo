"""Standard heap adapters preserve packed records and inline row state."""
# contract-group: kernel.order_keys
from std.testing import assert_equal
from dataframe.sorting.algorithms import Rec, SortKeys
from dataframe.sorting.top_n import TopNHeap
from dataframe.aggregation.function import _retain_top_rows
from std.memory import Pointer


def _heap_records[W: Int, VAR: Bool]() raises:
    comptime K = SortKeys[W, VAR]
    var left = TopNHeap(17, K.R, variable=VAR)
    var right = TopNHeap(17, K.R, variable=VAR)
    var bytes = List[UInt8](length=10, fill=97)
    for i in range(257):
        var rank = (i * 73) % 257
        var rec = Rec(0)
        comptime if VAR:
            bytes[8] = UInt8(rank >> 8)
            bytes[9] = UInt8(rank & 255)
            rec[0] = UInt64(0x6161616161616161)
            rec[1] = 10
            rec[2] = UInt64(Int(bytes.unsafe_ptr()))
        elif W == 1:
            rec[0] = UInt64(rank)
        else:
            comptime for w in range(W):
                rec[w] = UInt64(rank >> (8 * (W - 1 - w))) & 255
        rec[K.R - 1] = UInt64(i)
        if i % 2 == 0:
            if left.should_add[W, VAR](rec):
                left.add[W, VAR](rec)
        elif right.should_add[W, VAR](rec):
            right.add[W, VAR](rec)
    # The source bytes have been reused for every input; retained variable
    # keys must own copies, including through arena compaction and combine.
    left.compact_keys[W, VAR]()
    right.compact_keys[W, VAR]()
    right.finalize[W, VAR]()
    left.combine[W, VAR](right)
    left.finalize[W, VAR]()
    var rows = left.payloads[W, VAR]()
    assert_equal(len(rows), 17)
    for rank in range(17):
        assert_equal(rows[rank], (rank * 169) % 257)
    var empty = TopNHeap(0, K.R, variable=VAR)
    empty.add[W, VAR](Rec(0))
    assert_equal(empty.size, 0)
    _ = bytes^


def test_standard_heaps_keep_all_record_widths_and_owned_variable_keys() raises:
    _heap_records[1, False]()
    _heap_records[2, False]()
    _heap_records[3, False]()
    _heap_records[1, True]()


def test_inline_top_rows_keep_lexicographic_pairs_and_adjacent_guards() raises:
    comptime GUARD = UInt64(0xA55AA55AA55AA55A)
    for k in [1, 2, 16]:
        var words = List[UInt64](length=2 * k + 3, fill=GUARD)
        words[1] = 0
        var state = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=Int(words.unsafe_ptr().unsafe_offset(1))
        )
        for row in range(127, -1, -1):
            _retain_top_rows(state, k, UInt64(row % 4), UInt64(row))
        assert_equal(Int(state[]), k)
        for i in range(k):
            assert_equal(state[unsafe_offset=1 + 2 * i], UInt64(0))
            assert_equal(state[unsafe_offset=2 + 2 * i], UInt64(4 * i))
        assert_equal(words[0], GUARD)
        assert_equal(words[len(words) - 1], GUARD)
