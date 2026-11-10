"""Winning Int64 rows are immutable across repeated comparisons and groups."""
# contract-group: execution.backends
from std.memory import Pointer
from std.testing import assert_equal
from dataframe.expr import COL
from dataframe_metal.accelerator.gpu.multi_reduce import Bytes
from dataframe_metal.accelerator.gpu.group_cached import (
    TicketCache,
    update_ticket,
)
from dataframe_metal.accelerator.gpu.group_reduce import update_int64_ticket


def test_metal_group_ticket_cache_keeps_fields_and_external_updates_separate() raises:
    var left = List[Int64](
        [Int64.MIN, Int64.MAX, 0, -1, 1, Int64(1) << 60, -(Int64(1) << 60), 1]
    )
    var right = List[Int64](
        [Int64.MAX, Int64.MIN, 1, 0, -1, -(Int64(1) << 60), Int64(1) << 60, 1]
    )
    var words = List[UInt64](length=32, fill=0)
    for field in range(2):
        words[16 + field * 8] = UInt64(COL)
        words[17 + field * 8] = UInt64(field + 1)
    var metadata = Pointer[UInt64, UntrackedOrigin[mut=False]](
        unsafe_from_address=Int(words.unsafe_ptr())
    )
    var a = Bytes(unsafe_from_address=Int(left.unsafe_ptr()))
    var b = Bytes(unsafe_from_address=Int(right.unsafe_ptr()))
    var empty = a
    var actual = List[UInt32](length=4, fill=UInt32.MAX)
    var expected = List[UInt32](length=4, fill=UInt32.MAX)
    var ap = Pointer[UInt32, MutUntrackedOrigin](
        unsafe_from_address=Int(actual.unsafe_ptr())
    )
    var ep = Pointer[UInt32, MutUntrackedOrigin](
        unsafe_from_address=Int(expected.unsafe_ptr())
    )
    var min_cache = TicketCache()
    var max_cache = TicketCache()
    for phase in range(4):
        for i in range(4):
            actual[i] = UInt32.MAX
            expected[i] = UInt32.MAX
        for step in range(64):
            var row = (step * 5 + phase) % len(left)
            for field in range(2):
                var value = left[row] if field == 0 else right[row]
                if step % 11 == 0:
                    var other = (row + 3) % len(left)
                    var external = left[other] if field == 0 else right[other]
                    update_int64_ticket[True](
                        ap.unsafe_offset(field * 2),
                        external,
                        metadata,
                        field,
                        other,
                        a,
                        b,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                    )
                    update_int64_ticket[True](
                        ep.unsafe_offset(field * 2),
                        external,
                        metadata,
                        field,
                        other,
                        a,
                        b,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                    )
                    update_int64_ticket[False](
                        ap.unsafe_offset(field * 2 + 1),
                        external,
                        metadata,
                        field,
                        other,
                        a,
                        b,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                    )
                    update_int64_ticket[False](
                        ep.unsafe_offset(field * 2 + 1),
                        external,
                        metadata,
                        field,
                        other,
                        a,
                        b,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,
                    )
                update_ticket[True, False, False](
                    min_cache,
                    ap.unsafe_offset(field * 2),
                    value,
                    metadata,
                    field,
                    row,
                    a,
                    b,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                )
                update_ticket[False, False, False](
                    max_cache,
                    ap.unsafe_offset(field * 2 + 1),
                    value,
                    metadata,
                    field,
                    row,
                    a,
                    b,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                )
                update_int64_ticket[True](
                    ep.unsafe_offset(field * 2),
                    value,
                    metadata,
                    field,
                    row,
                    a,
                    b,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                )
                update_int64_ticket[False](
                    ep.unsafe_offset(field * 2 + 1),
                    value,
                    metadata,
                    field,
                    row,
                    a,
                    b,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                )
                assert_equal(actual[field * 2], expected[field * 2])
                assert_equal(actual[field * 2 + 1], expected[field * 2 + 1])
