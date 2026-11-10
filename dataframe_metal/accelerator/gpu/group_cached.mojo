"""Batch repeated group rows and reuse immutable Int64 winning values."""
from std.memory import Pointer, bitcast
from std.atomic import Atomic
from max.gpu import global_idx, block_idx, thread_idx
from .multi_reduce import Bytes
from .compound_predicate import predicate_matches
from .group_tuple_hash import tuple_equal, claim_tuple
from .group_hash import NO_GROUP
from .group_state import (
    GROUP_COUNT,
    GROUP_MIN,
    GROUP_MAX,
    atomic_or,
    add_integer_sum,
)
from .group_reduce import _update_group_field, group_value
from ...group_layout import GroupLayout, GROUP_VALUE_WORDS


struct TicketCache(Movable):
    var rows: SIMD[DType.uint32, 4]
    var values: SIMD[DType.int64, 4]

    @always_inline
    def __init__(out self):
        self.rows = SIMD[DType.uint32, 4](UInt32.MAX)
        self.values = SIMD[DType.int64, 4](0)


@always_inline
def update_ticket[
    minimum: Bool, cases: Bool, simple_cases: Bool
](
    mut cache: TicketCache,
    ticket: Pointer[UInt32, MutUntrackedOrigin],
    candidate: Int64,
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    field: Int,
    row: Int,
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    v4: Bytes,
    v5: Bytes,
    v6: Bytes,
    v7: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    b4: Bytes,
    b5: Bytes,
    b6: Bytes,
    b7: Bytes,
    predicate: Bytes,
    predicate_bits: Bytes,
):
    var expected = Atomic[UInt32].load(ticket)
    while True:
        if expected != UInt32.MAX:
            var previous = Int64(0)
            if cache.rows[field] == expected:
                previous = cache.values[field]
            else:
                var value = group_value[False, cases, simple_cases](
                    metadata,
                    field,
                    Int(expected),
                    1,
                    True,
                    v0,
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7,
                    b0,
                    b1,
                    b2,
                    b3,
                    b4,
                    b5,
                    b6,
                    b7,
                    predicate,
                    predicate_bits,
                )
                previous = bitcast[DType.int64](value[0])
                cache.rows[field] = expected
                cache.values[field] = previous
            if candidate >= previous if minimum else candidate <= previous:
                return
        if Atomic[UInt32].compare_exchange[weak=True](
            ticket, expected, UInt32(row)
        ):
            cache.rows[field] = UInt32(row)
            cache.values[field] = candidate
            return


@always_inline
def flush_rows[
    global_state: Bool
](
    slot: UInt32,
    rows: UInt32,
    first_row: UInt32,
    layout: GroupLayout,
    scratch: Pointer[UInt32, MutUntrackedOrigin],
):
    if slot == NO_GROUP:
        return
    var index = Int(slot)
    comptime if not global_state:
        index = index * layout.blocks + Int(block_idx.x)
    _ = Atomic[UInt32].fetch_add(
        scratch.unsafe_offset(layout.rows_offset() + index), rows
    )
    _ = Atomic[UInt32].min(
        scratch.unsafe_offset(layout.first_offset() + Int(slot)), first_row
    )


def cached_grouped_partials[
    cases: Bool = False,
    simple_cases: Bool = False,
    global_state: Bool = True,
    value_types: Int = 7,
](
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    v4: Bytes,
    v5: Bytes,
    v6: Bytes,
    v7: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    b4: Bytes,
    b5: Bytes,
    b6: Bytes,
    b7: Bytes,
    predicate: Bytes,
    predicate_bits: Bytes,
    k0: Bytes,
    k1: Bytes,
    k2: Bytes,
    k3: Bytes,
    kb0: Bytes,
    kb1: Bytes,
    kb2: Bytes,
    kb3: Bytes,
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    scratch: Pointer[UInt32, MutUntrackedOrigin],
):
    var layout = GroupLayout(
        Int(metadata[unsafe_offset=3]),
        Int(metadata[unsafe_offset=6]),
        Int(metadata[unsafe_offset=2]),
        global_state,
    )
    var table = scratch.unsafe_offset(layout.table_offset())
    var states = scratch.unsafe_offset(layout.states_offset())
    var flags = UInt32(metadata[unsafe_offset=9])
    var cached_slot = NO_GROUP
    var cached_first = UInt32.MAX
    var cached_rows = UInt32(0)
    var min_cache = TicketCache()
    var max_cache = TicketCache()
    var row = global_idx.x
    while row < Int(metadata[unsafe_offset=0]):
        if Atomic[UInt32].load(scratch.unsafe_offset(1)):
            return
        if predicate_matches(
            metadata.unsafe_offset(Int(metadata[unsafe_offset=10])),
            0,
            row,
            v0,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            predicate,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            predicate_bits,
        ):
            var same = cached_slot != NO_GROUP and tuple_equal(
                metadata,
                Int(cached_first),
                row,
                k0,
                k1,
                k2,
                k3,
                kb0,
                kb1,
                kb2,
                kb3,
            )
            var slot = cached_slot
            if not same:
                flush_rows[global_state](
                    cached_slot, cached_rows, cached_first, layout, scratch
                )
                slot = claim_tuple(
                    metadata,
                    row,
                    k0,
                    k1,
                    k2,
                    k3,
                    kb0,
                    kb1,
                    kb2,
                    kb3,
                    table,
                    scratch,
                )
                if slot == NO_GROUP:
                    return
                cached_slot = slot
                cached_first = UInt32(row)
                cached_rows = 0
            cached_rows += 1
            comptime for kind in range(3):
                comptime if value_types & (1 << kind):
                    var begin = 0
                    var end = Int(metadata[unsafe_offset=7])
                    comptime if kind == 1:
                        begin = end
                        end += Int(metadata[unsafe_offset=8])
                    elif kind == 2:
                        begin = end + Int(metadata[unsafe_offset=8])
                        end = layout.fields
                    for field in range(begin, end):
                        comptime if kind < 2:
                            if _update_group_field[
                                kind == 0,
                                True,
                                cases,
                                simple_cases,
                                global_state,
                            ](
                                metadata,
                                slot,
                                UInt32(layout.blocks),
                                flags,
                                field,
                                row,
                                v0,
                                v1,
                                v2,
                                v3,
                                v4,
                                v5,
                                v6,
                                v7,
                                b0,
                                b1,
                                b2,
                                b3,
                                b4,
                                b5,
                                b6,
                                b7,
                                predicate,
                                predicate_bits,
                                states,
                                scratch,
                                layout.fields,
                            ):
                                return
                        else:
                            var want_sum = Bool(
                                flags & (UInt32(1) << UInt32(field))
                            )
                            var want_extrema = Bool(
                                (flags >> 4 | flags >> 8)
                                & (UInt32(1) << UInt32(field))
                            )
                            var value = group_value[False, cases, simple_cases](
                                metadata,
                                field,
                                row,
                                1,
                                want_sum or want_extrema,
                                v0,
                                v1,
                                v2,
                                v3,
                                v4,
                                v5,
                                v6,
                                v7,
                                b0,
                                b1,
                                b2,
                                b3,
                                b4,
                                b5,
                                b6,
                                b7,
                                predicate,
                                predicate_bits,
                            )
                            if value[2]:
                                atomic_or(scratch.unsafe_offset(1), value[2])
                                return
                            if value[1]:
                                var state_index = Int(slot)
                                comptime if not global_state:
                                    state_index = (
                                        state_index * layout.blocks
                                        + Int(block_idx.x)
                                    )
                                var state = states.unsafe_offset(
                                    (state_index * layout.fields + field)
                                    * GROUP_VALUE_WORDS
                                )
                                _ = Atomic[UInt32].fetch_add(
                                    state.unsafe_offset(GROUP_COUNT), UInt32(1)
                                )
                                var number = bitcast[DType.int64](value[0])
                                if want_sum:
                                    add_integer_sum(state, number)
                                if want_extrema:
                                    update_ticket[True, cases, simple_cases](
                                        min_cache,
                                        state.unsafe_offset(GROUP_MIN),
                                        number,
                                        metadata,
                                        field,
                                        row,
                                        v0,
                                        v1,
                                        v2,
                                        v3,
                                        v4,
                                        v5,
                                        v6,
                                        v7,
                                        b0,
                                        b1,
                                        b2,
                                        b3,
                                        b4,
                                        b5,
                                        b6,
                                        b7,
                                        predicate,
                                        predicate_bits,
                                    )
                                    update_ticket[False, cases, simple_cases](
                                        max_cache,
                                        state.unsafe_offset(GROUP_MAX),
                                        number,
                                        metadata,
                                        field,
                                        row,
                                        v0,
                                        v1,
                                        v2,
                                        v3,
                                        v4,
                                        v5,
                                        v6,
                                        v7,
                                        b0,
                                        b1,
                                        b2,
                                        b3,
                                        b4,
                                        b5,
                                        b6,
                                        b7,
                                        predicate,
                                        predicate_bits,
                                    )
        row += Int(metadata[unsafe_offset=1])
    flush_rows[global_state](
        cached_slot, cached_rows, cached_first, layout, scratch
    )
    if thread_idx.x == 0:
        _ = Atomic[UInt32].fetch_add(scratch.unsafe_offset(2), UInt32(1))
