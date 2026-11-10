"""Repeated-key aggregation with exact thread-local contribution combining."""
from std.memory import Pointer
from std.atomic import Atomic
from max.gpu import global_idx, block_idx, thread_idx
from .multi_reduce import Bytes
from .compound_predicate import predicate_matches
from .group_tuple_hash import claim_tuple, tuple_equal
from .group_hash import NO_GROUP
from .group_state import GROUP_MIN, GROUP_MAX, atomic_or
from .group_local_state import GroupLocalState
from .group_reduce import group_value, update_int64_ticket
from ...group_layout import GroupLayout, GROUP_VALUE_WORDS


@always_inline
def flush_cache[
    cases: Bool, simple_cases: Bool, global_state: Bool
](
    mut cache: GroupLocalState,
    layout: GroupLayout,
    flags: UInt32,
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    counts: Pointer[UInt32, MutUntrackedOrigin],
    first: Pointer[UInt32, MutUntrackedOrigin],
    states: Pointer[UInt32, MutUntrackedOrigin],
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
    if cache.slot == NO_GROUP:
        return
    var index = Int(cache.slot)
    comptime if not global_state:
        index = index * layout.blocks + Int(block_idx.x)
    _ = Atomic[UInt32].fetch_add(counts.unsafe_offset(index), cache.rows)
    _ = Atomic[UInt32].min(first.unsafe_offset(Int(cache.slot)), cache.first)
    var integer64_begin = Int(
        metadata[unsafe_offset=7] + metadata[unsafe_offset=8]
    )
    for field in range(layout.fields):
        var kind = 0 if field < Int(metadata[unsafe_offset=7]) else (
            1 if field < integer64_begin else 2
        )
        var want_sum = Bool(flags & (UInt32(1) << UInt32(field)))
        var want_extrema = Bool(
            (flags >> 4 | flags >> 8) & (UInt32(1) << UInt32(field))
        )
        var state = states.unsafe_offset(
            (index * layout.fields + field) * GROUP_VALUE_WORDS
        )
        cache.flush_field(field, kind, want_sum, want_extrema, state)
        if kind == 2 and want_extrema and cache.counts[field]:
            update_int64_ticket[True, cases, simple_cases](
                state.unsafe_offset(GROUP_MIN),
                cache.minimum[field],
                metadata,
                field,
                Int(cache.min_row[field]),
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
            update_int64_ticket[False, cases, simple_cases](
                state.unsafe_offset(GROUP_MAX),
                cache.maximum[field],
                metadata,
                field,
                Int(cache.max_row[field]),
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


def cached_grouped_partials[
    cases: Bool = False,
    simple_cases: Bool = False,
    global_state: Bool = True,
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
        GROUP_VALUE_WORDS,
    )
    var table = scratch.unsafe_offset(layout.table_offset())
    var first = scratch.unsafe_offset(layout.first_offset())
    var counts = scratch.unsafe_offset(layout.rows_offset())
    var states = scratch.unsafe_offset(layout.states_offset())
    var flags = UInt32(metadata[unsafe_offset=9])
    var cache = GroupLocalState()
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
            var same = cache.slot != NO_GROUP and tuple_equal(
                metadata,
                Int(cache.first),
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
            var slot = cache.slot
            if not same:
                flush_cache[cases, simple_cases, global_state](
                    cache,
                    layout,
                    flags,
                    metadata,
                    counts,
                    first,
                    states,
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
                cache = GroupLocalState()
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
                cache.slot = slot
                cache.first = UInt32(row)
            cache.rows += 1
            var state_index = Int(slot)
            comptime if not global_state:
                state_index = state_index * layout.blocks + Int(block_idx.x)
            for field in range(layout.fields):
                var kind = 0 if field < Int(metadata[unsafe_offset=7]) else (
                    1 if field
                    < Int(
                        metadata[unsafe_offset=7] + metadata[unsafe_offset=8]
                    ) else 2
                )
                var want_sum = Bool(flags & (UInt32(1) << UInt32(field)))
                var want_extrema = Bool(
                    (flags >> 4 | flags >> 8) & (UInt32(1) << UInt32(field))
                )
                var value = (UInt64(0), False, UInt32(0))
                if kind == 0:
                    value = group_value[True, cases, simple_cases](
                        metadata,
                        field,
                        row,
                        2,
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
                else:
                    value = group_value[False, cases, simple_cases](
                        metadata,
                        field,
                        row,
                        UInt32(0 if kind == 1 else 1),
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
                    cache.add(
                        field,
                        kind,
                        value[0],
                        UInt32(row),
                        want_sum,
                        want_extrema,
                        states.unsafe_offset(
                            (state_index * layout.fields + field)
                            * GROUP_VALUE_WORDS
                        ),
                    )
        row += Int(metadata[unsafe_offset=1])
    flush_cache[cases, simple_cases, global_state](
        cache,
        layout,
        flags,
        metadata,
        counts,
        first,
        states,
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
    if thread_idx.x == 0:
        _ = Atomic[UInt32].fetch_add(scratch.unsafe_offset(2), UInt32(1))
