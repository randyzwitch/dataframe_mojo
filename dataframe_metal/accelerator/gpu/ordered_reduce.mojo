"""GPU ordered tickets, block-local merging and filtered group-local ranks.

UInt32 atomics publish immutable original-row tickets. No per-row group ID or
value buffer is materialized. A second input scan counts positions before the
winning ticket, so ARG_MIN/MAX return filtered/group-local rather than original
input offsets.
"""
from std.atomic import Atomic
from std.memory import Pointer
from max.gpu import global_idx, block_idx
from ...group_layout import GroupLayout
from .multi_reduce import Bytes
from .compound_predicate import _pointer, predicate_matches
from .filter_sum import _valid
from .group_tuple_hash import claim_tuple
from .group_hash import NO_GROUP
from .ordered_value import ordered_bits, ordered_better, update_ordered_arg


@always_inline
def ordered_layout(
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]]
) -> GroupLayout:
    return GroupLayout(
        Int(metadata[unsafe_offset=3]),
        Int(metadata[unsafe_offset=2]),
        Int(metadata[unsafe_offset=6]),
        Bool(metadata[unsafe_offset=7]),
    )


@always_inline
def ordered_state(
    layout: GroupLayout, slot: UInt32, field: Int, block: Int
) -> Int:
    var partition = 0 if layout.global_state else block
    return (
        layout.states_offset()
        + (
            (Int(slot) * layout.state_blocks() + partition) * layout.fields
            + field
        )
        * 32
    )


def initialize_ordered(
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    scratch: Pointer[UInt32, MutUntrackedOrigin],
):
    var layout = ordered_layout(metadata)
    var row = global_idx.x
    var total = layout.words() + layout.capacity * layout.fields * 8
    if row >= total:
        return
    var value = UInt32(0)
    if row >= layout.first_offset() and row < layout.rows_offset():
        value = UInt32.MAX
    elif row >= layout.states_offset() and row < layout.words():
        if (row - layout.states_offset()) % 32 == 0:
            value = UInt32.MAX
    elif row >= layout.words() and (row - layout.words()) % 8 == 0:
        value = UInt32.MAX
    scratch[unsafe_offset=row] = value


def ordered_partials(
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
    var layout = ordered_layout(metadata)
    var row = global_idx.x
    while row < Int(metadata[unsafe_offset=0]):
        if predicate_matches(
            metadata.unsafe_offset(Int(metadata[unsafe_offset=5])),
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
            var slot = UInt32(0)
            if metadata[unsafe_offset=11]:
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
                    scratch.unsafe_offset(layout.table_offset()),
                    scratch,
                )
            if slot != NO_GROUP:
                _ = Atomic[UInt32].min(
                    scratch.unsafe_offset(layout.first_offset() + Int(slot)),
                    UInt32(row),
                )
                for field in range(layout.fields):
                    var at = 16 + field * 4
                    var input = Int(metadata[unsafe_offset=at])
                    var descriptor = metadata[unsafe_offset=at + 1]
                    var tag = UInt32(descriptor >> 33)
                    var flags = metadata[unsafe_offset=at + 2]
                    var operations = UInt32(flags >> 8)
                    var data = _pointer(
                        input, v0, v1, v2, v3, v4, v5, v6, v7, predicate
                    )
                    var present = True
                    if (descriptor >> 32) & 1:
                        present = _valid(
                            _pointer(
                                input,
                                b0,
                                b1,
                                b2,
                                b3,
                                b4,
                                b5,
                                b6,
                                b7,
                                predicate_bits,
                            ),
                            UInt32(descriptor),
                            row,
                        )
                    var state = scratch.unsafe_offset(
                        ordered_state(layout, slot, field, Int(block_idx.x))
                    )
                    if present or not Bool(flags & 1):
                        if operations & 1:
                            _ = Atomic[UInt32].min(state, UInt32(row) + 1)
                        if operations & 2:
                            _ = Atomic[UInt32].max(
                                state.unsafe_offset(1), UInt32(row) + 1
                            )
                    if present and operations & 12:
                        var raw = ordered_bits(data, tag, row)
                        if operations & 4:
                            update_ordered_arg(
                                state.unsafe_offset(2),
                                raw,
                                data,
                                tag,
                                row,
                                True,
                            )
                        if operations & 8:
                            update_ordered_arg(
                                state.unsafe_offset(3),
                                raw,
                                data,
                                tag,
                                row,
                                False,
                            )
        row += Int(metadata[unsafe_offset=1])


def merge_ordered(
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
    var layout = ordered_layout(metadata)
    var index = global_idx.x
    if index >= layout.capacity * layout.fields:
        return
    var slot = index // layout.fields
    var field = index % layout.fields
    if metadata[unsafe_offset=11]:
        if scratch[unsafe_offset=layout.table_offset() + slot] == 0:
            return
    elif slot != 0:
        return
    var at = 16 + field * 4
    var data = _pointer(
        Int(metadata[unsafe_offset=at]),
        v0,
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        predicate,
    )
    var tag = UInt32(metadata[unsafe_offset=at + 1] >> 33)
    var first = UInt32.MAX
    var last = UInt32(0)
    var smallest = UInt32(0)
    var largest = UInt32(0)
    var minimum_rank = UInt32(0)
    var maximum_rank = UInt32(0)
    for partition in range(layout.state_blocks()):
        var offset = ordered_state(layout, UInt32(slot), field, partition)
        minimum_rank += scratch[unsafe_offset=offset + 4]
        maximum_rank += scratch[unsafe_offset=offset + 5]
        first = min(first, scratch[unsafe_offset=offset])
        last = max(last, scratch[unsafe_offset=offset + 1])
        var candidate = scratch[unsafe_offset=offset + 2]
        if candidate:
            if smallest == 0 or ordered_better(
                ordered_bits(data, tag, Int(candidate - 1)),
                candidate - 1,
                ordered_bits(data, tag, Int(smallest - 1)),
                smallest - 1,
                tag,
                True,
            ):
                smallest = candidate
        candidate = scratch[unsafe_offset=offset + 3]
        if candidate:
            if largest == 0 or ordered_better(
                ordered_bits(data, tag, Int(candidate - 1)),
                candidate - 1,
                ordered_bits(data, tag, Int(largest - 1)),
                largest - 1,
                tag,
                False,
            ):
                largest = candidate
    var target = layout.words() + index * 8
    scratch[unsafe_offset=target] = first
    scratch[unsafe_offset=target + 1] = last
    scratch[unsafe_offset=target + 2] = smallest
    scratch[unsafe_offset=target + 3] = largest
    scratch[unsafe_offset=target + 4] = minimum_rank
    scratch[unsafe_offset=target + 5] = maximum_rank


def ordered_ranks(
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
    var layout = ordered_layout(metadata)
    var row = global_idx.x
    while row < Int(metadata[unsafe_offset=0]):
        if predicate_matches(
            metadata.unsafe_offset(Int(metadata[unsafe_offset=5])),
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
            var slot = UInt32(0)
            if metadata[unsafe_offset=11]:
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
                    scratch.unsafe_offset(layout.table_offset()),
                    scratch,
                )
            if slot != NO_GROUP:
                for field in range(layout.fields):
                    var at = 16 + field * 4
                    var flags = metadata[unsafe_offset=at + 2]
                    var operations = UInt32(flags >> 8)
                    if not (operations & 12):
                        continue
                    var present = True
                    var descriptor = metadata[unsafe_offset=at + 1]
                    if flags & 1 and (descriptor >> 32) & 1:
                        var input = Int(metadata[unsafe_offset=at])
                        present = _valid(
                            _pointer(
                                input,
                                b0,
                                b1,
                                b2,
                                b3,
                                b4,
                                b5,
                                b6,
                                b7,
                                predicate_bits,
                            ),
                            UInt32(descriptor),
                            row,
                        )
                    if not present:
                        continue
                    var target = (
                        layout.words() + (Int(slot) * layout.fields + field) * 8
                    )
                    var state = scratch.unsafe_offset(
                        ordered_state(layout, slot, field, Int(block_idx.x))
                    )
                    if (
                        operations & 4
                        and UInt32(row) + 1 < scratch[unsafe_offset=target + 2]
                    ):
                        _ = Atomic[UInt32].fetch_add(
                            state.unsafe_offset(4), UInt32(1)
                        )
                    if (
                        operations & 8
                        and UInt32(row) + 1 < scratch[unsafe_offset=target + 3]
                    ):
                        _ = Atomic[UInt32].fetch_add(
                            state.unsafe_offset(5), UInt32(1)
                        )
        row += Int(metadata[unsafe_offset=1])
