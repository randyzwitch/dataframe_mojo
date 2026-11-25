"""Group direct numeric column reductions into state shared by native spans."""
from max.gpu import global_idx
from std.atomic import Atomic
from std.memory import Pointer
from ...group_layout import GroupLayout
from .filter_sum import _valid
from .group_hash import NO_GROUP
from .group_state import (
    GROUP_STATE_WORDS,
    add_float_sum,
    add_integer_sum,
    add_ordered_extrema,
)
from .multi_reduce import Bytes, read_integer
from .word_group import claim_word_group

# Low descriptor bits select the native element type and requested updates.
# The high 32 bits hold this individual view's validity offset.
comptime FIELD_NULLABLE = UInt64(4)
comptime FIELD_SUM = UInt64(8)
comptime FIELD_EXTREMA = UInt64(16)


@always_inline
def _update(
    values: Bytes,
    bits: Bytes,
    descriptor: UInt64,
    row: Int,
    state: Pointer[UInt32, MutUntrackedOrigin],
):
    if descriptor & FIELD_NULLABLE and not _valid(
        bits, UInt32(descriptor >> 32), row
    ):
        return
    _ = Atomic[UInt32].fetch_add(state, UInt32(1))
    var tag = UInt32(descriptor & 3)
    if descriptor & FIELD_SUM:
        if tag == 2:
            add_float_sum(
                state, values.unsafe_bitcast[UInt32]()[unsafe_offset=row]
            )
        else:
            add_integer_sum(state, read_integer(values, tag, row))
    if descriptor & FIELD_EXTREMA:
        add_ordered_extrema(
            state, values.unsafe_bitcast[UInt32]()[unsafe_offset=row], tag == 2
        )


def chunked_numeric_partials(
    keys: Bytes,
    key_bits: Bytes,
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    key_descriptor: UInt64,
    d0: UInt64,
    d1: UInt64,
    d2: UInt64,
    d3: UInt64,
    base: UInt32,
    length: UInt32,
    capacity: UInt32,
    fields: UInt32,
    scratch: Pointer[UInt32, MutUntrackedOrigin],
):
    var row = global_idx.x
    if row >= Int(length):
        return
    if row == 0:
        _ = Atomic[UInt32].fetch_add(scratch.unsafe_offset(2), UInt32(1))
    var layout = GroupLayout(Int(capacity), Int(fields), 1, True)
    var present = not (key_descriptor & FIELD_NULLABLE) or _valid(
        key_bits, UInt32(key_descriptor >> 32), row
    )
    var key = keys.unsafe_bitcast[UInt32]()[
        unsafe_offset=row
    ] if present else UInt32(0)
    var slot = claim_word_group(
        key,
        present,
        scratch.unsafe_offset(layout.table_offset()),
        capacity,
        scratch,
    )
    if slot == NO_GROUP:
        return
    var first = scratch.unsafe_offset(layout.first_offset() + Int(slot))
    var ticket = base + UInt32(row)
    if ticket < Atomic[UInt32].load(first):
        _ = Atomic[UInt32].min(first, ticket)
    _ = Atomic[UInt32].fetch_add(
        scratch.unsafe_offset(layout.rows_offset() + Int(slot)), UInt32(1)
    )
    var state = scratch.unsafe_offset(
        layout.states_offset() + Int(slot) * Int(fields) * GROUP_STATE_WORDS
    )
    _update(v0, b0, d0, row, state)
    if fields > 1:
        _update(v1, b1, d1, row, state.unsafe_offset(GROUP_STATE_WORDS))
    if fields > 2:
        _update(v2, b2, d2, row, state.unsafe_offset(2 * GROUP_STATE_WORDS))
    if fields > 3:
        _update(v3, b3, d3, row, state.unsafe_offset(3 * GROUP_STATE_WORDS))
