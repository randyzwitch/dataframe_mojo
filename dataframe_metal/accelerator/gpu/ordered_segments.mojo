"""Grouped ordered Int32 scans over independently mapped input windows."""
from max.gpu import global_idx
from std.atomic import Atomic
from std.memory import Pointer
from .filter_sum import _valid
from .group_hash import NO_GROUP
from .multi_reduce import Bytes
from .word_group import claim_word_group


def ordered_segment[
    phase: Int
](
    key_values: Bytes,
    key_bits: Bytes,
    values: Bytes,
    bits: Bytes,
    key_offset: UInt32,
    value_offset: UInt32,
    nullable: UInt32,
    base: UInt32,
    rows: UInt32,
    capacity: UInt32,
    fields: UInt32,
    policies: UInt32,
    table: Pointer[UInt32, MutUntrackedOrigin],
    first: Pointer[UInt32, MutUntrackedOrigin],
    states: Pointer[UInt32, MutUntrackedOrigin],
    control: Pointer[UInt32, MutUntrackedOrigin],
):
    var i = global_idx.x
    if i >= Int(rows):
        return
    if i == 0:
        _ = Atomic[UInt32].fetch_add(
            control.unsafe_offset(2 + phase), UInt32(1)
        )
    var valid_key = not (nullable & 1) or _valid(key_bits, key_offset, i)
    var raw_key = key_values.unsafe_bitcast[UInt32]()[
        unsafe_offset=i
    ] if valid_key else UInt32(0)
    var slot = claim_word_group(raw_key, valid_key, table, capacity, control)
    if slot == NO_GROUP:
        return
    var row = base + UInt32(i)
    comptime if phase == 0:
        _ = Atomic[UInt32].max(first.unsafe_offset(Int(slot)), ~row)
    var present = not (nullable & 2) or _valid(bits, value_offset, i)
    var ordered = UInt32(0)
    comptime if phase < 2:
        # At most two policies share the payload (ordinary and DROP_NULLS).
        if present and policies & 0x0C0C:
            ordered = values.unsafe_bitcast[UInt32]()[unsafe_offset=i] ^ UInt32(
                0x80000000
            )
    for field in range(Int(fields)):
        var ops = (policies >> UInt32(field * 8)) & 0xFF
        var included = present or not (ops & 16)
        var state = states.unsafe_offset((Int(slot) * Int(fields) + field) * 8)
        comptime if phase == 0:
            if included and ops & 1:
                _ = Atomic[UInt32].max(state, ~row)
            if included and ops & 2:
                _ = Atomic[UInt32].max(state.unsafe_offset(1), row + 1)
            if present and ops & 4:
                _ = Atomic[UInt32].max(state.unsafe_offset(6), ~ordered)
            if present and ops & 8:
                _ = Atomic[UInt32].max(state.unsafe_offset(7), ordered)
        elif phase == 1:
            # All segments have now contributed their extrema. An earlier
            # global row wins even if its value ties in a later allocation.
            if present and ops & 4 and ~ordered == state[unsafe_offset=6]:
                _ = Atomic[UInt32].max(state.unsafe_offset(2), ~row)
            if present and ops & 8 and ordered == state[unsafe_offset=7]:
                _ = Atomic[UInt32].max(state.unsafe_offset(3), ~row)
        else:
            # ARG_MIN/MAX index the group's input after optional DROP_NULLS.
            if (
                included
                and ops & 4
                and state[unsafe_offset=2]
                and row < ~state[unsafe_offset=2]
            ):
                _ = Atomic[UInt32].fetch_add(state.unsafe_offset(4), UInt32(1))
            if (
                included
                and ops & 8
                and state[unsafe_offset=3]
                and row < ~state[unsafe_offset=3]
            ):
                _ = Atomic[UInt32].fetch_add(state.unsafe_offset(5), UInt32(1))
