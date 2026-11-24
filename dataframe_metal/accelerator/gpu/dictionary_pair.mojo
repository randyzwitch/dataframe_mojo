"""Direct-address SUM for two bounded dictionaries, including null keys."""
from max.gpu import global_idx
from std.atomic import Atomic
from std.memory import Pointer
from .filter_sum import _valid
from .group_hash import atomic_add_signed96
from .multi_reduce import Bytes, read_integer

# Complement of the first row, non-null count, and an exact signed 96-bit sum.
comptime PAIR_WORDS = 5


def dictionary_pair_sum(
    left: Bytes,
    right: Bytes,
    values: Bytes,
    left_bits: Bytes,
    right_bits: Bytes,
    value_bits: Bytes,
    left_offset: UInt32,
    right_offset: UInt32,
    value_offset: UInt32,
    flags: UInt32,
    left_labels: UInt32,
    right_labels: UInt32,
    value_tag: UInt32,
    base: UInt32,
    length: UInt32,
    state: Pointer[UInt32, MutUntrackedOrigin],
):
    var i = global_idx.x
    if i >= Int(length):
        return
    var slots = (left_labels + 1) * (right_labels + 1)
    var control = state.unsafe_offset(Int(slots) * PAIR_WORDS)
    if i == 0:
        _ = Atomic[UInt32].fetch_add(control, UInt32(1))
    var a = left_labels
    if not (flags & 1) or _valid(left_bits, left_offset, i):
        a = left.unsafe_bitcast[UInt32]()[unsafe_offset=i]
        if a >= left_labels:
            _ = Atomic[UInt32].max(control.unsafe_offset(1), UInt32(1))
            return
    var b = right_labels
    if not (flags & 2) or _valid(right_bits, right_offset, i):
        b = right.unsafe_bitcast[UInt32]()[unsafe_offset=i]
        if b >= right_labels:
            _ = Atomic[UInt32].max(control.unsafe_offset(1), UInt32(1))
            return
    var slot = state.unsafe_offset(Int(a * (right_labels + 1) + b) * PAIR_WORDS)
    var ticket = ~(base + UInt32(i))
    if ticket > Atomic[UInt32].load(slot):
        _ = Atomic[UInt32].max(slot, ticket)
    if not (flags & 4) or _valid(value_bits, value_offset, i):
        _ = Atomic[UInt32].fetch_add(slot.unsafe_offset(1), UInt32(1))
        atomic_add_signed96(
            slot.unsafe_offset(2), read_integer(values, value_tag, i)
        )
