"""Collision-checked native tuple hashing with immutable original-row tickets."""
from std.atomic import Atomic
from std.memory import Pointer
from .multi_reduce import Bytes, read_integer
from .filter_sum import _valid
from .group_hash import _key_hash, NO_GROUP
from .group_state import atomic_or
from ...group_layout import (
    GENERAL_GROUP_LIMIT,
    GENERAL_TABLE_LIMIT,
    GROUP_CAPACITY_FAULT,
)

comptime MAX_GENERAL_GROUPS = GENERAL_GROUP_LIMIT
comptime MAX_GENERAL_CAPACITY = GENERAL_TABLE_LIMIT


@always_inline
def _key_buffer(
    index: Int, k0: Bytes, k1: Bytes, k2: Bytes, k3: Bytes
) -> Bytes:
    if index == 0:
        return k0
    if index == 1:
        return k1
    if index == 2:
        return k2
    return k3


@always_inline
def _key_valid(bits: Bytes, descriptor: UInt64, row: Int) -> Bool:
    return (
        _valid(bits, UInt32(descriptor >> 2), row) if descriptor & 2 else True
    )


@always_inline
def tuple_hash(
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    row: Int,
    k0: Bytes,
    k1: Bytes,
    k2: Bytes,
    k3: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
) -> UInt32:
    var hashed = UInt32(0x9E37_79B9)
    comptime for key in range(4):
        if key < Int(metadata[unsafe_offset=11]):
            var descriptor = metadata[unsafe_offset=12 + key]
            var bits = _key_buffer(key, b0, b1, b2, b3)
            var valid = _key_valid(bits, descriptor, row)
            var part = UInt32(0xD1B5_4A35)
            if valid:
                var values = _key_buffer(key, k0, k1, k2, k3)
                part = _key_hash(
                    read_integer(values, UInt32(descriptor & 1), row)
                ) ^ UInt32(0xA24B_AED5)
            hashed = (hashed ^ part) * UInt32(0x9E37_79B9)
            hashed ^= hashed >> 16
    return hashed


@always_inline
def tuple_equal(
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    left: Int,
    right: Int,
    k0: Bytes,
    k1: Bytes,
    k2: Bytes,
    k3: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
) -> Bool:
    comptime for key in range(4):
        if key < Int(metadata[unsafe_offset=11]):
            var descriptor = metadata[unsafe_offset=12 + key]
            var bits = _key_buffer(key, b0, b1, b2, b3)
            var lv = _key_valid(bits, descriptor, left)
            var rv = _key_valid(bits, descriptor, right)
            if lv != rv:
                return False
            if lv:
                var values = _key_buffer(key, k0, k1, k2, k3)
                var tag = UInt32(descriptor & 1)
                if read_integer(values, tag, left) != read_integer(
                    values, tag, right
                ):
                    return False
    return True


@always_inline
def claim_tuple(
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    row: Int,
    k0: Bytes,
    k1: Bytes,
    k2: Bytes,
    k3: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    table: Pointer[UInt32, MutUntrackedOrigin],
    control: Pointer[UInt32, MutUntrackedOrigin],
) -> UInt32:
    var capacity = UInt32(metadata[unsafe_offset=3])
    var limit = UInt32(metadata[unsafe_offset=4])
    var slot = tuple_hash(metadata, row, k0, k1, k2, k3, b0, b1, b2, b3) & (
        capacity - 1
    )
    var probes = UInt32(0)
    while probes < capacity:
        if Atomic[UInt32].load(control.unsafe_offset(1)):
            return NO_GROUP
        var address = table.unsafe_offset(Int(slot))
        var owner = Atomic[UInt32].load(address)
        if owner == 0:
            var expected = UInt32(0)
            if Atomic[UInt32].compare_exchange[weak=True](
                address, expected, UInt32(row) + 1
            ):
                if Atomic[UInt32].fetch_add(control, UInt32(1)) >= limit:
                    atomic_or(control.unsafe_offset(1), GROUP_CAPACITY_FAULT)
                    return NO_GROUP
                return slot
            # A spurious weak-CAS failure must not skip an empty slot.
            if expected == 0:
                continue
            owner = expected
        if tuple_equal(
            metadata, row, Int(owner) - 1, k0, k1, k2, k3, b0, b1, b2, b3
        ):
            return slot
        slot = (slot + 1) & (capacity - 1)
        probes += 1
    atomic_or(control.unsafe_offset(1), GROUP_CAPACITY_FAULT)
    return NO_GROUP
