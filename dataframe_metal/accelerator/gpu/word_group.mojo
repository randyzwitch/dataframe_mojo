"""Persistent group slots for nullable 32-bit keys in independent chunks."""
from std.atomic import Atomic
from std.memory import Pointer
from .group_hash import NO_GROUP, _key_hash, mark_capacity_fault


@always_inline
def claim_word_group(
    raw: UInt32,
    present: Bool,
    table: Pointer[UInt32, MutUntrackedOrigin],
    capacity: UInt32,
    control: Pointer[UInt32, MutUntrackedOrigin],
) -> UInt32:
    """Claim or find a slot using only the key bits and the shared table.

    The caller zeroes a power-of-two table and control[0:2] once per attempt.
    All segments share both buffers. Slot 0 represents null and slot 1 represents
    valid zero. Those slots use 1 as an occupancy marker. Remaining slots hold
    the complete nonzero key, published in a single 32-bit atomic operation.
    """
    var slot = UInt32(0)
    var key = UInt32(1)
    if present:
        if raw == 0:
            slot = 1
        else:
            key = raw
            slot = max(UInt32(2), _key_hash(Int64(raw)) & (capacity - 1))
    var fault = control.unsafe_offset(1)
    var probes = UInt32(0)
    while probes < capacity:
        if Atomic[UInt32].load(fault):
            return NO_GROUP
        var address = table.unsafe_offset(Int(slot))
        var found = Atomic[UInt32].load(address)
        if found == key:
            return slot
        if found == 0:
            if Atomic[UInt32].compare_exchange[weak=True](address, found, key):
                if (
                    Atomic[UInt32].fetch_add(control, UInt32(1))
                    >= capacity // 2
                ):
                    mark_capacity_fault(fault)
                    return NO_GROUP
                return slot
            if found == key:
                return slot
            # A weak CAS can fail with the slot still empty. Retry here so no
            # competing thread can publish the same key in a later slot.
            if found == 0:
                continue
        slot = 2 if slot + 1 == capacity else slot + 1
        probes += 1
    mark_capacity_fault(fault)
    return NO_GROUP
