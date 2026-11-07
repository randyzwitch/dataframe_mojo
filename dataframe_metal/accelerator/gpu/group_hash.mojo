"""Bounded integer-key grouping using immutable-row tickets and UInt32 atomics.

A claimed table entry is immediately readable: it names a row in the pinned
key input. No 64-bit key publication, lock, or CPU key-discovery pass is needed.
"""
from std.atomic import Atomic
from std.memory import Pointer, bitcast
from max.gpu import global_idx, block_idx
from .multi_reduce import Bytes, read_integer
from .filter_sum import _valid
from ...group_layout import GROUP_CAPACITY_FAULT

comptime GROUP_LIMIT = 256
comptime HASH_CAPACITY = 512
comptime GROUP_SLOTS = HASH_CAPACITY + 1
comptime NO_GROUP = UInt32.MAX


@always_inline
def _key_hash(value: Int64) -> UInt32:
    var raw = bitcast[DType.uint64](value)
    var hashed = UInt32(raw) ^ (UInt32(raw >> 32) * 0x9E37_79B9)
    hashed ^= hashed >> 16
    hashed *= 0x7FEB_352D
    hashed ^= hashed >> 15
    hashed *= 0x846C_A68B
    return hashed ^ (hashed >> 16)


@always_inline
def mark_capacity_fault(fault: Pointer[UInt32, MutUntrackedOrigin]):
    var expected = Atomic[UInt32].load(fault)
    while not (expected & GROUP_CAPACITY_FAULT):
        if Atomic[UInt32].compare_exchange[weak=True](
            fault, expected, expected | GROUP_CAPACITY_FAULT
        ):
            return


@always_inline
def claim_group(
    keys: Bytes,
    validity: Bytes,
    tag: UInt32,
    bit_offset: UInt32,
    nullable: UInt32,
    row: Int,
    table: Pointer[UInt32, MutUntrackedOrigin],
    distinct: Pointer[UInt32, MutUntrackedOrigin],
    fault: Pointer[UInt32, MutUntrackedOrigin],
) -> UInt32:
    var valid = True
    if nullable:
        valid = _valid(validity, bit_offset, row)
    var value = read_integer(keys, tag, row) if valid else Int64(0)
    var slot = UInt32(1) + (
        _key_hash(value) & UInt32(HASH_CAPACITY - 1)
    ) if valid else UInt32(0)
    var probes = 0
    while probes < HASH_CAPACITY:
        if Atomic[UInt32].load(fault):
            return NO_GROUP
        var address = table.unsafe_offset(Int(slot))
        var owner = Atomic[UInt32].load(address)
        if owner == 0:
            var expected = UInt32(0)
            if Atomic[UInt32].compare_exchange[weak=True](
                address, expected, UInt32(row) + 1
            ):
                if Atomic[UInt32].fetch_add(distinct, UInt32(1)) >= GROUP_LIMIT:
                    mark_capacity_fault(fault)
                    return NO_GROUP
                return slot
            # Metal exposes weak CAS only. A spurious failure must retry the
            # same slot instead of inventing a duplicate group farther on.
            if expected == 0:
                continue
            owner = expected
        if not valid or read_integer(keys, tag, Int(owner) - 1) == value:
            return slot
        slot = 1 if slot == HASH_CAPACITY else slot + 1
        probes += 1
    mark_capacity_fault(fault)
    return NO_GROUP


@always_inline
def record_group(
    slot: UInt32,
    row: Int,
    blocks: UInt32,
    counts: Pointer[UInt32, MutUntrackedOrigin],
    first: Pointer[UInt32, MutUntrackedOrigin],
):
    _ = Atomic[UInt32].fetch_add(
        counts.unsafe_offset(Int(slot) * Int(blocks) + Int(block_idx.x)),
        UInt32(1),
    )
    var address = first.unsafe_offset(Int(slot))
    if UInt32(row) < Atomic[UInt32].load(address):
        _ = Atomic[UInt32].min(address, UInt32(row))


def hash_groups(
    keys: Bytes,
    validity: Bytes,
    tag: UInt32,
    bit_offset: UInt32,
    nullable: UInt32,
    rows: UInt32,
    stride: UInt32,
    blocks: UInt32,
    table: Pointer[UInt32, MutUntrackedOrigin],
    distinct: Pointer[UInt32, MutUntrackedOrigin],
    fault: Pointer[UInt32, MutUntrackedOrigin],
    counts: Pointer[UInt32, MutUntrackedOrigin],
    first: Pointer[UInt32, MutUntrackedOrigin],
):
    var row = global_idx.x
    while row < Int(rows):
        var slot = claim_group(
            keys,
            validity,
            tag,
            bit_offset,
            nullable,
            row,
            table,
            distinct,
            fault,
        )
        if slot == NO_GROUP:
            return
        record_group(slot, row, blocks, counts, first)
        row += Int(stride)


@always_inline
def atomic_add_signed64(
    words: Pointer[UInt32, MutUntrackedOrigin], value: Int64
):
    var raw = bitcast[DType.uint64](value)
    var lower = UInt32(raw)
    var previous = Atomic[UInt32].fetch_add(words, lower)
    var carry = UInt32(previous + lower < previous)
    _ = Atomic[UInt32].fetch_add(
        words.unsafe_offset(1), UInt32(raw >> 32) + carry
    )


@always_inline
def atomic_add_signed96(
    words: Pointer[UInt32, MutUntrackedOrigin], value: Int64
):
    var raw = bitcast[DType.uint64](value)
    var lower = UInt32(raw)
    var middle = UInt32(raw >> 32)
    var previous = Atomic[UInt32].fetch_add(words, lower)
    var carry = UInt32(previous + lower < previous)
    previous = Atomic[UInt32].fetch_add(words.unsafe_offset(1), middle)
    var upper = UInt32.MAX if value < 0 else UInt32(0)
    upper += UInt32(previous + middle < previous)
    if carry:
        previous = Atomic[UInt32].fetch_add(words.unsafe_offset(1), UInt32(1))
        upper += UInt32(previous == UInt32.MAX)
    _ = Atomic[UInt32].fetch_add(words.unsafe_offset(2), upper)
