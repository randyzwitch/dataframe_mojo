"""Exact distinct records shared by ordered submissions of independent input spans."""
from std.atomic import Atomic
from std.memory import Pointer
from max.gpu import global_idx
from .distinct_reduce import distinct_valid
from .distinct_value import distinct_canonical, distinct_hash
from .group_hash import NO_GROUP
from .group_state import atomic_or
from .multi_reduce import Bytes
from .ordered_value import ordered_bits
from .word_group import claim_word_group
from ...distinct_layout import DISTINCT_CAPACITY_FAULT, MAX_DISTINCT_PROBES

comptime Words = Pointer[UInt32, MutUntrackedOrigin]


@always_inline
def count_new_value(count: Words, control: Words, capacity: UInt32):
    if (
        Atomic[UInt32].fetch_add(control.unsafe_offset(3), UInt32(1))
        >= capacity // 2
    ):
        atomic_or(control.unsafe_offset(1), DISTINCT_CAPACITY_FAULT)
    else:
        _ = Atomic[UInt32].fetch_add(count, UInt32(1))


def chunked_distinct_partials(
    keys: Bytes,
    key_bits: Bytes,
    values: Bytes,
    value_bits: Bytes,
    key_descriptor: UInt64,
    value_descriptor: UInt64,
    base: UInt32,
    length: UInt32,
    groups: UInt32,
    capacity: UInt32,
    scratch: Words,
):
    var row = global_idx.x
    if row >= Int(length):
        return
    if row == 0:
        _ = Atomic[UInt32].fetch_add(scratch.unsafe_offset(2), UInt32(1))
    if Atomic[UInt32].load(scratch.unsafe_offset(1)):
        return
    var group_table = scratch.unsafe_offset(4 + 3 * Int(groups))
    var key_valid = distinct_valid(key_bits, key_descriptor, row)
    var key = keys.unsafe_bitcast[UInt32]()[
        unsafe_offset=row
    ] if key_valid else UInt32(0)
    var group = claim_word_group(key, key_valid, group_table, groups, scratch)
    if group == NO_GROUP:
        return
    var ticket = base + UInt32(row) + 1
    var first = scratch.unsafe_offset(4 + Int(group))
    _ = Atomic[UInt32].max(first, ~(ticket - 1))
    var count = first.unsafe_offset(Int(groups))
    if not distinct_valid(value_bits, value_descriptor, row):
        # Null participates in the original table's load limit and in the
        # all-values count exactly once per group, even across dispatches.
        var flag = count.unsafe_offset(Int(groups))
        var seen = Atomic[UInt32].load(flag)
        while seen == 0:
            if Atomic[UInt32].compare_exchange[weak=True](
                flag, seen, UInt32(1)
            ):
                count_new_value(count, scratch, capacity)
                break
        return

    var tag = UInt32(value_descriptor >> 33)
    var value = distinct_canonical(ordered_bits(values, tag, row), tag)
    var wide = (tag & 3) == 1
    var record_words = 3 + Int(wide)
    var slot = distinct_hash(value, True, tag, group) & (capacity - 1)
    var records = group_table.unsafe_offset(Int(groups))
    var probes = 0
    while probes < MAX_DISTINCT_PROBES:
        if Atomic[UInt32].load(scratch.unsafe_offset(1)):
            return
        var record = records.unsafe_offset(Int(slot) * record_words)
        var owner = Atomic[UInt32].load(record)
        if owner == 0:
            if Atomic[UInt32].compare_exchange[weak=True](
                record, owner, ticket
            ):
                record[unsafe_offset=1] = group
                record[unsafe_offset=2] = UInt32(value)
                if wide:
                    record[unsafe_offset=3] = UInt32(value >> 32)
                count_new_value(count, scratch, capacity)
                return
            if owner == 0:
                continue

        # CAS publishes the row ticket before the remaining words. Threads
        # in its own dispatch compare immutable input rows. Once that dispatch
        # has finished, its compact record is ready for every later span.
        var previous = owner - 1
        if previous < base:
            if (
                record[unsafe_offset=1] == group
                and record[unsafe_offset=2] == UInt32(value)
                and (not wide or record[unsafe_offset=3] == UInt32(value >> 32))
            ):
                return
        else:
            var local = Int(previous - base)
            if (
                distinct_valid(key_bits, key_descriptor, local) == key_valid
                and (
                    not key_valid
                    or keys.unsafe_bitcast[UInt32]()[unsafe_offset=local] == key
                )
                and distinct_canonical(ordered_bits(values, tag, local), tag)
                == value
            ):
                return
        slot = (slot + 1) & (capacity - 1)
        probes += 1
    atomic_or(scratch.unsafe_offset(1), DISTINCT_CAPACITY_FAULT)
