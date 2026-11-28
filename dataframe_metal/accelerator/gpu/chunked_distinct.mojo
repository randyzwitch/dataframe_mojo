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
from ...group_layout import GROUP_CAPACITY_FAULT
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
    var group_table = scratch.unsafe_offset(4 + 3 * Int(groups))
    var key_valid = distinct_valid(key_bits, key_descriptor, row)
    var key = keys.unsafe_bitcast[UInt32]()[
        unsafe_offset=row
    ] if key_valid else UInt32(0)
    var value_valid = distinct_valid(value_bits, value_descriptor, row)
    var tag = UInt32(value_descriptor >> 33)
    var value = distinct_canonical(
        ordered_bits(values, tag, row), tag
    ) if value_valid else UInt64(0)
    # The first row of each equal (key, value) run publishes its exact pair.
    # Row zero always represents its own span; owners can change there.
    var same_key = False
    if row > 0:
        var prev = row - 1
        same_key = distinct_valid(
            key_bits, key_descriptor, prev
        ) == key_valid and (
            not key_valid
            or keys.unsafe_bitcast[UInt32]()[unsafe_offset=prev] == key
        )
        if (
            same_key
            and distinct_valid(value_bits, value_descriptor, prev)
            == value_valid
        ):
            if (
                not value_valid
                or distinct_canonical(ordered_bits(values, tag, prev), tag)
                == value
            ):
                return
    # Finish discovering groups even if the value table filled earlier.
    # The host can then retain the original group table on a value-only retry.
    var group = claim_word_group[GROUP_CAPACITY_FAULT](
        key, key_valid, group_table, groups, scratch
    )
    if group == NO_GROUP:
        return
    var ticket = base + UInt32(row) + 1
    var first = scratch.unsafe_offset(4 + Int(group))
    # Only the first row of a key run can establish an earlier group.
    var first_ticket = ~(ticket - 1)
    if not same_key and first_ticket > Atomic[UInt32].load(first):
        _ = Atomic[UInt32].max(first, first_ticket)
    var count = first.unsafe_offset(Int(groups))
    if not value_valid:
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
