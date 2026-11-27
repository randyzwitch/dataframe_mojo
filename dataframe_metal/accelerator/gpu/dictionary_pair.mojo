"""Direct-address SUM for one or two bounded dictionaries, including null keys."""
from max.gpu import global_idx
from std.atomic import Atomic
from std.memory import Pointer
from .filter_sum import _valid
from .group_hash import atomic_add_signed96, _key_hash
from .multi_reduce import Bytes, read_integer

# Complement of the first row, non-null count, and an exact signed 96-bit sum.
comptime PAIR_WORDS = 5


@always_inline
def _claim_dictionary_codes(
    codes: UInt32,
    table: Pointer[UInt32, MutUntrackedOrigin],
    capacity: UInt32,
    control: Pointer[UInt32, MutUntrackedOrigin],
) -> UInt32:
    # The all-null tuple has its own slot. Every other packed key is nonzero,
    # so one 32-bit CAS publishes the entire tuple on Apple GPUs.
    if codes == 0:
        return 0
    var index = _key_hash(Int64(codes)) & (capacity - 1)
    var probes = 0
    while probes < Int(capacity):
        if Atomic[UInt32].load(control.unsafe_offset(3)):
            return UInt32.MAX
        var address = table.unsafe_offset(Int(index))
        var found = Atomic[UInt32].load(address)
        if found == codes:
            return index + 1
        if found == 0:
            if Atomic[UInt32].compare_exchange[weak=True](
                address, found, codes
            ):
                if (
                    Atomic[UInt32].fetch_add(
                        control.unsafe_offset(2), UInt32(1)
                    )
                    >= 65536
                ):
                    _ = Atomic[UInt32].max(control.unsafe_offset(3), UInt32(1))
                    return UInt32.MAX
                return index + 1
            if found == 0:
                continue
            if found == codes:
                return index + 1
        probes += 1
        index = (index + 1) & (capacity - 1)
    _ = Atomic[UInt32].max(control.unsafe_offset(3), UInt32(1))
    return UInt32.MAX


@always_inline
def _dictionary_slot[
    single_key: Bool, sparse: Bool
](
    left: Bytes,
    right: Bytes,
    left_bits: Bytes,
    right_bits: Bytes,
    left_offset: UInt32,
    right_offset: UInt32,
    flags: UInt32,
    left_labels: UInt32,
    right_labels: UInt32,
    i: Int,
    slots: UInt32,
    table: Pointer[UInt32, MutUntrackedOrigin],
    control: Pointer[UInt32, MutUntrackedOrigin],
) -> UInt32:
    var a = left_labels
    if not (flags & 1) or _valid(left_bits, left_offset, i):
        a = left.unsafe_bitcast[UInt32]()[unsafe_offset=i]
        if a >= left_labels:
            _ = Atomic[UInt32].max(control.unsafe_offset(1), UInt32(1))
            return UInt32.MAX
    var b = right_labels
    comptime if not single_key:
        if not (flags & 2) or _valid(right_bits, right_offset, i):
            b = right.unsafe_bitcast[UInt32]()[unsafe_offset=i]
            if b >= right_labels:
                _ = Atomic[UInt32].max(control.unsafe_offset(1), UInt32(1))
                return UInt32.MAX
    var index = UInt32(0)
    comptime if sparse:
        var ca = a + 1 if a < left_labels else UInt32(0)
        var cb = b + 1 if b < right_labels else UInt32(0)
        index = _claim_dictionary_codes(
            (ca << 16) | cb, table, slots - 1, control
        )
    else:
        index = a * (right_labels + 1) + b
    return index


def dictionary_pair_sum[
    single_key: Bool = False, sparse: Bool = False
](
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
    slots: UInt32,
    state: Pointer[UInt32, MutUntrackedOrigin],
):
    var i = global_idx.x
    if i >= Int(length):
        return
    var table = state.unsafe_offset(Int(slots) * PAIR_WORDS)
    var control = table.unsafe_offset(Int(slots - 1)) if sparse else table
    if i == 0:
        _ = Atomic[UInt32].fetch_add(control, UInt32(1))
    var index = _dictionary_slot[single_key, sparse](
        left,
        right,
        left_bits,
        right_bits,
        left_offset,
        right_offset,
        flags,
        left_labels,
        right_labels,
        i,
        slots,
        table,
        control,
    )
    if index == UInt32.MAX:
        return
    var slot = state.unsafe_offset(Int(index) * PAIR_WORDS)
    var ticket = ~(base + UInt32(i))
    if ticket > Atomic[UInt32].load(slot):
        _ = Atomic[UInt32].max(slot, ticket)
    if not (flags & 4) or _valid(value_bits, value_offset, i):
        _ = Atomic[UInt32].fetch_add(slot.unsafe_offset(1), UInt32(1))
        atomic_add_signed96(
            slot.unsafe_offset(2), read_integer(values, value_tag, i)
        )


@always_inline
def _update_dictionary_field(
    row: Int,
    value: Bytes,
    bits: Bytes,
    descriptor: UInt64,
    slot: Pointer[UInt32, MutUntrackedOrigin],
):
    # Offset zero is the first-row ticket and never names a value field.
    var offset = Int((descriptor >> 8) & 255)
    if not offset:
        return
    if descriptor & 4 and not _valid(bits, UInt32(descriptor >> 32), row):
        return
    var field = slot.unsafe_offset(offset)
    var count = Bool(descriptor & 16)
    if count:
        _ = Atomic[UInt32].fetch_add(field, UInt32(1))
    if descriptor & 8:
        atomic_add_signed96(
            field.unsafe_offset(Int(count)),
            read_integer(value, UInt32(descriptor & 3), row),
        )


def dictionary_fused[
    single_key: Bool, sparse: Bool
](
    left: Bytes,
    right: Bytes,
    left_bits: Bytes,
    right_bits: Bytes,
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    d0: UInt64,
    d1: UInt64,
    d2: UInt64,
    d3: UInt64,
    left_offset: UInt32,
    right_offset: UInt32,
    flags: UInt32,
    left_labels: UInt32,
    right_labels: UInt32,
    base: UInt32,
    length: UInt32,
    slots: UInt32,
    stride: UInt32,
    state: Pointer[UInt32, MutUntrackedOrigin],
):
    var i = global_idx.x
    if i >= Int(length):
        return
    var table = state.unsafe_offset(Int(slots) * Int(stride))
    var control = table.unsafe_offset(Int(slots - 1)) if sparse else table
    if i == 0:
        _ = Atomic[UInt32].fetch_add(control, UInt32(1))
    var index = _dictionary_slot[single_key, sparse](
        left,
        right,
        left_bits,
        right_bits,
        left_offset,
        right_offset,
        flags,
        left_labels,
        right_labels,
        i,
        slots,
        table,
        control,
    )
    if index == UInt32.MAX:
        return
    var slot = state.unsafe_offset(Int(index) * Int(stride))
    var ticket = ~(base + UInt32(i))
    if ticket > Atomic[UInt32].load(slot):
        _ = Atomic[UInt32].max(slot, ticket)
    if flags & 4:
        _ = Atomic[UInt32].fetch_add(slot.unsafe_offset(1), UInt32(1))
    _update_dictionary_field(i, v0, b0, d0, slot)
    _update_dictionary_field(i, v1, b1, d1, slot)
    _update_dictionary_field(i, v2, b2, d2, slot)
    _update_dictionary_field(i, v3, b3, d3, slot)
