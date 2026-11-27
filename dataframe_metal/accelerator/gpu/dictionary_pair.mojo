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
def _dictionary_sum_row[
    single_key: Bool = False, sparse: Bool = False
](
    i: Int,
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
    if i >= Int(length):
        return
    var table = state.unsafe_offset(Int(slots) * PAIR_WORDS)
    var control = table.unsafe_offset(Int(slots - 1)) if sparse else table
    if i == 0:
        _ = Atomic[UInt32].fetch_add(control, UInt32(1))
    var a = left_labels
    if not (flags & 1) or _valid(left_bits, left_offset, i):
        a = left.unsafe_bitcast[UInt32]()[unsafe_offset=i]
        if a >= left_labels:
            _ = Atomic[UInt32].max(control.unsafe_offset(1), UInt32(1))
            return
    var b = right_labels
    comptime if not single_key:
        if not (flags & 2) or _valid(right_bits, right_offset, i):
            b = right.unsafe_bitcast[UInt32]()[unsafe_offset=i]
            if b >= right_labels:
                _ = Atomic[UInt32].max(control.unsafe_offset(1), UInt32(1))
                return
    var index = UInt32(0)
    comptime if sparse:
        var ca = a + 1 if a < left_labels else UInt32(0)
        var cb = b + 1 if b < right_labels else UInt32(0)
        index = _claim_dictionary_codes(
            (ca << 16) | cb, table, slots - 1, control
        )
        if index == UInt32.MAX:
            return
    else:
        index = a * (right_labels + 1) + b
    var slot = state.unsafe_offset(Int(index) * PAIR_WORDS)
    var ticket = ~(base + UInt32(i))
    if ticket > Atomic[UInt32].load(slot):
        _ = Atomic[UInt32].max(slot, ticket)
    if not (flags & 4) or _valid(value_bits, value_offset, i):
        _ = Atomic[UInt32].fetch_add(slot.unsafe_offset(1), UInt32(1))
        atomic_add_signed96(
            slot.unsafe_offset(2), read_integer(values, value_tag, i)
        )


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
    _dictionary_sum_row[single_key, sparse](
        global_idx.x,
        left,
        right,
        values,
        left_bits,
        right_bits,
        value_bits,
        left_offset,
        right_offset,
        value_offset,
        flags,
        left_labels,
        right_labels,
        value_tag,
        base,
        length,
        slots,
        state,
    )


def dictionary_pair_sum_two_spans[
    single_key: Bool, sparse: Bool
](
    left0: Bytes,
    right0: Bytes,
    values0: Bytes,
    lbits0: Bytes,
    rbits0: Bytes,
    vbits0: Bytes,
    left1: Bytes,
    right1: Bytes,
    values1: Bytes,
    lbits1: Bytes,
    rbits1: Bytes,
    vbits1: Bytes,
    left_offsets: UInt64,
    right_offsets: UInt64,
    value_offsets: UInt64,
    flags: UInt32,
    left_labels: UInt32,
    right_labels: UInt32,
    value_tag: UInt32,
    base: UInt32,
    first_rows: UInt32,
    second_rows: UInt32,
    slots: UInt32,
    state: Pointer[UInt32, MutUntrackedOrigin],
):
    # Each lane reads its owner's original buffers. The two spans share the
    # same atomic result slots; first-row tickets retain their global offsets.
    var i = global_idx.x
    if i < Int(first_rows):
        _dictionary_sum_row[single_key, sparse](
            i,
            left0,
            right0,
            values0,
            lbits0,
            rbits0,
            vbits0,
            UInt32(left_offsets),
            UInt32(right_offsets),
            UInt32(value_offsets),
            flags & 7,
            left_labels,
            right_labels,
            value_tag,
            base,
            first_rows,
            slots,
            state,
        )
    else:
        _dictionary_sum_row[single_key, sparse](
            i - Int(first_rows),
            left1,
            right1,
            values1,
            lbits1,
            rbits1,
            vbits1,
            UInt32(left_offsets >> 32),
            UInt32(right_offsets >> 32),
            UInt32(value_offsets >> 32),
            flags >> 3,
            left_labels,
            right_labels,
            value_tag,
            base + first_rows,
            second_rows,
            slots,
            state,
        )
