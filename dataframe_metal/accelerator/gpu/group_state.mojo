"""Bounded per-group/per-workgroup reduction state using UInt32 atomics.

Float32 sums use exact exponent bins. No floating atomic arithmetic or
64-bit atomic operation is required. A queue drain precedes host decoding.
"""
from std.atomic import Atomic
from std.bit import count_leading_zeros, count_trailing_zeros
from std.memory import Pointer, bitcast
from max.gpu import global_idx
from .group_hash import atomic_add_signed64, atomic_add_signed96
from .float_predicate import _ordered

comptime GROUP_STATE_WORDS = 32
comptime GROUP_COUNT = 0
comptime GROUP_SUM = 1
comptime GROUP_MIN = 4
comptime GROUP_MAX = 5
comptime GROUP_SPECIAL = 6
comptime GROUP_NUMERIC_COUNT = 7
comptime GROUP_LEAST_BIT = 8
comptime GROUP_GREATEST_BIT = 9
comptime GROUP_BINS = 10


@always_inline
def atomic_or(words: Pointer[UInt32, MutUntrackedOrigin], bits: UInt32):
    var expected = Atomic[UInt32].load(words)
    while (expected & bits) != bits:
        if Atomic[UInt32].compare_exchange[weak=True](
            words, expected, expected | bits
        ):
            return


@always_inline
def initial_group_word(offset: Int) -> UInt32:
    return UInt32.MAX if offset == GROUP_MIN or offset == GROUP_MAX else (
        UInt32(512) if offset == GROUP_LEAST_BIT else UInt32(0)
    )


def initialize_group_states(
    states: Pointer[UInt32, MutUntrackedOrigin],
    words: UInt32,
):
    var row = global_idx.x
    if row < Int(words):
        states[unsafe_offset=row] = initial_group_word(row % GROUP_STATE_WORDS)


@always_inline
def add_integer_sum(state: Pointer[UInt32, MutUntrackedOrigin], value: Int64):
    atomic_add_signed96(state.unsafe_offset(GROUP_SUM), value)


@always_inline
def add_float_sum(state: Pointer[UInt32, MutUntrackedOrigin], raw: UInt32):
    var exponent = (raw >> 23) & 255
    if exponent == 255:
        atomic_or(
            state.unsafe_offset(GROUP_SPECIAL),
            UInt32(1) if raw
            & 0x7FFFFF else (UInt32(4) if raw >> 31 else UInt32(2)),
        )
        return
    var significand = raw & 0x7FFFFF
    if exponent:
        significand |= 0x800000
    if significand == 0:
        return
    var shift = exponent - 1 if exponent else UInt32(0)
    # These bounds only widen. A stale load can cause an extra atomic update,
    # but cannot hide a value that would improve the final bound.
    var least = state.unsafe_offset(GROUP_LEAST_BIT)
    var low = shift + UInt32(count_trailing_zeros(significand))
    if low < Atomic[UInt32].load(least):
        _ = Atomic[UInt32].min(least, low)
    var greatest = state.unsafe_offset(GROUP_GREATEST_BIT)
    var high = shift + 31 - UInt32(count_leading_zeros(significand))
    if high > Atomic[UInt32].load(greatest):
        _ = Atomic[UInt32].max(greatest, high)
    var index = Int(shift >> 5)
    var shifted = UInt64(significand) << UInt64(shift & 31)
    var sign = Int64(-1) if raw >> 31 else Int64(1)
    var lower = Int64(shifted & 0xFFFF_FFFF) * sign
    var upper = Int64(shifted >> 32) * sign
    if lower:
        atomic_add_signed64(state.unsafe_offset(GROUP_BINS + index * 2), lower)
    if upper:
        atomic_add_signed64(
            state.unsafe_offset(GROUP_BINS + (index + 1) * 2), upper
        )


@always_inline
def add_ordered_extrema(
    state: Pointer[UInt32, MutUntrackedOrigin], raw: UInt32, floating: Bool
):
    var key = raw ^ 0x8000_0000
    if floating:
        var magnitude = raw & 0x7FFF_FFFF
        if magnitude > 0x7F80_0000:
            atomic_or(state.unsafe_offset(GROUP_SPECIAL), UInt32(1))
            return
        if magnitude == 0:
            atomic_or(
                state.unsafe_offset(GROUP_SPECIAL),
                UInt32(16) if raw >> 31 else UInt32(8),
            )
        key = _ordered(raw)
    _ = Atomic[UInt32].fetch_add(
        state.unsafe_offset(GROUP_NUMERIC_COUNT), UInt32(1)
    )
    var minimum = state.unsafe_offset(GROUP_MIN)
    if key < Atomic[UInt32].load(minimum):
        _ = Atomic[UInt32].min(minimum, key)
    # MAX uses complemented ordered keys so both extrema can start at MAX.
    var maximum_inverse = state.unsafe_offset(GROUP_MAX)
    if ~key < Atomic[UInt32].load(maximum_inverse):
        _ = Atomic[UInt32].min(maximum_inverse, ~key)
