"""Exact grouped square states using only UInt32 atomics.

The first 32 words retain the ordinary aggregate layout. Eighteen signed
64-bit bins hold unsigned square words; fewer than 2**31 input rows keep
each bin below 2**63. Integer first moments round inputs as CPU Float64
casts before accumulating two signed radix-2**32 bins.
"""
from std.atomic import Atomic
from std.bit import count_leading_zeros
from std.memory import Pointer
from .group_hash import atomic_add_signed64
from .group_state import atomic_or
from .moment_squares import (
    float32_square_words,
    integer_moment_magnitude,
    integer_square_words,
)

comptime GROUP_SQUARES = 32
comptime GROUP_MOMENT_SUM = 68
comptime GROUP_MOMENT_LARGEST = 72
comptime GROUP_MOMENT_SPECIAL = 73


@always_inline
def add_group_float_moment(
    state: Pointer[UInt32, MutUntrackedOrigin], raw: UInt32
):
    var square = float32_square_words(raw)
    if square[4]:
        atomic_or(state.unsafe_offset(GROUP_MOMENT_SPECIAL), UInt32(1))
        return
    _ = Atomic[UInt32].max(
        state.unsafe_offset(GROUP_MOMENT_LARGEST), raw & 0x7FFF_FFFF
    )
    comptime for word in range(3):
        var part = square[1]
        comptime if word == 1:
            part = square[2]
        elif word == 2:
            part = square[3]
        if part:
            atomic_add_signed64(
                state.unsafe_offset(
                    GROUP_SQUARES + (Int(square[0]) + word) * 2
                ),
                Int64(part),
            )


@always_inline
def add_group_integer_moment(
    state: Pointer[UInt32, MutUntrackedOrigin], value: Int64
):
    var rounded = integer_moment_magnitude(value)
    var magnitude = rounded[0]
    var sign = Int64(-1) if rounded[1] else Int64(1)
    atomic_add_signed64(
        state.unsafe_offset(GROUP_MOMENT_SUM),
        Int64(magnitude & 0xFFFF_FFFF) * sign,
    )
    atomic_add_signed64(
        state.unsafe_offset(GROUP_MOMENT_SUM + 2),
        Int64(magnitude >> 32) * sign,
    )
    var square = integer_square_words(magnitude)
    comptime for word in range(4):
        atomic_add_signed64(
            state.unsafe_offset(GROUP_SQUARES + word * 2), Int64(square[word])
        )
    # A bit-width bound avoids a UInt64 atomic maximum. It is conservative
    # for the subsequent numerical compatibility check, never an estimate.
    _ = Atomic[UInt32].max(
        state.unsafe_offset(GROUP_MOMENT_LARGEST),
        UInt32(64 - Int(count_leading_zeros(magnitude))),
    )
