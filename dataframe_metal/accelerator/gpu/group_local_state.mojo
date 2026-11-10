"""One exact repeated-key contribution cache per GPU thread."""
from std.atomic import Atomic
from std.memory import Pointer, bitcast
from std.bit import count_leading_zeros, count_trailing_zeros
from .group_hash import atomic_add_signed64, NO_GROUP
from .group_state import (
    GROUP_COUNT,
    GROUP_SUM,
    GROUP_MIN,
    GROUP_MAX,
    GROUP_SPECIAL,
    GROUP_NUMERIC_COUNT,
    GROUP_LEAST_BIT,
    GROUP_GREATEST_BIT,
    GROUP_BINS,
    atomic_or,
)
from .float_predicate import _ordered


@always_inline
def _add96(state: Pointer[UInt32, MutUntrackedOrigin], lo: UInt64, hi: Int64):
    var upper = hi + Int64(lo >> 32)
    var a = UInt32(lo)
    var b = UInt32(bitcast[DType.uint64](upper))
    var c = UInt32(bitcast[DType.uint64](upper) >> 32)
    var old = Atomic[UInt32].fetch_add(state, a)
    var carry = UInt32(old + a < old)
    old = Atomic[UInt32].fetch_add(state.unsafe_offset(1), b)
    c += UInt32(old + b < old)
    if carry:
        old = Atomic[UInt32].fetch_add(state.unsafe_offset(1), UInt32(1))
        c += UInt32(old == UInt32.MAX)
    _ = Atomic[UInt32].fetch_add(state.unsafe_offset(2), c)


struct GroupLocalState(Movable):
    var slot: UInt32
    var first: UInt32
    var rows: UInt32
    var counts: SIMD[DType.uint32, 4]
    var hi: SIMD[DType.int64, 4]
    var lo: SIMD[DType.uint64, 4]
    var minimum: SIMD[DType.int64, 4]
    var maximum: SIMD[DType.int64, 4]
    var min_row: SIMD[DType.uint32, 4]
    var max_row: SIMD[DType.uint32, 4]
    var numeric: SIMD[DType.uint32, 4]
    var special: SIMD[DType.uint32, 4]
    var least: SIMD[DType.uint32, 4]
    var greatest: SIMD[DType.uint32, 4]
    var base: SIMD[DType.uint32, 4]
    var bin0: SIMD[DType.int64, 4]
    var bin1: SIMD[DType.int64, 4]
    var bin2: SIMD[DType.int64, 4]

    @always_inline
    def __init__(out self):
        self.slot = NO_GROUP
        self.first = UInt32.MAX
        self.rows = 0
        self.counts = SIMD[DType.uint32, 4](0)
        self.hi = SIMD[DType.int64, 4](0)
        self.lo = SIMD[DType.uint64, 4](0)
        self.minimum = SIMD[DType.int64, 4](Int64.MAX)
        self.maximum = SIMD[DType.int64, 4](Int64.MIN)
        self.min_row = SIMD[DType.uint32, 4](UInt32.MAX)
        self.max_row = SIMD[DType.uint32, 4](UInt32.MAX)
        self.numeric = SIMD[DType.uint32, 4](0)
        self.special = SIMD[DType.uint32, 4](0)
        self.least = SIMD[DType.uint32, 4](512)
        self.greatest = SIMD[DType.uint32, 4](0)
        self.base = SIMD[DType.uint32, 4](512)
        self.bin0 = SIMD[DType.int64, 4](0)
        self.bin1 = SIMD[DType.int64, 4](0)
        self.bin2 = SIMD[DType.int64, 4](0)

    @always_inline
    def flush_bins(
        mut self, field: Int, state: Pointer[UInt32, MutUntrackedOrigin]
    ):
        if self.base[field] != 512:
            var offset = GROUP_BINS + Int(self.base[field]) * 2
            if self.bin0[field]:
                atomic_add_signed64(
                    state.unsafe_offset(offset), self.bin0[field]
                )
            if self.bin1[field]:
                atomic_add_signed64(
                    state.unsafe_offset(offset + 2), self.bin1[field]
                )
            if self.bin2[field]:
                atomic_add_signed64(
                    state.unsafe_offset(offset + 4), self.bin2[field]
                )
        self.bin0[field] = 0
        self.bin1[field] = 0
        self.bin2[field] = 0

    @always_inline
    def add(
        mut self,
        field: Int,
        kind: Int,
        bits: UInt64,
        row: UInt32,
        want_sum: Bool,
        want_extrema: Bool,
        state: Pointer[UInt32, MutUntrackedOrigin],
    ):
        self.counts[field] += 1
        if not want_sum and not want_extrema:
            return
        if kind == 0:
            var raw = UInt32(bits)
            var exponent = (raw >> 23) & 255
            if exponent == 255 and (want_sum or Bool(raw & 0x7FFFFF)):
                self.special[field] |= UInt32(1) if raw & 0x7FFFFF else (
                    UInt32(4) if raw >> 31 else UInt32(2)
                )
            if want_extrema:
                var magnitude = raw & 0x7FFFFFFF
                if magnitude <= 0x7F800000:
                    if magnitude == 0:
                        self.special[field] |= UInt32(
                            16
                        ) if raw >> 31 else UInt32(8)
                    var key = Int64(_ordered(raw))
                    self.minimum[field] = min(self.minimum[field], key)
                    self.maximum[field] = max(self.maximum[field], key)
                    self.numeric[field] += 1
            if not want_sum or exponent == 255:
                return
            var significand = raw & 0x7FFFFF
            if exponent:
                significand |= 0x800000
            if not significand:
                return
            var shift = exponent - 1 if exponent else UInt32(0)
            self.least[field] = min(
                self.least[field],
                shift + UInt32(count_trailing_zeros(significand)),
            )
            self.greatest[field] = max(
                self.greatest[field],
                shift + 31 - UInt32(count_leading_zeros(significand)),
            )
            var index = shift >> 5
            if index < self.base[field] or index + 1 >= self.base[field] + 3:
                self.flush_bins(field, state)
                self.base[field] = index
            var shifted = UInt64(significand) << UInt64(shift & 31)
            var sign = Int64(-1) if raw >> 31 else Int64(1)
            var lower = Int64(shifted & 0xFFFFFFFF) * sign
            var upper = Int64(shifted >> 32) * sign
            var offset = index - self.base[field]
            if offset == 0:
                self.bin0[field] += lower
                self.bin1[field] += upper
            else:
                self.bin1[field] += lower
                self.bin2[field] += upper
        else:
            var value = bitcast[DType.int64](bits)
            if want_sum:
                self.hi[field] += value >> 32
                self.lo[field] += UInt64(value & 0xFFFFFFFF)
            if want_extrema:
                self.numeric[field] += 1
                var key = (
                    Int64(UInt32(bits) ^ 0x80000000) if kind == 1 else value
                )
                if (
                    key < self.minimum[field]
                    or self.min_row[field] == UInt32.MAX
                ):
                    self.minimum[field] = key
                    self.min_row[field] = row
                if (
                    key > self.maximum[field]
                    or self.max_row[field] == UInt32.MAX
                ):
                    self.maximum[field] = key
                    self.max_row[field] = row

    @always_inline
    def flush_field(
        mut self,
        field: Int,
        kind: Int,
        want_sum: Bool,
        want_extrema: Bool,
        state: Pointer[UInt32, MutUntrackedOrigin],
    ):
        if not self.counts[field]:
            return
        _ = Atomic[UInt32].fetch_add(
            state.unsafe_offset(GROUP_COUNT), self.counts[field]
        )
        if want_sum:
            if kind == 0:
                self.flush_bins(field, state)
                _ = Atomic[UInt32].min(
                    state.unsafe_offset(GROUP_LEAST_BIT), self.least[field]
                )
                _ = Atomic[UInt32].max(
                    state.unsafe_offset(GROUP_GREATEST_BIT),
                    self.greatest[field],
                )
            else:
                _add96(
                    state.unsafe_offset(GROUP_SUM),
                    self.lo[field],
                    self.hi[field],
                )
        if self.special[field]:
            atomic_or(state.unsafe_offset(GROUP_SPECIAL), self.special[field])
        if want_extrema and kind != 2 and self.numeric[field]:
            _ = Atomic[UInt32].fetch_add(
                state.unsafe_offset(GROUP_NUMERIC_COUNT), self.numeric[field]
            )
            _ = Atomic[UInt32].min(
                state.unsafe_offset(GROUP_MIN), UInt32(self.minimum[field])
            )
            _ = Atomic[UInt32].min(
                state.unsafe_offset(GROUP_MAX), ~UInt32(self.maximum[field])
            )
