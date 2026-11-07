"""Exact host subtraction for GPU sums and squared sums.

Float32 values are integer multiples of 2**-149. Eighteen radix-2**32
square bins cover their entire finite range. With at most UInt32.MAX rows,
20 words hold n * sum(x*x) and sum(x)**2, including cancellation to zero.
No floating arithmetic occurs until after their exact subtraction.
"""
from std.memory import bitcast
from std.math import sqrt
from .accelerator.faults import data_fault

comptime MOMENT_WORDS = 20


struct MomentMagnitude(Copyable, Movable):
    var digits: List[UInt32]

    def __init__(out self):
        self.digits = List[UInt32](length=MOMENT_WORDS, fill=0)

    @staticmethod
    def from_bins(bins: List[Int128]) raises -> Self:
        if len(bins) > 18:
            raise data_fault("moment bins exceed the native numeric range")
        var result = Self()
        var carry = Int128(0)
        for i in range(MOMENT_WORDS):
            var value = carry
            if i < len(bins):
                # Each row contributes at most one unsigned 32-bit word to
                # a bin. This also makes signed normalization overflow-free.
                if bins[i] < -Int128(UInt64.MAX) or bins[i] > Int128(
                    UInt64.MAX
                ):
                    raise data_fault("moment bin exceeds the row-count bound")
                value += bins[i]
            result.digits[i] = UInt32(value & Int128(0xFFFF_FFFF))
            carry = value >> 32
        if carry != 0 and carry != -1:
            raise data_fault("moment magnitude exceeds its bounded storage")
        if carry < 0:
            var add = UInt64(1)
            for i in range(MOMENT_WORDS):
                var value = UInt64(~result.digits[i]) + add
                result.digits[i] = UInt32(value & 0xFFFF_FFFF)
                add = value >> 32
        return result^

    def used(self) -> Int:
        var length = MOMENT_WORDS
        while length and self.digits[length - 1] == 0:
            length -= 1
        return length

    def compare(self, other: Self) -> Int:
        for i in reversed(range(MOMENT_WORDS)):
            if self.digits[i] != other.digits[i]:
                return -1 if self.digits[i] < other.digits[i] else 1
        return 0

    def multiply(self, other: Self) raises -> Self:
        var result = Self()
        for i in range(self.used()):
            var carry = UInt64(0)
            for j in range(other.used()):
                var at = i + j
                if at >= MOMENT_WORDS:
                    raise data_fault(
                        "moment product exceeds its bounded storage"
                    )
                # (2**32-1)**2 + 2*(2**32-1) fits UInt64 exactly.
                var word = (
                    UInt64(self.digits[i]) * UInt64(other.digits[j])
                    + UInt64(result.digits[at])
                    + carry
                )
                result.digits[at] = UInt32(word & 0xFFFF_FFFF)
                carry = word >> 32
            var at = i + other.used()
            while carry:
                if at >= MOMENT_WORDS:
                    raise data_fault(
                        "moment product exceeds its bounded storage"
                    )
                var word = UInt64(result.digits[at]) + carry
                result.digits[at] = UInt32(word & 0xFFFF_FFFF)
                carry = word >> 32
                at += 1
        return result^

    def subtract(self, other: Self) raises -> Self:
        if self.compare(other) < 0:
            raise data_fault("moment partials produced a negative variance")
        var result = Self()
        var borrow = UInt64(0)
        for i in range(MOMENT_WORDS):
            var left = UInt64(self.digits[i])
            var right = UInt64(other.digits[i]) + borrow
            result.digits[i] = UInt32((left - right) & 0xFFFF_FFFF)
            borrow = UInt64(left < right)
        return result^

    def scaled(self, exponent: Int) raises -> Float64:
        """Round the magnitude once, then apply an exact binary scale."""
        var top = self.used() - 1
        if top < 0:
            return 0
        var leading = self.digits[top]
        var bits = 0
        while leading:
            leading >>= 1
            bits += 1
        var width = top * 32 + bits
        var shift = max(0, width - 53)
        var significand = UInt64(0)
        for i in reversed(range(shift, width)):
            significand = (significand << 1) | UInt64(
                (self.digits[i // 32] >> UInt32(i % 32)) & 1
            )
        if shift:
            var halfway = (
                self.digits[(shift - 1) // 32] >> UInt32((shift - 1) % 32)
            ) & 1
            var sticky = False
            for i in range(shift - 1):
                sticky = sticky or Bool(
                    (self.digits[i // 32] >> UInt32(i % 32)) & 1
                )
            if halfway and (sticky or Bool(significand & 1)):
                significand += 1
        var power = shift + exponent + 1023
        if power <= 0 or power >= 2047:
            raise data_fault("moment scale exceeds native Float64 finishing")
        return Float64(significand) * bitcast[DType.float64](
            UInt64(power) << 52
        )


def finish_variance(
    sums: List[Int128],
    squares: List[Int128],
    count: Int,
    ddof: Int,
    unit_exponent: Int,
    standard_deviation: Bool = False,
    non_finite: Bool = False,
) raises -> Optional[Float64]:
    """Finish exact moments, including arbitrary cancellation and ddof.

    Numerator rounding, denominator rounding and division each contribute at
    most one Float64 rounding. The denominator is formed exactly in Int128.
    This is a mathematical variance; matching CPU accumulation behavior for
    supported input families remains the caller's responsibility.
    """
    if count < 0 or count > Int(UInt32.MAX):
        raise data_fault("variance row count exceeds the GPU counter range")
    var denominator = Int128(count) - Int128(ddof)
    if denominator <= 0:
        return None
    if denominator > Int128(Int64.MAX):
        raise data_fault("variance ddof needs CPU integer adjudication")
    if non_finite:
        raise data_fault("non-finite variance needs CPU adjudication")
    if unit_exponent < -149 or unit_exponent > 0:
        raise data_fault("variance scale exceeds native numeric inputs")
    if count == 0:
        # The CPU permits negative ddof, including on empty input.
        return Float64(0)
    for word in squares:
        if word < 0:
            raise data_fault("variance squared sums must be nonnegative")
    var total = MomentMagnitude.from_bins(sums)
    var squared = MomentMagnitude.from_bins(squares)
    var population = MomentMagnitude.from_bins(List[Int128]([Int128(count)]))
    var first = squared.multiply(population)
    var second = total.multiply(total)
    var numerator = first.subtract(second)
    var result = numerator.scaled(unit_exponent * 2) / (
        (Int128(count) * denominator).cast[DType.float64]()
    )
    return sqrt(result) if standard_deviation else result
