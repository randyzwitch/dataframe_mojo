"""Exact decimal helpers implemented with paired UInt64 words.

Ported from native/dfmetal/dfmetal_decimal.h. Wide values store the low word
first and use two's-complement signed arithmetic where specified. No device
UInt128 instruction, native Metal helper, or external call is required.

Arithmetic error codes preserve the existing contract: 1 is a scaling or
product overflow, 2 is a precision overflow, 3 is unary overflow, and 4 is a
zero divisor whose quotient must be null. Cast/comparison validity is separate.
"""
from . import floating as f


struct Wide(ImplicitlyCopyable):
    var low: UInt64
    var high: UInt64

    def __init__(out self, low: UInt64, high: UInt64 = 0):
        self.low = low
        self.high = high


struct Quotient(Copyable):
    var value: UInt64
    var remainder: UInt64

    def __init__(out self, value: UInt64, remainder: UInt64):
        self.value = value
        self.remainder = remainder


struct Division(Copyable):
    var value: Wide
    var remainder: Wide

    def __init__(out self, value: Wide, remainder: Wide):
        self.value = value
        self.remainder = remainder


struct Cast(Copyable):
    var value: Wide
    var valid: Bool

    def __init__(out self, value: Wide, valid: Bool):
        self.value = value
        self.valid = valid


struct Arithmetic(Copyable):
    var value: Wide
    var error: UInt32

    def __init__(out self, value: Wide, error: UInt32):
        self.value = value
        self.error = error


struct Comparison(Copyable):
    var value: Bool
    var valid: Bool

    def __init__(out self, value: Bool, valid: Bool):
        self.value = value
        self.valid = valid


def divide(low: UInt64, high: UInt64, divisor: UInt64) -> Quotient:
    """Divide a magnitude with high < divisor <= 10**18, without ulong '/'."""
    var value = UInt64(0)
    var remainder = high
    for bit in range(63, -1, -1):
        remainder = (remainder << 1) | ((low >> UInt64(bit)) & 1)
        if remainder >= divisor:
            remainder -= divisor
            value |= UInt64(1) << UInt64(bit)
    return Quotient(value, remainder)


def mean(low: UInt64, high: UInt64, count: UInt64, scale: UInt32) -> UInt64:
    """Decimal32/64 mean, retaining the CPU's exact conversion sequence."""
    if count == 0:
        return 0
    var negative = (high >> 63) != 0
    var magnitude = Wide(low, high)
    if negative:
        magnitude = negate(magnitude)
    var per_row = divide(magnitude.low, magnitude.high, count)
    var unit = UInt64(1)
    for _ in range(Int(scale)):
        unit *= 10
    var parts = divide(per_row.value, 0, unit)
    var fractional = f.add(
        f.from_unsigned(parts.remainder),
        f.div(f.from_unsigned(per_row.remainder), f.from_unsigned(count)),
    )
    var value = f.add(
        f.from_unsigned(parts.value),
        f.div(fractional, f.from_unsigned(unit)),
    )
    return value ^ f.SIGN if negative else value


def wide_ge(a: Wide, b: Wide) -> Bool:
    return a.high > b.high or (a.high == b.high and a.low >= b.low)


def wide_equal(a: Wide, b: Wide) -> Bool:
    return a.low == b.low and a.high == b.high


def wide_subtract(a: Wide, b: Wide) -> Wide:
    return Wide(a.low - b.low, a.high - b.high - UInt64(a.low < b.low))


def wide_divide(numerator: Wide, divisor: Wide) -> Division:
    """Divide magnitudes; nonzero divisor is below 2**127 for decimals.

    A remainder smaller than the divisor fits in the two-word state after
    shifting. The numerator may occupy all 128 bits.
    """
    var quotient = Wide(0)
    var remainder = Wide(0)
    for bit in range(127, -1, -1):
        var incoming = (
            numerator.high >> UInt64(bit - 64)
        ) & 1 if bit >= 64 else (numerator.low >> UInt64(bit)) & 1
        remainder = Wide(
            (remainder.low << 1) | incoming,
            (remainder.high << 1) | (remainder.low >> 63),
        )
        if wide_ge(remainder, divisor):
            remainder = wide_subtract(remainder, divisor)
            if bit >= 64:
                quotient.high |= UInt64(1) << UInt64(bit - 64)
            else:
                quotient.low |= UInt64(1) << UInt64(bit)
    return Division(quotient, remainder)


def wide_pow10(scale: UInt32) -> Wide:
    var value = Wide(1)
    for _ in range(Int(scale)):
        value = Wide(
            value.low * 10,
            value.high * 10 + f.product(value.low, 10)[0],
        )
    return value


def negate(value: Wide) -> Wide:
    var low = ~value.low + 1
    return Wide(low, ~value.high + UInt64(low == 0))


def magnitude(value: Wide) -> Wide:
    return negate(value) if value.high >> 63 else value


def increment(value: Wide) -> Wide:
    var low = value.low + 1
    return Wide(low, value.high + UInt64(low == 0))


def half_even(value: Wide, scale: UInt32) -> Wide:
    var negative = (value.high >> 63) != 0
    var unit = wide_pow10(scale)
    var divided = wide_divide(magnitude(value), unit)
    var rest = wide_subtract(unit, divided.remainder)
    if wide_ge(divided.remainder, rest) and (
        not wide_equal(divided.remainder, rest) or (divided.value.low & 1) != 0
    ):
        divided.value = increment(divided.value)
    return negate(divided.value) if negative else divided.value


def scale_checked(value: Wide, scale: UInt32) -> Cast:
    if scale == 0:
        return Cast(value, True)
    var factor = wide_pow10(scale)
    var permitted = wide_divide(
        Wide(UInt64.MAX, 0x7FFFFFFFFFFFFFFF), factor
    ).value
    var absolute = magnitude(value)
    if wide_ge(absolute, permitted) and not wide_equal(absolute, permitted):
        return Cast(Wide(0), False)
    var scaled = value
    for _ in range(Int(scale)):
        scaled = Wide(
            scaled.low * 10,
            scaled.high * 10 + f.product(scaled.low, 10)[0],
        )
    return Cast(scaled, True)


def add_subtract(
    left: Wide,
    right: Wide,
    left_scale: UInt32,
    right_scale: UInt32,
    subtract: Bool,
) -> Arithmetic:
    var common = max(left_scale, right_scale)
    var a = scale_checked(left, common - left_scale)
    var b = scale_checked(right, common - right_scale)
    if not a.valid or not b.valid:
        return Arithmetic(Wide(0), 1)
    # CPU arithmetic wraps at Int128 before checking decimal precision.
    var value = wide_subtract(a.value, b.value) if subtract else wide_add(
        a.value, b.value
    )
    return Arithmetic(
        value, UInt32(2) if precision_overflow(value) else UInt32(0)
    )


def unary(value: Wide, bits: UInt32, absolute: Bool) -> Arithmetic:
    var minimum = (
        Wide(0, 0x8000000000000000) if bits
        == 128 else Wide(0x8000000000000000, UInt64.MAX) if bits
        == 64 else Wide(0xFFFFFFFF80000000, UInt64.MAX)
    )
    if wide_equal(value, minimum):
        return Arithmetic(Wide(0), 3)
    return Arithmetic(
        negate(value) if not absolute or (value.high >> 63) != 0 else value,
        0,
    )


def wide_add(a: Wide, b: Wide) -> Wide:
    var low = a.low + b.low
    return Wide(low, a.high + b.high + UInt64(low < a.low))


def wide_multiply(a: Wide, b: Wide) -> Wide:
    return Wide(
        a.low * b.low,
        a.high * b.low + a.low * b.high + f.product(a.low, b.low)[0],
    )


def signed_less(a: Wide, b: Wide) -> Bool:
    var sa = (a.high >> 63) != 0
    var sb = (b.high >> 63) != 0
    if sa != sb:
        return sa
    return a.high < b.high or (a.high == b.high and a.low < b.low)


def is_zero(value: Wide) -> Bool:
    return (value.low | value.high) == 0


def precision_overflow(value: Wide) -> Bool:
    return wide_ge(magnitude(value), wide_pow10(38))


def signed_divide(a: Wide, b: Wide) -> Division:
    """Return truncating signed quotient and floor-modulo signed remainder."""
    var sa = (a.high >> 63) != 0
    var sb = (b.high >> 63) != 0
    var divisor = magnitude(b)
    var parts = wide_divide(magnitude(a), divisor)
    if sa != sb:
        parts.value = negate(parts.value)
    if not is_zero(parts.remainder):
        if sa != sb:
            parts.remainder = wide_subtract(divisor, parts.remainder)
        if sb:
            parts.remainder = negate(parts.remainder)
    return parts^


def round_signed(value: Wide, remainder: Wide, divisor: Wide) -> Wide:
    var rest = wide_subtract(divisor, remainder)
    if signed_less(rest, remainder) or (
        wide_equal(rest, remainder) and (value.low & 1) != 0
    ):
        return increment(value)
    return value


def multiply(
    left: Wide, right: Wide, left_scale: UInt32, right_scale: UInt32
) -> Arithmetic:
    var common = max(left_scale, right_scale)
    if (
        not scale_checked(left, common - left_scale).valid
        or not scale_checked(right, common - right_scale).valid
    ):
        return Arithmetic(Wide(0), 1)
    # Preserve the CPU's signed bounds and abs(Int128.MIN) behavior.
    if wide_equal(left, Wide(0, 0x8000000000000000)):
        return Arithmetic(Wide(0), 1)
    if is_zero(left):
        return Arithmetic(Wide(0), 0)
    var maximum = Wide(0, 0x8000000000000000) if right.high >> 63 else Wide(
        UInt64.MAX, 0x7FFFFFFFFFFFFFFF
    )
    var permitted = wide_divide(maximum, magnitude(left)).value
    var absolute = magnitude(right)
    if wide_ge(absolute, permitted) and not wide_equal(absolute, permitted):
        return Arithmetic(Wide(0), 1)
    var value = wide_multiply(left, right)
    var scale = min(left_scale, right_scale)
    if scale:
        var minimum = wide_equal(value, Wide(0, 0x8000000000000000))
        if minimum and scale <= 18:
            value = Wide(0)
        else:
            value = half_even(value, scale)
            if minimum:
                value = negate(value)
    return Arithmetic(
        value, UInt32(2) if precision_overflow(value) else UInt32(0)
    )


def divide_rows(
    left: Wide, right: Wide, left_scale: UInt32, right_scale: UInt32
) -> Arithmetic:
    var common = max(left_scale, right_scale)
    if (
        not scale_checked(left, common - left_scale).valid
        or not scale_checked(right, common - right_scale).valid
    ):
        return Arithmetic(Wide(0), 1)
    if is_zero(right):
        return Arithmetic(Wide(0), 4)  # CPU null quotient.
    var negative = ((left.high ^ right.high) >> 63) != 0
    var numerator = magnitude(left)
    var divisor = magnitude(right)
    var parts = signed_divide(numerator, divisor)
    var value = parts.value
    var remainder = parts.remainder
    if precision_overflow(value):
        return Arithmetic(Wide(0), 2)
    var shift = common + right_scale - left_scale
    for _ in range(Int(shift)):
        value = wide_multiply(value, Wide(10))
        if precision_overflow(value):
            return Arithmetic(Wide(0), 2)
        remainder = wide_multiply(remainder, Wide(10))
        parts = signed_divide(remainder, divisor)
        value = wide_add(value, parts.value)
        remainder = parts.remainder
    value = round_signed(value, remainder, divisor)
    if precision_overflow(value):
        return Arithmetic(Wide(0), 2)
    return Arithmetic(negate(value) if negative else value, 0)


def compare(
    left: Wide,
    right: Wide,
    left_scale: UInt32,
    right_scale: UInt32,
    op: UInt32,
) -> Comparison:
    var common = max(left_scale, right_scale)
    var a = scale_checked(left, common - left_scale)
    var b = scale_checked(right, common - right_scale)
    if not a.valid or not b.valid:
        return Comparison(False, False)
    var equal = wide_equal(a.value, b.value)
    var less = signed_less(a.value, b.value)
    var value = not less
    if op == 9:
        value = equal
    elif op == 23:
        value = not equal
    elif op == 20:
        value = less
    elif op == 22:
        value = less or equal
    elif op == 8:
        value = not less and not equal
    return Comparison(value, True)


def cast(
    value: Wide,
    from_scale: UInt32,
    to_scale: UInt32,
    precision: UInt32,
    bits: UInt32,
    is_unsigned: Bool,
    is_boolean: Bool,
) -> Cast:
    var scaled = value
    if from_scale > to_scale:
        scaled = half_even(scaled, from_scale - to_scale)
    else:
        # CPU Int128 rescaling has a fixed-width intermediate.
        for _ in range(Int(from_scale), Int(to_scale)):
            scaled = Wide(
                scaled.low * 10,
                scaled.high * 10 + f.product(scaled.low, 10)[0],
            )
    var negative = (scaled.high >> 63) != 0
    var absolute = magnitude(scaled)
    if precision:
        return Cast(scaled, not wide_ge(absolute, wide_pow10(precision)))
    if is_boolean:
        return Cast(Wide(UInt64(not is_zero(absolute))), True)
    var maximum = (
        UInt64.MAX if bits == 64 else (UInt64(1) << UInt64(bits)) - 1
    ) if is_unsigned else (UInt64(1) << UInt64(bits - 1)) - UInt64(not negative)
    var valid = (
        absolute.high == 0
        and absolute.low <= maximum
        and not (is_unsigned and negative)
    )
    return Cast(scaled, valid)


def mean128(low: UInt64, high: UInt64, count: UInt64, scale: UInt32) -> UInt64:
    """Decimal128 mean with separate exact quotient/remainder conversions."""
    if count == 0:
        return 0
    var negative = (high >> 63) != 0
    var per_row = wide_divide(magnitude(Wide(low, high)), Wide(count))
    var unit = wide_pow10(scale)
    var parts = wide_divide(per_row.value, unit)
    var fractional = f.add(
        f.from_wide(parts.remainder.low, parts.remainder.high, True),
        f.div(f.from_unsigned(per_row.remainder.low), f.from_unsigned(count)),
    )
    var value = f.add(
        f.from_wide(parts.value.low, parts.value.high, True),
        f.div(fractional, f.from_wide(unit.low, unit.high, True)),
    )
    return value ^ f.SIGN if negative else value
