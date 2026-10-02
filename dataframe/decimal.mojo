"""Fixed-point decimal128 parsing, formatting, and arithmetic helpers."""
from .dtype import DataType


@always_inline
def pow10(scale: Int) -> Int128:
    """10**scale (1 for scale <= 0). Built from 10**18 steps and a short
    64-bit loop: at most two 128-bit multiplies, where a loop of 38 of them
    for `check_precision`'s 10**38 was most of a decimal sum (#385)."""
    var value = Int128(1)
    var left = scale
    while left >= 18:
        value *= Int128(1_000_000_000_000_000_000)
        left -= 18
    var small = Int64(1)
    for _ in range(left):
        small *= 10
    return value * Int128(small)


def precision_limit(precision: Int) -> Int128:
    return pow10(precision)


def check_precision(value: Int128, dtype: DataType) raises -> Int128:
    return check_limit(value, precision_limit(dtype.precision()), dtype)


@always_inline
def check_limit(value: Int128, limit: Int128, dtype: DataType) raises -> Int128:
    """`check_precision` with `limit` (`precision_limit` of `dtype`) read
    once per column by the caller, not computed again for every row."""
    if value <= -limit or value >= limit:
        raise Error(
            "value exceeds decimal precision " + String(dtype.precision())
        )
    return value


def parse_decimal(text: StringSlice, dtype: DataType) raises -> Int128:
    """Parse plain decimal text exactly at dtype.scale(); excess digits fail."""
    var bytes = text.as_bytes()
    if len(bytes) == 0:
        raise Error("empty decimal value")
    var i = 0
    var negative = False
    if bytes[0] == 45 or bytes[0] == 43:
        negative = bytes[0] == 45
        i = 1
    var value = Int128(0)
    var digits = 0
    var fraction = -1
    while i < len(bytes):
        var byte = bytes[i]
        if byte == 46 and fraction < 0:
            fraction = 0
            i += 1
            continue
        if byte < 48 or byte > 57:
            raise Error("invalid decimal value '" + String(text) + "'")
        if fraction >= 0:
            fraction += 1
            if fraction > dtype.scale():
                raise Error(
                    "decimal has more than "
                    + String(dtype.scale())
                    + " fractional digits"
                )
        value = value * 10 + Int128(byte - 48)
        digits += 1
        i += 1
    if digits == 0:
        raise Error("invalid decimal value '" + String(text) + "'")
    var seen_scale = max(fraction, 0)
    value *= pow10(dtype.scale() - seen_scale)
    if negative:
        value = -value
    return check_precision(value, dtype)


def format_decimal(value: Int128, scale: Int) -> String:
    var negative = value < 0
    var magnitude = -value if negative else value
    var digits = String(magnitude)
    if scale == 0:
        return ("-" if negative else "") + digits
    while digits.byte_length() <= scale:
        digits = "0" + digits
    var split = digits.byte_length() - scale
    return (
        ("-" if negative else "")
        + String(digits[byte=0:split])
        + "."
        + String(digits[byte = split : digits.byte_length()])
    )


def decimal_mean(total: Int128, count: Int64, scale: Int) -> Float64:
    """The mean of `count` decimals whose unscaled sum is `total`, as the
    nearest Float64 up to the final addition (#341).

    Dividing first by the count and then by 10^scale, in Int128 and keeping
    both remainders, never forms count x 10^scale (which overflows Int128
    at large scales) and rounds only when the parts are converted and added.
    Converting `total` to Float64 first would lose digits above 2^53.
    """
    # Work on the magnitude so the whole and fractional parts share a sign;
    # floor division of a negative total would subtract them and cancel
    # digits. Int128.MIN has no magnitude, but check_precision keeps sums
    # within 38 digits, far from it.
    var negative = total < 0
    var magnitude = -total if negative else total
    var divisor = Int128(count)
    var per_row = magnitude // divisor
    var left = magnitude - per_row * divisor
    var unit = Int128(1)
    for _ in range(scale):
        unit *= 10
    var whole = per_row // unit
    var digits = per_row - whole * unit
    var mean = Float64(whole) + (
        Float64(digits) + Float64(left) / Float64(count)
    ) / Float64(unit)
    return -mean if negative else mean


def round_half_even(
    magnitude: Int128, remainder: Int128, divisor: Int128
) -> Int128:
    """The quotient `magnitude` rounded half to even, given the nonnegative
    `remainder` left over from dividing by `divisor` (#342). Polars rounds
    decimal products, quotients and scale-reducing casts this way."""
    var rest = (
        divisor - remainder
    )  # compared without doubling, which could overflow
    if remainder > rest or (remainder == rest and magnitude % 2 == 1):
        return magnitude + 1
    return magnitude


def divide_half_even(numerator: Int128, divisor: Int128) -> Int128:
    """The quotient numerator / divisor rounded half to even; divisor must be positive.
    Works on the magnitude, so a negative tie rounds like a positive one
    (-0.625 to -0.62, as 0.625 to 0.62)."""
    var negative = numerator < 0
    var magnitude = -numerator if negative else numerator
    # Both fit 64 bits: one hardware division, not a 128-bit library call.
    if magnitude <= Int128(Int64.MAX) and divisor <= Int128(Int64.MAX):
        var m = Int64(magnitude)
        var d = Int64(divisor)
        var q = m // d
        var r = m - q * d
        var rest = d - r
        if r > rest or (r == rest and q % 2 == 1):
            q += 1
        return Int128(-q) if negative else Int128(q)
    var quotient = magnitude // divisor
    var rounded = round_half_even(
        quotient, magnitude - quotient * divisor, divisor
    )
    return -rounded if negative else rounded
