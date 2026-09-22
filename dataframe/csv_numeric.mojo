"""CSV-only numeric conversion paths modeled on Polars 1.44.2 dependencies.

`parse_csv_float32` and `parse_csv_float64` are deliberately separate from
dataframe.parse, so the clean CSV decoder can use fast-float2's direct
semantics without changing legacy cast behavior.

Algorithm translations derive from fast-float2 0.2.4 under its MIT option;
its retained text is in ``third_party/FAST_FLOAT2_LICENSE``.
"""
from std.memory import bitcast
from std.bit import count_leading_zeros
from std.builtin.globals import global_constant

from dataframe.csv_decimal import parse_csv_decimal
from std.collections.string._parsing_numbers.constants import (
    SMALLEST_POWER_OF_5,
    get_power_of_5,
)


comptime _F32_MANTISSA_BITS = 23
comptime _F32_MIN_EXPONENT = -127
comptime _F32_INFINITY_POWER = 255
comptime _F32_SMALLEST_POWER_OF_TEN = -65
comptime _F32_LARGEST_POWER_OF_TEN = 38
comptime _F32_MAX_EXACT_INTEGER = UInt64(1 << 24)
comptime _DECIMAL_MAX_DIGITS = 768


@fieldwise_init
struct _Product128:
    var low: UInt64
    var high: UInt64


@fieldwise_init
struct _Adjusted32:
    var mantissa: UInt64
    var power2: Int


@fieldwise_init
struct _ScannedNumber:
    var negative: Bool
    var mantissa: UInt64
    var exponent: Int
    var many_digits: Bool


@always_inline
def _ascii_lower(byte: UInt8) -> UInt8:
    return byte | UInt8(0x20)


# fast-float2 parse_inf_nan: return (none=0, nan=1, inf=2, negative).
def _scan_special(text: StringSlice) -> Tuple[Int, Bool]:
    var b = text.as_bytes()
    var start = 0
    var negative = False
    if len(b) > 0 and (b[0] == 43 or b[0] == 45):
        negative = b[0] == 45
        start = 1
    var n = len(b) - start
    if n == 3 and (
        _ascii_lower(b[start]) == 110
        and _ascii_lower(b[start + 1]) == 97
        and _ascii_lower(b[start + 2]) == 110
    ):
        return (1, negative)
    if n == 3 and (
        _ascii_lower(b[start]) == 105
        and _ascii_lower(b[start + 1]) == 110
        and _ascii_lower(b[start + 2]) == 102
    ):
        return (2, negative)
    if n == 8 and (
        _ascii_lower(b[start]) == 105
        and _ascii_lower(b[start + 1]) == 110
        and _ascii_lower(b[start + 2]) == 102
        and _ascii_lower(b[start + 3]) == 105
        and _ascii_lower(b[start + 4]) == 110
        and _ascii_lower(b[start + 5]) == 105
        and _ascii_lower(b[start + 6]) == 116
        and _ascii_lower(b[start + 7]) == 121
    ):
        return (2, negative)
    return (0, False)


# fast-float2 number.rs::parse_number translated for a strict CSV field.
# It builds the common <=19-digit mantissa during its first grammar pass.  Only
# the source algorithm's many-digit path reparses the first 19 digits.
@always_inline
def _is_decimal_digit(byte: UInt8) -> Bool:
    return byte >= 48 and byte <= 57


@always_inline
def _parse_8digits(word: UInt64) -> UInt64:
    # fast-float2 number.rs::parse_8digits.  The caller has already established
    # that every byte is an ASCII digit and that this 8-byte load is in bounds.
    var value = word - UInt64(0x3030303030303030)
    value = value * 10 + (value >> 8)
    var left = (value & UInt64(0x000000FF000000FF)) * UInt64(0x000F424000000064)
    var right = ((value >> 16) & UInt64(0x000000FF000000FF)) * UInt64(
        0x0000271000000001
    )
    return UInt64(UInt32((left + right) >> 32))


@always_inline
def _try_parse_8digits(
    bytes: Span[UInt8, _], start: Int, mantissa: UInt64
) -> Tuple[Int, UInt64]:
    # number.rs::try_parse_8digits uses a bounds-checked unaligned load, then
    # its packed 8-digit reduction.  Returning zero leaves scalar parsing at
    # the first non-eight-digit group, matching the Rust cursor behavior.
    if start + 8 > len(bytes):
        return (0, mantissa)
    var word = bitcast[DType.uint64, 1](
        bytes.unsafe_ptr().unsafe_load[width=8](start)
    )
    if (
        (word + UInt64(0x4646464646464646))
        | (word - UInt64(0x3030303030303030))
    ) & UInt64(0x8080808080808080) != 0:
        return (0, mantissa)
    return (8, mantissa * UInt64(100000000) + _parse_8digits(word))


@always_inline
def _parse_to_19(
    bytes: Span[UInt8, _], start: Int, mantissa: UInt64
) -> Tuple[Int, UInt64]:
    # number.rs::try_parse_19digits: stop as soon as 19 digits are present.
    var i = start
    var value = mantissa
    while i < len(bytes) and value < UInt64(1000000000000000000):
        var byte = bytes[i]
        if not _is_decimal_digit(byte):
            break
        value = value * 10 + UInt64(byte - 48)
        i += 1
    return (i, value)


def _scan_csv_number(text: StringSlice) -> Tuple[Bool, _ScannedNumber]:
    var bytes = text.as_bytes()
    if len(bytes) == 0:
        return (False, _ScannedNumber(False, 0, 0, False))

    # parse_number: optional sign, followed by the integer component.
    var i = 0
    var negative = False
    if bytes[0] == 45:
        negative = True
        i = 1
        if i == len(bytes):
            return (False, _ScannedNumber(False, 0, 0, False))
    elif bytes[0] == 43:
        i = 1
        if i == len(bytes):
            return (False, _ScannedNumber(False, 0, 0, False))

    var mantissa = UInt64(0)
    var digits_start = i
    while i < len(bytes) and _is_decimal_digit(bytes[i]):
        # number.rs::try_parse_digits deliberately wraps here.  The uncommon
        # many-digit branch below discards it and derives a bounded mantissa.
        mantissa = mantissa * 10 + UInt64(bytes[i] - 48)
        i += 1
    var integer_digits = i - digits_start
    var int_end = i

    # parse_number's fractional branch uses up to two packed 8-byte groups
    # before its scalar cursor.  The scalar tail also validates full CSV input.
    var fractional_digits = 0
    var exponent = 0
    if i < len(bytes) and bytes[i] == 46:
        i += 1
        var fractional_start = i
        var block = _try_parse_8digits(bytes, i, mantissa)
        i += block[0]
        mantissa = block[1]
        if block[0] == 8:
            block = _try_parse_8digits(bytes, i, mantissa)
            i += block[0]
            mantissa = block[1]
        while i < len(bytes) and _is_decimal_digit(bytes[i]):
            mantissa = mantissa * 10 + UInt64(bytes[i] - 48)
            i += 1
        fractional_digits = i - fractional_start
        exponent = -fractional_digits

    var digits = integer_digits + fractional_digits
    if digits == 0:
        return (False, _ScannedNumber(False, 0, 0, False))

    # number.rs::parse_scientific, with strict full consumption rather than its
    # prefix-accepting public parse contract.
    var explicit = 0
    if i < len(bytes) and (bytes[i] == 101 or bytes[i] == 69):
        i += 1
        var explicit_negative = False
        if i < len(bytes) and (bytes[i] == 45 or bytes[i] == 43):
            explicit_negative = bytes[i] == 45
            i += 1
        var exp_digits = 0
        while i < len(bytes) and _is_decimal_digit(bytes[i]):
            if explicit < 65536:
                explicit = explicit * 10 + Int(bytes[i] - 48)
            exp_digits += 1
            i += 1
        if exp_digits == 0:
            return (False, _ScannedNumber(False, 0, 0, False))
        if explicit_negative:
            explicit = -explicit
        exponent += explicit
    if i != len(bytes):
        return (False, _ScannedNumber(False, 0, 0, False))

    if digits <= 19:
        return (True, _ScannedNumber(negative, mantissa, exponent, False))

    # Exact fast-float2 uncommon path.  Leading zeroes and a decimal point
    # reduce the count before the 19-digit reparse, as in number.rs.
    var excess = digits - 19
    var p = digits_start
    while p < len(bytes) and (bytes[p] == 48 or bytes[p] == 46):
        if bytes[p] == 48:
            excess -= 1
        p += 1
    if excess <= 0:
        return (True, _ScannedNumber(negative, mantissa, exponent, False))

    mantissa = 0
    var parsed = _parse_to_19(bytes, digits_start, mantissa)
    i = parsed[0]
    mantissa = parsed[1]
    if mantissa >= UInt64(1000000000000000000):
        # int_end.offset_from(cursor): digits remaining before the decimal.
        exponent = int_end - i
    else:
        if i >= len(bytes) or bytes[i] != 46:
            return (False, _ScannedNumber(False, 0, 0, False))
        i += 1
        var fraction_start = i
        parsed = _parse_to_19(bytes, i, mantissa)
        i = parsed[0]
        mantissa = parsed[1]
        exponent = -(i - fraction_start)
    exponent += explicit
    return (True, _ScannedNumber(negative, mantissa, exponent, True))


@always_inline
def _float32_from_bits(bits: UInt32) -> Float32:
    return bitcast[DType.float32](bits)


@always_inline
def _float32_zero(negative: Bool) -> Float32:
    return _float32_from_bits(UInt32(0x80000000) if negative else UInt32(0))


@always_inline
def _float32_inf(negative: Bool) -> Float32:
    return _float32_from_bits(
        UInt32(0xFF800000) if negative else UInt32(0x7F800000)
    )


@always_inline
def _float32_nan() -> Float32:
    return _float32_from_bits(UInt32(0x7FC00000))


comptime _POW10_F32: SIMD[DType.float32, 16] = SIMD[DType.float32, 16](
    1e0,
    1e1,
    1e2,
    1e3,
    1e4,
    1e5,
    1e6,
    1e7,
    1e8,
    1e9,
    1e10,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
comptime _POW10_F64: SIMD[DType.float64, 32] = SIMD[DType.float64, 32](
    1e0,
    1e1,
    1e2,
    1e3,
    1e4,
    1e5,
    1e6,
    1e7,
    1e8,
    1e9,
    1e10,
    1e11,
    1e12,
    1e13,
    1e14,
    1e15,
    1e16,
    1e17,
    1e18,
    1e19,
    1e20,
    1e21,
    1e22,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
comptime _INT_POW10: SIMD[DType.uint64, 16] = SIMD[DType.uint64, 16](
    1,
    10,
    100,
    1000,
    10000,
    100000,
    1000000,
    10000000,
    100000000,
    1000000000,
    10000000000,
    100000000000,
    1000000000000,
    10000000000000,
    100000000000000,
    1000000000000000,
)


@always_inline
def _pow10_f32(k: Int) -> Float32:
    return global_constant[_POW10_F32]()[k & 15]


@always_inline
def _pow10_f64(k: Int) -> Float64:
    return global_constant[_POW10_F64]()[k & 31]


@always_inline
def _int_pow10(k: Int) -> UInt64:
    return global_constant[_INT_POW10]()[k]


@always_inline
def _power(q: Int) -> Int:
    return ((q * 217706) >> 16) + 63


@always_inline
def _full_multiply(a: UInt64, b: UInt64) -> _Product128:
    var product = UInt128(a) * UInt128(b)
    return _Product128(UInt64(product), UInt64(product >> 64))


@always_inline
def _product_approx(q: Int, w: UInt64) -> _Product128:
    # fast-float2 binary::compute_product_approx, using the same power table
    # shipped by the installed Mojo Lemire implementation.
    var index = 2 * (q - SMALLEST_POWER_OF_5)
    var first = _full_multiply(w, get_power_of_5(index))
    var precision_mask = UInt64(0xFFFFFFFFFFFFFFFF) >> UInt64(
        _F32_MANTISSA_BITS + 3
    )
    if (first.high & precision_mask) == precision_mask:
        var second = _full_multiply(w, get_power_of_5(index + 1))
        first.low += second.high
        if second.high > first.low:
            first.high += 1
    return first^


@always_inline
def _make_float32(mantissa: UInt64, exponent: Int) -> Float32:
    var bits = UInt32(mantissa & UInt64((1 << _F32_MANTISSA_BITS) - 1))
    bits |= UInt32(exponent + 127) << 23
    return _float32_from_bits(bits)


@always_inline
def _compute_float32(q: Int, input: UInt64) -> _Adjusted32:
    # Direct translation of fast-float2 0.2.4 binary::compute_float for f32.
    if input == 0 or q < _F32_SMALLEST_POWER_OF_TEN:
        return _Adjusted32(0, 0)
    if q > _F32_LARGEST_POWER_OF_TEN:
        return _Adjusted32(0, _F32_INFINITY_POWER)
    var lz = Int(count_leading_zeros(input))
    var w = input << UInt64(lz)
    var product = _product_approx(q, w)
    if product.low == UInt64(0xFFFFFFFFFFFFFFFF) and (q < -27 or q > 55):
        return _Adjusted32(0, -1)
    var upper = Int(product.high >> 63)
    var shift = upper + 64 - _F32_MANTISSA_BITS - 3
    var mantissa = product.high >> UInt64(shift)
    var power2 = _power(q) + upper - lz - _F32_MIN_EXPONENT
    if power2 <= 0:
        if -power2 + 1 >= 64:
            return _Adjusted32(0, 0)
        mantissa >>= UInt64(-power2 + 1)
        mantissa += mantissa & 1
        mantissa >>= 1
        power2 = 1 if mantissa >= UInt64(1 << _F32_MANTISSA_BITS) else 0
        return _Adjusted32(mantissa, power2)
    if (
        product.low <= 1
        and q >= -17
        and q <= 10
        and (mantissa & 3) == 1
        and (mantissa << UInt64(shift)) == product.high
    ):
        mantissa &= ~UInt64(1)
    mantissa += mantissa & 1
    mantissa >>= 1
    if mantissa >= UInt64(2 << _F32_MANTISSA_BITS):
        mantissa = UInt64(1 << _F32_MANTISSA_BITS)
        power2 += 1
    mantissa &= ~UInt64(1 << _F32_MANTISSA_BITS)
    if power2 >= _F32_INFINITY_POWER:
        return _Adjusted32(0, _F32_INFINITY_POWER)
    return _Adjusted32(mantissa, power2)


@always_inline
def _try_fast_float32(
    mantissa: UInt64, exponent: Int, negative: Bool, many_digits: Bool
) -> Tuple[Bool, Float32]:
    # number.rs::Number::try_fast_path for Float32: one guarded attempt,
    # including the source checked-multiply disguised path.
    if (
        many_digits
        or mantissa > _F32_MAX_EXACT_INTEGER
        or exponent < -10
        or exponent > 17
    ):
        return (False, Float32(0))
    var value = Float32(mantissa)
    if exponent <= 10:
        if exponent < 0:
            value /= _pow10_f32(-exponent)
        else:
            value *= _pow10_f32(exponent)
    else:
        var product = UInt128(mantissa) * UInt128(_int_pow10(exponent - 10))
        if product > UInt128(_F32_MAX_EXACT_INTEGER):
            return (False, Float32(0))
        value = Float32(UInt64(product)) * _pow10_f32(10)
    return (True, -value if negative else value)


comptime _SIMPLE_SHIFTS: SIMD[DType.uint8, 32] = SIMD[DType.uint8, 32](
    0,
    3,
    6,
    9,
    13,
    16,
    19,
    23,
    26,
    29,
    33,
    36,
    39,
    43,
    46,
    49,
    53,
    56,
    59,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
)


@always_inline
def _shift_amount(n: Int) -> Int:
    # simple.rs::{POWERS,MAX_SHIFT}: table for 0..18, otherwise 60.
    if n < 19:
        return Int(global_constant[_SIMPLE_SHIFTS]()[n])
    return 60


def _long_float32(text: StringSlice) raises -> Float32:
    # Direct simple::parse_long_mantissa control flow. CsvDecimal owns the
    # matching fast-float2 batched decimal shifts and fixed shift metadata.
    var negative = text.as_bytes()[0] == 45
    var d = parse_csv_decimal(text)
    if d.num_digits == 0 or d.decimal_point < -324:
        return _float32_zero(negative)
    if d.decimal_point >= 310:
        return _float32_inf(negative)
    var exp2 = 0
    while d.decimal_point > 0:
        var shift = _shift_amount(d.decimal_point)
        d.right_shift(shift)
        if d.decimal_point < -2047:
            return _float32_zero(negative)
        exp2 += shift
    while d.decimal_point <= 0:
        var shift = _shift_amount(-d.decimal_point)
        if d.decimal_point == 0:
            if d.digits[0] >= 5:
                break
            shift = 2 if d.digits[0] <= 1 else 1
        d.left_shift(shift)
        if d.decimal_point > 2047:
            return _float32_inf(negative)
        exp2 -= shift
    exp2 -= 1
    while _F32_MIN_EXPONENT + 1 > exp2:
        var shift = min(_F32_MIN_EXPONENT + 1 - exp2, 60)
        d.right_shift(shift)
        exp2 += shift
    if exp2 - _F32_MIN_EXPONENT >= _F32_INFINITY_POWER:
        return _float32_inf(negative)
    d.left_shift(_F32_MANTISSA_BITS + 1)
    var mantissa = d.round()
    if mantissa >= UInt64(1 << (_F32_MANTISSA_BITS + 1)):
        d.right_shift(1)
        exp2 += 1
        mantissa = d.round()
        if exp2 - _F32_MIN_EXPONENT >= _F32_INFINITY_POWER:
            return _float32_inf(negative)
    var power2 = exp2 - _F32_MIN_EXPONENT
    if mantissa < UInt64(1 << _F32_MANTISSA_BITS):
        power2 -= 1
    mantissa &= UInt64((1 << _F32_MANTISSA_BITS) - 1)
    var value = _make_float32(mantissa, power2 - 127)
    return -value if negative else value


def parse_csv_float32(text: StringSlice) raises -> Float32:
    """Strict CSV Float32 without a Float64 or standard-parser fallback."""
    var scanned = _scan_csv_number(text)
    if not scanned[0]:
        var special = _scan_special(text)
        if special[0] == 1:
            return _float32_from_bits(
                UInt32(0xFFC00000) if special[1] else UInt32(0x7FC00000)
            )
        if special[0] == 2:
            return _float32_inf(special[1])
        raise Error("invalid floating-point value")
    var number = _ScannedNumber(
        scanned[1].negative,
        scanned[1].mantissa,
        scanned[1].exponent,
        scanned[1].many_digits,
    )
    var fast = _try_fast_float32(
        number.mantissa, number.exponent, number.negative, number.many_digits
    )
    if fast[0]:
        return fast[1]
    var adjusted = _compute_float32(number.exponent, number.mantissa)
    var needs_long = adjusted.power2 < 0
    if number.many_digits:
        var next = _compute_float32(number.exponent, number.mantissa + 1)
        needs_long = (
            needs_long
            or adjusted.mantissa != next.mantissa
            or adjusted.power2 != next.power2
        )
    if needs_long:
        return _long_float32(text)
    var value = _make_float32(adjusted.mantissa, adjusted.power2 - 127)
    return -value if number.negative else value


# Algorithm source: fast-float2 0.2.4, pinned by Polars 1.44.2. The direct
# translation below follows number.rs, binary.rs, and simple.rs for Float64.
# CsvDecimal supplies simple.rs's batched decimal shifts and metadata. See the
# retained third_party license.
comptime _F64_MANTISSA_BITS = 52
comptime _F64_MIN_EXPONENT = -1023
comptime _F64_INFINITY_POWER = 2047
comptime _F64_SMALLEST_POWER_OF_TEN = -342
comptime _F64_LARGEST_POWER_OF_TEN = 308
comptime _F64_MAX_EXACT_INTEGER = UInt64(1 << 53)


@fieldwise_init
struct _Adjusted64:
    var mantissa: UInt64
    var power2: Int


@always_inline
def _float64_from_bits(bits: UInt64) -> Float64:
    return bitcast[DType.float64](bits)


@always_inline
def _float64_zero(negative: Bool) -> Float64:
    return _float64_from_bits(UInt64(1 << 63) if negative else UInt64(0))


@always_inline
def _float64_inf(negative: Bool) -> Float64:
    return _float64_from_bits(
        UInt64(0xFFF0000000000000) if negative else UInt64(0x7FF0000000000000)
    )


@always_inline
def _float64_nan() -> Float64:
    return _float64_from_bits(UInt64(0x7FF8000000000000))


@always_inline
def _product_approx64(q: Int, w: UInt64) -> _Product128:
    var index = 2 * (q - SMALLEST_POWER_OF_5)
    var first = _full_multiply(w, get_power_of_5(index))
    var precision_mask = UInt64(0xFFFFFFFFFFFFFFFF) >> UInt64(
        _F64_MANTISSA_BITS + 3
    )
    if (first.high & precision_mask) == precision_mask:
        var second = _full_multiply(w, get_power_of_5(index + 1))
        first.low += second.high
        if second.high > first.low:
            first.high += 1
    return first^


@always_inline
def _make_float64(mantissa: UInt64, exponent: Int) -> Float64:
    var bits = mantissa & UInt64((1 << _F64_MANTISSA_BITS) - 1)
    bits |= UInt64(exponent + 1023) << 52
    return _float64_from_bits(bits)


@always_inline
def _compute_float64(q: Int, input: UInt64) -> _Adjusted64:
    if input == 0 or q < _F64_SMALLEST_POWER_OF_TEN:
        return _Adjusted64(0, 0)
    if q > _F64_LARGEST_POWER_OF_TEN:
        return _Adjusted64(0, _F64_INFINITY_POWER)
    var lz = Int(count_leading_zeros(input))
    var w = input << UInt64(lz)
    var product = _product_approx64(q, w)
    if product.low == UInt64(0xFFFFFFFFFFFFFFFF) and (q < -27 or q > 55):
        return _Adjusted64(0, -1)
    var upper = Int(product.high >> 63)
    var shift = upper + 64 - _F64_MANTISSA_BITS - 3
    var mantissa = product.high >> UInt64(shift)
    var power2 = _power(q) + upper - lz - _F64_MIN_EXPONENT
    if power2 <= 0:
        if -power2 + 1 >= 64:
            return _Adjusted64(0, 0)
        mantissa >>= UInt64(-power2 + 1)
        mantissa += mantissa & 1
        mantissa >>= 1
        power2 = 1 if mantissa >= UInt64(1 << _F64_MANTISSA_BITS) else 0
        return _Adjusted64(mantissa, power2)
    if (
        product.low <= 1
        and q >= -4
        and q <= 23
        and (mantissa & 3) == 1
        and (mantissa << UInt64(shift)) == product.high
    ):
        mantissa &= ~UInt64(1)
    mantissa += mantissa & 1
    mantissa >>= 1
    if mantissa >= UInt64(2 << _F64_MANTISSA_BITS):
        mantissa = UInt64(1 << _F64_MANTISSA_BITS)
        power2 += 1
    mantissa &= ~UInt64(1 << _F64_MANTISSA_BITS)
    if power2 >= _F64_INFINITY_POWER:
        return _Adjusted64(0, _F64_INFINITY_POWER)
    return _Adjusted64(mantissa, power2)


@always_inline
def _try_fast_float64(
    mantissa: UInt64, exponent: Int, negative: Bool, many_digits: Bool
) -> Tuple[Bool, Float64]:
    # number.rs::Number::try_fast_path for Float64, including checked disguise.
    if (
        many_digits
        or mantissa > _F64_MAX_EXACT_INTEGER
        or exponent < -22
        or exponent > 37
    ):
        return (False, Float64(0))
    var value = Float64(mantissa)
    if exponent <= 22:
        if exponent < 0:
            value /= _pow10_f64(-exponent)
        else:
            value *= _pow10_f64(exponent)
    else:
        var product = UInt128(mantissa) * UInt128(_int_pow10(exponent - 22))
        if product > UInt128(_F64_MAX_EXACT_INTEGER):
            return (False, Float64(0))
        value = Float64(UInt64(product)) * _pow10_f64(22)
    return (True, -value if negative else value)


def _long_float64(text: StringSlice) raises -> Float64:
    # Direct simple::parse_long_mantissa control flow through CsvDecimal.
    var negative = text.as_bytes()[0] == 45
    var d = parse_csv_decimal(text)
    if d.num_digits == 0 or d.decimal_point < -324:
        return _float64_zero(negative)
    if d.decimal_point >= 310:
        return _float64_inf(negative)
    var exp2 = 0
    while d.decimal_point > 0:
        var shift = _shift_amount(d.decimal_point)
        d.right_shift(shift)
        if d.decimal_point < -2047:
            return _float64_zero(negative)
        exp2 += shift
    while d.decimal_point <= 0:
        var shift = _shift_amount(-d.decimal_point)
        if d.decimal_point == 0:
            if d.digits[0] >= 5:
                break
            shift = 2 if d.digits[0] <= 1 else 1
        d.left_shift(shift)
        if d.decimal_point > 2047:
            return _float64_inf(negative)
        exp2 -= shift
    exp2 -= 1
    while _F64_MIN_EXPONENT + 1 > exp2:
        var shift = min(_F64_MIN_EXPONENT + 1 - exp2, 60)
        d.right_shift(shift)
        exp2 += shift
    if exp2 - _F64_MIN_EXPONENT >= _F64_INFINITY_POWER:
        return _float64_inf(negative)
    d.left_shift(_F64_MANTISSA_BITS + 1)
    var mantissa = d.round()
    if mantissa >= UInt64(1 << (_F64_MANTISSA_BITS + 1)):
        d.right_shift(1)
        exp2 += 1
        mantissa = d.round()
        if exp2 - _F64_MIN_EXPONENT >= _F64_INFINITY_POWER:
            return _float64_inf(negative)
    var power2 = exp2 - _F64_MIN_EXPONENT
    if mantissa < UInt64(1 << _F64_MANTISSA_BITS):
        power2 -= 1
    mantissa &= UInt64((1 << _F64_MANTISSA_BITS) - 1)
    var value = _make_float64(mantissa, power2 - 1023)
    return -value if negative else value


def parse_csv_float64(text: StringSlice) raises -> Float64:
    """Strict CSV Float64 without a standard-parser fallback."""
    var scanned = _scan_csv_number(text)
    if not scanned[0]:
        var special = _scan_special(text)
        if special[0] == 1:
            return _float64_from_bits(
                UInt64(0xFFF8000000000000) if special[1] else UInt64(
                    0x7FF8000000000000
                )
            )
        if special[0] == 2:
            return _float64_inf(special[1])
        raise Error("invalid floating-point value")
    var number = _ScannedNumber(
        scanned[1].negative,
        scanned[1].mantissa,
        scanned[1].exponent,
        scanned[1].many_digits,
    )
    var fast = _try_fast_float64(
        number.mantissa, number.exponent, number.negative, number.many_digits
    )
    if fast[0]:
        return fast[1]
    var adjusted = _compute_float64(number.exponent, number.mantissa)
    var needs_long = adjusted.power2 < 0
    if number.many_digits:
        var next = _compute_float64(number.exponent, number.mantissa + 1)
        needs_long = (
            needs_long
            or adjusted.mantissa != next.mantissa
            or adjusted.power2 != next.power2
        )
    if needs_long:
        return _long_float64(text)
    var value = _make_float64(adjusted.mantissa, adjusted.power2 - 1023)
    return -value if number.negative else value
