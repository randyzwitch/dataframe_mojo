#ifndef DFM_DECIMAL_SOURCE_H
#define DFM_DECIMAL_SOURCE_H

// Shared integer-only C++/MSL helpers for the CPU decimal mean contract.
static const char *dfm_decimal_source = R"metal(
struct DecimalQuotient { ulong value; ulong remainder; };
// The callers prove high < divisor and divisor <= 10**18. A shifted
// remainder therefore fits one word. Avoid native variable ulong division,
// which is not reliable in the M1 grouped pipeline compiler.
inline DecimalQuotient decimal_divide(ulong low, ulong high, ulong divisor) {
  ulong value = 0, remainder = high;
  for (int bit = 63; bit >= 0; --bit) {
    remainder = (remainder << 1) | ((low >> uint(bit)) & 1UL);
    if (remainder >= divisor) {
      remainder -= divisor;
      value |= 1UL << uint(bit);
    }
  }
  return {value, remainder};
}
// Decimal32/64 means use a signed exact total with magnitude <= count*2**63.
// Divide before conversion, retaining both remainders. This preserves the
// exact Float64 conversion/division/addition order of CPU decimal_mean.
inline ulong f64_decimal_mean(ulong low, ulong high, ulong count, uint scale) {
  if (!count) return 0;
  bool negative = (high >> 63) != 0;
  if (negative) {
    low = ~low + 1UL;
    high = ~high + ulong(low == 0);
  }
  DecimalQuotient per_row = decimal_divide(low, high, count);
  ulong unit = 1;
  for (uint i = 0; i < scale; ++i) unit *= 10UL;
  DecimalQuotient parts = decimal_divide(per_row.value, 0, unit);
  ulong fractional = f64_add(f64_from_unsigned(parts.remainder),
                            f64_div(f64_from_unsigned(per_row.remainder),
                                    f64_from_unsigned(count)));
  ulong value = f64_add(f64_from_unsigned(parts.value),
                       f64_div(fractional, f64_from_unsigned(unit)));
  return negative ? value ^ 0x8000000000000000UL : value;
}

struct DecimalWideDivision { ulong2 value; ulong2 remainder; };
inline bool decimal_wide_ge(ulong2 a, ulong2 b) {
  return a.y > b.y || (a.y == b.y && a.x >= b.x);
}
inline ulong2 decimal_wide_subtract(ulong2 a, ulong2 b) {
  ulong low = a.x - b.x;
  return ulong2(low, a.y - b.y - ulong(a.x < b.x));
}
// Divide a nonnegative 128-bit value by a nonzero 128-bit value. Decimal128
// precision keeps every intermediate below 10**38 (< 2**127), so shifting a
// remainder smaller than the divisor cannot overflow the two-word state.
inline DecimalWideDivision decimal_wide_divide(ulong2 numerator,
                                                ulong2 divisor) {
  ulong2 quotient = ulong2(0), remainder = ulong2(0);
  for (int bit = 127; bit >= 0; --bit) {
    ulong incoming = bit >= 64
                         ? (numerator.y >> uint(bit - 64)) & 1UL
                         : (numerator.x >> uint(bit)) & 1UL;
    remainder = ulong2((remainder.x << 1) | incoming,
                       (remainder.y << 1) | (remainder.x >> 63));
    if (decimal_wide_ge(remainder, divisor)) {
      remainder = decimal_wide_subtract(remainder, divisor);
      if (bit >= 64)
        quotient.y |= 1UL << uint(bit - 64);
      else
        quotient.x |= 1UL << uint(bit);
    }
  }
  return {quotient, remainder};
}
inline ulong2 decimal_wide_pow10(uint scale) {
  ulong2 value = ulong2(1, 0);
  for (uint i = 0; i < scale; ++i) {
    ulong low = value.x * 10UL;
    value = ulong2(low, value.y * 10UL + mulhi(value.x, 10UL));
  }
  return value;
}
inline ulong2 decimal_negate(ulong2 value) {
  ulong low = ~value.x + 1UL;
  return ulong2(low, ~value.y + ulong(low == 0));
}
inline ulong2 decimal_magnitude(ulong2 value) {
  return (value.y >> 63) ? decimal_negate(value) : value;
}
inline ulong2 decimal_increment(ulong2 value) {
  ulong low = value.x + 1UL;
  return ulong2(low, value.y + ulong(low == 0));
}
inline ulong2 decimal_half_even(ulong2 value, uint scale) {
  bool negative = (value.y >> 63) != 0;
  ulong2 unit = decimal_wide_pow10(scale);
  DecimalWideDivision divided = decimal_wide_divide(decimal_magnitude(value), unit);
  ulong2 rest = decimal_wide_subtract(unit, divided.remainder);
  if (decimal_wide_ge(divided.remainder, rest) &&
      (divided.remainder.x != rest.x || divided.remainder.y != rest.y ||
       (divided.value.x & 1UL)))
    divided.value = decimal_increment(divided.value);
  return negative ? decimal_negate(divided.value) : divided.value;
}
struct DecimalCast { ulong2 value; bool valid; };
inline DecimalCast decimal_scale_checked(ulong2 value, uint scale) {
  if (!scale) return {value, true};
  ulong2 factor = decimal_wide_pow10(scale);
  ulong2 maximum = ulong2(~0UL, 0x7fffffffffffffffUL);
  ulong2 permitted = decimal_wide_divide(maximum, factor).value;
  ulong2 magnitude = decimal_magnitude(value);
  if (decimal_wide_ge(magnitude, permitted) &&
      (magnitude.x != permitted.x || magnitude.y != permitted.y))
    return {ulong2(0), false};
  for (uint i = 0; i < scale; ++i)
    value = ulong2(value.x * 10UL, value.y * 10UL + mulhi(value.x, 10UL));
  return {value, true};
}
struct DecimalComparison { bool value; bool valid; };
struct DecimalArithmetic { ulong2 value; uint error; };
inline DecimalArithmetic decimal_add_subtract(ulong2 left, ulong2 right,
                                               uint leftScale, uint rightScale,
                                               bool subtract) {
  uint common = leftScale > rightScale ? leftScale : rightScale;
  DecimalCast a = decimal_scale_checked(left, common - leftScale);
  DecimalCast b = decimal_scale_checked(right, common - rightScale);
  if (!a.valid || !b.valid) return {ulong2(0), 1};
  ulong2 value = ulong2(0);
  if (subtract) value = decimal_wide_subtract(a.value, b.value);
  else {
    ulong low = a.value.x + b.value.x;
    value = ulong2(low, a.value.y + b.value.y + ulong(low < a.value.x));
  }
  // CPU addition/subtraction wraps at Int128 before its precision check.
  return {value, decimal_wide_ge(decimal_magnitude(value), decimal_wide_pow10(38)) ? 2U : 0U};
}
inline DecimalArithmetic decimal_unary(ulong2 value, uint bits, bool absolute) {
  ulong2 minimum = bits == 128 ? ulong2(0, 0x8000000000000000UL) :
      bits == 64 ? ulong2(0x8000000000000000UL, ~0UL) :
                   ulong2(0xffffffff80000000UL, ~0UL);
  if (value.x == minimum.x && value.y == minimum.y) return {ulong2(0), 3};
  return {(!absolute || (value.y >> 63)) ? decimal_negate(value) : value, 0};
}
inline ulong2 decimal_wide_add(ulong2 a, ulong2 b) {
  ulong low = a.x + b.x;
  return ulong2(low, a.y + b.y + ulong(low < a.x));
}
inline ulong2 decimal_wide_multiply(ulong2 a, ulong2 b) {
  return ulong2(a.x * b.x, a.y * b.x + a.x * b.y + mulhi(a.x, b.x));
}
inline bool decimal_signed_less(ulong2 a, ulong2 b) {
  bool sa = (a.y >> 63) != 0, sb = (b.y >> 63) != 0;
  return sa != sb ? sa : (a.y < b.y || (a.y == b.y && a.x < b.x));
}
inline bool decimal_is_zero(ulong2 value) { return !(value.x | value.y); }
inline bool decimal_precision_overflow(ulong2 value) {
  return decimal_wide_ge(decimal_magnitude(value), decimal_wide_pow10(38));
}
// Mojo '/' truncates signed integer division, while '%' is floor modulo.
// Return that pair explicitly; it differs from C++ signed div/rem.
inline DecimalWideDivision decimal_signed_divide(ulong2 a, ulong2 b) {
  bool sa = (a.y >> 63) != 0, sb = (b.y >> 63) != 0;
  ulong2 divisor = decimal_magnitude(b);
  DecimalWideDivision parts = decimal_wide_divide(decimal_magnitude(a), divisor);
  if (sa != sb) parts.value = decimal_negate(parts.value);
  if (!decimal_is_zero(parts.remainder)) {
    if (sa != sb) parts.remainder = decimal_wide_subtract(divisor, parts.remainder);
    if (sb) parts.remainder = decimal_negate(parts.remainder);
  }
  return parts;
}
inline ulong2 decimal_round_signed(ulong2 value, ulong2 remainder, ulong2 divisor) {
  ulong2 rest = decimal_wide_subtract(divisor, remainder);
  bool equal = rest.x == remainder.x && rest.y == remainder.y;
  if (decimal_signed_less(rest, remainder) || (equal && (value.x & 1UL)))
    value = decimal_increment(value);
  return value;
}
inline DecimalArithmetic decimal_multiply(ulong2 left, ulong2 right,
                                           uint leftScale, uint rightScale) {
  uint common = leftScale > rightScale ? leftScale : rightScale;
  if (!decimal_scale_checked(left, common-leftScale).valid ||
      !decimal_scale_checked(right, common-rightScale).valid)
    return {ulong2(0), 1};
  // Preserve _decimal_mul's signed bounds and its abs(Int128.MIN) behavior.
  bool leftMinimum = left.x == 0 && left.y == 0x8000000000000000UL;
  if (leftMinimum) return {ulong2(0), 1};
  if (decimal_is_zero(left)) return {ulong2(0), 0};
  ulong2 maximum = (right.y >> 63) ? ulong2(0, 0x8000000000000000UL)
                                  : ulong2(~0UL, 0x7fffffffffffffffUL);
  ulong2 permitted = decimal_wide_divide(maximum, decimal_magnitude(left)).value;
  ulong2 magnitude = decimal_magnitude(right);
  if (decimal_wide_ge(magnitude, permitted) &&
      (magnitude.x != permitted.x || magnitude.y != permitted.y))
    return {ulong2(0), 1};
  ulong2 value = decimal_wide_multiply(left, right);
  uint scale = leftScale < rightScale ? leftScale : rightScale;
  if (scale) {
    bool minimum = value.x == 0 && value.y == 0x8000000000000000UL;
    if (minimum && scale <= 18) value = ulong2(0);
    else {
      value = decimal_half_even(value, scale);
      if (minimum) value = decimal_negate(value);
    }
  }
  return {value, decimal_precision_overflow(value) ? 2U : 0U};
}
inline DecimalArithmetic decimal_divide_rows(ulong2 left, ulong2 right,
                                              uint leftScale, uint rightScale) {
  uint common = leftScale > rightScale ? leftScale : rightScale;
  if (!decimal_scale_checked(left, common-leftScale).valid ||
      !decimal_scale_checked(right, common-rightScale).valid)
    return {ulong2(0), 1};
  if (decimal_is_zero(right)) return {ulong2(0), 4}; // CPU null quotient.
  bool negative = ((left.y ^ right.y) >> 63) != 0;
  ulong2 numerator = decimal_magnitude(left), divisor = decimal_magnitude(right);
  DecimalWideDivision parts = decimal_signed_divide(numerator, divisor);
  ulong2 value = parts.value, remainder = parts.remainder;
  if (decimal_precision_overflow(value)) return {ulong2(0), 2};
  uint shift = common + rightScale - leftScale;
  for (uint i = 0; i < shift; ++i) {
    value = decimal_wide_multiply(value, ulong2(10, 0));
    if (decimal_precision_overflow(value)) return {ulong2(0), 2};
    remainder = decimal_wide_multiply(remainder, ulong2(10, 0));
    parts = decimal_signed_divide(remainder, divisor);
    value = decimal_wide_add(value, parts.value);
    remainder = parts.remainder;
  }
  value = decimal_round_signed(value, remainder, divisor);
  if (decimal_precision_overflow(value)) return {ulong2(0), 2};
  return {negative ? decimal_negate(value) : value, 0};
}
inline DecimalComparison decimal_compare(ulong2 left, ulong2 right,
                                          uint leftScale, uint rightScale,
                                          uint op) {
  uint common = leftScale > rightScale ? leftScale : rightScale;
  DecimalCast a = decimal_scale_checked(left, common - leftScale);
  DecimalCast b = decimal_scale_checked(right, common - rightScale);
  if (!a.valid || !b.valid) return {false, false};
  bool equal = a.value.x == b.value.x && a.value.y == b.value.y;
  bool sa = (a.value.y >> 63) != 0, sb = (b.value.y >> 63) != 0;
  bool less = sa != sb ? sa :
      (a.value.y < b.value.y || (a.value.y == b.value.y && a.value.x < b.value.x));
  return {op == 9 ? equal : op == 23 ? !equal : op == 20 ? less :
          op == 22 ? less || equal : op == 8 ? !less && !equal : !less, true};
}
inline DecimalCast decimal_cast(ulong2 value, uint fromScale, uint toScale,
                                uint precision, uint bits, bool isUnsigned,
                                bool isBoolean) {
  if (fromScale > toScale) {
    value = decimal_half_even(value, fromScale - toScale);
  } else {
    // Match CPU Int128 rescaling, including its fixed-width intermediate.
    for (uint i = fromScale; i < toScale; ++i)
      value = ulong2(value.x * 10UL, value.y * 10UL + mulhi(value.x, 10UL));
  }
  bool negative = (value.y >> 63) != 0;
  ulong2 magnitude = decimal_magnitude(value);
  if (precision)
    return {value, !decimal_wide_ge(magnitude, decimal_wide_pow10(precision))};
  if (isBoolean)
    return {ulong2(ulong(magnitude.x != 0 || magnitude.y != 0), 0), true};
  ulong maximum = isUnsigned
      ? (bits == 64 ? ~0UL : (1UL << bits) - 1UL)
      : (1UL << (bits - 1)) - ulong(!negative);
  bool valid = magnitude.y == 0 && magnitude.x <= maximum &&
               !(isUnsigned && negative);
  return {value, valid};
}
// Decimal128 totals and scale divisors may both exceed one word. Retain the
// CPU decimal_mean quotient/remainder sequence and each separate Float64
// conversion rather than dividing rounded binary64 values.
inline ulong f64_decimal128_mean(ulong low, ulong high, ulong count,
                                 uint scale) {
  if (!count) return 0;
  bool negative = (high >> 63) != 0;
  if (negative) {
    low = ~low + 1UL;
    high = ~high + ulong(low == 0);
  }
  DecimalWideDivision per_row =
      decimal_wide_divide(ulong2(low, high), ulong2(count, 0));
  ulong2 unit = decimal_wide_pow10(scale);
  DecimalWideDivision parts = decimal_wide_divide(per_row.value, unit);
  ulong fractional = f64_add(
      f64_from_wide(parts.remainder.x, parts.remainder.y, true),
      f64_div(f64_from_unsigned(per_row.remainder.x),
              f64_from_unsigned(count)));
  ulong value = f64_add(
      f64_from_wide(parts.value.x, parts.value.y, true),
      f64_div(fractional, f64_from_wide(unit.x, unit.y, true)));
  return negative ? value ^ 0x8000000000000000UL : value;
}
)metal";
#endif
