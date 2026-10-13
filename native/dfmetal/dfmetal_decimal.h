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
