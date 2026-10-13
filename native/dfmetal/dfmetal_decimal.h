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
)metal";
#endif
