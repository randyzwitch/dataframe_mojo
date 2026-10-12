#pragma once

// Integer-only binary64 primitives, shared verbatim by generated Metal shaders
// and the differential arithmetic tests. No native floating arithmetic is used.
// Values are IEEE 754 encodings; rounding is nearest, ties to even. Exception
// flags are not exposed by the dataframe API. NaN inputs retain their payload
// and are quieted; invalid operations return a quiet NaN.
static constexpr const char *dfm_float64_source = R"metal(
inline ulong f64_abs(ulong a) { return a & 0x7fffffffffffffffUL; }
inline bool f64_nan(ulong a) { return f64_abs(a) > 0x7ff0000000000000UL; }
inline bool f64_inf(ulong a) { return f64_abs(a) == 0x7ff0000000000000UL; }
inline ulong f64_quiet(ulong a) { return a | 0x0008000000000000UL; }
inline ulong f64_invalid() { return 0x7ff8000000000000UL; }
inline ulong f64_jam(ulong a, uint distance) {
  if (!distance) return a;
  if (distance >= 64) return ulong(a != 0);
  return (a >> distance) | ulong((a << (64 - distance)) != 0);
}
struct F64Parts { ulong significand; int exponent; };
inline F64Parts f64_parts(ulong a) {
  F64Parts p;
  p.significand = a & 0x000fffffffffffffUL;
  int encoded = int((a >> 52) & 2047);
  p.exponent = encoded ? encoded - 1023 : -1022;
  if (encoded) p.significand |= 0x0010000000000000UL;
  else if (p.significand) {
    while (p.significand < 0x0010000000000000UL) {
      p.significand <<= 1;
      --p.exponent;
    }
  }
  return p;
}
// Extended significand has its leading bit at 55 and three rounding bits.
inline ulong f64_pack(ulong sign, int exponent, ulong significand) {
  if (!significand) return sign;
  while (significand >= 0x0100000000000000UL) {
    significand = f64_jam(significand, 1);
    ++exponent;
  }
  while (significand < 0x0080000000000000UL) {
    significand <<= 1;
    --exponent;
  }
  if (exponent < -1022) {
    significand = f64_jam(significand, uint(-1022 - exponent));
    exponent = -1022;
  }
  ulong rounded = significand >> 3;
  ulong tail = significand & 7;
  rounded += ulong(tail > 4 || (tail == 4 && (rounded & 1)));
  if (rounded >= 0x0020000000000000UL) {
    rounded >>= 1;
    ++exponent;
  }
  if (exponent > 1023) return sign | 0x7ff0000000000000UL;
  ulong encoded = rounded >= 0x0010000000000000UL ? ulong(exponent + 1023) : 0;
  return sign | (encoded << 52) | (rounded & 0x000fffffffffffffUL);
}
inline ulong f64_sqrt(ulong a) {
  if (f64_nan(a)) return f64_quiet(a);
  if (!f64_abs(a)) return a;
  if (a >> 63) return f64_invalid();
  if (f64_inf(a)) return a;
  F64Parts p = f64_parts(a);
  if (p.exponent & 1) { p.significand <<= 1; --p.exponent; }
  // Extract the 53 root digits of significand * 2^52, two radicand
  // bits at a time. The radicand is 106 bits; root and remainder fit ulong.
  ulong high = p.significand >> 12, low = p.significand << 52;
  ulong root = 0, remainder = 0;
  for (int digit = 52; digit >= 0; --digit) {
    uint shift = uint(digit * 2);
    ulong pair = (shift >= 64 ? high >> (shift - 64) : low >> shift) & 3;
    remainder = (remainder << 2) | pair;
    ulong trial = (root << 2) | 1;
    root <<= 1;
    if (remainder >= trial) { remainder -= trial; ++root; }
  }
  // (root + 1/2)^2 has a quarter fractional bit, so an integer
  // radicand cannot tie. Compare the exact remainder to root to round.
  root += ulong(remainder > root);
  return f64_pack(0, p.exponent / 2, root << 3);
}
inline ulong f64_add(ulong a, ulong b) {
  if (f64_nan(a)) return f64_quiet(a);
  if (f64_nan(b)) return f64_quiet(b);
  ulong signa = a & 0x8000000000000000UL;
  ulong signb = b & 0x8000000000000000UL;
  if (f64_inf(a) || f64_inf(b)) {
    if (f64_inf(a) && f64_inf(b) && signa != signb) return f64_invalid();
    return f64_inf(a) ? a : b;
  }
  if (!f64_abs(a) && !f64_abs(b)) return signa & signb;
  if (!f64_abs(a)) return b;
  if (!f64_abs(b)) return a;
  if (f64_abs(a) < f64_abs(b)) {
    ulong temporary = a; a = b; b = temporary;
    temporary = signa; signa = signb; signb = temporary;
  }
  F64Parts x = f64_parts(a), y = f64_parts(b);
  ulong left = x.significand << 3;
  ulong right = f64_jam(y.significand << 3, uint(x.exponent - y.exponent));
  if (signa != signb && left == right) return 0;
  return f64_pack(signa, x.exponent, signa == signb ? left + right : left - right);
}
inline ulong f64_sub(ulong a, ulong b) {
  // Preserve the original sign of a NaN right operand when quieting it.
  if (f64_nan(a)) return f64_quiet(a);
  if (f64_nan(b)) return f64_quiet(b);
  return f64_add(a, b ^ 0x8000000000000000UL);
}
struct F64Product { ulong high; ulong low; };
inline F64Product f64_product(ulong a, ulong b) {
  ulong a0 = uint(a), a1 = a >> 32, b0 = uint(b), b1 = b >> 32;
  ulong bottom = a0 * b0;
  ulong middle = a1 * b0 + (bottom >> 32);
  ulong carry = middle >> 32;
  middle = (middle & 0xffffffffUL) + a0 * b1;
  F64Product p;
  p.high = a1 * b1 + carry + (middle >> 32);
  p.low = (middle << 32) | uint(bottom);
  return p;
}
inline ulong f64_mul(ulong a, ulong b) {
  if (f64_nan(a)) return f64_quiet(a);
  if (f64_nan(b)) return f64_quiet(b);
  ulong sign = (a ^ b) & 0x8000000000000000UL;
  if (f64_inf(a) || f64_inf(b)) {
    if (!f64_abs(a) || !f64_abs(b)) return f64_invalid();
    return sign | 0x7ff0000000000000UL;
  }
  if (!f64_abs(a) || !f64_abs(b)) return sign;
  F64Parts x = f64_parts(a), y = f64_parts(b);
  F64Product product = f64_product(x.significand, y.significand);
  bool upper = (product.high & (1UL << 41)) != 0;
  uint shift = upper ? 50 : 49;
  ulong significand = (product.high << (64 - shift)) | (product.low >> shift);
  significand |= ulong((product.low << (64 - shift)) != 0);
  return f64_pack(sign, x.exponent + y.exponent + int(upper), significand);
}
inline F64Product f64_wide_jam(F64Product value, uint distance) {
  if (!distance) return value;
  ulong high=value.high, low=value.low;
  if (distance < 64) {
    value.high=high >> distance;
    value.low=(low >> distance) | (high << (64-distance)) |
              ulong((low << (64-distance)) != 0);
  } else if (distance == 64) {
    value.high=0; value.low=high | ulong(low != 0);
  } else if (distance < 128) {
    value.high=0;
    value.low=(high >> (distance-64)) |
              ulong(low != 0 || (high << (128-distance)) != 0);
  } else {
    value.high=0; value.low=ulong(high != 0 || low != 0);
  }
  return value;
}
// Round the extended significand directly to binary32. Rounding to binary64
// first would lose the sticky bits needed at binary32 halfway boundaries.
inline uint f32_pack(ulong sign, int exponent, ulong significand) {
  uint sign32 = uint(sign >> 32);
  if (!significand) return sign32;
  while (significand >= (1UL << 56)) {
    significand = f64_jam(significand, 1); ++exponent;
  }
  while (significand < (1UL << 55)) {
    significand <<= 1; --exponent;
  }
  if (exponent < -126) {
    significand = f64_jam(significand, uint(-126 - exponent));
    exponent = -126;
  }
  uint rounded = uint(significand >> 32);
  ulong tail = significand & 0xffffffffUL;
  rounded += uint(tail > 0x80000000UL ||
                  (tail == 0x80000000UL && (rounded & 1)));
  if (rounded >= 0x01000000U) { rounded >>= 1; ++exponent; }
  if (exponent > 127) return sign32 | 0x7f800000U;
  uint encoded = rounded >= 0x00800000U ? uint(exponent + 127) : 0;
  return sign32 | (encoded << 23) | (rounded & 0x007fffffU);
}
inline uint f64_to_float32(ulong value);
inline ulong f64_fma_result(ulong value, bool narrow) {
  return narrow ? ulong(f64_to_float32(value)) : value;
}
inline ulong f64_fma_core(ulong a, ulong b, ulong c, bool narrow) {
  if (f64_nan(a)) return f64_fma_result(f64_quiet(a),narrow);
  if (f64_nan(b)) return f64_fma_result(f64_quiet(b),narrow);
  if (f64_nan(c)) return f64_fma_result(f64_quiet(c),narrow);
  ulong sign=(a^b)&0x8000000000000000UL;
  ulong other_sign=c&0x8000000000000000UL;
  if (f64_inf(a) || f64_inf(b)) {
    if (!f64_abs(a) || !f64_abs(b) || (f64_inf(c) && sign != other_sign))
      return f64_fma_result(f64_invalid(),narrow);
    return f64_fma_result(sign|0x7ff0000000000000UL,narrow);
  }
  if (f64_inf(c)) return f64_fma_result(c,narrow);
  if (!f64_abs(a) || !f64_abs(b)) return f64_fma_result(f64_add(sign,c),narrow);
  if (!f64_abs(c)) return f64_fma_result(f64_mul(a,b),narrow);
  F64Parts x=f64_parts(a), y=f64_parts(b), z=f64_parts(c);
  F64Product product=f64_product(x.significand,y.significand);
  bool upper=(product.high&(1UL<<41)) != 0;
  int exponent=x.exponent+y.exponent+int(upper);
  uint shift=upper?15:16;
  // Keep the exact 106-bit product and align the addend at bit 120.
  // Extra low bits retain cancellation information and a sticky remainder.
  product.high=(product.high<<shift)|(product.low>>(64-shift));
  product.low <<= shift;
  F64Product addend;
  addend.high=z.significand<<4; addend.low=0;
  if (exponent >= z.exponent)
    addend=f64_wide_jam(addend,uint(exponent-z.exponent));
  else {
    product=f64_wide_jam(product,uint(z.exponent-exponent));
    exponent=z.exponent;
  }
  if (sign == other_sign) {
    ulong low=product.low+addend.low;
    product.high += addend.high+ulong(low<product.low);
    product.low=low;
  } else {
    if (product.high < addend.high ||
        (product.high == addend.high && product.low < addend.low)) {
      F64Product temporary=product; product=addend; addend=temporary;
      sign=other_sign;
    }
    ulong borrow=ulong(product.low<addend.low);
    product.low -= addend.low;
    product.high -= addend.high+borrow;
    if (!product.high && !product.low) return 0;
  }
  if (product.high >= (1UL<<57)) {
    product=f64_wide_jam(product,1); ++exponent;
  }
  while (product.high < (1UL<<56)) {
    product.high=(product.high<<1)|(product.low>>63);
    product.low <<= 1; --exponent;
  }
  ulong significand=(product.high>>1) |
                    ulong((product.high&1) != 0 || product.low != 0);
  return narrow ? ulong(f32_pack(sign,exponent,significand))
                : f64_pack(sign,exponent,significand);
}
inline ulong f64_fma(ulong a, ulong b, ulong c) {
  return f64_fma_core(a,b,c,false);
}
inline ulong f64_div(ulong a, ulong b) {
  if (f64_nan(a)) return f64_quiet(a);
  if (f64_nan(b)) return f64_quiet(b);
  ulong sign = (a ^ b) & 0x8000000000000000UL;
  if ((f64_inf(a) && f64_inf(b)) || (!f64_abs(a) && !f64_abs(b)))
    return f64_invalid();
  if (f64_inf(a) || !f64_abs(b)) return sign | 0x7ff0000000000000UL;
  if (!f64_abs(a) || f64_inf(b)) return sign;
  F64Parts x = f64_parts(a), y = f64_parts(b);
  int exponent = x.exponent - y.exponent;
  ulong remainder = x.significand;
  if (remainder < y.significand) { remainder <<= 1; --exponent; }
  ulong quotient = 0;
  for (uint i = 0; i < 56; ++i) {
    quotient <<= 1;
    if (remainder >= y.significand) { remainder -= y.significand; quotient |= 1; }
    remainder <<= 1;
  }
  quotient |= ulong(remainder != 0);
  return f64_pack(sign, exponent, quotient);
}
inline bool f64_equal(ulong a, ulong b) {
  return !f64_nan(a) && !f64_nan(b) && (a == b || (!f64_abs(a) && !f64_abs(b)));
}
inline bool f64_less(ulong a, ulong b) {
  if (f64_nan(a) || f64_nan(b) || f64_equal(a, b)) return false;
  bool sa = (a >> 63) != 0, sb = (b >> 63) != 0;
  return sa != sb ? sa : (sa ? a > b : a < b);
}
inline ulong f64_from_unsigned(ulong value) {
  if (!value) return 0;
  uint leading = 0;
  for (ulong rest = value; rest >>= 1;) ++leading;
  ulong significand = leading > 55 ? f64_jam(value, leading - 55)
                                   : value << (55 - leading);
  return f64_pack(0, int(leading), significand);
}
inline ulong f64_from_signed(long value) {
  ulong sign = value < 0 ? 0x8000000000000000UL : 0;
  ulong magnitude = value < 0 ? 0UL - ulong(value) : ulong(value);
  return sign | f64_from_unsigned(magnitude);
}
// Round a two-word integer directly to binary64, retaining guard/round/sticky.
// Shared C++/MSL source avoids floating-point intermediates and double rounding.
inline ulong f64_from_wide(ulong low, ulong high, bool is_unsigned) {
  ulong sign = !is_unsigned && (high >> 63) ? 0x8000000000000000UL : 0;
  if (sign) {
    low = ~low + 1UL;
    high = ~high + ulong(low == 0);
  }
  if (!(low | high)) return 0;
  uint leading = high ? 64 : 0;
  for (ulong rest = high ? high : low; rest >>= 1;) ++leading;
  ulong significand;
  if (leading <= 55) significand = low << (55 - leading);
  else {
    uint distance = leading - 55;
    if (distance < 64) {
      significand = (high << (64 - distance)) | (low >> distance);
      significand |= ulong((low << (64 - distance)) != 0);
    } else significand = f64_jam(high, distance - 64) | ulong(low != 0);
  }
  return f64_pack(sign, int(leading), significand);
}
inline ulong f64_from_float32(uint value) {
  ulong sign = ulong(value & 0x80000000U) << 32;
  uint fraction = value & 0x007fffffU;
  int encoded = int((value >> 23) & 255);
  if (encoded == 255)
    return sign | 0x7ff0000000000000UL | (ulong(fraction) << 29) |
           (fraction ? 0x0008000000000000UL : 0UL);
  if (!encoded && !fraction) return sign;
  int exponent = encoded ? encoded - 127 : -126;
  if (encoded) fraction |= 0x00800000U;
  else while (fraction < 0x00800000U) { fraction <<= 1; --exponent; }
  return sign | (ulong(exponent + 1023) << 52) |
         (ulong(fraction & 0x007fffffU) << 29);
}
inline ulong f64_word(ulong value, long type) {
  if (type == 11) return value;
  if (type == 1) return f64_from_float32(uint(value));
  if (type == 2 || type == 3 || type == 5 || type == 6)
    return f64_from_signed(long(value));
  return f64_from_unsigned(value);
}
inline uint f64_to_float32(ulong value) {
  uint sign = uint(value >> 32) & 0x80000000U;
  if (f64_nan(value))
    return sign | 0x7f800000U | uint((value >> 29) & 0x007fffffUL) | 0x00400000U;
  if (f64_inf(value)) return sign | 0x7f800000U;
  if (!f64_abs(value)) return sign;
  F64Parts p = f64_parts(value);
  ulong significand = f64_jam(p.significand, 26);
  if (p.exponent < -126) {
    significand = f64_jam(significand, uint(-126 - p.exponent));
    p.exponent = -126;
  }
  uint rounded = uint(significand >> 3);
  uint tail = uint(significand & 7);
  rounded += uint(tail > 4 || (tail == 4 && (rounded & 1)));
  if (rounded >= 0x01000000U) { rounded >>= 1; ++p.exponent; }
  if (p.exponent > 127) return sign | 0x7f800000U;
  uint encoded = rounded >= 0x00800000U ? uint(p.exponent + 127) : 0;
  return sign | (encoded << 23) | (rounded & 0x007fffffU);
}
struct F64Integer { ulong word; bool valid; };
inline F64Integer f64_to_integer(ulong value, uint width, bool is_unsigned) {
  F64Integer result;
  result.word = 0;
  result.valid = !f64_nan(value) && !f64_inf(value);
  if (!result.valid || !f64_abs(value)) return result;
  F64Parts p = f64_parts(value);
  if (p.exponent < 0) return result;
  if (p.exponent >= 64) { result.valid = false; return result; }
  ulong magnitude = p.exponent >= 52 ? p.significand << uint(p.exponent - 52)
                                       : p.significand >> uint(52 - p.exponent);
  bool negative = (value >> 63) != 0;
  ulong maximum = is_unsigned ? (width == 64 ? ~0UL : (1UL << width) - 1)
      : (negative ? 1UL << (width - 1) : (1UL << (width - 1)) - 1);
  result.valid = magnitude <= maximum && !(is_unsigned && negative && magnitude);
  result.word = negative ? 0UL - magnitude : magnitude;
  return result;
}
inline ulong f64_integral(ulong value, int op) {
  if (f64_nan(value)) return f64_quiet(value);
  ulong absolute = f64_abs(value), sign = value & 0x8000000000000000UL;
  int exponent = int((value >> 52) & 2047) - 1023;
  if (exponent >= 52 || !absolute) return value;
  if (exponent < 0)
    return op == 55 ? (sign ? 0xbff0000000000000UL : sign)
                    : (sign ? sign : 0x3ff0000000000000UL);
  ulong unit = 1UL << uint(52 - exponent);
  ulong rest = absolute & (unit - 1), whole = value & ~(unit - 1);
  if (rest && ((op == 55 && sign) || (op == 56 && !sign))) whole += unit;
  return whole;
}
inline uint f32_floor_div(uint a, uint b) {
  // The CPU floors the quotient after binary32 rounding, which can differ
  // from flooring a binary64 quotient near an integer boundary.
  uint quotient = f64_to_float32(f64_div(f64_from_float32(a), f64_from_float32(b)));
  return f64_to_float32(f64_integral(f64_from_float32(quotient), 55));
}
inline uint f32_fma(uint a, uint b, uint c) {
  return uint(f64_fma_core(f64_from_float32(a),f64_from_float32(b),
                           f64_from_float32(c),true));
}
inline uint f32_mod(uint a, uint b) {
  ulong x = f64_from_float32(a), y = f64_from_float32(b);
  if (f64_nan(x)) return a | 0x00400000U;
  if (f64_nan(y)) return b | 0x00400000U;
  if (f64_inf(x) || !f64_abs(y)) return 0x7fc00000U;
  uint quotient = f64_to_float32(f64_div(x,y));
  ulong whole = f64_from_float32(quotient);
  whole = f64_integral(whole, whole >> 63 ? 56 : 55);
  uint remainder = f32_fma(f64_to_float32(whole)^0x80000000U,b,a);
  ulong result = f64_from_float32(remainder);
  bool adjust = (f64_less(x,0) != f64_less(y,0)) && !f64_equal(result,0);
  return f64_to_float32(f64_add(result,adjust ? y : 0UL));
}
inline ulong f64_mod(ulong a, ulong b) {
  // Match the CPU SIMD operation sequence, including a rounded quotient,
  // its truncation, fused multiply/subtract, and final addition of zero.
  // This intentionally differs from native fmod for extreme quotients.
  if (f64_nan(a)) return f64_quiet(a);
  if (f64_nan(b)) return f64_quiet(b);
  if (f64_inf(a) || !f64_abs(b)) return f64_invalid();
  ulong quotient = f64_div(a,b);
  quotient = f64_integral(quotient, quotient >> 63 ? 56 : 55);
  ulong remainder = f64_fma(quotient^0x8000000000000000UL,b,a);
  bool adjust = (f64_less(a,0) != f64_less(b,0)) && !f64_equal(remainder,0);
  return f64_add(remainder, adjust ? b : 0UL);
}
inline ulong f64_round(ulong value) {
  // Match the CPU's floor(abs(x)+0.5) expression, including its intermediate
  // rounding and the special handling of already integral values and -0.
  if (f64_abs(value) >= 0x4330000000000000UL) return value;
  ulong rounded = f64_integral(f64_add(f64_abs(value), 0x3fe0000000000000UL), 55);
  return value >> 63 && f64_abs(value) ? rounded ^ 0x8000000000000000UL : rounded;
}
inline ulong f64_decimal_round(ulong value, ulong scale, bool negative_decimals) {
  // Match the CPU's original-value early return and every rounded binary64
  // intermediate. The scale is a bound scalar constant, never row data.
  if (f64_abs(value) >= 0x4330000000000000UL) return value;
  ulong scaled = negative_decimals ? f64_div(value, scale) : f64_mul(value, scale);
  ulong rounded = f64_integral(f64_add(f64_abs(scaled), 0x3fe0000000000000UL),55);
  if (f64_less(scaled,0)) rounded ^= 0x8000000000000000UL;
  return negative_decimals ? f64_mul(rounded,scale) : f64_div(rounded,scale);
}
)metal";
