"""IEEE binary64 arithmetic using integer words, including on Apple GPUs.

Ported from native/dfmetal/dfmetal_float64.h. Operations round to nearest,
ties to even, preserve gradual underflow, and quiet input NaNs with payloads.
No floating-point instruction or external function implements these primitives.
"""

comptime SIGN = UInt64(0x8000000000000000)
comptime INF = UInt64(0x7FF0000000000000)
comptime FRACTION = UInt64(0x000FFFFFFFFFFFFF)
comptime HIDDEN = UInt64(0x0010000000000000)


def absolute(a: UInt64) -> UInt64:
    return a & ~SIGN


def is_nan(a: UInt64) -> Bool:
    return absolute(a) > INF


def is_inf(a: UInt64) -> Bool:
    return absolute(a) == INF


def quiet(a: UInt64) -> UInt64:
    return a | 0x0008000000000000


def invalid() -> UInt64:
    return 0x7FF8000000000000


def jam(a: UInt64, distance: UInt32) -> UInt64:
    if distance == 0:
        return a
    if distance >= 64:
        return UInt64(a != 0)
    return (a >> UInt64(distance)) | UInt64((a << UInt64(64 - distance)) != 0)


def parts(a: UInt64) -> Tuple[UInt64, Int32]:
    var significand = a & FRACTION
    var encoded = Int32((a >> 52) & 2047)
    var exponent = encoded - 1023 if encoded else Int32(-1022)
    if encoded:
        significand |= HIDDEN
    elif significand:
        while significand < HIDDEN:
            significand <<= 1
            exponent -= 1
    return (significand, exponent)


def pack(sign: UInt64, exponent: Int32, significand: UInt64) -> UInt64:
    if significand == 0:
        return sign
    var sig = significand
    var exp = exponent
    while sig >= 0x0100000000000000:
        sig = jam(sig, 1)
        exp += 1
    while sig < 0x0080000000000000:
        sig <<= 1
        exp -= 1
    if exp < -1022:
        sig = jam(sig, UInt32(-1022 - exp))
        exp = -1022
    var rounded = sig >> 3
    var tail = sig & 7
    rounded += UInt64(tail > 4 or (tail == 4 and (rounded & 1) != 0))
    if rounded >= 0x0020000000000000:
        rounded >>= 1
        exp += 1
    if exp > 1023:
        return sign | INF
    var encoded = UInt64(exp + 1023) if rounded >= HIDDEN else UInt64(0)
    return sign | (encoded << 52) | (rounded & FRACTION)


def add(a: UInt64, b: UInt64) -> UInt64:
    if is_nan(a):
        return quiet(a)
    if is_nan(b):
        return quiet(b)
    var left = a
    var right = b
    var signa = a & SIGN
    var signb = b & SIGN
    if is_inf(a) or is_inf(b):
        if is_inf(a) and is_inf(b) and signa != signb:
            return invalid()
        return a if is_inf(a) else b
    if absolute(a) == 0 and absolute(b) == 0:
        return signa & signb
    if absolute(a) == 0:
        return b
    if absolute(b) == 0:
        return a
    if absolute(a) < absolute(b):
        left = b
        right = a
        signa = b & SIGN
        signb = a & SIGN
    var x = parts(left)
    var y = parts(right)
    var l = x[0] << 3
    var r = jam(y[0] << 3, UInt32(x[1] - y[1]))
    if signa != signb and l == r:
        return 0
    return pack(signa, x[1], l + r if signa == signb else l - r)


def sub(a: UInt64, b: UInt64) -> UInt64:
    if is_nan(a):
        return quiet(a)
    if is_nan(b):
        return quiet(b)
    return add(a, b ^ SIGN)


def product(a: UInt64, b: UInt64) -> Tuple[UInt64, UInt64]:
    var a0 = a & 0xFFFFFFFF
    var a1 = a >> 32
    var b0 = b & 0xFFFFFFFF
    var b1 = b >> 32
    var bottom = a0 * b0
    var middle = a1 * b0 + (bottom >> 32)
    var carry = middle >> 32
    middle = (middle & 0xFFFFFFFF) + a0 * b1
    return (
        a1 * b1 + carry + (middle >> 32),
        (middle << 32) | (bottom & 0xFFFFFFFF),
    )


def mul(a: UInt64, b: UInt64) -> UInt64:
    if is_nan(a):
        return quiet(a)
    if is_nan(b):
        return quiet(b)
    var sign = (a ^ b) & SIGN
    if is_inf(a) or is_inf(b):
        if absolute(a) == 0 or absolute(b) == 0:
            return invalid()
        return sign | INF
    if absolute(a) == 0 or absolute(b) == 0:
        return sign
    var x = parts(a)
    var y = parts(b)
    var p = product(x[0], y[0])
    var upper = (p[0] & (UInt64(1) << 41)) != 0
    var shift = UInt64(50) if upper else UInt64(49)
    var sig = (p[0] << (64 - shift)) | (p[1] >> shift)
    sig |= UInt64((p[1] << (64 - shift)) != 0)
    return pack(sign, x[1] + y[1] + Int32(upper), sig)


def div(a: UInt64, b: UInt64) -> UInt64:
    if is_nan(a):
        return quiet(a)
    if is_nan(b):
        return quiet(b)
    var sign = (a ^ b) & SIGN
    if (is_inf(a) and is_inf(b)) or (absolute(a) == 0 and absolute(b) == 0):
        return invalid()
    if is_inf(a) or absolute(b) == 0:
        return sign | INF
    if absolute(a) == 0 or is_inf(b):
        return sign
    var x = parts(a)
    var y = parts(b)
    var exponent = x[1] - y[1]
    var remainder = x[0]
    if remainder < y[0]:
        remainder <<= 1
        exponent -= 1
    var quotient = UInt64(0)
    for _ in range(56):
        quotient <<= 1
        if remainder >= y[0]:
            remainder -= y[0]
            quotient |= 1
        remainder <<= 1
    quotient |= UInt64(remainder != 0)
    return pack(sign, exponent, quotient)


def sqrt(a: UInt64) -> UInt64:
    if is_nan(a):
        return quiet(a)
    if absolute(a) == 0:
        return a
    if a & SIGN:
        return invalid()
    if is_inf(a):
        return a
    var p = parts(a)
    var sig = p[0]
    var exponent = p[1]
    if exponent & 1:
        sig <<= 1
        exponent -= 1
    var high = sig >> 12
    var low = sig << 52
    var root = UInt64(0)
    var remainder = UInt64(0)
    for digit in range(52, -1, -1):
        var shift = UInt64(digit * 2)
        var pair = (high >> (shift - 64) if shift >= 64 else low >> shift) & 3
        remainder = (remainder << 2) | pair
        var trial = (root << 2) | 1
        root <<= 1
        if remainder >= trial:
            remainder -= trial
            root += 1
    root += UInt64(remainder > root)
    return pack(0, exponent // 2, root << 3)


def equal(a: UInt64, b: UInt64) -> Bool:
    return (
        not is_nan(a)
        and not is_nan(b)
        and (a == b or (absolute(a) == 0 and absolute(b) == 0))
    )


def less(a: UInt64, b: UInt64) -> Bool:
    if is_nan(a) or is_nan(b) or equal(a, b):
        return False
    var sa = a >> 63 != 0
    var sb = b >> 63 != 0
    return sa if sa != sb else (a > b if sa else a < b)


def from_unsigned(value: UInt64) -> UInt64:
    if value == 0:
        return 0
    var leading = UInt32(0)
    var rest = value >> 1
    while rest:
        leading += 1
        rest >>= 1
    var sig = jam(value, leading - 55) if leading > 55 else value << UInt64(
        55 - leading
    )
    return pack(0, Int32(leading), sig)


def from_signed(value: Int64) -> UInt64:
    var word = value.cast[DType.uint64]()
    var magnitude = UInt64(0) - word if value < 0 else word
    return (SIGN if value < 0 else UInt64(0)) | from_unsigned(magnitude)


def from_float32(value: UInt32) -> UInt64:
    var sign = UInt64(value & 0x80000000) << 32
    var fraction = value & 0x007FFFFF
    var encoded = Int32((value >> 23) & 255)
    if encoded == 255:
        return (
            sign
            | INF
            | (UInt64(fraction) << 29)
            | (UInt64(0x0008000000000000) if fraction else UInt64(0))
        )
    if encoded == 0 and fraction == 0:
        return sign
    var exponent = encoded - 127 if encoded else Int32(-126)
    if encoded:
        fraction |= 0x00800000
    else:
        while fraction < 0x00800000:
            fraction <<= 1
            exponent -= 1
    return (
        sign
        | (UInt64(exponent + 1023) << 52)
        | (UInt64(fraction & 0x007FFFFF) << 29)
    )


def word(value: UInt64, dtype: Int64) -> UInt64:
    if dtype == 11:
        return value
    if dtype == 1:
        return from_float32(UInt32(value))
    if dtype == 2 or dtype == 3 or dtype == 5 or dtype == 6:
        return from_signed(value.cast[DType.int64]())
    return from_unsigned(value)


def to_float32(value: UInt64) -> UInt32:
    var sign = UInt32(value >> 32) & 0x80000000
    if is_nan(value):
        return (
            sign | 0x7F800000 | UInt32((value >> 29) & 0x007FFFFF) | 0x00400000
        )
    if is_inf(value):
        return sign | 0x7F800000
    if absolute(value) == 0:
        return sign
    var p = parts(value)
    var sig = jam(p[0], 26)
    var exponent = p[1]
    if exponent < -126:
        sig = jam(sig, UInt32(-126 - exponent))
        exponent = -126
    var rounded = UInt32(sig >> 3)
    var tail = UInt32(sig & 7)
    rounded += UInt32(tail > 4 or (tail == 4 and (rounded & 1) != 0))
    if rounded >= 0x01000000:
        rounded >>= 1
        exponent += 1
    if exponent > 127:
        return sign | 0x7F800000
    var encoded = UInt32(exponent + 127) if rounded >= 0x00800000 else UInt32(0)
    return sign | (encoded << 23) | (rounded & 0x007FFFFF)


def integral(value: UInt64, floor: Bool) -> UInt64:
    if is_nan(value):
        return quiet(value)
    var a = absolute(value)
    var sign = value & SIGN
    var exponent = Int32((value >> 52) & 2047) - 1023
    if exponent >= 52 or a == 0:
        return value
    if exponent < 0:
        if floor:
            return UInt64(0xBFF0000000000000) if sign else sign
        return sign if sign else UInt64(0x3FF0000000000000)
    var unit = UInt64(1) << UInt64(52 - exponent)
    var rest = a & (unit - 1)
    var whole = value & ~(unit - 1)
    if rest and ((floor and sign != 0) or (not floor and sign == 0)):
        whole += unit
    return whole


def round_integer(value: UInt64) -> UInt64:
    if absolute(value) >= 0x4330000000000000:
        return value
    var rounded = integral(add(absolute(value), 0x3FE0000000000000), True)
    return rounded ^ SIGN if value >> 63 and absolute(value) != 0 else rounded


def decimal_round(value: UInt64, scale: UInt64, negative: Bool) -> UInt64:
    if absolute(value) >= 0x4330000000000000:
        return value
    var scaled = div(value, scale) if negative else mul(value, scale)
    var rounded = integral(add(absolute(scaled), 0x3FE0000000000000), True)
    if less(scaled, 0):
        rounded ^= SIGN
    return mul(rounded, scale) if negative else div(rounded, scale)


def wide_jam(
    high: UInt64, low: UInt64, distance: UInt32
) -> Tuple[UInt64, UInt64]:
    if distance == 0:
        return (high, low)
    var shift = UInt64(distance)
    if distance < 64:
        return (
            high >> shift,
            (low >> shift)
            | (high << (64 - shift))
            | UInt64((low << (64 - shift)) != 0),
        )
    if distance == 64:
        return (0, high | UInt64(low != 0))
    if distance < 128:
        return (
            0,
            (high >> (shift - 64))
            | UInt64(low != 0 or (high << (128 - shift)) != 0),
        )
    return (0, UInt64(high != 0 or low != 0))


def pack32(sign: UInt64, exponent: Int32, significand: UInt64) -> UInt32:
    # Round once to binary32; going through pack() loses FMA tie information.
    var sign32 = UInt32(sign >> 32)
    if significand == 0:
        return sign32
    var sig = significand
    var exp = exponent
    while sig >= UInt64(1) << 56:
        sig = jam(sig, 1)
        exp += 1
    while sig < UInt64(1) << 55:
        sig <<= 1
        exp -= 1
    if exp < -126:
        sig = jam(sig, UInt32(-126 - exp))
        exp = -126
    var rounded = UInt32(sig >> 32)
    var tail = sig & 0xFFFFFFFF
    rounded += UInt32(
        tail > 0x80000000 or (tail == 0x80000000 and (rounded & 1) != 0)
    )
    if rounded >= 0x01000000:
        rounded >>= 1
        exp += 1
    if exp > 127:
        return sign32 | 0x7F800000
    var encoded = UInt32(exp + 127) if rounded >= 0x00800000 else UInt32(0)
    return sign32 | (encoded << 23) | (rounded & 0x007FFFFF)


def fma_result(value: UInt64, narrow: Bool) -> UInt64:
    return UInt64(to_float32(value)) if narrow else value


def fma(a: UInt64, b: UInt64, c: UInt64, narrow: Bool = False) -> UInt64:
    if is_nan(a):
        return fma_result(quiet(a), narrow)
    if is_nan(b):
        return fma_result(quiet(b), narrow)
    if is_nan(c):
        return fma_result(quiet(c), narrow)
    var sign = (a ^ b) & SIGN
    var other_sign = c & SIGN
    if is_inf(a) or is_inf(b):
        if (
            absolute(a) == 0
            or absolute(b) == 0
            or (is_inf(c) and sign != other_sign)
        ):
            return fma_result(invalid(), narrow)
        return fma_result(sign | INF, narrow)
    if is_inf(c):
        return fma_result(c, narrow)
    if absolute(a) == 0 or absolute(b) == 0:
        return fma_result(add(sign, c), narrow)
    if absolute(c) == 0:
        return fma_result(mul(a, b), narrow)
    var x = parts(a)
    var y = parts(b)
    var z = parts(c)
    var p = product(x[0], y[0])
    var upper = (p[0] & (UInt64(1) << 41)) != 0
    var exponent = x[1] + y[1] + Int32(upper)
    var shift = UInt64(15) if upper else UInt64(16)
    var ph = (p[0] << shift) | (p[1] >> (64 - shift))
    var pl = p[1] << shift
    var ah = z[0] << 4
    var al = UInt64(0)
    if exponent >= z[1]:
        var aligned = wide_jam(ah, al, UInt32(exponent - z[1]))
        ah = aligned[0]
        al = aligned[1]
    else:
        var aligned = wide_jam(ph, pl, UInt32(z[1] - exponent))
        ph = aligned[0]
        pl = aligned[1]
        exponent = z[1]
    if sign == other_sign:
        var low = pl + al
        ph += ah + UInt64(low < pl)
        pl = low
    else:
        if ph < ah or (ph == ah and pl < al):
            var high = ph
            var low = pl
            ph = ah
            pl = al
            ah = high
            al = low
            sign = other_sign
        var borrow = UInt64(pl < al)
        pl -= al
        ph -= ah + borrow
        if ph == 0 and pl == 0:
            return 0
    if ph >= UInt64(1) << 57:
        var aligned = wide_jam(ph, pl, 1)
        ph = aligned[0]
        pl = aligned[1]
        exponent += 1
    while ph < UInt64(1) << 56:
        ph = (ph << 1) | (pl >> 63)
        pl <<= 1
        exponent -= 1
    var sig = (ph >> 1) | UInt64((ph & 1) != 0 or pl != 0)
    return UInt64(pack32(sign, exponent, sig)) if narrow else pack(
        sign, exponent, sig
    )


def modulo(a: UInt64, b: UInt64, narrow: Bool = False) -> UInt64:
    # Follow the dataframe CPU sequence: rounded quotient, truncation,
    # fused multiply/subtract, then the sign adjustment (including +zero).
    if is_nan(a):
        return fma_result(quiet(a), narrow)
    if is_nan(b):
        return fma_result(quiet(b), narrow)
    if is_inf(a) or absolute(b) == 0:
        return fma_result(invalid(), narrow)
    var quotient = div(a, b)
    if narrow:
        quotient = from_float32(to_float32(quotient))
    quotient = integral(quotient, quotient >> 63 == 0)
    var remainder = fma(quotient ^ SIGN, b, a, narrow)
    if narrow:
        remainder = from_float32(UInt32(remainder))
    var adjust = (less(a, 0) != less(b, 0)) and not equal(remainder, 0)
    return fma_result(add(remainder, b if adjust else UInt64(0)), narrow)


def to_integer(
    value: UInt64, width: UInt32, unsigned: Bool
) -> Tuple[UInt64, Bool]:
    if is_nan(value) or is_inf(value):
        return (0, False)
    if absolute(value) == 0:
        return (0, True)
    var p = parts(value)
    if p[1] < 0:
        return (0, True)
    if p[1] >= 64:
        return (0, False)
    var magnitude = p[0] << UInt64(p[1] - 52) if p[1] >= 52 else p[0] >> UInt64(
        52 - p[1]
    )
    var negative = value >> 63 != 0
    var maximum = UInt64.MAX
    if unsigned:
        if width != 64:
            maximum = (UInt64(1) << UInt64(width)) - 1
    else:
        maximum = (UInt64(1) << UInt64(width - 1)) - UInt64(not negative)
    var valid = magnitude <= maximum and not (
        unsigned and negative and magnitude != 0
    )
    return (UInt64(0) - magnitude if negative else magnitude, valid)


def from_wide(low: UInt64, high: UInt64, unsigned: Bool) -> UInt64:
    var sign = SIGN if not unsigned and high >> 63 != 0 else UInt64(0)
    var l = low
    var h = high
    if sign:
        l = ~l + 1
        h = ~h + UInt64(l == 0)
    if (l | h) == 0:
        return 0
    var leading = UInt32(64) if h else UInt32(0)
    var rest = (h if h else l) >> 1
    while rest:
        leading += 1
        rest >>= 1
    var sig = UInt64(0)
    if leading <= 55:
        sig = l << UInt64(55 - leading)
    else:
        var distance = UInt64(leading - 55)
        if distance < 64:
            sig = (h << (64 - distance)) | (l >> distance)
            sig |= UInt64((l << (64 - distance)) != 0)
        else:
            sig = jam(h, UInt32(distance - 64)) | UInt64(l != 0)
    return pack(sign, Int32(leading), sig)


def exp(a: UInt64) -> UInt64:
    if is_nan(a):
        return quiet(a)
    if is_inf(a):
        return UInt64(0) if a >> 63 else a
    if less(0x4086300000000000, a):
        return INF
    if less(a, 0xC087700000000000):
        return 0
    var rounded = round_integer(mul(a, 0x3FF71547652B82FE))
    var k = to_integer(rounded, 64, False)[0].cast[DType.int64]()
    var count = from_signed(k)
    var r = fma(count ^ SIGN, 0x3FE62E42FEFA39EF, a)
    r = fma(count ^ SIGN, 0x3C7ABC9E3B39803F, r)
    var total = UInt64(0x3FF0000000000000)
    var term = total
    for n in range(1, 25):
        term = div(mul(term, r), from_unsigned(UInt64(n)))
        total = add(total, term)
    var p = parts(total)
    return pack(0, p[1] + Int32(k), p[0] << 3)


def log(a: UInt64) -> UInt64:
    if is_nan(a):
        return quiet(a)
    if absolute(a) == 0:
        return SIGN | INF
    if a >> 63:
        return invalid()
    if is_inf(a):
        return a
    var p = parts(a)
    var exponent = p[1]
    var m = UInt64(0x3FF0000000000000) | (p[0] & FRACTION)
    if less(0x3FF6A09E667F3BCD, m):
        m -= HIDDEN
        exponent += 1
    var y = div(sub(m, 0x3FF0000000000000), add(m, 0x3FF0000000000000))
    var square = mul(y, y)
    var term = y
    var total = y
    for n in range(3, 50, 2):
        term = mul(term, square)
        total = add(total, div(term, from_unsigned(UInt64(n))))
    var count = from_signed(Int64(exponent))
    var result = fma(count, 0x3C7ABC9E3B39803F, mul(total, 0x4000000000000000))
    return fma(count, 0x3FE62E42FEFA39EF, result)


comptime Double = Tuple[UInt64, UInt64]


def double_sum(a: UInt64, b: UInt64) -> Double:
    var s = add(a, b)
    var v = sub(s, a)
    return (s, add(sub(a, sub(s, v)), sub(b, v)))


def double_add(a: Double, b: Double) -> Double:
    var s = double_sum(a[0], b[0])
    return double_sum(s[0], add(s[1], add(a[1], b[1])))


def double_neg(a: Double) -> Double:
    return (a[0] ^ SIGN, a[1] ^ SIGN)


def double_mul(a: Double, b: Double) -> Double:
    var p = mul(a[0], b[0])
    var error = fma(a[0], b[0], p ^ SIGN)
    error = add(error, add(mul(a[0], b[1]), mul(a[1], b[0])))
    error = add(error, mul(a[1], b[1]))
    return double_sum(p, error)


def double_div(a: Double, b: Double) -> Double:
    var q = div(a[0], b[0])
    var residual = double_add(a, double_neg(double_mul(b, (q, UInt64(0)))))
    return double_sum(q, div(add(residual[0], residual[1]), b[0]))


def power_log(a: UInt64) -> Double:
    var p = parts(a)
    var exponent = p[1]
    var m = UInt64(0x3FF0000000000000) | (p[0] & FRACTION)
    if less(0x3FF6A09E667F3BCD, m):
        m -= HIDDEN
        exponent += 1
    var y = double_div(
        double_sum(m, 0xBFF0000000000000), double_sum(m, 0x3FF0000000000000)
    )
    var square = double_mul(y, y)
    var term = y
    var total = y
    for n in range(3, 50, 2):
        term = double_mul(term, square)
        total = double_add(
            total, double_div(term, (from_unsigned(UInt64(n)), UInt64(0)))
        )
    return double_add(
        double_mul(total, (UInt64(0x4000000000000000), UInt64(0))),
        double_mul(
            (from_signed(Int64(exponent)), UInt64(0)),
            (UInt64(0x3FE62E42FEFA39EF), UInt64(0x3C7ABC9E3B39803F)),
        ),
    )


def power_exp(a: Double) -> UInt64:
    var k = to_integer(round_integer(mul(a[0], 0x3FF71547652B82FE)), 64, False)[
        0
    ].cast[DType.int64]()
    var count: Double = (from_signed(k), 0)
    var ln2: Double = (0x3FE62E42FEFA39EF, 0x3C7ABC9E3B39803F)
    var r = double_add(a, double_neg(double_mul(count, ln2)))
    var term: Double = (0x3FF0000000000000, 0)
    var total = term
    for n in range(1, 25):
        term = double_div(
            double_mul(term, r), (from_unsigned(UInt64(n)), UInt64(0))
        )
        total = double_add(total, term)
    var p = parts(total[0])
    var sig = p[0] << 3
    if absolute(total[1]):
        sig = sig - 1 if less(total[1], 0) else sig + 1
    return pack(0, p[1] + Int32(k), sig)


def power(a: UInt64, b: UInt64) -> UInt64:
    comptime ONE = UInt64(0x3FF0000000000000)
    var x = absolute(a)
    var y = absolute(b)
    if y == 0 or a == ONE:
        return ONE
    if is_nan(a):
        return quiet(a)
    if is_nan(b):
        return quiet(b)
    if y == INF:
        if x == ONE:
            return ONE
        return INF if less(ONE, x) != Bool(b >> 63) else UInt64(0)
    var exponent = parts(y)
    var integral_exponent = exponent[1] >= 0
    var odd = False
    if integral_exponent and exponent[1] <= 52:
        var shift = UInt64(52 - exponent[1])
        integral_exponent = (exponent[0] & ((UInt64(1) << shift) - 1)) == 0
        odd = integral_exponent and ((exponent[0] >> shift) & 1) != 0
    var sign = SIGN if a >> 63 and odd else UInt64(0)
    if x == 0:
        return sign | (INF if b >> 63 else UInt64(0))
    if x == INF:
        return sign | (UInt64(0) if b >> 63 else INF)
    if a >> 63 and not integral_exponent:
        return invalid()
    if b == ONE:
        return a
    if b == (ONE | SIGN):
        return div(ONE, a)
    if b == 0x3FE0000000000000:
        return sqrt(x)
    var base = parts(x)
    if integral_exponent and base[0] == HIDDEN:
        var p = mul(from_signed(Int64(base[1])), b)
        if not less(p, 0x4090000000000000):
            return sign | INF
        if not less(0xC090CC0000000000, p):
            return sign
        return pack(
            sign,
            Int32(to_integer(p, 64, False)[0].cast[DType.int64]()),
            0x0080000000000000,
        )
    if b == 0x4000000000000000:
        return mul(a, a)
    var approximate = mul(b, log(x))
    if less(0x408F400000000000, absolute(approximate)):
        return sign | (UInt64(0) if approximate >> 63 else INF)
    return sign | power_exp(double_mul((b, UInt64(0)), power_log(x)))
