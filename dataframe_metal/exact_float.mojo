"""Host finishing for exact Float32 exponent bins used by Float64 means."""
from std.memory import bitcast
from std.utils.numerics import inf


def finish_float_bins(
    totals: List[Int128], nan: Bool, positive: Bool, negative: Bool
) -> Float64:
    if nan or (positive and negative):
        return Float64(0) / Float64(0)
    if positive:
        return inf[DType.float64]()
    if negative:
        return -inf[DType.float64]()
    # Normalize signed radix-2**32 bins before rounding, so cancellation
    # across bins cannot destroy small contributions in floating arithmetic.
    var digits = List[UInt64](length=10, fill=0)
    var carry = Int128(0)
    for i in range(10):
        var value = carry
        if i < 9:
            value += totals[i]
        digits[i] = UInt64(value & Int128(0xFFFF_FFFF))
        carry = value >> 32
    var negative_total = carry < 0
    if negative_total:
        var add = UInt64(1)
        for i in range(10):
            var value = (digits[i] ^ UInt64(0xFFFF_FFFF)) + add
            digits[i] = value & 0xFFFF_FFFF
            add = value >> 32
    var top = 9
    while top >= 0 and digits[top] == 0:
        top -= 1
    if top < 0:
        return 0
    var leading = digits[top]
    var bits = 0
    while leading:
        leading >>= 1
        bits += 1
    var width = top * 32 + bits
    var shift = max(0, width - 53)
    var significand = UInt64(0)
    for i in reversed(range(shift, width)):
        significand = (significand << 1) | (
            (digits[i // 32] >> UInt64(i % 32)) & 1
        )
    if shift:
        var halfway = (
            digits[(shift - 1) // 32] >> UInt64((shift - 1) % 32)
        ) & 1
        var sticky = False
        for i in range(shift - 1):
            sticky = sticky or Bool((digits[i // 32] >> UInt64(i % 32)) & 1)
        if halfway and (sticky or Bool(significand & 1)):
            significand += 1
    # The exponent is in [-149, 106], safely inside Float64's normal range.
    var power = UInt64(shift - 149 + 1023) << 52
    var value = Float64(significand) * bitcast[DType.float64](power)
    return -value if negative_total else value
