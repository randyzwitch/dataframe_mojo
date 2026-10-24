"""Finish a proven greedy suffix with at most three ASCII exit bytes."""
from std.bit import count_leading_zeros, count_trailing_zeros
from std.memory import pack_bits


@always_inline
def _exit[
    reverse: Bool, count: Int
](
    text: Span[UInt8, ImmutAnyOrigin], var at: Int, bound: Int, code: Int32
) -> Int:
    var packed = UInt32(code)
    var a = UInt8(packed & 127)
    var b = UInt8((packed >> 7) & 127)
    var c = UInt8((packed >> 14) & 127)
    var av = SIMD[DType.uint8, 16](a)
    var bv = SIMD[DType.uint8, 16](b)
    var cv = SIMD[DType.uint8, 16](c)
    var p = text.unsafe_ptr()
    comptime if reverse:
        while at - bound >= 16:
            var block = p.unsafe_load[width=16](at - 16)
            var hits = block.eq(av)
            comptime if count > 1:
                hits |= block.eq(bv)
            comptime if count > 2:
                hits |= block.eq(cv)
            var mask = UInt64(pack_bits[DType.uint16](hits))
            if mask:
                return at - 16 + 64 - Int(count_leading_zeros(mask))
            at -= 16
        while at > bound:
            at -= 1
            var byte = p[unsafe_offset=at]
            if (
                byte == a
                or (count > 1 and byte == b)
                or (count > 2 and byte == c)
            ):
                return at + 1
    else:
        while bound - at >= 16:
            var block = p.unsafe_load[width=16](at)
            var hits = block.eq(av)
            comptime if count > 1:
                hits |= block.eq(bv)
            comptime if count > 2:
                hits |= block.eq(cv)
            var mask = UInt64(pack_bits[DType.uint16](hits))
            if mask:
                return at + Int(count_trailing_zeros(mask))
            at += 16
        while at < bound:
            var byte = p[unsafe_offset=at]
            if (
                byte == a
                or (count > 1 and byte == b)
                or (count > 2 and byte == c)
            ):
                return at
            at += 1
    return bound


@always_inline
def suffix_end[
    reverse: Bool
](text: Span[UInt8, ImmutAnyOrigin], at: Int, bound: Int, code: Int32) -> Int:
    var count = Int((UInt32(code) >> 21) & 3)
    if count == 0:
        return bound
    if count == 1:
        return _exit[reverse, 1](text, at, bound, code)
    if count == 2:
        return _exit[reverse, 2](text, at, bound, code)
    return _exit[reverse, 3](text, at, bound, code)
