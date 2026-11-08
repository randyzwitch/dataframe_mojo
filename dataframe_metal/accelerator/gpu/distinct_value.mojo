"""Exact native distinct keys: one NaN class, one zero class, separate nulls."""
from std.memory import bitcast
from .group_hash import _key_hash


@always_inline
def distinct_canonical(raw: UInt64, tag: UInt32) -> UInt64:
    if tag == 2:
        var bits = UInt32(raw)
        var magnitude = bits & 0x7FFF_FFFF
        if magnitude == 0:
            return UInt64(0)
        if magnitude > 0x7F80_0000:
            return UInt64(0x7FC0_0000)
        return UInt64(bits)
    if tag == 0:
        return UInt64(UInt32(raw))
    if tag == 3:
        return UInt64(raw != 0)
    return raw


@always_inline
def distinct_same(
    a: UInt64, av: Bool, b: UInt64, bv: Bool, tag: UInt32
) -> Bool:
    if not av or not bv:
        return av == bv
    return distinct_canonical(a, tag) == distinct_canonical(b, tag)


@always_inline
def distinct_hash(
    raw: UInt64, valid: Bool, tag: UInt32, group: UInt32
) -> UInt32:
    var key = distinct_canonical(raw, tag) if valid else UInt64(0)
    return (
        group
        ^ _key_hash(bitcast[DType.int64](key))
        ^ (UInt32(0xA24BAED5) if valid else UInt32(0xD1B54A35))
    ) * UInt32(0x9E3779B9)
