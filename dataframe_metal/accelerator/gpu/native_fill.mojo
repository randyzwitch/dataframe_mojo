"""Leaf-only fill operations with independent validity and optional shared loads."""
from std.memory import bitcast
from dataframe.expr import FILL_NAN
from .multi_reduce import Bytes, read_integer
from .filter_sum import _valid


@always_inline
def native_fill_bits(
    left: Bytes,
    right: Bytes,
    left_bits: Bytes,
    right_bits: Bytes,
    left_offset: UInt32,
    right_offset: UInt32,
    row: Int,
    tag: UInt32,
    op: UInt32,
    flags: UInt32,
    side: UInt32,
    literal: UInt64,
    want: Bool,
    reuse: Bool = False,
    cached: UInt32 = 0,
    cached_valid: Bool = True,
) -> Tuple[UInt64, Bool]:
    var ls = side & 3 == 1
    var rs = side & 3 == 2
    var lv = not Bool(side & 4) if ls else (
        cached_valid if reuse else not Bool(flags & 1)
        or _valid(left_bits, left_offset, row)
    )
    var rv = not Bool(side & 4) if rs else (
        not Bool(flags & 2) or _valid(right_bits, right_offset, row)
    )
    var a = UInt64(0)
    if lv and (want or op == UInt32(FILL_NAN)):
        a = literal if ls else (
            UInt64(cached) if reuse else (
                UInt64(left.unsafe_bitcast[UInt32]()[unsafe_offset=row]) if tag
                == 2 else bitcast[DType.uint64](read_integer(left, tag, row))
            )
        )
    var replace = not lv
    if op == UInt32(FILL_NAN):
        # FILL_NAN preserves input nulls; a NaN replacement may itself be null.
        if not lv:
            return (UInt64(0), False)
        replace = (UInt32(a) & 0x7FFF_FFFF) > 0x7F80_0000
    if not replace:
        return (a, lv)
    if not rv or not want:
        return (UInt64(0), rv)
    var b = literal if rs else (
        UInt64(right.unsafe_bitcast[UInt32]()[unsafe_offset=row]) if tag
        == 2 else bitcast[DType.uint64](read_integer(right, tag, row))
    )
    return (b, True)
