"""Shared signed Int64 arithmetic; callers supply their error vocabulary."""


@always_inline
def add[message: StaticString](a: Int64, b: Int64) raises -> Int64:
    if (b > 0 and a > Int64.MAX - b) or (b < 0 and a < Int64.MIN - b):
        raise Error(message)
    return a + b


@always_inline
def sub[message: StaticString](a: Int64, b: Int64) raises -> Int64:
    # Test the difference directly: negating MIN would itself overflow.
    if (b > 0 and a < Int64.MIN + b) or (b < 0 and a > Int64.MAX + b):
        raise Error(message)
    return a - b


@always_inline
def mul[message: StaticString](a: Int64, b: Int64) raises -> Int64:
    # Unsigned magnitudes represent abs(MIN). Only negative products may
    # have magnitude 2**63; zero retains the normal signed result.
    var negative = (a < 0) != (b < 0)
    var left = a.cast[DType.uint64]()
    var right = b.cast[DType.uint64]()
    left = UInt64(0) - left if a < 0 else left
    right = UInt64(0) - right if b < 0 else right
    var limit = (UInt64(1) << 63) - UInt64(not negative)
    if right != 0 and left > limit // right:
        raise Error(message)
    var magnitude = left * right
    var result = UInt64(0) - magnitude if negative else magnitude
    return result.cast[DType.int64]()
