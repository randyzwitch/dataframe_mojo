"""Pure floating-point operations shared by CPU and accelerator kernels."""
from .expr import ADD, SUB, MUL, GT, LT, GE, LE, EQ


@always_inline
def arithmetic[
    D: DType, width: Int
](op: Int, x: SIMD[D, width], y: SIMD[D, width]) -> SIMD[D, width]:
    if op == ADD:
        return x + y
    if op == SUB:
        return x - y
    if op == MUL:
        return x * y
    return x / y


@always_inline
def compare[
    D: DType, width: Int
](op: Int, x: SIMD[D, width], y: SIMD[D, width]) -> SIMD[DType.bool, width]:
    if op == GT:
        return x.gt(y)
    if op == LT:
        return x.lt(y)
    if op == GE:
        return x.ge(y)
    if op == LE:
        return x.le(y)
    if op == EQ:
        return x.eq(y)
    # Ordered SIMD ne excludes NaNs; IEEE != must include them.
    return ~x.eq(y)
