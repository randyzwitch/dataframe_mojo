"""Bit-based native null/NaN selection and total missing-value equality."""
from dataframe.expr import FILL_NULL


@always_inline
def fill_value_bits(
    left: UInt64, left_valid: Bool, right: UInt64, right_valid: Bool, op: Int
) -> Tuple[UInt64, Bool]:
    var replace = not left_valid if op == FILL_NULL else (
        left_valid and (UInt32(left) & 0x7FFF_FFFF) > 0x7F80_0000
    )
    return (right, right_valid) if replace else (left, left_valid)


@always_inline
def equal_missing_integer(
    left: Int64, left_valid: Bool, right: Int64, right_valid: Bool
) -> Bool:
    if not left_valid or not right_valid:
        return left_valid == right_valid
    return left == right


@always_inline
def equal_missing_float32(
    left: UInt32, left_valid: Bool, right: UInt32, right_valid: Bool
) -> Bool:
    if not left_valid or not right_valid:
        return left_valid == right_valid
    var a = left & 0x7FFF_FFFF
    var b = right & 0x7FFF_FFFF
    if a > 0x7F80_0000 or b > 0x7F80_0000:
        return a > 0x7F80_0000 and b > 0x7F80_0000
    return left == right or (a == 0 and b == 0)
