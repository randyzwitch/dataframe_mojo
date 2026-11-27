"""Implicit numeric promotion: the types two numeric operands of different
types convert to before an operation combines them.

Two rules exist, because combining values and choosing between them need
different things:

- A common type holds every value of both inputs. Branches of a
  conditional, fill values and comparisons use it: the result is one of the
  inputs, or a test on them, so the smallest type that holds both is exact.
- An arithmetic type is what `+ - * // % **` compute in. When one operand's
  type holds every value of the other, the operation computes in that type.
  Otherwise there is no narrow type both sides widen to cheaply, and the
  operation computes in Int64, which leaves room for the result of the
  narrow operands and keeps every such mix on one well-tested width.

Integers mixed with floats compute in the float type (Float32 stays Float32,
since the float operand already decides the precision). Integers mixed with
decimals become decimals of scale 0 with enough digits for every value of the
integer type, after which the decimal rules decide the result. Decimals mixed
with floats compute in the float type.

Strings, booleans, temporal types and categoricals never promote: mixing
them with other types stays an error. Promotion happens before the operation,
so integer overflow is still checked at the promoted width.
"""
from .dtype import DataType
from .decimal import common_decimal, decimal_width_for


def integer_digits(dtype: DataType) -> Int:
    """Decimal digits needed for every value of an integer type."""
    var bits = dtype.bit_width()
    if bits == 8:
        return 3
    if bits == 16:
        return 5
    if bits == 32:
        return 10
    if bits == 64:
        return 20 if dtype.is_unsigned() else 19
    return 38


def _integer_decimal(dtype: DataType) raises -> DataType:
    """The decimal of scale 0 that holds every value of an integer type."""
    var digits = integer_digits(dtype)
    return DataType.decimal(digits, 0, decimal_width_for(digits))


def _promotable(dtype: DataType) -> Bool:
    """Integer, float and decimal types up to 64-bit integers (Int128 is
    not an arithmetic type)."""
    if dtype.is_integer():
        return dtype.bit_width() <= 64
    return dtype.is_float() or dtype.is_decimal()


def _holds(wide: DataType, narrow: DataType) -> Bool:
    """Whether integer type `wide` holds every value of `narrow`."""
    if wide.is_unsigned() == narrow.is_unsigned():
        return wide.bit_width() >= narrow.bit_width()
    # A signed type holds an unsigned one only with a spare bit for the
    # sign; an unsigned type never holds negative values.
    return not wide.is_unsigned() and wide.bit_width() > narrow.bit_width()


def _signed(bits: Int) -> DataType:
    if bits <= 8:
        return DataType.INT8
    if bits <= 16:
        return DataType.INT16
    if bits <= 32:
        return DataType.INT32
    return DataType.INT64


def _integer_type(
    left: DataType, right: DataType, arithmetic: Bool
) -> Optional[DataType]:
    if _holds(left, right):
        return left
    if _holds(right, left):
        return right
    # One side is signed and the other unsigned, and neither holds the
    # other. The common type is the narrowest signed type wider than the
    # unsigned side; arithmetic computes in Int64.
    var unsigned = left if left.is_unsigned() else right
    if unsigned.bit_width() >= 64:
        # UInt64 beside a signed type needs 128 bits.
        return None
    if arithmetic:
        return DataType.INT64
    var signed = right if left.is_unsigned() else left
    return _signed(max(signed.bit_width(), 2 * unsigned.bit_width()))


def promoted_operands(
    left: DataType, right: DataType, arithmetic: Bool
) raises -> Optional[Tuple[DataType, DataType]]:
    """The types `left` and `right` convert to before an operation combines
    them, or None when they do not promote (equal types, two decimals, which
    the decimal rules combine as they are, or a non-numeric operand).

    `arithmetic` picks the arithmetic type over the common type for two
    integers. Raises when both are numeric but no supported type holds both,
    which is UInt64 beside a signed integer.
    """
    if left == right or not _promotable(left) or not _promotable(right):
        return None
    if left.is_decimal() and right.is_decimal():
        return None
    if left.is_float() or right.is_float():
        var wide = DataType.FLOAT64 if (
            left == DataType.FLOAT64 or right == DataType.FLOAT64
        ) else DataType.FLOAT32
        return Optional((wide, wide))
    if left.is_decimal():
        return Optional((left, _integer_decimal(right)))
    if right.is_decimal():
        return Optional((_integer_decimal(left), right))
    var found = _integer_type(left, right, arithmetic)
    if not found:
        raise Error(
            "no common type holds both "
            + left.name()
            + " and "
            + right.name()
            + " (it would need 128-bit integers); cast one operand"
        )
    var common = found.value()
    return Optional((common, common))


def common_type(left: DataType, right: DataType) raises -> Optional[DataType]:
    """The one type that holds every value of `left` and of `right`, for
    columns stacked or matched across inputs: `left` when they are equal,
    the common numeric type when both are numeric, and None otherwise
    (including UInt64 beside a signed integer)."""
    if left == right:
        return left
    if left.is_decimal() and right.is_decimal():
        return common_decimal(left, right)
    if not _promotable(left) or not _promotable(right):
        return None
    if (
        left.is_integer()
        and right.is_integer()
        and not _integer_type(left, right, False)
    ):
        return None
    var pair = promoted_operands(left, right, False).value()
    if pair[0].is_decimal() and pair[1].is_decimal():
        return common_decimal(pair[0], pair[1])
    return pair[0]
