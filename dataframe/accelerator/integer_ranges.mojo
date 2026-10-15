"""Conservative integer error proofs using types and expression literals only."""
from std.math import abs
from std.memory import bitcast
from dataframe.binding import BoundExpr
from dataframe.dtype import DataType
from dataframe.decimal import pow10, divide_half_even
from dataframe.expr import (
    COL,
    LIT_INT,
    ADD,
    SUB,
    MUL,
    DIV,
    NEG,
    ABS,
    FLOORDIV,
    POW,
    CAST,
    CLIP_LOW,
    CLIP_HIGH,
    FILL_NULL,
    KEEP_NULLS,
    WHEN,
    LIT_BOOL,
    is_comparison,
)


@fieldwise_init
struct IntegerRange(Copyable):
    var low: Int128
    var high: Int128
    var known: Bool


def _type_range(dtype: DataType) -> IntegerRange:
    if dtype == DataType.BOOL:
        return IntegerRange(0, 1, True)
    if not dtype.is_integer() or dtype.bit_width() > 64:
        return IntegerRange(0, 0, False)
    var bits = dtype.bit_width()
    if dtype.is_unsigned():
        return IntegerRange(0, (Int128(1) << Int128(bits)) - 1, True)
    var edge = Int128(1) << Int128(bits - 1)
    return IntegerRange(-edge, edge - 1, True)


def _decimal_cast_safe(
    source: DataType, target: DataType, bounds: IntegerRange
) -> Bool:
    var source_bounds = bounds.copy()
    if source.is_decimal():
        # with_dtype can retag raw storage without validating precision. Prove
        # against the physical range, never the declared decimal digit count.
        source_bounds = _type_range(source.physical())
        if target == DataType.BOOL or target.is_float():
            return True
    if not source_bounds.known:
        return False
    var from_scale = source.scale() if source.is_decimal() else 0
    if target == DataType.BOOL or target.is_float():
        return True
    var to_scale = target.scale() if target.is_decimal() else 0
    var limits = _type_range(target)
    if target.is_decimal():
        var maximum = pow10(target.precision()) - 1
        limits = IntegerRange(-maximum, maximum, True)
    if not limits.known:
        return False
    if to_scale > from_scale:
        var factor = pow10(to_scale - from_scale)
        return (
            source_bounds.low >= 0
            or abs(source_bounds.low) <= abs(limits.low) // factor
        ) and source_bounds.high <= limits.high // factor
    if from_scale > to_scale:
        var divisor = pow10(from_scale - to_scale)
        source_bounds = IntegerRange(
            divide_half_even(source_bounds.low, divisor),
            divide_half_even(source_bounds.high, divisor),
            True,
        )
    return limits.low <= source_bounds.low and source_bounds.high <= limits.high


def _decimal_scale_safe(dtype: DataType, scale: Int) -> Bool:
    if scale == dtype.scale():
        return True
    var bounds = _type_range(dtype.physical())
    if not bounds.known:
        return False
    return max(abs(bounds.low), abs(bounds.high)) <= (
        Int128.MAX // pow10(scale - dtype.scale())
    )


def proven_integer_nodes(bound: BoundExpr) -> List[Bool]:
    """Prove that individual checked nodes cannot fail for any valid operand.

    This does not inspect source values or remove evaluations. Unknown nodes
    retain their entire result type range; each potentially throwing child
    still needs its own proof or an existing observable terminal boundary.
    """
    var ranges = List[IntegerRange]()
    var safe = List[Bool]()
    for i in range(len(bound.expr._nodes)):
        ref node = bound.expr._nodes[i]
        var result = _type_range(bound.dtypes[i])
        var limits = result.copy()
        var proved = False
        var a = IntegerRange(0, 0, False)
        var b = IntegerRange(0, 0, False)
        if node.left >= 0:
            a = ranges[node.left].copy()
        if node.right >= 0:
            b = ranges[node.right].copy()
        if result.known:
            if node.op == LIT_INT or node.op == LIT_BOOL:
                var value = Int128(node.integer)
                if bound.dtypes[i].is_unsigned():
                    value = Int128(bitcast[DType.uint64](node.integer))
                if limits.low <= value and value <= limits.high:
                    result = IntegerRange(value, value, True)
            elif node.op == WHEN and b.known:
                var other = IntegerRange(0, 0, False)
                if node.extra >= 0:
                    other = ranges[node.extra].copy()
                if other.known:
                    result = IntegerRange(
                        min(b.low, other.low), max(b.high, other.high), True
                    )
            elif a.known:
                if node.op == CAST or node.op == KEEP_NULLS:
                    result = a.copy()
                    proved = node.op == CAST
                elif node.op == NEG:
                    result = IntegerRange(-a.high, -a.low, True)
                    proved = True
                elif node.op == ABS:
                    var lo = Int128(0) if a.low <= 0 and a.high >= 0 else min(
                        abs(a.low), abs(a.high)
                    )
                    result = IntegerRange(
                        lo, max(abs(a.low), abs(a.high)), True
                    )
                    proved = True
                elif b.known:
                    if node.op == ADD:
                        result = IntegerRange(
                            a.low + b.low, a.high + b.high, True
                        )
                        proved = True
                    elif node.op == SUB:
                        result = IntegerRange(
                            a.low - b.high, a.high - b.low, True
                        )
                        proved = True
                    elif node.op == MUL:
                        # Check magnitudes before multiplication: UInt64 pairs
                        # can exceed signed Int128 even in metadata arithmetic.
                        var lo = limits.high
                        var hi = limits.low
                        proved = True
                        for x in List[Int128]([a.low, a.high]):
                            for y in List[Int128]([b.low, b.high]):
                                var negative = (x < 0) != (y < 0)
                                var edge = (
                                    -limits.low if negative else limits.high
                                )
                                if y != 0 and abs(x) > edge // abs(y):
                                    proved = False
                                elif proved:
                                    var product = x * y
                                    lo = min(lo, product)
                                    hi = max(hi, product)
                        if proved:
                            result = IntegerRange(lo, hi, True)
                    elif node.op == FLOORDIV:
                        # Zero divisors produce nulls. The only integer floor
                        # division overflow is the signed minimum divided by -1.
                        var operands_fit = (
                            limits.low <= a.low
                            and a.high <= limits.high
                            and limits.low <= b.low
                            and b.high <= limits.high
                        )
                        var excludes_minus_one = b.high < -1 or b.low > -1
                        var negation_fits = (
                            limits.low <= -a.high and -a.low <= limits.high
                        )
                        proved = operands_fit and (
                            excludes_minus_one or negation_fits
                        )
                    elif node.op == POW and b.low == b.high:
                        if b.low == 0:
                            result = IntegerRange(1, 1, True)
                            proved = True
                        elif b.low == 1:
                            result = a.copy()
                            proved = True
                    elif node.op == CLIP_LOW:
                        result = IntegerRange(
                            max(a.low, b.low), max(a.high, b.high), True
                        )
                    elif node.op == CLIP_HIGH:
                        result = IntegerRange(
                            min(a.low, b.low), min(a.high, b.high), True
                        )
                    elif node.op == FILL_NULL:
                        result = IntegerRange(
                            min(a.low, b.low), max(a.high, b.high), True
                        )
            if result.low < limits.low or result.high > limits.high:
                result = limits.copy()
                proved = False
        if (
            node.op == CAST
            and node.left >= 0
            and (
                bound.dtypes[i].is_decimal()
                or bound.dtypes[node.left].is_decimal()
            )
        ):
            proved = _decimal_cast_safe(
                bound.dtypes[node.left], bound.dtypes[i], a
            )
        if bound.dtypes[i].is_decimal():
            result = _type_range(bound.dtypes[i].physical())
            if node.op != COL:
                result = IntegerRange(0, 0, False)
            if (node.op == NEG or node.op == ABS) and a.known:
                var limits = _type_range(bound.dtypes[i].physical())
                if bound.dtypes[i].decimal_width() == 128:
                    limits = IntegerRange(Int128.MIN, Int128.MAX, True)
                if limits.known and a.low > limits.low:
                    var low = -a.high
                    var high = -a.low
                    if node.op == ABS:
                        low = Int128(0) if a.low <= 0 and a.high >= 0 else min(
                            abs(a.low), abs(a.high)
                        )
                        high = max(abs(a.low), abs(a.high))
                    result = IntegerRange(low, high, True)
                    proved = True
            if (node.op == ADD or node.op == SUB) and a.known and b.known:
                var scale = bound.dtypes[i].scale()
                var left_factor = pow10(scale - bound.dtypes[node.left].scale())
                var right_factor = pow10(
                    scale - bound.dtypes[node.right].scale()
                )
                if (
                    max(abs(a.low), abs(a.high)) <= Int128.MAX // left_factor
                    and max(abs(b.low), abs(b.high))
                    <= Int128.MAX // right_factor
                ):
                    var low = a.low * left_factor
                    var high = a.high * left_factor
                    var other_low = b.low * right_factor
                    var other_high = b.high * right_factor
                    if node.op == SUB:
                        var previous_low = other_low
                        other_low = -other_high
                        other_high = -previous_low
                    var maximum = pow10(bound.dtypes[i].precision()) - 1
                    if (
                        low >= -maximum
                        and high <= maximum
                        and other_low >= -maximum
                        and other_high <= maximum
                    ):
                        if (other_low >= 0 or low >= -maximum - other_low) and (
                            other_high <= 0 or high <= maximum - other_high
                        ):
                            result = IntegerRange(
                                low + other_low, high + other_high, True
                            )
                            proved = True
        if (
            bound.dtypes[i].is_decimal()
            and node.op == MUL
            and a.known
            and b.known
        ):
            var largest_a = max(abs(a.low), abs(a.high))
            var largest_b = max(abs(b.low), abs(b.high))
            var maximum = pow10(bound.dtypes[i].precision()) - 1
            if node.op == MUL and (
                largest_b == 0 or largest_a <= Int128.MAX // largest_b
            ):
                var low = min(
                    min(a.low * b.low, a.low * b.high),
                    min(a.high * b.low, a.high * b.high),
                )
                var high = max(
                    max(a.low * b.low, a.low * b.high),
                    max(a.high * b.low, a.high * b.high),
                )
                if low >= -maximum and high <= maximum:
                    result = IntegerRange(low, high, True)
                    proved = True
        if is_comparison(node.op) and node.left >= 0 and node.right >= 0:
            var left_type = bound.dtypes[node.left]
            var right_type = bound.dtypes[node.right]
            if left_type.is_decimal() and right_type.is_decimal():
                var scale = max(left_type.scale(), right_type.scale())
                proved = _decimal_scale_safe(
                    left_type, scale
                ) and _decimal_scale_safe(right_type, scale)
        ranges.append(result^)
        safe.append(proved)
    return safe^
