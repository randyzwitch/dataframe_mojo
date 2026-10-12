"""Conservative integer error proofs using types and expression literals only."""
from std.math import abs
from std.memory import bitcast
from dataframe.binding import BoundExpr
from dataframe.dtype import DataType
from dataframe.expr import (
    LIT_INT,
    ADD,
    SUB,
    MUL,
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
        ranges.append(result^)
        safe.append(proved)
    return safe^
