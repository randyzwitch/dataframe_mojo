"""Raw magnitude proofs for valid Decimal values at successful barriers.

Output precision is not a source-value bound. Only physical storage, exact
literal words, retained SUM receipts, and observed checked arithmetic qualify.
The magnitude type is independent of the result's physical storage width.
"""
from dataframe.binding import BoundExpr
from dataframe.dtype import DataType
from dataframe.decimal import pow10
from dataframe.expr import (
    COL,
    LIT_INT,
    ADD,
    SUB,
    MUL,
    CAST,
    NEG,
    ABS,
    is_comparison,
)


@fieldwise_init
struct Magnitude(Copyable):
    var maximum: Int128
    var known: Bool


def unknown() -> Magnitude:
    return Magnitude(0, False)


def physical(dtype: DataType) -> Magnitude:
    if not dtype.is_decimal() or dtype.decimal_width() not in [32, 64]:
        return unknown()
    # Include the signed minimum; never abs() an arbitrary Int128 payload.
    return Magnitude(Int128(1) << Int128(dtype.decimal_width() - 1), True)


def product(a: Magnitude, b: Magnitude) -> Magnitude:
    if not a.known or not b.known or a.maximum < 0 or b.maximum < 0:
        return unknown()
    if a.maximum == 0 or b.maximum == 0:
        return Magnitude(0, True)
    if a.maximum > Int128.MAX // b.maximum:
        return unknown()
    return Magnitude(a.maximum * b.maximum, True)


def addition(a: Magnitude, b: Magnitude) -> Magnitude:
    if not a.known or not b.known or a.maximum > Int128.MAX - b.maximum:
        return unknown()
    return Magnitude(a.maximum + b.maximum, True)


def align(value: Magnitude, source_scale: Int, target_scale: Int) -> Magnitude:
    if source_scale < 0 or target_scale < source_scale or target_scale > 38:
        return unknown()
    return product(value, Magnitude(pow10(target_scale - source_scale), True))


def sum_bound(value: Magnitude, rows: Int) -> Magnitude:
    """The original GROUP input cardinality bounds every prefix and merge."""
    if rows < 0 or rows > Int(UInt32.MAX):
        return unknown()
    return product(value, Magnitude(Int128(rows), True))


def precision_safe(value: Magnitude, dtype: DataType) -> Bool:
    return (
        value.known
        and dtype.is_decimal()
        and value.maximum < pow10(dtype.precision())
    )


def _literal(bound: BoundExpr, at: Int) -> Magnitude:
    ref node = bound.expr._nodes[at]
    # Exactly one sign-extended signed64 raw word. An arbitrary Decimal128
    # literal requires a separate exact-word proof; its declared precision is
    # insufficient. Canonical literal folding produces this representation.
    if node.text2.byte_length() and node.text2 != String(
        -1 if node.integer < 0 else 0
    ):
        return unknown()
    var raw = Int128(node.integer)
    return Magnitude(-raw if raw < 0 else raw, True)


def analyze(
    bound: BoundExpr,
    columns: List[Magnitude],
    *,
    observed_success: Bool = False,
) -> Tuple[List[Magnitude], List[Bool]]:
    """Prove original intermediates using bound source slots and raw scales.

    observed_success is solely for an issuer after an original checked
    producer barrier: a successful ADD/SUB/MUL result passed check_limit.
    Consumers always use False and must prove ALL intermediate operations.
    No data values are read. Unknown operators, casts with rounding and
    conditional branches cannot issue new magnitude bounds.
    """
    var values = List[Magnitude](length=len(bound.expr._nodes), fill=unknown())
    var safe = List[Bool](length=len(bound.expr._nodes), fill=False)
    for at in range(len(bound.expr._nodes)):
        ref node = bound.expr._nodes[at]
        var dtype = bound.dtypes[at]
        if node.op == COL and dtype.is_decimal():
            var slot = bound.sources[at]
            values[at] = physical(dtype)
            if slot >= 0 and slot < len(columns) and columns[slot].known:
                values[at] = columns[slot].copy()
            safe[at] = True
            continue
        if node.op == LIT_INT and dtype.is_decimal():
            values[at] = _literal(bound, at)
            safe[at] = True
            continue
        if node.left < 0 or node.left >= at:
            continue
        var left = values[node.left].copy()
        var left_type = bound.dtypes[node.left]
        if not left_type.is_decimal():
            continue
        if node.op in [NEG, ABS] and dtype.is_decimal():
            values[at] = left.copy()
            safe[at] = left.known and (
                dtype.decimal_width() == 128
                or left.maximum
                < (Int128(1) << Int128(dtype.decimal_width() - 1))
            )
            if not safe[at]:
                values[at] = unknown()
            continue
        if (
            node.op == CAST
            and dtype.is_decimal()
            and dtype.scale() >= left_type.scale()
        ):
            var cast_value = align(left, left_type.scale(), dtype.scale())
            if precision_safe(cast_value, dtype):
                values[at] = cast_value.copy()
                safe[at] = True
            continue
        if node.right < 0 or node.right >= at:
            continue
        var right = values[node.right].copy()
        var right_type = bound.dtypes[node.right]
        if not right_type.is_decimal():
            continue
        var candidate: Magnitude
        if node.op == MUL and dtype.is_decimal():
            if dtype.scale() != left_type.scale() + right_type.scale():
                continue
            candidate = product(left, right)
        elif node.op in [ADD, SUB] or is_comparison(node.op):
            var scale = max(left_type.scale(), right_type.scale())
            var a = align(left, left_type.scale(), scale)
            var b = align(right, right_type.scale(), scale)
            if is_comparison(node.op):
                safe[at] = a.known and b.known
                continue
            if not dtype.is_decimal() or dtype.scale() != scale:
                continue
            candidate = addition(a, b)
        else:
            continue
        if precision_safe(candidate, dtype):
            values[at] = candidate.copy()
            safe[at] = True
        if observed_success:
            # This is an observed arithmetic occurrence, never a source tag.
            var checked_maximum = pow10(dtype.precision()) - 1
            values[at] = Magnitude(
                min(
                    candidate.maximum, checked_maximum
                ) if candidate.known else checked_maximum,
                True,
            )
    return (values^, safe^)
