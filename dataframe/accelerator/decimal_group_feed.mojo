"""Commuting precision-fault proof for narrow Decimal aggregate operands.

ADD/SUB scale intermediates are bounded by 2*2**63*10**18 < 2**127;
MUL intermediates by 2**126. Successful results fit their checked precision
at most18 and physical32/64 storage. Thus only precision category8 can fail.
Every <=UInt32.MAX-row Decimal SUM/MEAN prefix and merge is below2**95,
well inside precision38. With infallible keys/states/siblings and immutable
sources, scheduling may commute precision sites without changing the public
GPU exception: it reports a generic precision category, not CPU dtype digits.
Original occurrence ranks are retained; all faults precede any consumer.
"""
from dataframe import DataFrame, DataType, Series
from dataframe.binding import BoundExpr, bind
from dataframe.expr import (
    SUM,
    MEAN,
    COUNT,
    LEN,
    NULL_COUNT,
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_NULL,
    ADD,
    SUB,
    MUL,
    CAST,
    WHEN,
    FILL_NULL,
    IS_NULL,
    IS_NOT_NULL,
    EQ,
    NE,
    GT,
    GE,
    LT,
    LE,
    AND,
    OR,
    XOR,
    NOT,
    INT_IS_IN,
    subtree,
)
from .integer_ranges import proven_integer_nodes
from . import ordered_rows
from dataframe.decimal import pow10
from dataframe.lazy import LazyFrame, AGG
from .metadata import schema_frame
from .decimal_prefix import prefix_infallible

comptime MATERIALIZATION = "metal_narrow_decimal_group_feed"
comptime ORDERED_MODE = 7


def infallible_sum(dtype: DataType, rows: Int) -> Bool:
    if not dtype.is_decimal() or rows < 0 or rows > Int(UInt32.MAX):
        return False
    var width = dtype.decimal_width()
    if width != 32 and width != 64:
        return False
    var maximum = Int128(1) << Int128(width - 1)
    return Int128(rows) <= (pow10(38) - 1) // maximum


def narrow_value(bound: BoundExpr, at: Int) -> Bool:
    """Physical64 storage or one exact sign-extended signed64 literal.

    money() deliberately has Arrow Decimal128 dtype. Literal folding retains
    that dtype and its high word, so precision alone is never a value bound.
    Only a canonical sign extension proves the same physical64 magnitude.
    """
    var dtype = bound.dtypes[at]
    if not dtype.is_decimal() or dtype.scale() > 18:
        return False
    if dtype.decimal_width() <= 64:
        return True
    ref node = bound.expr._nodes[at]
    return node.op == LIT_INT and (
        not node.text2.byte_length()
        or node.text2 == String(-1 if node.integer < 0 else 0)
    )


def _selection_branch(bound: BoundExpr, child: Int, target: DataType) -> Bool:
    if child < 0 or child >= len(bound.dtypes):
        return False
    var source = bound.dtypes[child]
    if source == target:
        return True
    if (
        not source.is_decimal()
        or not target.is_decimal()
        or source.scale() != target.scale()
        or bound.expr._nodes[child].op != LIT_INT
        or not narrow_value(bound, child)
    ):
        return False
    # CPU casts both valid branches before selecting, even when unchosen.
    # A canonical signed64 literal at the same scale needs no word rescale
    # and cannot raise if its exact raw value fits the target precision.
    ref node = bound.expr._nodes[child]
    if node.text2.byte_length() and node.text2 != String(
        -1 if node.integer < 0 else 0
    ):
        return False
    var raw = Int128(node.integer)
    if (source.decimal_width() == 32 or target.decimal_width() == 32) and (
        raw < Int128(Int32.MIN) or raw > Int128(Int32.MAX)
    ):
        return False
    var limit = pow10(target.precision())
    return raw > -limit and raw < limit


def operand(bound: BoundExpr) -> Tuple[Bool, Bool]:
    if not ordered_rows.eligible(bound):
        return (False, False)
    var proven = proven_integer_nodes(bound)
    var checked = False
    for at in range(len(bound.expr._nodes)):
        ref node = bound.expr._nodes[at]
        var dtype = bound.dtypes[at]
        if node.op in [COL, LIT_INT, LIT_FLOAT, LIT_BOOL, LIT_NULL]:
            if dtype.is_decimal() and not narrow_value(bound, at):
                return (False, False)
            continue
        if node.op == CAST and proven[at]:
            continue
        if node.op == WHEN:
            if (
                node.extra < 0
                or bound.dtypes[node.left] != DataType.BOOL
                or not _selection_branch(bound, node.right, dtype)
                or not _selection_branch(bound, node.extra, dtype)
            ):
                return (False, False)
            continue
        if node.op == FILL_NULL:
            if (
                node.left < 0
                or node.right < 0
                or not _selection_branch(bound, node.left, dtype)
                or not _selection_branch(bound, node.right, dtype)
            ):
                return (False, False)
            continue
        if node.op in [IS_NULL, IS_NOT_NULL, AND, OR, XOR, NOT]:
            continue
        if (
            node.op == INT_IS_IN
            and node.left >= 0
            and bound.dtypes[node.left].is_integer()
            and dtype == DataType.BOOL
        ):
            continue
        if node.op in [EQ, NE, GT, GE, LT, LE]:
            if (
                bound.dtypes[node.left].is_decimal()
                or bound.dtypes[node.right].is_decimal()
            ):
                return (False, False)
            continue
        if node.op not in [ADD, SUB, MUL] or not dtype.is_decimal():
            return (False, False)
        if dtype.decimal_width() > 64 or dtype.scale() > 18:
            return (False, False)
        for child in [node.left, node.right]:
            if child < 0 or not bound.dtypes[child].is_decimal():
                return (False, False)
            if not narrow_value(bound, child):
                return (False, False)
        checked |= not proven[at]
    return (True, checked)


def eligible(
    expressions: List[BoundExpr], schema: List[Series], keys: List[String]
) raises -> Bool:
    var checked = False
    for name in keys:
        var key = DataFrame(schema.copy()).column(name)
        if (
            key.dtype().is_categorical()
            or key.dtype()._is_bytes()
            or key.dtype().is_nested()
        ):
            return False
        # Coded String keys use the existing original-code category owner at
        # lowering. This is independent of the commuting operand-fault proof;
        # publication drops that private identity exactly as CPU GROUP does.
    for bound in expressions:
        var last = len(bound.expr._nodes) - 1
        if last < 0:
            return False
        ref root = bound.expr._nodes[last]
        if root.op not in [SUM, MEAN, COUNT, LEN, NULL_COUNT] or root.left < 0:
            return False
        var input = bind(subtree(bound.expr, root.left), schema)
        var proof = operand(input)
        if not proof[0]:
            return False
        if root.op in [COUNT, LEN, NULL_COUNT] and proof[1]:
            # Count fast paths may avoid reading a direct input. Keep all
            # count operands infallible rather than inventing observations.
            return False
        checked |= proof[1]
        if root.op == SUM or root.op == MEAN:
            if not infallible_sum(
                input.dtypes[len(input.dtypes) - 1], Int(UInt32.MAX)
            ):
                return False
    return checked


def producer(query: LazyFrame, index: Int) raises -> Bool:
    ref operation = query._nodes[index]
    if operation.kind != AGG or not len(operation.names):
        return False
    var possible = False
    for expression in operation.exprs:
        for node in expression._nodes:
            possible |= node.op in [ADD, SUB, MUL]
    if not possible:
        return False
    if not prefix_infallible(query, operation.left):
        return False
    for key in operation.window_keys:
        if (
            key.dtype.is_categorical()
            or key.dtype._is_bytes()
            or key.dtype.is_nested()
        ):
            return False
    ref source = query._nodes[operation.left]
    for field in source.text_fields:
        if field.name in operation.names and (
            (field.category and not field.category.value().string_codes)
            or field.dtype._is_bytes()
            or field.dtype.is_nested()
        ):
            return False
    var schema = schema_frame(query, operation.left)
    var bounds = List[BoundExpr]()
    for expression in operation.exprs:
        bounds.append(bind(expression, schema._columns))
    return eligible(bounds, schema._columns, operation.names)
