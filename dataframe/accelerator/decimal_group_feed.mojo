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
from dataframe import DataFrame, DataType, Series, StringColumn
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
    subtree,
)
from .integer_ranges import proven_integer_nodes
from . import ordered_rows
from dataframe.decimal import pow10

comptime MATERIALIZATION = "metal_narrow_decimal_group_feed"
comptime ORDERED_MODE = 7


def infallible_sum(dtype: DataType, rows: Int) -> Bool:
    if not dtype.is_decimal() or rows < 0 or rows > Int(UInt32.MAX):
        return False
    var width = dtype.decimal_width()
    if width != 32 and width != 64:
        return False
    var maximum = Int128(1) << (width - 1)
    return Int128(rows) <= (pow10(38) - 1) // maximum


def operand(bound: BoundExpr) -> Tuple[Bool, Bool]:
    if not ordered_rows.eligible(bound):
        return (False, False)
    var proven = proven_integer_nodes(bound)
    var checked = False
    for at in range(len(bound.expr._nodes)):
        ref node = bound.expr._nodes[at]
        var dtype = bound.dtypes[at]
        if node.op in [COL, LIT_INT, LIT_FLOAT, LIT_BOOL, LIT_NULL]:
            if dtype.is_decimal() and (
                dtype.decimal_width() > 64 or dtype.scale() > 18
            ):
                return (False, False)
            continue
        if node.op == CAST and proven[at]:
            continue
        if node.op in [WHEN, IS_NULL, IS_NOT_NULL, AND, OR, XOR, NOT]:
            # Masks select original arithmetic occurrences. These operators
            # neither rescale Decimal operands nor introduce another fault.
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
            if (
                bound.dtypes[child].decimal_width() > 64
                or bound.dtypes[child].scale() > 18
            ):
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
        # An ordinary String hashes raw bytes. Coded String and CAT keys
        # require their separate original-code certificate, never inference.
        for chunk in key.chunks():
            if (
                chunk._data.isa[StringColumn]()
                and chunk._data[StringColumn]._codes
            ):
                return False
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
