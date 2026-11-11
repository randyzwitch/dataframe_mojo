"""Canonical native membership sets for the public numeric is_in shapes."""
from std.collections import Dict
from std.memory import bitcast
from dataframe import DataFrame, DataType, Expr
from dataframe.binding import bind, BoundExpr
from dataframe.expr import (
    COL,
    EQ,
    OR,
    FILL_NULL,
    KEEP_NULLS,
    INT_IS_IN,
    LIT_BOOL,
    LIT_INT,
    LIT_FLOAT,
    LIT_NULL,
)
from .filter_plan import supported_dtype
from .expression_plan import _integer_fits_float32

comptime MAX_MEMBERS = 4096


@fieldwise_init
struct NativeMembership(Copyable, Movable):
    var column: String
    var dtype: DataType
    var keys: List[UInt64]


def integer_key(value: Int64) -> UInt64:
    return bitcast[DType.uint64](value) ^ (UInt64(1) << 63)


def float_key(raw: UInt32) -> UInt64:
    var bits = raw
    if (bits & 0x7FFF_FFFF) == 0:
        bits = 0
    return UInt64(~bits if bits & 0x8000_0000 else bits ^ UInt32(0x8000_0000))


def _canonical(values: List[UInt64]) -> Optional[List[UInt64]]:
    # This is metadata canonicalization, bounded by distinct candidates.
    var seen = Dict[UInt64, Bool]()
    var keys = List[UInt64]()
    for value in values:
        if value not in seen:
            if len(keys) == MAX_MEMBERS:
                return None
            seen[value] = True
            keys.append(value)
    sort(keys)
    return keys^


def compile_membership(
    expression: Expr, at: Int, frame: DataFrame
) raises -> Optional[NativeMembership]:
    if at < 0 or at >= len(expression._nodes):
        return None
    ref root = expression._nodes[at]
    if root.op != INT_IS_IN and root.op != KEEP_NULLS:
        return None
    if root.left < 0 or root.left >= at:
        return None
    ref column = expression._nodes[root.left]
    if column.op != COL or column.text not in frame.columns():
        return None
    var dtype = frame.column(column.text).dtype()
    if not supported_dtype(dtype):
        return None
    var values = List[UInt64]()
    if root.op == INT_IS_IN:
        if (
            len(root.dtypes) != 1
            or not root.dtypes[0]
            or root.dtypes[0].value() != dtype
        ):
            return None
        if dtype != DataType.INT32 and dtype != DataType.INT64:
            return None
        for part in root.text.split(","):
            var value: Int64
            try:
                value = Int64(atol(part))
            except:
                return None
            if dtype == DataType.INT32 and (
                value < Int64(Int32.MIN) or value > Int64(Int32.MAX)
            ):
                return None
            values.append(integer_key(value))
    else:
        # Expr.is_in's fallback is KEEP_NULLS(input, OR(EQ.fill_null(False),
        # ... False)). Recognize that complete shape, not arbitrary OR trees.
        if root.right < 0 or root.right >= at:
            return None
        var pending = List[Int]([root.right])
        var literals = List[Int]()
        while pending:
            var index = pending.pop()
            ref node = expression._nodes[index]
            if node.op == OR:
                if (
                    node.left < 0
                    or node.left >= index
                    or node.right < 0
                    or node.right >= index
                ):
                    return None
                pending.append(node.left)
                pending.append(node.right)
            elif node.op == LIT_BOOL and node.integer == 0:
                pass
            elif node.op == FILL_NULL:
                if (
                    node.left < 0
                    or node.left >= index
                    or node.right < 0
                    or node.right >= index
                ):
                    return None
                ref fill = expression._nodes[node.right]
                ref compare = expression._nodes[node.left]
                if fill.op != LIT_BOOL or fill.integer != 0 or compare.op != EQ:
                    return None
                if (
                    compare.left < 0
                    or compare.left >= node.left
                    or compare.right < 0
                    or compare.right >= node.left
                ):
                    return None
                ref input = expression._nodes[compare.left]
                if input.op != COL or input.text != column.text:
                    return None
                literals.append(compare.right)
            else:
                return None
        # CPU binding validates typed literal equality (including nulls) before
        # we omit nonmatching null/NaN candidates or deduplicate the set.
        var bound: BoundExpr
        try:
            bound = bind(expression, frame._columns, fuse=False)
        except:
            return None
        for index in literals:
            ref literal = expression._nodes[index]
            if literal.op == LIT_NULL:
                continue
            if bound.dtypes[index] != dtype:
                return None
            if dtype == DataType.FLOAT32:
                if literal.op != LIT_INT and literal.op != LIT_FLOAT:
                    return None
                if literal.op == LIT_INT and not _integer_fits_float32(
                    literal.integer
                ):
                    return None
                var value = (
                    Float64(literal.integer) if literal.op
                    == LIT_INT else literal.floating
                )
                if value != value:
                    continue
                var narrow = Float32(value)
                if Float64(narrow) != value:
                    return None
                values.append(float_key(bitcast[DType.uint32](narrow)))
            else:
                if literal.op != LIT_INT:
                    return None
                if dtype == DataType.INT32 and (
                    literal.integer < Int64(Int32.MIN)
                    or literal.integer > Int64(Int32.MAX)
                ):
                    return None
                values.append(integer_key(literal.integer))
    var canonical = _canonical(values)
    if not canonical:
        return None
    return NativeMembership(column.text.copy(), dtype, canonical.take())
