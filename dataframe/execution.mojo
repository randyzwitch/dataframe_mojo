"""Bounded eager execution; expression nodes dispatch once per column batch.

Reductions are separate passes, not per-row callbacks or per-group dataframes.
The reference reduction schedule is serial; Float64 results may be reassociated
by future parallel implementations. Int64 sums use exact wide states and check final overflow.
"""
from .expr import (
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_STRING,
    ADD,
    SUB,
    MUL,
    GT,
    EQ,
    SUM,
    COUNT,
    LT,
    GE,
    LE,
    NE,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    CLIP_LOW,
    CLIP_HIGH,
    NEG,
    ABS,
    SQRT,
    EXP,
    LOG,
    FLOOR,
    CEIL,
    ROUND,
    LIT_NULL,
    AND,
    OR,
    XOR,
    FILL_NULL,
    FILL_NAN,
    KEEP_NULLS,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    ANY,
    ALL,
    NULL_COUNT,
    WHEN,
    STR_CONCAT,
    is_reduction,
    is_string_op,
)
from .str_kernels import string_op, concat_strings
from .binding import BoundExpr, ROWS, AGGREGATE
from .column import Column
from .series import Series
from .expr_kernels import binary, unary, choose, fit_mask
from .aggregate import Reducer


def _empty(dtype: String) raises -> Series:
    if dtype == "int64":
        return Series("", Column[Int64]([]))
    if dtype == "float64":
        return Series("", Column[Float64]([]))
    if dtype == "bool":
        return Series("", Column[Bool]([]))
    if dtype == "string":
        return Series("", Column[String]([]))
    raise Error("Unknown expression dtype")


def _binary_op[
    width: Int
](op: Int, left: Series, right: Series, mask: List[Bool]) raises -> Series:
    """Map a runtime opcode to its compile-time specialized kernel."""
    if op == ADD:
        return binary[ADD, width](left, right, mask)
    if op == SUB:
        return binary[SUB, width](left, right, mask)
    if op == MUL:
        return binary[MUL, width](left, right, mask)
    if op == DIV:
        return binary[DIV, width](left, right, mask)
    if op == FLOORDIV:
        return binary[FLOORDIV, width](left, right, mask)
    if op == MOD:
        return binary[MOD, width](left, right, mask)
    if op == POW:
        return binary[POW, width](left, right, mask)
    if op == CLIP_LOW:
        return binary[CLIP_LOW, width](left, right, mask)
    if op == CLIP_HIGH:
        return binary[CLIP_HIGH, width](left, right, mask)
    if op == GT:
        return binary[GT, width](left, right, mask)
    if op == LT:
        return binary[LT, width](left, right, mask)
    if op == GE:
        return binary[GE, width](left, right, mask)
    if op == LE:
        return binary[LE, width](left, right, mask)
    if op == EQ:
        return binary[EQ, width](left, right, mask)
    if op == NE:
        return binary[NE, width](left, right, mask)
    if op == AND:
        return binary[AND, width](left, right, mask)
    if op == OR:
        return binary[OR, width](left, right, mask)
    if op == XOR:
        return binary[XOR, width](left, right, mask)
    if op == FILL_NULL:
        return binary[FILL_NULL, width](left, right, mask)
    if op == FILL_NAN:
        return binary[FILL_NAN, width](left, right, mask)
    if op == KEEP_NULLS:
        return binary[KEEP_NULLS, width](left, right, mask)
    raise Error("Unsupported binary expression node")


def _unary_op[
    width: Int
](op: Int, input: Series, integer: Int64, mask: List[Bool]) raises -> Series:
    if op == NEG:
        return unary[NEG, width](input, integer, mask)
    if op == ABS:
        return unary[ABS, width](input, integer, mask)
    if op == SQRT:
        return unary[SQRT, width](input, integer, mask)
    if op == EXP:
        return unary[EXP, width](input, integer, mask)
    if op == LOG:
        return unary[LOG, width](input, integer, mask)
    if op == FLOOR:
        return unary[FLOOR, width](input, integer, mask)
    if op == CEIL:
        return unary[CEIL, width](input, integer, mask)
    if op == ROUND:
        return unary[ROUND, width](input, integer, mask)
    if op == NOT:
        return unary[NOT, width](input, integer, mask)
    if op == IS_NULL:
        return unary[IS_NULL, width](input, integer, mask)
    if op == IS_NOT_NULL:
        return unary[IS_NOT_NULL, width](input, integer, mask)
    if op == IS_NAN:
        return unary[IS_NAN, width](input, integer, mask)
    if op == IS_NOT_NAN:
        return unary[IS_NOT_NAN, width](input, integer, mask)
    if op == IS_FINITE:
        return unary[IS_FINITE, width](input, integer, mask)
    if op == IS_INFINITE:
        return unary[IS_INFINITE, width](input, integer, mask)
    raise Error("Unsupported unary expression node")


def _eval[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    aggregates: List[Series],
    index: Int,
    offset: Int,
    length: Int,
    grouped: Bool,
    mask: List[Bool],
) raises -> Series:
    """Evaluate one node for rows [offset, offset + length).

    Returns `length` values, or one value for scalar results that callers
    broadcast. Aggregate nodes read precomputed states instead of recursing.
    `mask` (empty means all rows) marks rows whose result can be observed;
    conditional branches narrow it so unselected rows never raise.
    """
    ref node = bound.expr._nodes[index]
    if node.op == COL:
        return columns[bound.sources[index]].slice(offset, length)
    if node.op == LIT_INT:
        return Series("", Column[Int64]([node.integer]))
    if node.op == LIT_FLOAT:
        return Series("", Column[Float64]([node.floating]))
    if node.op == LIT_BOOL:
        return Series("", Column[Bool]([Bool(node.integer)]))
    if node.op == LIT_STRING:
        return Series("", Column[String]([node.text]))
    if node.op == LIT_NULL:
        return Series.full_null("", node.text, 1)
    if is_reduction(node.op):
        if grouped:
            return aggregates[index].slice(offset, length)
        return aggregates[index].copy()
    if node.op == WHEN:
        return _conditional[width](
            bound, columns, aggregates, index, offset, length, grouped, mask
        )
    var left = _eval[width](
        bound, columns, aggregates, node.left, offset, length, grouped, mask
    )
    if is_string_op(node.op):
        return string_op(node, left)
    if node.right < 0:
        return _unary_op[width](node.op, left, node.integer, mask)
    var right = _eval[width](
        bound, columns, aggregates, node.right, offset, length, grouped, mask
    )
    if node.op == STR_CONCAT:
        return concat_strings(left, right, node.text)
    return _binary_op[width](node.op, left, right, mask)


def _conditional[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    aggregates: List[Series],
    index: Int,
    offset: Int,
    length: Int,
    grouped: Bool,
    mask: List[Bool],
) raises -> Series:
    ref node = bound.expr._nodes[index]
    var shape = bound.shapes[index]
    var size = (
        length if shape == ROWS or (grouped and shape == AGGREGATE) else 1
    )
    var active = fit_mask(mask, size)
    var predicate = _eval[width](
        bound, columns, aggregates, node.left, offset, length, grouped, mask
    )
    ref flags = predicate._data[Column[Bool]]
    var selected = List[Bool](capacity=size)
    var then_mask = List[Bool](capacity=size)
    var other_mask = List[Bool](capacity=size)
    for i in range(size):
        var p = 0 if len(flags) == 1 else i
        var take = flags._valid(p) and flags._values[p]
        var observed = len(active) == 0 or active[i]
        selected.append(take)
        then_mask.append(observed and take)
        other_mask.append(observed and not take)
    var then = _eval[width](
        bound,
        columns,
        aggregates,
        node.right,
        offset,
        length,
        grouped,
        then_mask,
    )
    var other: Series
    if node.extra >= 0:
        other = _eval[width](
            bound,
            columns,
            aggregates,
            node.extra,
            offset,
            length,
            grouped,
            other_mask,
        )
    else:
        other = Series.full_null("", bound.dtypes[index], 1)
    return choose(selected, then, other)


def _batch[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    aggregates: List[Series],
    target: Int,
    offset: Int,
    length: Int,
    grouped: Bool,
) raises -> Series:
    return _eval[width](
        bound,
        columns,
        aggregates,
        target,
        offset,
        length,
        grouped,
        List[Bool](),
    )


def evaluate[
    width: Int = 4
](
    bound: BoundExpr,
    columns: List[Series],
    height: Int,
    *,
    batch_size: Int = 1024,
    grouped: Bool = False,
    groups: List[Int] = List[Int](),
    group_count: Int = 1,
) raises -> Series:
    """Evaluate a bound expression; temporary vectors are bounded by batch_size.

    Full output and O(groups * aggregates) states are materialized. Each
    aggregate currently scans its own input; shared-subexpression fusion and
    parallel state merging are future execution changes, not API changes.
    """
    if batch_size <= 0:
        raise Error("batch_size must be positive")
    if grouped and (len(groups) != height or group_count < 0):
        raise Error("Invalid group mapping")
    var states = List[Series]()
    for i in range(len(bound.expr._nodes)):
        states.append(_empty(bound.dtypes[i]))
    for node_index in range(len(bound.expr._nodes)):
        var node = bound.expr._nodes[node_index].copy()
        if not is_reduction(node.op):
            continue
        var reducer = Reducer(
            node.op,
            bound.dtypes[node.left],
            group_count,
            node.min_count,
            node.integer,
            node.floating,
            node.text,
        )
        for offset in range(0, height, batch_size):
            var chunk = _batch[width](
                bound,
                columns,
                states,
                node.left,
                offset,
                min(batch_size, height - offset),
                False,
            )
            reducer.update(chunk, offset, grouped, groups)
        states[node_index] = reducer.finish()
    var root = len(bound.expr._nodes) - 1
    var result = _empty(bound.dtypes[root])
    var size = height if bound.shape() == ROWS else 1
    if grouped:
        size = group_count
    for offset in range(0, size, batch_size):
        var chunk = _batch[width](
            bound,
            columns,
            states,
            root,
            offset,
            min(batch_size, size - offset),
            grouped,
        )
        result._append_series(chunk)
    result._name = bound.expr._name
    return result^
