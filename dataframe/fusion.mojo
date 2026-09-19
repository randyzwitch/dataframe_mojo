"""Fused Float64 elementwise kernels.

A fusible subtree (Float64 columns and literals combined with + - * / and at
most one comparison at the root) is compiled into a small register program
and evaluated one SIMD vector at a time, loading directly from the source
column buffers. No intermediate column is materialized for inner nodes, and
source values are never copied into batch slices. Validity is the AND of the
leaf columns' validity, which matches null propagation through every fused
operation; null rows get a zero payload. Results are identical to the
unfused kernels, which remain available with bind(..., fuse=False).
"""
from .dtype import DataType
from .binding import BoundExpr
from .column import Column
from .expr import COL, LIT_FLOAT, ADD, SUB, MUL, DIV, GT, LT, GE, LE, EQ, NE
from .series import Series


@fieldwise_init
struct _Step(Copyable):
    var op: Int
    var left: Int
    var right: Int
    var source: Int
    var literal: Float64


def _program(bound: BoundExpr, root: Int) -> List[_Step]:
    """Reachable nodes in topological order, with operands as step slots."""
    var keep = List[Bool](length=root + 1, fill=False)
    keep[root] = True
    for reverse in range(root + 1):
        var i = root - reverse
        if keep[i]:
            ref node = bound.expr._nodes[i]
            if node.left >= 0:
                keep[node.left] = True
            if node.right >= 0:
                keep[node.right] = True
    var slot = List[Int](length=root + 1, fill=-1)
    var steps = List[_Step]()
    for i in range(root + 1):
        if not keep[i]:
            continue
        ref node = bound.expr._nodes[i]
        slot[i] = len(steps)
        steps.append(
            _Step(
                node.op,
                slot[node.left] if node.left >= 0 else -1,
                slot[node.right] if node.right >= 0 else -1,
                bound.sources[i],
                node.floating,
            )
        )
    return steps^


def _apply[
    width: Int
](
    op: Int, x: SIMD[DType.float64, width], y: SIMD[DType.float64, width]
) -> SIMD[DType.float64, width]:
    if op == ADD:
        return x + y
    if op == SUB:
        return x - y
    if op == MUL:
        return x * y
    return x / y


def _compare[
    width: Int
](
    op: Int, x: SIMD[DType.float64, width], y: SIMD[DType.float64, width]
) -> SIMD[DType.bool, width]:
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
    # SIMD ne is ordered; IEEE requires NaN != NaN.
    return ~x.eq(y)


def _run[
    width: Int
](
    steps: List[_Step],
    columns: List[Series],
    row: Int,
    mut registers: List[SIMD[DType.float64, width]],
):
    """Evaluate every arithmetic step for `width` rows starting at row."""
    for k in range(len(steps)):
        ref step = steps[k]
        if step.op == COL:
            registers[k] = (
                columns[step.source]
                ._data[Column[Float64]]
                ._values.unsafe_ptr()
                .unsafe_load[width=width](row)
            )
        elif step.op == LIT_FLOAT:
            registers[k] = SIMD[DType.float64, width](step.literal)
        elif (
            step.op == ADD or step.op == SUB or step.op == MUL or step.op == DIV
        ):
            registers[k] = _apply[width](
                step.op, registers[step.left], registers[step.right]
            )


def fused[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    root: Int,
    offset: Int,
    length: Int,
) raises -> Series:
    var steps = _program(bound, root)
    var predicate = bound.dtypes[root] == DataType.BOOL
    var valid = List[Bool](length=length, fill=True)
    for step in steps:
        if step.op == COL:
            ref column = columns[step.source]._data[Column[Float64]]
            for i in range(length):
                valid[i] = valid[i] and column._valid(offset + i)
    var values = List[Float64](length=0 if predicate else length, fill=0)
    var flags = List[Bool](length=length if predicate else 0, fill=False)
    var last = len(steps) - 1
    var wide = List[SIMD[DType.float64, width]](
        length=len(steps), fill=SIMD[DType.float64, width](0)
    )
    var narrow = List[SIMD[DType.float64, 1]](
        length=len(steps), fill=SIMD[DType.float64, 1](0)
    )
    var main = length - length % width
    for start in range(0, main, width):
        _run[width](steps, columns, offset + start, wide)
        if predicate:
            var result = _compare[width](
                steps[last].op, wide[steps[last].left], wide[steps[last].right]
            )
            comptime for lane in range(width):
                flags[start + lane] = result[lane]
        else:
            values.unsafe_ptr().unsafe_offset(start).unsafe_store(wide[last])
    for i in range(main, length):
        _run[1](steps, columns, offset + i, narrow)
        if predicate:
            flags[i] = _compare[1](
                steps[last].op,
                narrow[steps[last].left],
                narrow[steps[last].right],
            )[0]
        else:
            values[i] = narrow[last][0]
    for i in range(length):
        if not valid[i]:
            if predicate:
                flags[i] = False
            else:
                values[i] = 0
    if predicate:
        return Series("", Column[Bool](flags^, valid))
    return Series("", Column[Float64](values^, valid))
