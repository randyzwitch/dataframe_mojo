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
from std.memory import pack_bits
from .dtype import DataType
from .binding import BoundExpr
from .bool_column import BoolColumn
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
                ._ptr()
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
    # Evaluate inside physical source chunks when a large caller leaves them
    # chunked. Each local window starts at row zero, even if source chunk
    # boundaries differ between columns.
    var chunked = False
    for step in steps:
        if step.op == COL and columns[step.source].is_chunked():
            chunked = True
            break
    if chunked and length > 0:
        var output = Series("", BoolColumn(List[Bool]())) if bound.dtypes[
            root
        ] == DataType.BOOL else Series("", Column[Float64]([]))
        var cursor = offset
        var end = offset + length
        while cursor < end:
            var local_columns = columns.copy()
            var local_offsets = List[Int](length=len(columns), fill=cursor)
            var segment_end = end
            for step in steps:
                if step.op != COL:
                    continue
                var source = step.source
                if columns[source].is_chunked():
                    var part = columns[source]._chunk_at(cursor)
                    segment_end = min(
                        segment_end, cursor + len(part[0]) - part[1]
                    )
                    local_columns[source] = part[0].copy()
                    local_offsets[source] = part[1]
            var segment_length = segment_end - cursor
            for step in steps:
                if step.op == COL:
                    var source = step.source
                    local_columns[source] = local_columns[source].slice(
                        local_offsets[source], segment_length
                    )
            var piece = fused[width](
                bound, local_columns^, root, 0, segment_length
            )
            if cursor == offset and segment_end == end:
                return piece^
            output._append_series(piece)
            cursor = segment_end
        return output^
    var predicate = bound.dtypes[root] == DataType.BOOL
    var packed_valid = List[UInt8](length=(length + 7) // 8, fill=255)
    for step in steps:
        if step.op == COL:
            ref column = columns[step.source]._data[Column[Float64]]
            ref source_bits = column._bits[]
            if len(source_bits) == 0:
                continue
            var base = column._offset + offset
            for byte in range(len(packed_valid)):
                var bit = base + byte * 8
                var index = bit // 8
                var shift = bit % 8
                var value = UInt16(source_bits[index]) >> UInt16(shift)
                if shift > 0 and index + 1 < len(source_bits):
                    value |= UInt16(source_bits[index + 1]) << UInt16(8 - shift)
                packed_valid[byte] &= UInt8(value & UInt16(255))
    var values = List[Float64](length=0 if predicate else length, fill=0)
    var packed_values = List[UInt8](
        length=(length + 7) // 8 if predicate else 0, fill=0
    )
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
            comptime if width == 8:
                packed_values[start // 8] |= UInt8(
                    pack_bits[DType.uint8](result)
                ) << UInt8(start % 8)
            else:
                comptime for lane in range(width):
                    if result[lane]:
                        var row = start + lane
                        packed_values[row // 8] |= UInt8(1) << UInt8(row % 8)
        else:
            values.unsafe_ptr().unsafe_offset(start).unsafe_store(wide[last])
    for i in range(main, length):
        _run[1](steps, columns, offset + i, narrow)
        if predicate:
            if _compare[1](
                steps[last].op,
                narrow[steps[last].left],
                narrow[steps[last].right],
            )[0]:
                packed_values[i // 8] |= UInt8(1) << UInt8(i % 8)
        else:
            values[i] = narrow[last][0]
    for byte in range(len(packed_valid)):
        if packed_valid[byte] == 255:
            continue
        for lane in range(8):
            var i = byte * 8 + lane
            if i >= length:
                break
            if packed_valid[byte] & (UInt8(1) << UInt8(lane)) == 0:
                if not predicate:
                    values[i] = 0
    if predicate:
        for byte in range(len(packed_values)):
            packed_values[byte] &= packed_valid[byte]
        return Series(
            "",
            BoolColumn(
                values=packed_values^, bits=packed_valid^, length=length
            ),
        )
    return Series("", Column[Float64](values=values^, bits=packed_valid^))
