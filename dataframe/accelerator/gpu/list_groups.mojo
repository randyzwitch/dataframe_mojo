"""Scoped GPU grouping layout for resident List producers.

Existing ordering/grouping kernels establish typed key equivalence and stable
row runs. A second stable merge orders groups by their first source row. No key
values leave the device; GroupBy maintain_order=False permits this same order.
"""
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer, Pointer
from dataframe import Expr
from dataframe.accelerator.rows import ROW_GROUP_KEY
from .residency import Scope, Table, checked_add, checked_mul
from .runtime import _preflight
from .program import Program
from .device_lists import GroupOwner
from . import ordering, grouping


def _terminal(
    starts: Pointer[UInt32, MutAnyOrigin], groups: UInt32, rows: UInt32
):
    if global_idx.x == 0:
        starts[unsafe_offset=Int(groups)] = rows


def _merge_first(
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    order: Pointer[UInt32, ImmutAnyOrigin],
    target: Pointer[UInt32, MutAnyOrigin],
    groups: UInt32,
    run_width: UInt32,
):
    var row = global_idx.x
    if row >= Int(groups):
        return
    var width = Int(run_width)
    var base = row // (2 * width) * (2 * width)
    var middle = min(base + width, Int(groups))
    var end = min(base + 2 * width, Int(groups))
    var left = row < middle
    var first = middle if left else base
    var lo = first
    var hi = end if left else middle
    var own = order[unsafe_offset=row]
    var own_first = index[unsafe_offset=Int(starts[unsafe_offset=Int(own)])]
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        var other = order[unsafe_offset=mid]
        var other_first = index[
            unsafe_offset=Int(starts[unsafe_offset=Int(other)])
        ]
        if other_first < own_first:
            lo = mid + 1
        else:
            hi = mid
    target[
        unsafe_offset=base + row - (base if left else middle) + lo - first
    ] = own


def build_groups(
    source: Table, keys: List[String], scope: Scope
) raises -> Optional[ArcPointer[GroupOwner]]:
    """Return actual GPU stable runs and first-occurrence canonical group IDs."""
    if not len(keys):
        raise Error("Grouped List requires at least one key")
    # Binding is metadata-only, using the same descriptors as normal grouping.
    var plan = _preflight(
        source.schema.lazy()
        .group_by(keys, maintain_order=True)
        .agg(List[Expr]()),
        source_rows=source.rows,
    )
    if len(plan.steps) != 2 and len(plan.steps) != 1:
        raise Error("Grouped List key-only grouping plan changed")
    var start = -1
    for step in plan.steps:
        if step.grouped:
            if start >= 0:
                raise Error("Grouped List requires one grouping header")
            start = step.start
    if start < 0:
        raise Error("Grouped List key-only grouping header is missing")
    if not source.rows:
        return None
    if not source.storage or not source.matrix()[].scope.same_query(scope):
        raise Error("Grouped List source belongs to another query context")
    var matrix = source.matrix()
    var program = Program(plan)
    for node in range(len(program.node_types)):
        if program.code[4 * node] != ROW_GROUP_KEY:
            continue
        var logical = Int(program.code[4 * node + 1])
        if logical < 0 or logical >= plan.source.width():
            raise Error("Grouped List key slot changed")
        var name = plan.source._columns[logical].name()
        var mapped = -1
        for column in range(source.schema.width()):
            if source.schema._columns[column].name() == name:
                mapped = column
                break
        if mapped < 0:
            raise Error("Grouped List key source is missing")
        program.code[4 * node + 1] = source.columns[mapped]
    var rows = source.rows
    var blocks = (rows + 255) // 256
    var bytes = checked_add(
        checked_add(checked_mul(rows, 16), 4),
        checked_add(checked_mul(blocks, 8), 32),
    )
    var program_bytes = checked_add(
        checked_mul(len(program.code), 8),
        checked_add(
            checked_mul(len(program.node_types), 8),
            checked_mul(len(program.literals), 8),
        ),
    )
    _ = scope.check(checked_add(bytes, checked_mul(program_bytes, 2)))
    var ctx = scope.context()
    var index = ctx.enqueue_create_buffer[DType.uint32](rows)
    var spare = ctx.enqueue_create_buffer[DType.uint32](rows)
    var positions = ctx.enqueue_create_buffer[DType.uint32](rows)
    var starts = ctx.enqueue_create_buffer[DType.uint32](rows + 1)
    var blocks_buffer = ctx.enqueue_create_buffer[DType.uint32](2 * blocks)
    var metadata = ctx.enqueue_create_buffer[DType.uint32](4)
    var code = ctx.enqueue_create_buffer[DType.int64](len(program.code))
    var types = ctx.enqueue_create_buffer[DType.int64](len(program.node_types))
    var literals = ctx.enqueue_create_buffer[DType.uint64](
        len(program.literals)
    )
    var held_index = index.copy()
    var held_spare = spare.copy()
    var held_positions = positions.copy()
    var held_starts = starts.copy()
    var held_blocks = blocks_buffer.copy()
    var held_metadata = metadata.copy()
    var held_code = code.copy()
    var held_types = types.copy()
    var held_literals = literals.copy()
    try:
        ctx.enqueue_copy(code, program.code.unsafe_ptr())
        ctx.enqueue_copy(types, program.node_types.unsafe_ptr())
        ctx.enqueue_copy(literals, program.literals.unsafe_ptr())
        scope.record_upload(program_bytes)
        ctx.enqueue_function[ordering.order_init](
            index.unsafe_ptr(), UInt32(rows), grid_dim=blocks, block_dim=256
        )
        scope.record_launches()
        var width = 1
        while width < rows:
            ctx.enqueue_function[ordering.order_merge](
                matrix[].words.unsafe_ptr().unsafe_offset(source.offset),
                matrix[].validity.unsafe_ptr().unsafe_offset(source.offset),
                index.unsafe_ptr(),
                spare.unsafe_ptr(),
                code.unsafe_ptr(),
                types.unsafe_ptr(),
                UInt32(start + 1),
                UInt32(len(keys)),
                UInt32(rows),
                UInt32(matrix[].capacity),
                UInt32(width),
                UInt32(matrix[].slots),
                literals.unsafe_ptr(),
                grid_dim=blocks,
                block_dim=256,
            )
            var old = index^
            index = spare^
            spare = old^
            width *= 2
            scope.record_launches()
        ctx.enqueue_function[grouping.group_mark](
            matrix[].words.unsafe_ptr().unsafe_offset(source.offset),
            matrix[].validity.unsafe_ptr().unsafe_offset(source.offset),
            matrix[].errors.unsafe_ptr().unsafe_offset(source.offset),
            index.unsafe_ptr(),
            positions.unsafe_ptr(),
            blocks_buffer.unsafe_ptr(),
            code.unsafe_ptr(),
            types.unsafe_ptr(),
            UInt32(start),
            UInt32(len(keys)),
            UInt32(rows),
            UInt32(matrix[].capacity),
            UInt32(matrix[].slots),
            grid_dim=blocks,
            block_dim=256,
        )
        ctx.enqueue_function[grouping.group_offsets](
            blocks_buffer.unsafe_ptr(),
            metadata.unsafe_ptr(),
            UInt32(blocks),
            grid_dim=1,
            block_dim=1,
        )
        scope.record_launches(2)
        var shape = scope.read_control(metadata)
        scope.record_download(16)
        if shape[1]:
            raise Error(
                "Mojo GPU group key failed: checked arithmetic overflow or strict cast failed"
            )
        var groups = Int(shape[0])
        if groups < 1 or groups > rows:
            raise Error("Grouped List group count is invalid")
        ctx.enqueue_function[grouping.group_starts](
            matrix[].words.unsafe_ptr().unsafe_offset(source.offset),
            matrix[].validity.unsafe_ptr().unsafe_offset(source.offset),
            index.unsafe_ptr(),
            positions.unsafe_ptr(),
            blocks_buffer.unsafe_ptr(),
            starts.unsafe_ptr(),
            code.unsafe_ptr(),
            types.unsafe_ptr(),
            UInt32(start),
            UInt32(len(keys)),
            UInt32(rows),
            UInt32(matrix[].capacity),
            UInt32(matrix[].slots),
            grid_dim=blocks,
            block_dim=256,
        )
        ctx.enqueue_function[_terminal](
            starts.unsafe_ptr(),
            UInt32(groups),
            UInt32(rows),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[ordering.order_init](
            positions.unsafe_ptr(),
            UInt32(groups),
            grid_dim=(groups + 255) // 256,
            block_dim=256,
        )
        scope.record_launches(3)
        width = 1
        while width < groups:
            ctx.enqueue_function[_merge_first](
                index.unsafe_ptr(),
                starts.unsafe_ptr(),
                positions.unsafe_ptr(),
                spare.unsafe_ptr(),
                UInt32(groups),
                UInt32(width),
                grid_dim=(groups + 255) // 256,
                block_dim=256,
            )
            var old = positions^
            positions = spare^
            spare = old^
            width *= 2
            scope.record_launches()
        scope.synchronize()
        var result = ArcPointer(
            GroupOwner(scope, rows, groups, index^, starts^, positions^)
        )
        _ = program^
        _ = matrix^
        _ = held_index^
        _ = held_spare^
        _ = held_positions^
        _ = held_starts^
        _ = held_blocks^
        _ = held_metadata^
        _ = held_code^
        _ = held_types^
        _ = held_literals^
        return result^
    except error:
        scope.synchronize()
        _ = program^
        _ = matrix^
        _ = held_index^
        _ = held_spare^
        _ = held_positions^
        _ = held_starts^
        _ = held_blocks^
        _ = held_metadata^
        _ = held_code^
        _ = held_types^
        _ = held_literals^
        raise error^
