"""Scoped stable GPU grouping used by original-order floating reductions.

Extracted from the existing private grouped layout prototype. This module
owns only key runs, destination maps and row-group geometry; all row values
remain on device. No List producers, bucket scheduler or moment states.
"""
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer, Pointer
from std.collections import Optional
from dataframe import Expr
from dataframe.accelerator.rows import ROW_GROUP_KEY
from .residency import Scope, Table, checked_add, checked_mul
from .runtime import _preflight, _width
from .program import Program
from . import ordering, grouping


struct _Reservation(Movable):
    var scope: Scope
    var bytes: Int

    def __init__(out self, scope: Scope, bytes: Int) raises:
        scope._retain(bytes)
        self.scope = scope.copy()
        self.bytes = bytes

    def release(mut self, bytes: Int):
        self.scope._release(bytes)
        self.bytes -= bytes

    def __deinit__(deinit self):
        self.scope._release(self.bytes)


def descriptor_bytes(rows: Int) raises -> Int:
    return checked_add(checked_mul(rows + 1, 8), checked_mul(rows, 5))


struct DescriptorOwner(Movable):
    var scope: Scope
    var rows: Int
    var retained: Int
    var offsets: Optional[DeviceBuffer[DType.int64]]
    var valid: Optional[DeviceBuffer[DType.uint8]]
    var errors: Optional[DeviceBuffer[DType.uint32]]

    def __init__(
        out self,
        scope: Scope,
        rows: Int,
        var offsets: Optional[DeviceBuffer[DType.int64]] = None,
        var valid: Optional[DeviceBuffer[DType.uint8]] = None,
        var errors: Optional[DeviceBuffer[DType.uint32]] = None,
    ) raises:
        if rows < 0 or rows > Int(Int32.MAX):
            raise Error("Group parent count is invalid")
        if rows and (
            not offsets
            or not valid
            or not errors
            or len(offsets.value()) != rows + 1
            or len(valid.value()) != rows
            or len(errors.value()) != rows
        ):
            raise Error("Group descriptor geometry is invalid")
        if not rows and (offsets or valid or errors):
            raise Error("Empty Group descriptors must not allocate")
        var bytes = descriptor_bytes(rows) if rows else 0
        scope._retain(bytes)
        self.scope = scope.copy()
        self.rows = rows
        self.retained = bytes
        self.offsets = offsets^
        self.valid = valid^
        self.errors = errors^

    def __deinit__(deinit self):
        self.scope._release(self.retained)


struct GroupOwner(Movable):
    var scope: Scope
    var rows: Int
    var groups: Int
    var retained: Int
    var index: DeviceBuffer[DType.uint32]
    var starts: DeviceBuffer[DType.uint32]
    var order: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        scope: Scope,
        rows: Int,
        groups: Int,
        var index: DeviceBuffer[DType.uint32],
        var starts: DeviceBuffer[DType.uint32],
        var order: DeviceBuffer[DType.uint32],
    ) raises:
        if (
            rows < 0
            or rows > Int(Int32.MAX)
            or groups < 0
            or groups > Int(Int32.MAX)
            or len(index) != rows
            or len(starts) < groups + 1
            or len(order) < groups
        ):
            raise Error("Resident group layout geometry is invalid")
        # Group builders retain full N-capacity backing owners even when G<N.
        var bytes = checked_mul(
            checked_add(len(index), checked_add(len(starts), len(order))), 4
        )
        scope._retain(bytes)
        self.scope = scope.copy()
        self.rows = rows
        self.groups = groups
        self.retained = bytes
        self.index = index^
        self.starts = starts^
        self.order = order^

    def __deinit__(deinit self):
        self.scope._release(self.retained)


def validate_groups(
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    order: Pointer[UInt32, ImmutAnyOrigin],
    seen: Pointer[UInt32, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    groups: UInt32,
):
    """Linear control proof; no payload copy or whole-query CPU execution."""
    if global_idx.x != 0:
        return
    var failed = UInt32(
        starts[unsafe_offset=0] != 0
        or starts[unsafe_offset=Int(groups)] != rows
    )
    for row in range(Int(rows) + Int(groups)):
        seen[unsafe_offset=row] = 0
    for group in range(Int(groups)):
        var begin = starts[unsafe_offset=group]
        var end = starts[unsafe_offset=group + 1]
        if begin > end or end > rows:
            failed = 1
            continue
        var previous = UInt32(0)
        for at in range(Int(begin), Int(end)):
            var source = index[unsafe_offset=at]
            if source >= rows:
                failed = 1
                continue
            if seen[unsafe_offset=Int(source)] or (
                at > Int(begin) and source <= previous
            ):
                failed = 1
            seen[unsafe_offset=Int(source)] = 1
            previous = source
        var original = order[unsafe_offset=group]
        if original >= groups:
            failed = 1
        elif seen[unsafe_offset=Int(rows) + Int(original)]:
            failed = 1
        else:
            seen[unsafe_offset=Int(rows) + Int(original)] = 1
    control[unsafe_offset=0] = failed
    control[unsafe_offset=1] = rows


def grouped_offsets(
    starts: Pointer[UInt32, ImmutAnyOrigin],
    order: Pointer[UInt32, ImmutAnyOrigin],
    offsets: Pointer[Int64, MutAnyOrigin],
    valid: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    groups: UInt32,
):
    if global_idx.x != 0:
        return
    var total = Int64(0)
    offsets[unsafe_offset=0] = 0
    for group in range(Int(groups)):
        var original = Int(order[unsafe_offset=group])
        total += Int64(
            starts[unsafe_offset=original + 1] - starts[unsafe_offset=original]
        )
        offsets[unsafe_offset=group + 1] = total
        valid[unsafe_offset=group] = 1
        errors[unsafe_offset=group] = 0


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
        raise Error("Grouped float requires at least one key")
    # Binding is metadata-only, using the same descriptors as normal grouping.
    var plan = _preflight(
        source.schema.lazy()
        .group_by(keys, maintain_order=True)
        .agg(List[Expr]()),
        source_rows=source.rows,
    )
    if len(plan.steps) != 2 and len(plan.steps) != 1:
        raise Error("Grouped float key-only grouping plan changed")
    var start = -1
    for step in plan.steps:
        if step.grouped:
            if start >= 0:
                raise Error("Grouped float requires one grouping header")
            start = step.start
    if start < 0:
        raise Error("Grouped float key-only grouping header is missing")
    if not source.rows:
        return None
    if not source.storage or not source.matrix()[].scope.same_query(scope):
        raise Error("Grouped float source belongs to another query context")
    var matrix = source.matrix()
    var program = Program(plan)
    for node in range(len(program.node_types)):
        if program.code[4 * node] != ROW_GROUP_KEY:
            continue
        var logical = Int(program.code[4 * node + 1])
        if logical < 0 or logical >= plan.source.width():
            raise Error("Grouped float key slot changed")
        var name = plan.source._columns[logical].name()
        var mapped = -1
        for column in range(source.schema.width()):
            if source.schema._columns[column].name() == name:
                mapped = column
                break
        if mapped < 0:
            raise Error("Grouped float key source is missing")
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
            raise Error("Grouped float group count is invalid")
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


def _destinations(
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    order: Pointer[UInt32, ImmutAnyOrigin],
    offsets: Pointer[Int64, ImmutAnyOrigin],
    destination: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    groups: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var lo = 0
    var hi = Int(groups)
    while lo < hi:
        var middle = lo + (hi - lo) // 2
        if offsets[unsafe_offset=middle + 1] <= Int64(row):
            lo = middle + 1
        else:
            hi = middle
    var group = Int(order[unsafe_offset=lo])
    var at = (
        Int(starts[unsafe_offset=group]) + row - Int(offsets[unsafe_offset=lo])
    )
    destination[unsafe_offset=Int(index[unsafe_offset=at])] = UInt32(row)


struct DestinationOwner(Movable):
    var scope: Scope
    var rows: Int
    var groups: Int
    var descriptors: ArcPointer[DescriptorOwner]
    var layout: ArcPointer[GroupOwner]
    var map: DeviceBuffer[DType.uint32]
    var retained: Int

    def __init__(
        out self,
        scope: Scope,
        rows: Int,
        groups: Int,
        descriptors: ArcPointer[DescriptorOwner],
        layout: ArcPointer[GroupOwner],
        var map: DeviceBuffer[DType.uint32],
    ) raises:
        if (
            rows <= 0
            or groups <= 0
            or len(map) != rows
            or descriptors[].rows != groups
            or not descriptors[].scope.same_query(scope)
            or not layout[].scope.same_query(scope)
            or layout[].rows != rows
            or layout[].groups != groups
        ):
            raise Error("Grouped float destination geometry is invalid")
        var bytes = checked_mul(rows, 4)
        scope._retain(bytes)
        self.scope = scope.copy()
        self.rows = rows
        self.groups = groups
        self.descriptors = descriptors.copy()
        self.layout = layout.copy()
        self.map = map^
        self.retained = bytes

    def __deinit__(deinit self):
        self.scope._release(self.retained)


def build_destinations(
    layout: ArcPointer[GroupOwner], scope: Scope
) raises -> ArcPointer[DestinationOwner]:
    if not layout[].scope.same_query(scope):
        raise Error(
            "Grouped float destination belongs to another query context"
        )
    var rows = layout[].rows
    var groups = layout[].groups
    if rows <= 0 or groups <= 0:
        raise Error("Grouped float destination requires nonempty groups")
    var bytes = checked_add(
        descriptor_bytes(groups),
        checked_add(
            checked_mul(rows, 8), checked_add(checked_mul(groups, 4), 16)
        ),
    )
    _ = scope.check(bytes)
    var ctx = scope.context()
    var offsets = ctx.enqueue_create_buffer[DType.int64](groups + 1)
    var parents = ctx.enqueue_create_buffer[DType.uint8](groups)
    var parent_errors = ctx.enqueue_create_buffer[DType.uint32](groups)
    var map = ctx.enqueue_create_buffer[DType.uint32](rows)
    var seen = ctx.enqueue_create_buffer[DType.uint32](rows + groups)
    var control = ctx.enqueue_create_buffer[DType.uint32](2)
    var held_offsets = offsets.copy()
    var held_parents = parents.copy()
    var held_parent_errors = parent_errors.copy()
    var held_map = map.copy()
    var held_seen = seen.copy()
    var held_control = control.copy()
    var held_layout = layout.copy()
    try:
        ctx.enqueue_function[validate_groups](
            layout[].index.unsafe_ptr(),
            layout[].starts.unsafe_ptr(),
            layout[].order.unsafe_ptr(),
            seen.unsafe_ptr(),
            control.unsafe_ptr(),
            UInt32(rows),
            UInt32(groups),
            grid_dim=1,
            block_dim=1,
        )
        scope.record_launches()
        var proof = scope.read_control(control)
        scope.record_download(8)
        if proof[0]:
            raise Error(
                "Grouped float destination layout is not a stable permutation"
            )
        ctx.enqueue_function[grouped_offsets](
            layout[].starts.unsafe_ptr(),
            layout[].order.unsafe_ptr(),
            offsets.unsafe_ptr(),
            parents.unsafe_ptr(),
            parent_errors.unsafe_ptr(),
            UInt32(groups),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[_destinations](
            layout[].index.unsafe_ptr(),
            layout[].starts.unsafe_ptr(),
            layout[].order.unsafe_ptr(),
            offsets.unsafe_ptr(),
            map.unsafe_ptr(),
            UInt32(rows),
            UInt32(groups),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        scope.record_launches(2)
        scope.synchronize()
        var descriptor = ArcPointer(
            DescriptorOwner(scope, groups, offsets^, parents^, parent_errors^)
        )
        var result = ArcPointer(
            DestinationOwner(scope, rows, groups, descriptor, layout, map^)
        )
        _ = held_offsets^
        _ = held_parents^
        _ = held_parent_errors^
        _ = held_map^
        _ = held_seen^
        _ = held_control^
        _ = held_layout^
        return result^
    except error:
        scope.synchronize()
        _ = held_offsets^
        _ = held_parents^
        _ = held_parent_errors^
        _ = held_map^
        _ = held_seen^
        _ = held_control^
        _ = held_layout^
        raise error^


def _row_groups(
    index: Pointer[UInt32, ImmutAnyOrigin],
    destinations: Pointer[UInt32, ImmutAnyOrigin],
    offsets: Pointer[Int64, ImmutAnyOrigin],
    groups_out: Pointer[UInt32, MutAnyOrigin],
    ordinals: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    groups: UInt32,
    begin: UInt32,
    whole: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var original = UInt32(row) if whole else index[
        unsafe_offset=Int(begin) + row
    ]
    var destination = destinations[unsafe_offset=Int(original)]
    var lo = 0
    var hi = Int(groups)
    while lo < hi:
        var middle = lo + (hi - lo) // 2
        if offsets[unsafe_offset=middle + 1] <= Int64(destination):
            lo = middle + 1
        else:
            hi = middle
    groups_out[unsafe_offset=row] = UInt32(lo)
    ordinals[unsafe_offset=row] = destination - UInt32(
        offsets[unsafe_offset=lo]
    )


struct PhaseGroups(Movable):
    var scope: Scope
    var rows: Int
    var groups: Int
    var active_count: Int
    var row_to_group: DeviceBuffer[DType.uint32]
    var group_ordinal: DeviceBuffer[DType.uint32]
    var group_sizes: DeviceBuffer[DType.uint32]
    var active_order: DeviceBuffer[DType.uint32]
    var retained: Int

    def __init__(
        out self,
        scope: Scope,
        rows: Int,
        groups: Int,
        active_count: Int,
        var row_to_group: DeviceBuffer[DType.uint32],
        var group_ordinal: DeviceBuffer[DType.uint32],
        var group_sizes: DeviceBuffer[DType.uint32],
        var active_order: DeviceBuffer[DType.uint32],
    ) raises:
        if (
            rows <= 0
            or groups <= 0
            or active_count <= 0
            or active_count > groups
            or len(row_to_group) != rows
            or len(group_ordinal) != rows
            or len(group_sizes) != groups
            or len(active_order) != groups
        ):
            raise Error("Grouped feed phase geometry is invalid")
        var bytes = checked_mul(checked_add(rows, groups), 8)
        scope._retain(bytes)
        self.scope = scope.copy()
        self.rows = rows
        self.groups = groups
        self.active_count = active_count
        self.row_to_group = row_to_group^
        self.group_ordinal = group_ordinal^
        self.group_sizes = group_sizes^
        self.active_order = active_order^
        self.retained = bytes

    def __deinit__(deinit self):
        self.scope._release(self.retained)


def _whole_sizes(
    offsets: Pointer[Int64, ImmutAnyOrigin],
    sizes: Pointer[UInt32, MutAnyOrigin],
    order: Pointer[UInt32, MutAnyOrigin],
    groups: UInt32,
):
    var group = global_idx.x
    if group < Int(groups):
        sizes[unsafe_offset=group] = UInt32(
            offsets[unsafe_offset=group + 1] - offsets[unsafe_offset=group]
        )
        order[unsafe_offset=group] = UInt32(group)


def _host_bytes(source: Table) raises -> Int:
    var result = 0
    for field in source.schema._columns:
        result = checked_add(
            result,
            checked_add(
                checked_mul(source.rows, _width(field.dtype())),
                (source.rows + 7) // 8,
            ),
        )
    return result
