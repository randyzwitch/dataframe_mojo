"""Private serial grouped feeds; ranked input matrices never become Tables.

This route certifies _agg_whole's 1024-row _feed batches below 32768 rows.
Outer bucket/job scheduling remains the caller's separate contract. Only a
successful finished state matrix is adopted into ordinary shared storage.
"""
from max.gpu import global_idx
from std.memory import ArcPointer, Pointer
from dataframe import DataFrame
from dataframe.accelerator.grouped_feeds import FeedPlan
from .program import Program
from .residency import (
    Scope,
    Matrix,
    Table,
    CELL_BYTES,
    checked_add,
    checked_mul,
    raise_value_error,
)
from .grouped_list_targets import DestinationOwner
from .grouped_phase_groups import PhaseGroups, _row_groups
from .rows import row_program_ordered
from . import grouped_feed as feed


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


def whole_geometry(
    destinations: ArcPointer[DestinationOwner], scope: Scope
) raises -> PhaseGroups:
    """Original source order and canonical first-occurrence group IDs."""
    if not destinations[].scope.same_query(scope):
        raise Error("Grouped feed geometry belongs to another query")
    var rows = destinations[].rows
    var groups = destinations[].groups
    if rows <= 0 or rows >= 32768 or groups <= 0 or groups > rows:
        raise Error("Grouped feed requires exact serial whole-source geometry")
    _ = scope.check(checked_mul(checked_add(rows, groups), 8))
    var ctx = scope.context()
    var ids = ctx.enqueue_create_buffer[DType.uint32](rows)
    var ordinals = ctx.enqueue_create_buffer[DType.uint32](rows)
    var sizes = ctx.enqueue_create_buffer[DType.uint32](groups)
    var order = ctx.enqueue_create_buffer[DType.uint32](groups)
    var descriptor = destinations[].descriptors.copy()
    var held = destinations.copy()
    try:
        ctx.enqueue_function[_whole_sizes](
            descriptor[].offsets.value().unsafe_ptr(),
            sizes.unsafe_ptr(),
            order.unsafe_ptr(),
            UInt32(groups),
            grid_dim=(groups + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[_row_groups](
            destinations[].map.unsafe_ptr().unsafe_mut_cast[False](),
            destinations[].map.unsafe_ptr().unsafe_mut_cast[False](),
            descriptor[].offsets.value().unsafe_ptr(),
            ids.unsafe_ptr(),
            ordinals.unsafe_ptr(),
            UInt32(rows),
            UInt32(groups),
            UInt32(0),
            UInt32(1),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        scope.record_launches(2)
        scope.synchronize()
    except error:
        scope.synchronize()
        _ = ids^
        _ = ordinals^
        _ = sizes^
        _ = order^
        _ = descriptor^
        _ = held^
        raise error^
    _ = descriptor^
    _ = held^
    return PhaseGroups(
        scope, rows, groups, groups, ids^, ordinals^, sizes^, order^
    )


def _source_zero_partials(
    errors: Pointer[UInt32, ImmutAnyOrigin],
    partials: Pointer[UInt32, MutAnyOrigin],
    cells: UInt32,
    groups: UInt32,
):
    var group = global_idx.x
    if group >= Int(groups):
        return
    var width = (Int(cells) + Int(groups) - 1) // Int(groups)
    var failure = UInt32(0)
    for cell in range(group * width, min(Int(cells), (group + 1) * width)):
        failure |= errors[unsafe_offset=cell]
    partials[unsafe_offset=group] = failure


def _source_zero_finish(
    partials: Pointer[UInt32, ImmutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    groups: UInt32,
):
    if global_idx.x != 0:
        return
    for at in range(feed.CONTROL_WORDS):
        control[unsafe_offset=at] = 0
    for group in range(Int(groups)):
        control[unsafe_offset=0] |= partials[unsafe_offset=group]


def execute(
    plan: FeedPlan, source: Table, geometry: PhaseGroups, scope: Scope
) raises -> Table:
    """Observe every batch operand before updates; publish zero-error states."""
    ref operand = plan.operand[]
    if (
        plan.rows < 0
        or plan.rows >= 32768
        or plan.batch_size != 1024
        or source.rows != plan.rows
        or geometry.rows != plan.rows
        or geometry.active_count != geometry.groups
        or not geometry.scope.same_query(scope)
        or not source.storage
        or not source.matrix()[].scope.same_query(scope)
        or len(operand.steps) != 1
        or len(operand.outputs) != 1
        or operand.reductions
        or operand.limit >= 0
        or operand.steps[0].filter
        or operand.steps[0].order
        or operand.steps[0].grouped
        or len(operand.ordered_ranks[]) != len(operand.literals)
        or not feed.admitted(plan.op, plan.input_kind, plan.output_kind)
    ):
        raise Error("Grouped feed private producer certificate is invalid")
    var rows = plan.rows
    var groups = geometry.groups
    var cells = checked_mul(rows, operand.slots)
    var input_cells = checked_mul(rows, operand.source.width())
    if rows <= 0 or groups <= 0 or input_cells <= 0:
        raise Error("Grouped feed nonempty phase geometry is required")
    var program = Program(operand)
    if (
        len(program.node_types) != len(operand.ordered_ranks[])
        or operand.steps[0].nodes > len(program.node_types)
        or operand.outputs[0].slot >= operand.slots
    ):
        raise Error("Grouped feed private program geometry changed")
    var mapping = List[Int64](capacity=operand.source.width())
    for field in operand.source._columns:
        var index = -1
        for at in range(source.schema.width()):
            if (
                source.schema._columns[at].name() == field.name()
                and source.schema._columns[at].dtype() == field.dtype()
            ):
                index = at
                break
        if index < 0:
            raise Error(
                "Grouped feed source field does not match its certificate"
            )
        mapping.append(Int64(index))
    var batches = (rows + plan.batch_size - 1) // plan.batch_size
    var resources = feed.resources(groups, 1, batches)
    var program_bytes = checked_add(
        checked_add(
            checked_mul(len(program.code), 8),
            checked_mul(len(program.literals), 8),
        ),
        checked_add(
            checked_mul(len(program.node_types), 8),
            checked_mul(len(program.ordered_ranks[]), 4),
        ),
    )
    var partial_count = min(1024, (input_cells + 255) // 256)
    var reserved = checked_add(
        checked_add(
            checked_mul(cells, CELL_BYTES), checked_mul(program_bytes, 2)
        ),
        checked_add(resources.peak_bytes, checked_mul(partial_count, 4)),
    )
    # copy_to separately charges its temporary host/device 48-byte descriptors.
    _ = scope.check(checked_add(reserved, checked_mul(len(mapping), 48)))
    var reservation = _Reservation(scope, reserved)
    var ctx = scope.context()
    var words = ctx.enqueue_create_buffer[DType.uint64](2 * cells)
    var validity = ctx.enqueue_create_buffer[DType.uint8](cells)
    var errors = ctx.enqueue_create_buffer[DType.uint32](cells)
    var code = ctx.enqueue_create_buffer[DType.int64](len(program.code))
    var literals = ctx.enqueue_create_buffer[DType.uint64](
        len(program.literals)
    )
    var types = ctx.enqueue_create_buffer[DType.int64](len(program.node_types))
    var ranks = ctx.enqueue_create_buffer[DType.uint32](
        len(program.ordered_ranks[])
    )
    var states = ctx.enqueue_create_buffer[DType.uint64](groups * 8)
    var result_words = ctx.enqueue_create_buffer[DType.uint64](2 * groups)
    var result_validity = ctx.enqueue_create_buffer[DType.uint8](groups)
    var result_errors = ctx.enqueue_create_buffer[DType.uint32](groups)
    var control = ctx.enqueue_create_buffer[DType.uint32](feed.CONTROL_WORDS)
    var partials = ctx.enqueue_create_buffer[DType.uint32](partial_count)
    var host_control = List[UInt32](length=feed.CONTROL_WORDS, fill=0)
    var held_ids = geometry.row_to_group.copy()
    var held_ordinals = geometry.group_ordinal.copy()
    var held_sizes = geometry.group_sizes.copy()
    var held_order = geometry.active_order.copy()
    try:
        _ = source.copy_to(
            scope, words, validity, errors, rows, operand.slots, mapping
        )
        ctx.enqueue_function[_source_zero_partials](
            errors.unsafe_ptr(),
            partials.unsafe_ptr(),
            UInt32(input_cells),
            UInt32(partial_count),
            grid_dim=(partial_count + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[_source_zero_finish](
            partials.unsafe_ptr(),
            control.unsafe_ptr(),
            UInt32(partial_count),
            grid_dim=1,
            block_dim=1,
        )
        scope.record_launches(2)
        scope.read_control_into(control, host_control)
        scope.record_download(feed.CONTROL_WORDS * 4)
        if host_control[0]:
            raise Error(
                "Grouped feed requires already observed successful inputs"
            )
        ctx.enqueue_copy(code, program.code.unsafe_ptr())
        ctx.enqueue_copy(literals, program.literals.unsafe_ptr())
        ctx.enqueue_copy(types, program.node_types.unsafe_ptr())
        ctx.enqueue_copy(ranks, program.ordered_ranks[].unsafe_ptr())
        scope.record_upload(program_bytes)
        ref step = operand.steps[0]
        ctx.enqueue_function[row_program_ordered](
            words.unsafe_ptr(),
            validity.unsafe_ptr(),
            errors.unsafe_ptr(),
            code.unsafe_ptr(),
            literals.unsafe_ptr(),
            types.unsafe_ptr(),
            ranks.unsafe_ptr(),
            UInt32(step.start),
            UInt32(step.nodes),
            UInt32(step.slot),
            UInt32(rows),
            UInt32(rows),
            UInt32(operand.slots),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[feed.initialize](
            states.unsafe_ptr(),
            result_words.unsafe_ptr(),
            result_validity.unsafe_ptr(),
            result_errors.unsafe_ptr(),
            control.unsafe_ptr(),
            held_order.unsafe_ptr(),
            UInt32(groups),
            UInt32(groups),
            UInt32(1),
            UInt32(0),
            grid_dim=1,
            block_dim=1,
        )
        for batch in range(batches):
            ctx.enqueue_function[feed.feed_batch](
                words.unsafe_ptr(),
                validity.unsafe_ptr(),
                errors.unsafe_ptr(),
                held_ids.unsafe_ptr(),
                held_ordinals.unsafe_ptr(),
                states.unsafe_ptr(),
                control.unsafe_ptr(),
                UInt32(operand.outputs[0].slot),
                UInt32(batch * plan.batch_size),
                UInt32(min(rows, (batch + 1) * plan.batch_size)),
                UInt32(rows),
                UInt32(operand.slots),
                UInt32(groups),
                UInt32(batch),
                plan.op,
                plan.input_kind,
                plan.parameter,
                grid_dim=1,
                block_dim=1,
            )
        ctx.enqueue_function[feed.finish](
            states.unsafe_ptr(),
            held_sizes.unsafe_ptr(),
            held_order.unsafe_ptr(),
            result_words.unsafe_ptr(),
            result_validity.unsafe_ptr(),
            result_errors.unsafe_ptr(),
            control.unsafe_ptr(),
            UInt32(groups),
            UInt32(groups),
            UInt32(batches),
            plan.op,
            plan.input_kind,
            plan.output_kind,
            plan.parameter,
            UInt32(1),
            UInt32(0),
            grid_dim=1,
            block_dim=1,
        )
        scope.record_launches(3 + batches)
        scope.read_control_into(control, host_control)
        scope.record_download(feed.CONTROL_WORDS * 4)
        if host_control[3] == feed.PROTOCOL:
            raise Error(
                "Grouped feed rejected invalid internal token or geometry"
            )
        raise_value_error(
            host_control[0],
            "Mojo GPU grouped operand strict cast failed or arithmetic overflow",
        )
    except error:
        scope.synchronize()
        _ = program^
        _ = words^
        _ = validity^
        _ = errors^
        _ = code^
        _ = literals^
        _ = types^
        _ = ranks^
        _ = states^
        _ = result_words^
        _ = result_validity^
        _ = result_errors^
        _ = control^
        _ = partials^
        _ = host_control^
        _ = held_ids^
        _ = held_ordinals^
        _ = held_sizes^
        _ = held_order^
        raise error^
    _ = program^
    _ = words^
    _ = validity^
    _ = errors^
    _ = code^
    _ = literals^
    _ = types^
    _ = ranks^
    _ = states^
    _ = control^
    _ = partials^
    _ = host_control^
    _ = held_ids^
    _ = held_ordinals^
    _ = held_sizes^
    _ = held_order^
    reservation.release(resources.result_bytes)
    var matrix = Matrix(
        scope, groups, 1, result_words^, result_validity^, result_errors^
    )
    return Table(
        matrix^, DataFrame([plan.prototype.copy()]), [Int64(0)], groups
    )
