"""GPU phase-local feed geometry in canonical global group IDs.

No key, group ID or input payload is read on the host. The sole readback is
active group count. Buffers keep global capacity so results from every CPU
bucket can scatter directly into one immutable canonical output owner.
"""
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer, Pointer
from .residency import Scope, checked_add, checked_mul
from .grouped_schedule import ScheduleOwner, HostPhases
from .grouped_list_targets import DestinationOwner


def _active(
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    order: Pointer[UInt32, ImmutAnyOrigin],
    phases: Pointer[UInt32, ImmutAnyOrigin],
    sizes: Pointer[UInt32, MutAnyOrigin],
    active: Pointer[UInt32, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    groups: UInt32,
    phase: UInt32,
):
    if global_idx.x != 0:
        return
    var count = UInt32(0)
    for group in range(Int(groups)):
        var run = Int(order[unsafe_offset=group])
        var begin = starts[unsafe_offset=run]
        var end = starts[unsafe_offset=run + 1]
        var present = (
            end > begin
            and phases[unsafe_offset=Int(index[unsafe_offset=Int(begin)])]
            == phase
        )
        sizes[unsafe_offset=group] = end - begin if present else UInt32(0)
        active[unsafe_offset=group] = UInt32.MAX
        if present:
            active[unsafe_offset=Int(count)] = UInt32(group)
            count += 1
    control[unsafe_offset=0] = count


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


def build_phase_groups(
    schedule: ArcPointer[ScheduleOwner],
    destinations: ArcPointer[DestinationOwner],
    controls: HostPhases,
    phase: Int,
    scope: Scope,
) raises -> PhaseGroups:
    if (
        not schedule[].scope.same_query(scope)
        or not destinations[].scope.same_query(scope)
        or not controls.scope.same_query(scope)
        or destinations[].rows != schedule[].rows
    ):
        raise Error("Grouped feed phase belongs to another query context")
    if (
        phase < 0
        or phase >= Int(schedule[].facts[4])
        or len(controls.order) != schedule[].buckets
        or len(controls.starts) != schedule[].buckets + 1
    ):
        raise Error("Grouped feed phase ordinal or controls are invalid")
    var begin = 0
    var rows = schedule[].rows
    if not schedule[].facts[0]:
        var bucket = Int(controls.order[phase])
        if bucket < 0 or bucket >= schedule[].buckets:
            raise Error("Grouped feed phase bucket is invalid")
        begin = Int(controls.starts[bucket])
        rows = Int(controls.starts[bucket + 1]) - begin
    if (
        begin < 0
        or rows <= 0
        or begin > schedule[].rows
        or rows > schedule[].rows - begin
    ):
        raise Error("Grouped feed phase range is invalid")
    var groups = destinations[].groups
    _ = scope.check(checked_add(checked_mul(checked_add(rows, groups), 8), 8))
    var layout = destinations[].layout.copy()
    var descriptor = destinations[].descriptors.copy()
    var ctx = scope.context()
    var ids = ctx.enqueue_create_buffer[DType.uint32](rows)
    var ordinals = ctx.enqueue_create_buffer[DType.uint32](rows)
    var sizes = ctx.enqueue_create_buffer[DType.uint32](groups)
    var order = ctx.enqueue_create_buffer[DType.uint32](groups)
    var control = ctx.enqueue_create_buffer[DType.uint32](1)
    var held_ids = ids.copy()
    var held_ordinals = ordinals.copy()
    var held_sizes = sizes.copy()
    var held_order = order.copy()
    var held_control = control.copy()
    var held_schedule = schedule.copy()
    var held_destinations = destinations.copy()
    try:
        ctx.enqueue_function[_active](
            layout[].index.unsafe_ptr(),
            layout[].starts.unsafe_ptr(),
            layout[].order.unsafe_ptr(),
            schedule[].phase_by_row.unsafe_ptr(),
            sizes.unsafe_ptr(),
            order.unsafe_ptr(),
            control.unsafe_ptr(),
            UInt32(groups),
            UInt32(phase),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[_row_groups](
            schedule[].index.unsafe_ptr(),
            destinations[].map.unsafe_ptr(),
            descriptor[].offsets.value().unsafe_ptr(),
            ids.unsafe_ptr(),
            ordinals.unsafe_ptr(),
            UInt32(rows),
            UInt32(groups),
            UInt32(begin),
            schedule[].facts[0],
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        scope.record_launches(2)
        var count = scope.read_control(control)
        scope.record_download(4)
        var result = PhaseGroups(
            scope, rows, groups, Int(count[0]), ids^, ordinals^, sizes^, order^
        )
        _ = layout^
        _ = descriptor^
        _ = held_ids^
        _ = held_ordinals^
        _ = held_sizes^
        _ = held_order^
        _ = held_control^
        _ = held_schedule^
        _ = held_destinations^
        return result^
    except error:
        scope.synchronize()
        _ = layout^
        _ = descriptor^
        _ = held_ids^
        _ = held_ordinals^
        _ = held_sizes^
        _ = held_order^
        _ = held_control^
        _ = held_schedule^
        _ = held_destinations^
        raise error^
