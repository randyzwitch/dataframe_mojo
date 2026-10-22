"""Private resident List targets filled in actual CPU bucket producer order.

Only a fully populated target can publish DeviceLists. Device bucket gathers
may have a different order from canonical output groups; the shared inverse
stable-group map places every child in its final group/source-order position.
"""
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer, Pointer
from dataframe import DataFrame, ListColumn, Series
from .residency import (
    Scope,
    Matrix,
    Table,
    CELL_BYTES,
    checked_add,
    checked_mul,
)
from .device_lists import (
    DescriptorOwner,
    DeviceLists,
    GroupOwner,
    descriptor_bytes,
    validate_groups,
    grouped_offsets,
    validate,
)
from .grouped_schedule import ScheduleOwner, HostPhases


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
            raise Error("Grouped List destination geometry is invalid")
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
        raise Error("Grouped List destination belongs to another query context")
    var rows = layout[].rows
    var groups = layout[].groups
    if rows <= 0 or groups <= 0:
        raise Error("Grouped List destination requires nonempty groups")
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
                "Grouped List destination layout is not a stable permutation"
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


@fieldwise_init
struct ListTarget(Movable):
    var _value: DeviceLists
    var _destinations: ArcPointer[DestinationOwner]
    var _schedule: ArcPointer[ScheduleOwner]
    var _next_phase: Int
    var _copied_rows: Int


def create_target(
    destinations: ArcPointer[DestinationOwner],
    schedule: ArcPointer[ScheduleOwner],
    child: Series,
    scope: Scope,
    name: String,
) raises -> ListTarget:
    if (
        not destinations[].scope.same_query(scope)
        or not schedule[].scope.same_query(scope)
        or destinations[].rows != schedule[].rows
    ):
        raise Error("Grouped List target geometry or query context differs")
    var offsets: List[Int64] = [0]
    var prototype = Series(
        name, ListColumn(offsets^, child.slice(0, 0).renamed("item"))
    )
    var empty = DeviceLists(
        ArcPointer(DescriptorOwner(scope, 0)),
        Table.empty(DataFrame([child.slice(0, 0)])),
        prototype.copy(),
        0,
        0,
    )
    validate(empty, scope)
    var rows = destinations[].rows
    _ = scope.check(checked_mul(rows, CELL_BYTES))
    var ctx = scope.context()
    var words = ctx.enqueue_create_buffer[DType.uint64](2 * rows)
    var valid = ctx.enqueue_create_buffer[DType.uint8](rows)
    var errors = ctx.enqueue_create_buffer[DType.uint32](rows)
    var columns: List[Int64] = [0]
    var output = Table(
        Matrix(scope, rows, 1, words^, valid^, errors^),
        DataFrame([child.slice(0, 0)]),
        columns^,
        rows,
    )
    var value = DeviceLists(
        destinations[].descriptors.copy(),
        output^,
        prototype^,
        0,
        destinations[].groups,
    )
    return ListTarget(value^, destinations.copy(), schedule.copy(), 0, 0)


def _append(
    words: Pointer[UInt64, ImmutAnyOrigin],
    valid: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    source_index: Pointer[UInt32, ImmutAnyOrigin],
    destination: Pointer[UInt32, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_valid: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    begin: UInt32,
    rows: UInt32,
    total: UInt32,
    whole: UInt32,
    capacity: UInt32,
    slots: UInt32,
    column: Int64,
    offset: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var original = UInt32(row) if whole else source_index[
        unsafe_offset=Int(begin) + row
    ]
    var target = Int(destination[unsafe_offset=Int(original)])
    var source = Int(column) * Int(capacity) + Int(offset) + row
    output_words[unsafe_offset=target] = words[unsafe_offset=source]
    output_words[unsafe_offset=Int(total) + target] = words[
        unsafe_offset=Int(capacity) * Int(slots) + source
    ]
    output_valid[unsafe_offset=target] = valid[unsafe_offset=source]
    output_errors[unsafe_offset=target] = errors[unsafe_offset=source]


def append_phase(
    mut target: ListTarget,
    child: Table,
    controls: HostPhases,
    phase: Int,
    scope: Scope,
) raises:
    ref schedule = target._schedule[]
    if (
        not schedule.scope.same_query(scope)
        or not controls.scope.same_query(scope)
        or not target._destinations[].scope.same_query(scope)
    ):
        raise Error("Grouped List target belongs to another query context")
    if (
        phase != target._next_phase
        or phase < 0
        or phase >= Int(schedule.facts[4])
        or len(controls.starts) != schedule.buckets + 1
        or len(controls.order) != schedule.buckets
    ):
        raise Error("Grouped List target phase order is invalid")
    var begin = 0
    var rows = schedule.rows
    if not schedule.facts[0]:
        var bucket = Int(controls.order[phase])
        if bucket < 0 or bucket >= schedule.buckets:
            raise Error("Grouped List target phase bucket is invalid")
        begin = Int(controls.starts[bucket])
        rows = Int(controls.starts[bucket + 1]) - begin
    if (
        begin < 0
        or rows <= 0
        or begin > schedule.rows
        or rows > schedule.rows - begin
        or child.rows != rows
        or child.schema.width() != 1
        or child.schema._columns[0].dtype()
        != target._value.prototype.dtype().inner()
        or not child.storage
        or len(child.columns) != 1
    ):
        raise Error("Grouped List target child geometry differs")
    var source = child.matrix()
    if not source[].scope.same_query(scope):
        raise Error(
            "Grouped List target child belongs to another query context"
        )
    if (
        child.columns[0] < 0
        or child.columns[0] >= Int64(source[].slots)
        or child.offset < 0
        or child.offset > source[].capacity
        or rows > source[].capacity - child.offset
    ):
        raise Error("Grouped List target child lies outside its matrix")
    # Each original expression is observed inside its CPU bucket before the
    # next projection is evaluated. Pure ROWS single-category errors preserve
    # the1024 batch/AST/row order; multi-category ASTs need ordered certificates.
    _ = scope.check(checked_mul(rows, 4))
    child.observe_errors()
    var output = target._value.child.matrix()
    var held_child = child.copy()
    var held_schedule = target._schedule.copy()
    var held_destinations = target._destinations.copy()
    var ctx = scope.context()
    try:
        ctx.enqueue_function[_append](
            source[].words.unsafe_ptr(),
            source[].validity.unsafe_ptr(),
            source[].errors.unsafe_ptr(),
            schedule.index.unsafe_ptr(),
            target._destinations[].map.unsafe_ptr(),
            output[].words.unsafe_ptr(),
            output[].validity.unsafe_ptr(),
            output[].errors.unsafe_ptr(),
            UInt32(begin),
            UInt32(rows),
            UInt32(schedule.rows),
            schedule.facts[0],
            UInt32(source[].capacity),
            UInt32(source[].slots),
            child.columns[0],
            UInt32(child.offset),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        scope.record_launches()
        scope.synchronize()
        target._copied_rows += rows
        target._next_phase += 1
        _ = held_child^
        _ = held_schedule^
        _ = held_destinations^
        _ = source^
        _ = output^
    except error:
        scope.synchronize()
        _ = held_child^
        _ = held_schedule^
        _ = held_destinations^
        _ = source^
        _ = output^
        raise error^


def finish_target(var target: ListTarget, scope: Scope) raises -> DeviceLists:
    if (
        not target._schedule[].scope.same_query(scope)
        or target._next_phase != Int(target._schedule[].facts[4])
        or target._copied_rows != target._schedule[].rows
    ):
        raise Error("Grouped List target cannot publish incomplete children")
    validate(target._value, scope)
    return target._value.copy()
