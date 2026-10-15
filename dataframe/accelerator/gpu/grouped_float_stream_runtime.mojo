"""Resident float pipeline producer with explicit captured CPU epoch metadata."""
from dataframe import DataFrame
from dataframe.accelerator.grouped_float_stream import FloatStreamPlan
from .residency import (
    Scope,
    Matrix,
    Table,
    CELL_BYTES,
    checked_add,
    checked_mul,
)
from .grouped_float_geometry import PhaseGroups
from .grouped_float_geometry import _Reservation
from . import grouped_float_stream as stream


def execute(
    plan: FloatStreamPlan,
    source: Table,
    geometry: PhaseGroups,
    scope: Scope,
    canonical_nan: UInt64,
) raises -> Table:
    if (
        source.rows != plan.rows
        or source.offset != 0
        or not source.storage
        or not source.matrix()[].scope.same_query(scope)
        or not geometry.scope.same_query(scope)
        or geometry.rows != plan.rows
        or geometry.active_count != geometry.groups
        or plan.key_column < 0
        or plan.key_column >= source.schema.width()
        or plan.value_column < 0
        or plan.value_column >= source.schema.width()
        or len(plan.ranges) % 2 != 0
    ):
        raise Error("Float stream original source geometry certificate changed")
    var n = plan.rows
    var g = geometry.groups
    var sizes = stream.resources(n, g)
    var range_bytes = checked_mul(len(plan.ranges), 4)
    var result_bytes = checked_mul(g, CELL_BYTES)
    var reserved = checked_add(
        checked_add(sizes.build_peak_bytes, checked_mul(range_bytes, 2)),
        checked_add(result_bytes, 8),
    )
    _ = scope.check(reserved)
    var reservation = _Reservation(scope, reserved)
    var ctx = scope.context()
    var ranges = ctx.enqueue_create_buffer[DType.uint32](len(plan.ranges))
    var host_ranges = plan.ranges.copy()
    var record = ctx.enqueue_create_buffer[DType.uint32](n)
    var previous = ctx.enqueue_create_buffer[DType.uint32](n)
    var heads = ctx.enqueue_create_buffer[DType.uint32](g)
    var epoch = ctx.enqueue_create_buffer[DType.uint32](g)
    var totals = ctx.enqueue_create_buffer[DType.uint64](n)
    var counts = ctx.enqueue_create_buffer[DType.uint32](n)
    var control = ctx.enqueue_create_buffer[DType.uint32](1)
    var host_control = List[UInt32](length=1, fill=0)
    var output = ctx.enqueue_create_buffer[DType.uint64](2 * g)
    var validity = ctx.enqueue_create_buffer[DType.uint8](g)
    var errors = ctx.enqueue_create_buffer[DType.uint32](g)
    var matrix = source.matrix()
    if checked_mul(matrix[].capacity, matrix[].slots) > Int(UInt32.MAX):
        raise Error("Float stream source matrix exceeds address geometry")
    var key_base = UInt32(
        Int(source.columns[plan.key_column]) * matrix[].capacity
    )
    var value_base = UInt32(
        Int(source.columns[plan.value_column]) * matrix[].capacity
    )
    var held_ids = geometry.row_to_group.copy()
    try:
        ctx.enqueue_copy(ranges, host_ranges.unsafe_ptr())
        scope.record_upload(range_bytes)
        ctx.enqueue_function[stream.build_epochs](
            matrix[].words.unsafe_ptr(),
            matrix[].validity.unsafe_ptr(),
            matrix[].errors.unsafe_ptr(),
            held_ids.unsafe_ptr(),
            ranges.unsafe_ptr(),
            record.unsafe_ptr(),
            previous.unsafe_ptr(),
            heads.unsafe_ptr(),
            epoch.unsafe_ptr(),
            control.unsafe_ptr(),
            key_base,
            UInt32(n),
            UInt32(g),
            UInt32(len(plan.ranges) // 2),
            plan.flush_groups,
            grid_dim=1,
            block_dim=1,
        )
        # Build-only metadata owners can be released once the queued epoch
        # construction finishes; the feed retains exactly 20N + 4G bytes.
        scope.synchronize()
        _ = epoch^
        _ = ranges^
        _ = host_ranges^
        reservation.release(
            checked_add(checked_mul(g, 4), checked_mul(range_bytes, 2))
        )
        ctx.enqueue_function[stream.update](
            matrix[].words.unsafe_ptr(),
            matrix[].validity.unsafe_ptr(),
            matrix[].errors.unsafe_ptr(),
            record.unsafe_ptr(),
            totals.unsafe_ptr(),
            counts.unsafe_ptr(),
            control.unsafe_ptr(),
            value_base,
            UInt32(n),
            plan.dtype,
            plan.op,
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[stream.finish](
            previous.unsafe_ptr(),
            heads.unsafe_ptr(),
            totals.unsafe_ptr(),
            counts.unsafe_ptr(),
            control.unsafe_ptr(),
            output.unsafe_ptr(),
            validity.unsafe_ptr(),
            errors.unsafe_ptr(),
            UInt32(n),
            UInt32(g),
            plan.dtype,
            plan.op,
            plan.minimum,
            canonical_nan,
            grid_dim=(g + 255) // 256,
            block_dim=256,
        )
        scope.record_launches(3)
        scope.read_control_into(control, host_control)
        scope.record_download(4)
        if host_control[0] == 3:
            raise Error(
                "Metal unsupported [float stream]: CPU sampled groups select the collected eager route"
            )
        if host_control[0]:
            raise Error("Float stream source or epoch protocol failed")
    except error:
        scope.synchronize()
        raise error^
    _ = matrix^
    _ = held_ids^
    _ = record^
    _ = previous^
    _ = heads^
    _ = totals^
    _ = counts^
    _ = control^
    _ = host_control^
    reservation.release(result_bytes)
    var result = Matrix(scope, g, 1, output^, validity^, errors^)
    return Table(result^, DataFrame([plan.prototype.copy()]), [Int64(0)], g)


from std.memory import ArcPointer
from .grouped_float_geometry import DestinationOwner
from .grouped_float_geometry import _whole_sizes
from .grouped_float_geometry import _row_groups


def whole_geometry(
    destinations: ArcPointer[DestinationOwner], scope: Scope
) raises -> PhaseGroups:
    """Whole raw source indices for a separately captured worker-one route.

    Unlike the eager feed geometry, the CPU streaming insert tables do not
    switch to bucket jobs at 32768 rows. Admission is decided by sampling
    and the independently captured single worker, then rechecked on device.
    """
    var n = destinations[].rows
    var g = destinations[].groups
    if (
        not destinations[].scope.same_query(scope)
        or n <= 0
        or n >= (1 << 30)
        or g <= 0
        or g > n
    ):
        raise Error("Float stream original source group geometry is invalid")
    _ = scope.check(checked_mul(checked_add(n, g), 8))
    var ctx = scope.context()
    var ids = ctx.enqueue_create_buffer[DType.uint32](n)
    var ordinals = ctx.enqueue_create_buffer[DType.uint32](n)
    var sizes = ctx.enqueue_create_buffer[DType.uint32](g)
    var order = ctx.enqueue_create_buffer[DType.uint32](g)
    var descriptor = destinations[].descriptors.copy()
    var held = destinations.copy()
    try:
        ctx.enqueue_function[_whole_sizes](
            descriptor[].offsets.value().unsafe_ptr(),
            sizes.unsafe_ptr(),
            order.unsafe_ptr(),
            UInt32(g),
            grid_dim=(g + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[_row_groups](
            destinations[].map.unsafe_ptr().unsafe_mut_cast[False](),
            destinations[].map.unsafe_ptr().unsafe_mut_cast[False](),
            descriptor[].offsets.value().unsafe_ptr(),
            ids.unsafe_ptr(),
            ordinals.unsafe_ptr(),
            UInt32(n),
            UInt32(g),
            UInt32(0),
            UInt32(1),
            grid_dim=(n + 255) // 256,
            block_dim=256,
        )
        scope.record_launches(2)
        scope.synchronize()
    except error:
        scope.synchronize()
        raise error^
    _ = descriptor^
    _ = held^
    return PhaseGroups(scope, n, g, g, ids^, ordinals^, sizes^, order^)


def uses_insert(source: Table, key_column: Int, scope: Scope) raises -> Bool:
    """Bounded device-only route choice before group/epoch allocation."""
    if (
        not source.storage
        or source.rows <= 0
        or key_column < 0
        or key_column >= len(source.columns)
    ):
        raise Error("Float sample source geometry is invalid")
    var matrix = source.matrix()
    if not matrix[].scope.same_query(scope):
        raise Error("Float sample source belongs to another query")
    var key_slot = Int(source.columns[key_column])
    if (
        source.rows >= (1 << 30)
        or source.offset < 0
        or source.offset > matrix[].capacity
        or source.rows > matrix[].capacity - source.offset
        or key_slot < 0
        or key_slot >= matrix[].slots
        or checked_mul(matrix[].capacity, matrix[].slots) > Int(UInt32.MAX)
    ):
        raise Error("Float sample source exceeds device address geometry")
    var samples = min(source.rows, 4096)
    var capacity = samples // 4 + 1
    var reservation = _Reservation(
        scope, checked_add(checked_mul(capacity, 4), 8)
    )
    var host = List[UInt32](length=1, fill=0)
    var ctx = scope.context()
    var representatives = ctx.enqueue_create_buffer[DType.uint32](capacity)
    var control = ctx.enqueue_create_buffer[DType.uint32](1)
    try:
        ctx.enqueue_function[stream.sample_insert](
            matrix[].words.unsafe_ptr(),
            matrix[].validity.unsafe_ptr(),
            representatives.unsafe_ptr(),
            control.unsafe_ptr(),
            UInt32(key_slot * matrix[].capacity + source.offset),
            UInt32(source.rows),
            grid_dim=1,
            block_dim=1,
        )
        scope.record_launches()
        scope.read_control_into(control, host)
        scope.record_download(4)
    except error:
        scope.synchronize()
        _ = representatives^
        _ = control^
        _ = host^
        _ = matrix^
        _ = reservation^
        raise error^
    var selected = host[0] != 0
    _ = representatives^
    _ = control^
    _ = host^
    _ = matrix^
    _ = reservation^
    return selected
