"""Persistent primitive List descriptors and stable grouped child gathering.

This isolated module consumes a validated group producer's index/starts/order.
It never derives CPU group order by sorting logical keys lexically. Host export
is explicit; offsets, parent validity/errors and children stay resident between
consumers. There is no new library or native ABI.
"""
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer, Pointer
from dataframe import DataFrame, DataType, ListColumn, Series
from dataframe.accelerator.gpu.residency import (
    Scope,
    Matrix,
    Table,
    CELL_BYTES,
    checked_add,
    checked_mul,
    raise_value_error,
)
from dataframe.accelerator.gpu.runtime import materialize
from .host_list_reference import ListRef, _validate


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
            raise Error("Device List parent count is invalid")
        if rows and (
            not offsets
            or not valid
            or not errors
            or len(offsets.value()) != rows + 1
            or len(valid.value()) != rows
            or len(errors.value()) != rows
        ):
            raise Error("Device List descriptor geometry is invalid")
        if not rows and (offsets or valid or errors):
            raise Error("Empty Device List descriptors must not allocate")
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


@fieldwise_init
struct DeviceLists(Copyable):
    var descriptors: ArcPointer[DescriptorOwner]
    var child: Table
    var prototype: Series
    var offset: Int
    var rows: Int


def validate(value: DeviceLists, scope: Scope) raises:
    ref owner = value.descriptors[]
    if not owner.scope.same_query(scope):
        raise Error("Device List belongs to another query context")
    if (
        value.offset < 0
        or value.rows < 0
        or value.offset > owner.rows
        or value.rows > owner.rows - value.offset
        or not value.prototype.dtype().is_list()
        or value.child.schema.width() != 1
        or value.child.schema._columns[0].dtype()
        != value.prototype.dtype().inner()
    ):
        raise Error("Device List view or child schema is invalid")
    var dtype = value.prototype.dtype().inner()
    if (
        dtype.is_nested()
        or dtype == DataType.STRING
        or dtype._is_bytes()
        or dtype.is_categorical()
        or (
            dtype.physical() == DataType.of(DType.int128)
            and not dtype.is_decimal()
        )
    ):
        raise Error("Device List requires fixed-width primitive child")
    if value.child.storage:
        var matrix = value.child.matrix()
        if not matrix[].scope.same_query(scope):
            raise Error("Device List child belongs to another query context")
        if (
            len(value.child.columns) != 1
            or value.child.columns[0] < 0
            or value.child.columns[0] >= Int64(matrix[].slots)
            or value.child.offset < 0
            or value.child.rows < 0
            or value.child.offset > matrix[].capacity
            or value.child.rows > matrix[].capacity - value.child.offset
        ):
            raise Error("Device List child view is outside its matrix")
    elif value.child.rows:
        raise Error("Device List child must be resident")


def stage_descriptors(
    value: ListRef,
    scope: Scope,
    parent_faults: List[UInt32] = List[UInt32](),
) raises -> DeviceLists:
    _validate(value, scope)
    var rows = value.rows
    if len(parent_faults) and len(parent_faults) != rows:
        raise Error("Device List parent fault dimensions differ")
    if not rows:
        return DeviceLists(
            ArcPointer(DescriptorOwner(scope, 0)),
            value.child.copy(),
            value.prototype.copy(),
            0,
            0,
        )
    var bytes = descriptor_bytes(rows)
    _ = scope.check(checked_mul(bytes, 2))
    var offsets = List[Int64](capacity=rows + 1)
    for row in range(rows + 1):
        offsets.append(value.offsets[][value.offset + row])
    var valid = List[UInt8](length=rows, fill=1)
    var errors = parent_faults.copy() if len(parent_faults) else List[UInt32](
        length=rows, fill=0
    )
    if len(value.bits[]):
        for row in range(rows):
            var at = row + value.offset
            valid[row] = (value.bits[][at // 8] >> UInt8(at & 7)) & UInt8(1)
    var ctx = scope.context()
    var doffsets = ctx.enqueue_create_buffer[DType.int64](rows + 1)
    var dvalid = ctx.enqueue_create_buffer[DType.uint8](rows)
    var derrors = ctx.enqueue_create_buffer[DType.uint32](rows)
    var held_offsets = doffsets.copy()
    var held_valid = dvalid.copy()
    var held_errors = derrors.copy()
    try:
        ctx.enqueue_copy(doffsets, offsets.unsafe_ptr())
        ctx.enqueue_copy(dvalid, valid.unsafe_ptr())
        ctx.enqueue_copy(derrors, errors.unsafe_ptr())
        scope.record_upload(bytes)
        scope.synchronize()
        var owner = DescriptorOwner(scope, rows, doffsets^, dvalid^, derrors^)
        var result = DeviceLists(
            ArcPointer(owner^),
            value.child.copy(),
            value.prototype.copy(),
            0,
            rows,
        )
        _ = offsets^
        _ = valid^
        _ = errors^
        _ = held_offsets^
        _ = held_valid^
        _ = held_errors^
        return result^
    except error:
        scope.synchronize()
        _ = offsets^
        _ = valid^
        _ = errors^
        _ = held_offsets^
        _ = held_valid^
        _ = held_errors^
        raise error^


def list_consumer[
    lengths: Bool
](
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    offsets: Pointer[Int64, ImmutAnyOrigin],
    parents: Pointer[UInt8, ImmutAnyOrigin],
    parent_errors: Pointer[UInt32, ImmutAnyOrigin],
    out_words: Pointer[UInt64, MutAnyOrigin],
    out_valid: Pointer[UInt8, MutAnyOrigin],
    out_errors: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    parent_offset: UInt32,
    capacity: UInt32,
    slots: UInt32,
    child_slot: Int64,
    child_offset: UInt32,
    index: Int64,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var parent = Int(parent_offset) + row
    var word = UInt64(0)
    var high = UInt64(0)
    var valid = UInt8(0)
    var failure = parent_errors[unsafe_offset=parent]
    if parents[unsafe_offset=parent]:
        var begin = offsets[unsafe_offset=parent]
        var count = offsets[unsafe_offset=parent + 1] - begin
        comptime if lengths:
            word = UInt64(count)
            valid = 1
        else:
            var at = index + count if index < 0 else index
            if at >= 0 and at < count:
                var cell = (
                    Int(child_slot) * Int(capacity)
                    + Int(child_offset)
                    + Int(begin + at)
                )
                valid = validity[unsafe_offset=cell]
                failure |= errors[unsafe_offset=cell]
                if valid:
                    word = words[unsafe_offset=cell]
                    high = words[
                        unsafe_offset=Int(capacity) * Int(slots) + cell
                    ]
    out_words[unsafe_offset=row] = word
    out_words[unsafe_offset=Int(rows) + row] = high
    out_valid[unsafe_offset=row] = valid
    out_errors[unsafe_offset=row] = failure


def consume[
    lengths: Bool
](
    value: DeviceLists, scope: Scope, name: String, index: Int64 = 0
) raises -> Table:
    validate(value, scope)
    var prototype = Series.full_null(name, DataType.INT64, 0)
    comptime if not lengths:
        prototype = value.child.schema._columns[0].renamed(name)
    var schema = DataFrame([prototype^])
    var rows = value.rows
    if not rows:
        return Table.empty(schema^)
    _ = scope.check(checked_mul(rows, CELL_BYTES))
    var owner = value.descriptors.copy()
    var child = value.child.copy()
    var matrix = child.storage.copy()
    var ctx = scope.context()
    var dw = ctx.enqueue_create_buffer[DType.uint64](2 * rows)
    var dv = ctx.enqueue_create_buffer[DType.uint8](rows)
    var de = ctx.enqueue_create_buffer[DType.uint32](rows)
    var held_words = dw.copy()
    var held_valid = dv.copy()
    var held_errors = de.copy()
    try:
        var words = (
            dw.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var valid = (
            dv.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var errors = (
            de.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var capacity = UInt32(0)
        var slots = UInt32(0)
        var column = Int64(0)
        if matrix:
            words = (
                matrix.value()[]
                .words.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
            )
            valid = (
                matrix.value()[]
                .validity.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
            )
            errors = (
                matrix.value()[]
                .errors.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
            )
            capacity = UInt32(matrix.value()[].capacity)
            slots = UInt32(matrix.value()[].slots)
            column = child.columns[0]
        ctx.enqueue_function[list_consumer[lengths]](
            words,
            valid,
            errors,
            owner[].offsets.value().unsafe_ptr(),
            owner[].valid.value().unsafe_ptr(),
            owner[].errors.value().unsafe_ptr(),
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            de.unsafe_ptr(),
            UInt32(rows),
            UInt32(value.offset),
            capacity,
            slots,
            column,
            UInt32(child.offset),
            index,
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        scope.record_launches()
        scope.synchronize()
        var columns: List[Int64] = [0]
        var result = Table(
            Matrix(scope, rows, 1, dw^, dv^, de^), schema^, columns^, rows
        )
        _ = owner^
        _ = child^
        _ = matrix^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        return result^
    except error:
        scope.synchronize()
        _ = owner^
        _ = child^
        _ = matrix^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        raise error^


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
            or len(starts) != groups + 1
            or len(order) != groups
        ):
            raise Error("Resident group layout geometry is invalid")
        var bytes = checked_mul(
            checked_add(rows, checked_add(2 * groups, 1)), 4
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


def grouped_gather(
    words: Pointer[UInt64, ImmutAnyOrigin],
    valid: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    order: Pointer[UInt32, ImmutAnyOrigin],
    offsets: Pointer[Int64, ImmutAnyOrigin],
    out_words: Pointer[UInt64, MutAnyOrigin],
    out_valid: Pointer[UInt8, MutAnyOrigin],
    out_errors: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    groups: UInt32,
    capacity: UInt32,
    slots: UInt32,
    column: Int64,
    child_offset: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    # upper_bound skips empty groups, including repeated zero offsets.
    var lo = 0
    var hi = Int(groups)
    while lo < hi:
        var middle = lo + (hi - lo) // 2
        if offsets[unsafe_offset=middle + 1] <= Int64(row):
            lo = middle + 1
        else:
            hi = middle
    var original = Int(order[unsafe_offset=lo])
    var position = (
        Int(starts[unsafe_offset=original])
        + row
        - Int(offsets[unsafe_offset=lo])
    )
    var source = (
        Int(column) * Int(capacity)
        + Int(child_offset)
        + Int(index[unsafe_offset=position])
    )
    out_words[unsafe_offset=row] = words[unsafe_offset=source]
    out_words[unsafe_offset=Int(rows) + row] = words[
        unsafe_offset=Int(capacity) * Int(slots) + source
    ]
    out_valid[unsafe_offset=row] = valid[unsafe_offset=source]
    out_errors[unsafe_offset=row] = errors[unsafe_offset=source]


def grouped_implode(
    child: Table, layout: ArcPointer[GroupOwner], scope: Scope, name: String
) raises -> DeviceLists:
    """Use actual CPU canonical group IDs supplied by the grouping producer.

    `layout.order[g]` maps CPU output group g to its stable index run. This can
    consume existing grouping.index/starts plus the exact CPU-order merge; it
    never assumes lexical key order is CPU group order. Operand errors must be
    fully observed before packaging, as CPU _implode calls _full eagerly.
    """
    if not layout[].scope.same_query(scope):
        raise Error("Resident group layout belongs to another query context")
    if child.schema.width() != 1 or child.rows != layout[].rows:
        raise Error("Grouped IMPLODE child cardinality differs")
    var empty_offsets: List[Int64] = [0]
    var prototype = Series(
        name,
        ListColumn(empty_offsets^, child.schema._columns[0].renamed("item")),
    )
    var shape = DeviceLists(
        ArcPointer(DescriptorOwner(scope, 0)),
        child.copy(),
        prototype.copy(),
        0,
        0,
    )
    validate(shape, scope)
    if not layout[].groups:
        if child.rows:
            raise Error("Grouped IMPLODE has rows without groups")
        return shape^
    var rows = child.rows
    var groups = layout[].groups
    var descriptors = descriptor_bytes(groups)
    var scratch = checked_add(checked_mul(checked_add(rows, groups), 4), 16)
    # Operand observation needs 4N host error bytes. Its phase is separate
    # from descriptor/gather scratch, and uses the same retained input owners.
    _ = scope.check(
        max(
            checked_mul(rows, 4),
            checked_add(
                descriptors, checked_add(checked_mul(rows, CELL_BYTES), scratch)
            ),
        )
    )
    child.observe_errors()
    var ctx = scope.context()
    var offsets = ctx.enqueue_create_buffer[DType.int64](groups + 1)
    var parents = ctx.enqueue_create_buffer[DType.uint8](groups)
    var parent_errors = ctx.enqueue_create_buffer[DType.uint32](groups)
    var words = ctx.enqueue_create_buffer[DType.uint64](2 * rows)
    var valid = ctx.enqueue_create_buffer[DType.uint8](rows)
    var errors = ctx.enqueue_create_buffer[DType.uint32](rows)
    var seen = ctx.enqueue_create_buffer[DType.uint32](rows + groups)
    var control = ctx.enqueue_create_buffer[DType.uint32](2)
    var held_offsets = offsets.copy()
    var held_parents = parents.copy()
    var held_parent_errors = parent_errors.copy()
    var held_words = words.copy()
    var held_valid = valid.copy()
    var held_errors = errors.copy()
    var held_layout = layout.copy()
    var held_child = child.copy()
    var matrix = child.storage.copy()
    var held_seen = seen.copy()
    var held_control = control.copy()
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
            raise Error("Grouped IMPLODE layout is not a stable permutation")
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
        scope.record_launches()
        if rows:
            ctx.enqueue_function[grouped_gather](
                matrix.value()[].words.unsafe_ptr(),
                matrix.value()[].validity.unsafe_ptr(),
                matrix.value()[].errors.unsafe_ptr(),
                layout[].index.unsafe_ptr(),
                layout[].starts.unsafe_ptr(),
                layout[].order.unsafe_ptr(),
                offsets.unsafe_ptr(),
                words.unsafe_ptr(),
                valid.unsafe_ptr(),
                errors.unsafe_ptr(),
                UInt32(rows),
                UInt32(groups),
                UInt32(matrix.value()[].capacity),
                UInt32(matrix.value()[].slots),
                child.columns[0],
                UInt32(child.offset),
                grid_dim=(rows + 255) // 256,
                block_dim=256,
            )
            scope.record_launches()
        scope.synchronize()
        var descriptor = DescriptorOwner(
            scope, groups, offsets^, parents^, parent_errors^
        )
        var gathered: Table
        if rows:
            var columns: List[Int64] = [0]
            gathered = Table(
                Matrix(scope, rows, 1, words^, valid^, errors^),
                child.schema.copy(),
                columns^,
                rows,
            )
        else:
            gathered = Table.empty(child.schema.copy())
        var result = DeviceLists(
            ArcPointer(descriptor^), gathered^, prototype^, 0, groups
        )
        _ = held_offsets^
        _ = held_parents^
        _ = held_parent_errors^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        _ = held_layout^
        _ = held_child^
        _ = matrix^
        _ = held_seen^
        _ = held_control^
        return result^
    except error:
        scope.synchronize()
        _ = held_offsets^
        _ = held_parents^
        _ = held_parent_errors^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        _ = held_layout^
        _ = held_child^
        _ = matrix^
        _ = held_seen^
        _ = held_control^
        raise error^


def empty_grouped(
    child: Series, scope: Scope, name: String
) raises -> DeviceLists:
    """Zero input produces zero grouped parents without device initialization."""
    if len(child):
        raise Error("Empty grouped IMPLODE needs an empty child prototype")
    var offsets: List[Int64] = [0]
    var prototype = Series(name, ListColumn(offsets^, child.renamed("item")))
    var result = DeviceLists(
        ArcPointer(DescriptorOwner(scope, 0)),
        Table.empty(DataFrame([child.copy()])),
        prototype^,
        0,
        0,
    )
    validate(result, scope)
    return result^


struct HostReservation(Movable):
    var scope: Scope
    var bytes: Int

    def __init__(out self, scope: Scope, bytes: Int) raises:
        scope._retain(bytes)
        self.scope = scope.copy()
        self.bytes = bytes

    def __deinit__(deinit self):
        self.scope._release(self.bytes)


def export(value: DeviceLists, scope: Scope) raises -> Series:
    """Explicit terminal export; no host List allocation between consumers."""
    validate(value, scope)
    var rows = value.rows
    if not rows:
        return value.prototype.copy()
    var bytes = checked_add(descriptor_bytes(rows), (rows + 7) // 8)
    _ = scope.check(bytes)
    # Keep host descriptors charged while child materialization computes its
    # own matrix/typed/scratch peak. Release after packaging shared host owners.
    var reservation = HostReservation(scope, bytes)
    var offsets = List[Int64](length=rows + 1, fill=0)
    var valid = List[UInt8](length=rows, fill=0)
    var errors = List[UInt32](length=rows, fill=0)
    var bits = List[UInt8](length=(rows + 7) // 8, fill=0)
    var owner = value.descriptors.copy()
    var ctx = scope.context()
    var frame = DataFrame([])
    try:
        var source_offsets = (
            owner[]
            .offsets.value()
            .create_sub_buffer[DType.int64](value.offset, rows + 1)
        )
        var source_valid = (
            owner[]
            .valid.value()
            .create_sub_buffer[DType.uint8](value.offset, rows)
        )
        var source_errors = (
            owner[]
            .errors.value()
            .create_sub_buffer[DType.uint32](value.offset, rows)
        )
        ctx.enqueue_copy(offsets.unsafe_ptr(), source_offsets)
        ctx.enqueue_copy(valid.unsafe_ptr(), source_valid)
        ctx.enqueue_copy(errors.unsafe_ptr(), source_errors)
        scope.synchronize()
        scope.record_download(descriptor_bytes(rows))
        for row in range(rows):
            raise_value_error(errors[row], "Device List parent failed")
            bits[row // 8] |= UInt8(valid[row] != 0) << UInt8(row & 7)
        frame = materialize(value.child)
    except error:
        scope.synchronize()
        _ = offsets^
        _ = valid^
        _ = errors^
        _ = bits^
        _ = owner^
        _ = reservation^
        raise error^
    # All queued work is complete. Construction may consume its host inputs
    # and raise; no catch path can then attempt to consume those inputs again.
    # The reservation's destructor covers publication failure too.
    var child = frame._columns[0].renamed("item")
    var result = Series(
        value.prototype.name(), ListColumn(offsets^, child^, bits^)
    )
    result._field_metadata = value.prototype._field_metadata
    _ = valid^
    _ = errors^
    _ = owner^
    _ = reservation^
    return result^
