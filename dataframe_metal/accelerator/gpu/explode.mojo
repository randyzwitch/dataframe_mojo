"""Logical primitive List EXPLODE over scoped descriptor/child owners.

Null and empty parents produce one null placeholder. All selected columns
must have equal LOGICAL counts, including zero versus one. Only bounded shape
controls cross to host; output child/ordinary values are gathered on device.
Requires the separately reviewed CPU logical-count repair for public parity.
"""
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.memory import Pointer
from dataframe import DataFrame, Series
from .residency import (
    Scope,
    Matrix,
    Table,
    CELL_BYTES,
    checked_add,
    checked_mul,
    raise_value_error,
)
from .resident_fields import ResidentFrame, validate_frame
from .device_lists import HostReservation


def _parent_control(
    offsets: Pointer[Int64, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    start: UInt32,
    rows: UInt32,
    children: UInt32,
):
    if global_idx.x != 0:
        return
    for at in range(4):
        control[unsafe_offset=at] = 0
    var previous = offsets[unsafe_offset=Int(start)]
    if previous < 0 or previous > Int64(children):
        control[unsafe_offset=0] = 1
        return
    for row in range(Int(rows)):
        var next = offsets[unsafe_offset=Int(start) + row + 1]
        if next < previous or next > Int64(children):
            control[unsafe_offset=0] = 1
            control[unsafe_offset=1] = UInt32(row)
            return
        var failure = errors[unsafe_offset=Int(start) + row]
        if failure:
            control[unsafe_offset=0] = 2
            control[unsafe_offset=1] = UInt32(row)
            control[unsafe_offset=2] = failure
            return
        previous = next


def _counts[
    first: Bool
](
    offsets: Pointer[Int64, ImmutAnyOrigin],
    valid: Pointer[UInt8, ImmutAnyOrigin],
    logical: Pointer[Int64, MutAnyOrigin],
    prefix: Pointer[Int64, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    start: UInt32,
    rows: UInt32,
    column: UInt32,
):
    if global_idx.x != 0:
        return
    comptime if first:
        for at in range(4):
            control[unsafe_offset=at] = 0
        control[unsafe_offset=1] = UInt32.MAX
        prefix[unsafe_offset=0] = 0
    var total = Int64(0)
    for row in range(Int(rows)):
        var at = Int(start) + row
        var count = offsets[unsafe_offset=at + 1] - offsets[
            unsafe_offset=at
        ] if valid[unsafe_offset=at] else Int64(0)
        comptime if first:
            logical[unsafe_offset=row] = count
            total += max(Int64(1), count)
            if total > Int64(Int32.MAX) and not control[unsafe_offset=0]:
                control[unsafe_offset=0] = 3
                control[unsafe_offset=1] = UInt32(row)
            prefix[unsafe_offset=row + 1] = total
        else:
            if count != logical[unsafe_offset=row] and (
                not control[unsafe_offset=0]
                or UInt32(row) < control[unsafe_offset=1]
            ):
                control[unsafe_offset=0] = 4
                control[unsafe_offset=1] = UInt32(row)
                control[unsafe_offset=2] = column
    comptime if first:
        control[unsafe_offset=3] = UInt32(total)


def _repeat(
    prefix: Pointer[Int64, ImmutAnyOrigin],
    parents: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    output: UInt32,
):
    var row = global_idx.x
    if row >= Int(output):
        return
    var lo = 0
    var hi = Int(rows)
    while lo < hi:
        var middle = lo + (hi - lo) // 2
        if prefix[unsafe_offset=middle + 1] <= Int64(row):
            lo = middle + 1
        else:
            hi = middle
    parents[unsafe_offset=row] = UInt32(lo)


def _gather[
    list_field: Bool
](
    words: Pointer[UInt64, ImmutAnyOrigin],
    valid: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    offsets: Pointer[Int64, ImmutAnyOrigin],
    logical: Pointer[Int64, ImmutAnyOrigin],
    prefix: Pointer[Int64, ImmutAnyOrigin],
    parents: Pointer[UInt32, ImmutAnyOrigin],
    out_words: Pointer[UInt64, MutAnyOrigin],
    out_valid: Pointer[UInt8, MutAnyOrigin],
    out_errors: Pointer[UInt32, MutAnyOrigin],
    capacity: UInt32,
    slots: UInt32,
    source_slot: UInt32,
    source_offset: UInt32,
    parent_offset: UInt32,
    output: UInt32,
    output_slots: UInt32,
    target_slot: UInt32,
):
    var row = global_idx.x
    if row >= Int(output):
        return
    var parent = Int(parents[unsafe_offset=row])
    var source = parent
    var present = True
    comptime if list_field:
        present = logical[unsafe_offset=parent] > 0
        source = (
            Int(offsets[unsafe_offset=Int(parent_offset) + parent])
            + row
            - Int(prefix[unsafe_offset=parent]) if present else 0
        )
    var target = Int(target_slot) * Int(output) + row
    var word = UInt64(0)
    var high = UInt64(0)
    var validity = UInt8(0)
    var failure = UInt32(0)
    if present:
        var cell = (
            Int(source_slot) * Int(capacity) + Int(source_offset) + source
        )
        validity = valid[unsafe_offset=cell]
        failure = errors[unsafe_offset=cell]
        if validity:
            word = words[unsafe_offset=cell]
            high = words[unsafe_offset=Int(capacity) * Int(slots) + cell]
    out_words[unsafe_offset=target] = word
    out_words[unsafe_offset=Int(output) * Int(output_slots) + target] = high
    out_valid[unsafe_offset=target] = validity
    out_errors[unsafe_offset=target] = failure


def execute(
    frame: ResidentFrame, columns: List[String], scope: Scope
) raises -> Table:
    validate_frame(frame, scope)
    if not len(columns):
        raise Error("explode needs at least one column")
    var selected = List[Bool](length=len(frame.fields), fill=False)
    var positions = List[Int]()
    for name in columns:
        var found = -1
        for index in range(len(frame.fields)):
            if frame.fields[index].prototype.name() == name:
                found = index
                break
        if found < 0:
            raise Error("Unknown column: " + name)
        if selected[found]:
            raise Error("explode: column listed twice: " + name)
        if not frame.fields[found].lists:
            raise Error("explode needs list columns; " + name)
        selected[found] = True
        positions.append(found)
    var prototypes = List[Series]()
    for index in range(len(frame.fields)):
        ref field = frame.fields[index]
        if field.lists:
            if not selected[index]:
                raise Error(
                    "Resident EXPLODE: unselected nested output requires recursive field owner"
                )
            prototypes.append(
                field.lists.value()
                .child.schema._columns[0]
                .renamed(field.prototype.name())
            )
        else:
            prototypes.append(
                field.table.value()
                .schema._columns[0]
                .renamed(field.prototype.name())
            )
    var schema = DataFrame(prototypes^)
    if not frame.rows:
        var result = Table.empty(schema^)
        for index in range(len(frame.fields)):
            ref field = frame.fields[index]
            if field.table:
                result.text[index] = field.table.value().text[0].copy()
                result.categories[index] = (
                    field.table.value().categories[0].copy()
                )
                result.category_leases.extend(
                    field.table.value().category_leases.copy()
                )
                result.grouped_keys[index] = (
                    field.table.value().grouped_keys[0].copy()
                )
        return result^
    if len(frame.fields) > 256:
        raise Error("Resident EXPLODE exceeds 256 primitive output columns")
    var scratch = checked_add(checked_mul(frame.rows, 16), 8 + 32)
    _ = scope.check(scratch)
    var reservation = HostReservation(scope, scratch)
    var ctx = scope.context()
    var logical = ctx.enqueue_create_buffer[DType.int64](frame.rows)
    var prefix = ctx.enqueue_create_buffer[DType.int64](frame.rows + 1)
    var control = ctx.enqueue_create_buffer[DType.uint32](4)
    var host = List[UInt32](length=4, fill=0)
    var parents = Optional[DeviceBuffer[DType.uint32]]()
    var words = Optional[DeviceBuffer[DType.uint64]]()
    var valid = Optional[DeviceBuffer[DType.uint8]]()
    var errors = Optional[DeviceBuffer[DType.uint32]]()
    var held = frame.copy()
    var output_rows: Int
    try:
        # Existing primitive producers are fully observed before cardinality.
        # List parent failures also remain observable beneath null parents.
        for field in held.fields:
            if field.table:
                field.table.value().observe_errors()
            else:
                ref value = field.lists.value()
                var owner = value.descriptors.copy()
                ctx.enqueue_function[_parent_control](
                    owner[].offsets.value().unsafe_ptr(),
                    owner[].errors.value().unsafe_ptr(),
                    control.unsafe_ptr(),
                    UInt32(value.offset),
                    UInt32(value.rows),
                    UInt32(value.child.rows),
                    grid_dim=1,
                    block_dim=1,
                )
                scope.record_launches()
                scope.read_control_into(control, host)
                scope.record_download(16)
                if host[0] == 1:
                    raise Error("Resident EXPLODE offsets exceed child storage")
                if host[0] == 2:
                    raise_value_error(
                        host[2],
                        "Resident EXPLODE parent strict cast failed or arithmetic overflow",
                    )
                _ = owner^
        for index in range(len(positions)):
            ref value = held.fields[positions[index]].lists.value()
            ref owner = value.descriptors[]
            if index == 0:
                ctx.enqueue_function[_counts[True]](
                    owner.offsets.value().unsafe_ptr(),
                    owner.valid.value().unsafe_ptr(),
                    logical.unsafe_ptr(),
                    prefix.unsafe_ptr(),
                    control.unsafe_ptr(),
                    UInt32(value.offset),
                    UInt32(frame.rows),
                    UInt32(index),
                    grid_dim=1,
                    block_dim=1,
                )
            else:
                ctx.enqueue_function[_counts[False]](
                    owner.offsets.value().unsafe_ptr(),
                    owner.valid.value().unsafe_ptr(),
                    logical.unsafe_ptr(),
                    prefix.unsafe_ptr(),
                    control.unsafe_ptr(),
                    UInt32(value.offset),
                    UInt32(frame.rows),
                    UInt32(index),
                    grid_dim=1,
                    block_dim=1,
                )
            scope.record_launches()
        scope.read_control_into(control, host)
        scope.record_download(16)
        if host[0] == 3:
            raise Error("Resident EXPLODE output exceeds Int32 row capacity")
        if host[0] == 4:
            raise Error(
                "explode: columns have different element counts at row "
                + String(host[1])
            )
        output_rows = Int(host[3])
        var cells = checked_mul(output_rows, len(frame.fields))
        _ = scope.check(
            checked_add(
                checked_mul(output_rows, 4), checked_mul(cells, CELL_BYTES)
            )
        )
        parents = ctx.enqueue_create_buffer[DType.uint32](output_rows)
        words = ctx.enqueue_create_buffer[DType.uint64](2 * cells)
        valid = ctx.enqueue_create_buffer[DType.uint8](cells)
        errors = ctx.enqueue_create_buffer[DType.uint32](cells)
        ctx.enqueue_function[_repeat](
            prefix.unsafe_ptr(),
            parents.value().unsafe_ptr(),
            UInt32(frame.rows),
            UInt32(output_rows),
            grid_dim=(output_rows + 255) // 256,
            block_dim=256,
        )
        scope.record_launches()
        for index in range(len(held.fields)):
            ref field = held.fields[index]
            var source = (
                field.lists.value()
                .child.copy() if field.lists else field.table.value()
                .copy()
            )
            # A zero-child List contributes placeholders only. Dummy source
            # pointers are valid allocations and never dereferenced on that path.
            var raw_words: Pointer[UInt64, ImmutAnyOrigin] = (
                words.value()
                .unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
            )
            var raw_valid: Pointer[UInt8, ImmutAnyOrigin] = (
                valid.value()
                .unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
            )
            var raw_errors: Pointer[UInt32, ImmutAnyOrigin] = (
                errors.value()
                .unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
            )
            var capacity = 0
            var slots = 0
            var slot = UInt32(0)
            if source.rows:
                var matrix = source.matrix()
                raw_words = (
                    matrix[]
                    .words.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin]()
                )
                raw_valid = (
                    matrix[]
                    .validity.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin]()
                )
                raw_errors = (
                    matrix[]
                    .errors.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin]()
                )
                capacity = matrix[].capacity
                slots = matrix[].slots
                slot = UInt32(source.columns[0])
            if field.lists:
                ref value = field.lists.value()
                ctx.enqueue_function[_gather[True]](
                    raw_words,
                    raw_valid,
                    raw_errors,
                    value.descriptors[]
                    .offsets.value()
                    .unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    logical.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    prefix.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    parents.value()
                    .unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    words.value().unsafe_ptr(),
                    valid.value().unsafe_ptr(),
                    errors.value().unsafe_ptr(),
                    UInt32(capacity),
                    UInt32(slots),
                    slot,
                    UInt32(source.offset),
                    UInt32(value.offset),
                    UInt32(output_rows),
                    UInt32(len(held.fields)),
                    UInt32(index),
                    grid_dim=(output_rows + 255) // 256,
                    block_dim=256,
                )
            else:
                ctx.enqueue_function[_gather[False]](
                    raw_words,
                    raw_valid,
                    raw_errors,
                    prefix.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    logical.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    prefix.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    parents.value()
                    .unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    words.value().unsafe_ptr(),
                    valid.value().unsafe_ptr(),
                    errors.value().unsafe_ptr(),
                    UInt32(capacity),
                    UInt32(slots),
                    slot,
                    UInt32(source.offset),
                    UInt32(0),
                    UInt32(output_rows),
                    UInt32(len(held.fields)),
                    UInt32(index),
                    grid_dim=(output_rows + 255) // 256,
                    block_dim=256,
                )
            scope.record_launches()
        scope.synchronize()
    except error:
        scope.synchronize()
        _ = logical^
        _ = prefix^
        _ = control^
        _ = host^
        _ = parents^
        _ = words^
        _ = valid^
        _ = errors^
        _ = held^
        _ = reservation^
        raise error^
    _ = logical^
    _ = prefix^
    _ = control^
    _ = host^
    _ = parents^
    _ = held^
    # Keep the scratch charge live through synchronization and disposal of all
    # scratch owners. A lexical local alone permits Mojo ASAP destruction.
    _ = reservation^
    var matrix = Matrix(
        scope,
        output_rows,
        len(frame.fields),
        words.take(),
        valid.take(),
        errors.take(),
    )
    var mapping = List[Int64]()
    for column in range(len(frame.fields)):
        mapping.append(Int64(column))
    var result = Table(matrix^, schema^, mapping^, output_rows)
    for index in range(len(frame.fields)):
        ref field = frame.fields[index]
        if field.table and len(field.table.value().text):
            result.text[index] = field.table.value().text[0].copy()
        if field.table:
            result.categories[index] = field.table.value().categories[0].copy()
            result.category_leases.extend(
                field.table.value().category_leases.copy()
            )
        if field.table and len(field.table.value().grouped_keys):
            result.grouped_keys[index] = (
                field.table.value().grouped_keys[0].copy()
            )
    return result^
