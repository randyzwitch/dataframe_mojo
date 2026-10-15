"""Isolated resident primitive ListRef storage and len/get device consumers.

Offsets/parent validity share immutable original CPU owners. Child payloads and
errors remain in a resident Table; consumers upload descriptors only. This is a
storage proof, not public List lowering or grouped IMPLODE enablement.
"""
from max.gpu import global_idx
from std.memory import ArcPointer, Pointer
from dataframe import DataFrame, DataType, ListColumn, Series
from dataframe.accelerator.gpu.residency import (
    Scope,
    Matrix,
    Table,
    CELL_BYTES,
    checked_add,
    checked_mul,
)
from dataframe.accelerator.gpu.runtime import _stage


@fieldwise_init
struct ListRef(Copyable):
    var scope: Scope
    var child: Table
    var offsets: ArcPointer[List[Int64]]
    var bits: ArcPointer[List[UInt8]]
    var offset: Int
    var rows: Int
    var prototype: Series


def _validate(value: ListRef, scope: Scope) raises:
    if not value.scope.same_query(scope):
        raise Error("ListRef belongs to another query context")
    var dtype = value.prototype.dtype().inner()
    if (
        dtype.is_nested()
        or dtype == DataType.STRING
        or dtype._is_bytes()
        or dtype.is_categorical()
    ):
        raise Error("ListRef requires fixed-width primitive child")
    if (
        value.child.rows < 0
        or value.child.rows > Int(Int32.MAX)
        or value.child.offset < 0
        or value.child.offset > Int(Int32.MAX)
    ):
        raise Error("ListRef child range is invalid")
    if (
        value.child.schema.width() != 1
        or value.child.schema._columns[0].dtype()
        != value.prototype.dtype().inner()
    ):
        raise Error("ListRef child dtype differs")
    if (
        value.rows < 0
        or value.rows > Int(Int32.MAX)
        or value.offset < 0
        or value.offset >= len(value.offsets[])
        or value.rows >= len(value.offsets[]) - value.offset
    ):
        raise Error("ListRef parent range is outside offsets")
    if len(value.bits[]) and (
        checked_add(checked_add(value.offset, value.rows), 7)
    ) // 8 > len(value.bits[]):
        raise Error("ListRef parent validity is shorter than range")
    var previous = value.offsets[][value.offset]
    if previous < 0:
        raise Error("ListRef child offsets are invalid")
    for at in range(1, value.rows + 1):
        var next = value.offsets[][value.offset + at]
        if next < previous:
            raise Error("ListRef child offsets are invalid")
        previous = next
    if previous > Int64(value.child.rows):
        raise Error("ListRef child offsets exceed resident child")
    if value.child.storage:
        var matrix = value.child.matrix()
        if not matrix[].scope.same_query(scope):
            raise Error("ListRef child belongs to another query context")
        if (
            len(value.child.columns) != 1
            or value.child.columns[0] < 0
            or value.child.columns[0] >= Int64(matrix[].slots)
            or value.child.offset < 0
            or value.child.rows < 0
            or value.child.offset > matrix[].capacity
            or value.child.rows > matrix[].capacity - value.child.offset
        ):
            raise Error("ListRef child is outside its immutable matrix")
    elif value.child.rows != 0:
        raise Error("ListRef needs device-backed primitive child storage")


def list_ref(source: Series, child: Table, scope: Scope) raises -> ListRef:
    if (
        source.is_chunked()
        or not source.dtype().is_list()
        or source.dtype().inner().is_nested()
    ):
        raise Error("ListRef requires a contiguous primitive ListColumn")
    var lists = source.list_column()
    if len(lists.child()) != child.rows:
        raise Error("ListRef original and resident child lengths differ")
    var result = ListRef(
        scope.copy(),
        child.copy(),
        lists._offsets.copy(),
        lists._bits.copy(),
        lists._offset,
        len(lists),
        source.slice(0, 0),
    )
    _validate(result, scope)
    return result^


def stage_child(
    child: Series, scope: Scope, faults: List[UInt32] = List[UInt32]()
) raises -> Table:
    """Transport a fixed-width primitive child once; no query evaluation."""
    var rows = len(child)
    if (
        rows > Int(Int32.MAX)
        or child.dtype().is_nested()
        or child.dtype() == DataType.STRING
        or child.dtype()._is_bytes()
        or child.dtype().is_categorical()
    ):
        raise Error("ListRef requires fixed-width primitive child")
    if len(faults) and len(faults) != rows:
        raise Error("ListRef child fault dimensions differ")
    if not rows:
        return Table.empty(DataFrame([child.slice(0, 0)]))
    _ = scope.check(checked_mul(rows, CELL_BYTES * 2))
    var words = List[UInt64](length=2 * rows, fill=0)
    var valid = List[UInt8](length=rows, fill=0)
    var errors = faults.copy() if len(faults) else List[UInt32](
        length=rows, fill=0
    )
    _stage(child, rows, 0, 0, words, valid)
    var ctx = scope.context()
    var dw = ctx.enqueue_create_buffer[DType.uint64](2 * rows)
    var dv = ctx.enqueue_create_buffer[DType.uint8](rows)
    var de = ctx.enqueue_create_buffer[DType.uint32](rows)
    var held_words = dw.copy()
    var held_valid = dv.copy()
    var held_errors = de.copy()
    try:
        ctx.enqueue_copy(dw, words.unsafe_ptr())
        ctx.enqueue_copy(dv, valid.unsafe_ptr())
        ctx.enqueue_copy(de, errors.unsafe_ptr())
        scope.record_upload(rows * CELL_BYTES)
        scope.synchronize()
        var columns: List[Int64] = [0]
        var result = Table(
            Matrix(scope, rows, 1, dw^, dv^, de^),
            DataFrame([child.slice(0, 0)]),
            columns^,
            rows,
        )
        _ = words^
        _ = valid^
        _ = errors^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        return result^
    except error:
        scope.synchronize()
        _ = words^
        _ = valid^
        _ = errors^
        _ = held_words^
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
    out_words: Pointer[UInt64, MutAnyOrigin],
    out_valid: Pointer[UInt8, MutAnyOrigin],
    out_errors: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    capacity: UInt32,
    slots: UInt32,
    child_slot: Int64,
    child_offset: UInt32,
    index: Int64,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var word = UInt64(0)
    var high = UInt64(0)
    var valid = UInt8(0)
    var failure = UInt32(0)
    if parents[unsafe_offset=row]:
        var begin = offsets[unsafe_offset=row]
        var end = offsets[unsafe_offset=row + 1]
        var count = end - begin
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
                # A selected null child still carries its checked failure.
                # Null parents and out-of-range indices read no child errors.
                failure = errors[unsafe_offset=cell]
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
](value: ListRef, scope: Scope, name: String, index: Int64 = 0) raises -> Table:
    _validate(value, scope)
    var prototype = Series.full_null(name, DataType.INT64, 0)
    comptime if not lengths:
        prototype = (
            value.prototype.list_column().child().slice(0, 0).renamed(name)
        )
    var schema = DataFrame([prototype^])
    var rows = value.rows
    if not rows:
        return Table.empty(schema^)
    var descriptors = checked_add(checked_mul(rows + 1, 8), rows)
    _ = scope.check(
        checked_add(checked_mul(descriptors, 2), checked_mul(rows, CELL_BYTES))
    )
    var offsets = List[Int64](capacity=rows + 1)
    for at in range(rows + 1):
        offsets.append(value.offsets[][value.offset + at])
    var parents = List[UInt8](length=rows, fill=1)
    if len(value.bits[]):
        for row in range(rows):
            var at = value.offset + row
            parents[row] = (value.bits[][at // 8] >> UInt8(at & 7)) & UInt8(1)
    var ctx = scope.context()
    var doffsets = ctx.enqueue_create_buffer[DType.int64](rows + 1)
    var dparents = ctx.enqueue_create_buffer[DType.uint8](rows)
    var dw = ctx.enqueue_create_buffer[DType.uint64](2 * rows)
    var dv = ctx.enqueue_create_buffer[DType.uint8](rows)
    var de = ctx.enqueue_create_buffer[DType.uint32](rows)
    var held_offsets = doffsets.copy()
    var held_parents = dparents.copy()
    var held_words = dw.copy()
    var held_valid = dv.copy()
    var held_errors = de.copy()
    var child_owner = value.child.copy()
    var matrix = child_owner.storage.copy()
    try:
        ctx.enqueue_copy(doffsets, offsets.unsafe_ptr())
        ctx.enqueue_copy(dparents, parents.unsafe_ptr())
        scope.record_upload(descriptors)
        var child_words = (
            dw.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var child_valid = (
            dv.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var child_errors = (
            de.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var capacity = UInt32(0)
        var slots = UInt32(0)
        var column = Int64(0)
        if matrix:
            child_words = (
                matrix.value()[]
                .words.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
            )
            child_valid = (
                matrix.value()[]
                .validity.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
            )
            child_errors = (
                matrix.value()[]
                .errors.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
            )
            capacity = UInt32(matrix.value()[].capacity)
            slots = UInt32(matrix.value()[].slots)
            column = child_owner.columns[0]
        ctx.enqueue_function[list_consumer[lengths]](
            child_words,
            child_valid,
            child_errors,
            doffsets.unsafe_ptr(),
            dparents.unsafe_ptr(),
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            de.unsafe_ptr(),
            UInt32(rows),
            capacity,
            slots,
            column,
            UInt32(child_owner.offset),
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
        _ = offsets^
        _ = parents^
        _ = child_owner^
        _ = matrix^
        _ = held_offsets^
        _ = held_parents^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        return result^
    except error:
        scope.synchronize()
        _ = offsets^
        _ = parents^
        _ = child_owner^
        _ = matrix^
        _ = held_offsets^
        _ = held_parents^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        raise error^
