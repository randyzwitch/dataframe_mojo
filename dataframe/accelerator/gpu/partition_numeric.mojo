"""Publish one pure numeric OVER state after its exact ranked phase observer."""
from max.gpu import global_idx
from std.memory import Pointer, ArcPointer
from dataframe import DataFrame, DataType, Series
from .residency import Scope, Table, Matrix, CELL_BYTES, checked_mul, checked_add, raise_value_error
from .ranked_rows import RankedRows, _Reservation
from .partition_layout import PartitionOwner
from .ordered_faults import rank, category
from . import partition_faults as faults


def publish(
    source_words: Pointer[UInt64, ImmutAnyOrigin],
    source_validity: Pointer[UInt8, ImmutAnyOrigin],
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    source_cells: UInt64,
    source_base: UInt64,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var source = Int(source_base) + row
    words[unsafe_offset=row] = source_words[unsafe_offset=source]
    words[unsafe_offset=Int(rows) + row] = source_words[unsafe_offset=Int(source_cells) + source]
    validity[unsafe_offset=row] = source_validity[unsafe_offset=source]
    errors[unsafe_offset=row] = 0


def execute(
    operand: RankedRows, layout: ArcPointer[PartitionOwner], scope: Scope,
    dtype: DataType, name: String, batch_rows: Int = 8192,
) raises -> Table:
    if not operand.scope.same_query(scope) or not layout[].scope.same_query(scope) or operand.rows != layout[].rows:
        raise Error("Numeric OVER state owner or geometry changed")
    var rows = operand.rows
    var slices = min(1024, (rows + 255) // 256)
    var result_bytes = checked_mul(rows, CELL_BYTES)
    var reservation = _Reservation(scope, checked_add(result_bytes, checked_add(checked_mul(slices, 32), 64)))
    var context = scope.context()
    var partials = context.enqueue_create_buffer[DType.uint32](8 * slices)
    var winner = context.enqueue_create_buffer[DType.uint32](8)
    var host = List[UInt32](length=8, fill=0)
    var words = context.enqueue_create_buffer[DType.uint64](2 * rows)
    var validity = context.enqueue_create_buffer[DType.uint8](rows)
    var errors = context.enqueue_create_buffer[DType.uint32](rows)
    try:
        context.enqueue_function[faults.operand_partials](
            operand.tokens.unsafe_ptr(), layout[].phase_by_row.unsafe_ptr(),
            layout[].ordinal_by_row.unsafe_ptr(), partials.unsafe_ptr(),
            UInt32(operand.target), UInt32(rows), UInt32(rows), UInt32(slices),
            UInt32(0), UInt32(batch_rows),
            grid_dim=(slices + 255) // 256, block_dim=256,
        )
        context.enqueue_function[faults.finish](partials.unsafe_ptr(), winner.unsafe_ptr(), UInt32(slices), grid_dim=1, block_dim=1)
        scope.record_launches(2)
        context.enqueue_copy(host.unsafe_ptr(), winner)
        scope.synchronize()
        scope.record_download(32)
        if host[0]:
            if host[0] != faults.OPERAND or not rank(host[7]) or not category(host[7]):
                raise Error("Numeric OVER state returned an invalid occurrence token")
            raise_value_error(category(host[7]), "Mojo GPU numeric OVER strict cast failed or arithmetic overflow")
        context.enqueue_function[publish](
            operand.words.unsafe_ptr(), operand.validity.unsafe_ptr(),
            words.unsafe_ptr(), validity.unsafe_ptr(), errors.unsafe_ptr(),
            UInt32(rows), UInt64(operand.cells), UInt64(operand.target * rows),
            grid_dim=(rows + 255) // 256, block_dim=256,
        )
        scope.record_launches()
        scope.synchronize()
    except error:
        scope.synchronize()
        raise error^
    reservation.release(result_bytes)
    var schema = DataFrame([Series.full_null(name, dtype, 0)])
    var columns: List[Int64] = [0]
    return Table(Matrix(scope, rows, 1, words^, validity^, errors^), schema^, columns^, rows)
