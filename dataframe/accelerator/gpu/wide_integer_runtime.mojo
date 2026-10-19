"""Native wide SUMs reuse bounded device partials across scalar outputs."""
from max.gpu.host import DeviceBuffer
from std.memory import bitcast
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, Series
from dataframe.accelerator.rows import RowPlan
from dataframe.accelerator.faults import data_fault
from dataframe.execution_report import ExecutionReport
from .query_scope import Scope, checked_add, checked_mul
from .typed_storage import NativeColumn, _NativeReservation, upload_native
from .wide_integer_plan import Program
from . import wide_integer_sum

comptime MAX_PARTIALS = 4096


def _submit[
    unsigned: Bool, nullable: Bool, add: Bool, multiply: Bool
](
    source: NativeColumn[DType.uint64 if unsigned else DType.int64],
    program: Program,
    partial_count: Int,
    output_index: Int,
    lows: DeviceBuffer[DType.int64],
    highs: DeviceBuffer[DType.int64],
    counts: DeviceBuffer[DType.uint32],
    faults: DeviceBuffer[DType.uint32],
    output: DeviceBuffer[DType.int64],
) raises:
    var scope = source.scope()
    var ctx = scope.context()
    var values = source.values()
    var bits = values.create_sub_buffer[DType.uint8](0, 1)
    comptime if nullable:
        bits = source.validity().value().copy()
    ctx.enqueue_function[
        wide_integer_sum.partials[unsigned, nullable, add, multiply]
    ](
        values, bits, UInt32(source.validity_bit_offset), UInt32(source.length),
        UInt32(partial_count * 256), program.addend, program.factor,
        lows, highs, counts, faults, grid_dim=partial_count, block_dim=256,
    )
    ctx.enqueue_function[wide_integer_sum.finish[unsigned]](
        lows, highs, counts, faults, UInt32(partial_count), UInt32(output_index),
        program.min_count, output, grid_dim=1, block_dim=256,
    )
    scope.record_launches(2)


def _dispatch[unsigned: Bool, nullable: Bool](
    source: NativeColumn[DType.uint64 if unsigned else DType.int64],
    program: Program,
    partial_count: Int,
    output_index: Int,
    lows: DeviceBuffer[DType.int64],
    highs: DeviceBuffer[DType.int64],
    counts: DeviceBuffer[DType.uint32],
    faults: DeviceBuffer[DType.uint32],
    output: DeviceBuffer[DType.int64],
) raises:
    if program.add:
        if program.multiply:
            _submit[unsigned, nullable, True, True](
                source, program, partial_count, output_index,
                lows, highs, counts, faults, output,
            )
        else:
            _submit[unsigned, nullable, True, False](
                source, program, partial_count, output_index,
                lows, highs, counts, faults, output,
            )
    elif program.multiply:
        _submit[unsigned, nullable, False, True](
            source, program, partial_count, output_index,
            lows, highs, counts, faults, output,
        )
    else:
        _submit[unsigned, nullable, False, False](
            source, program, partial_count, output_index,
            lows, highs, counts, faults, output,
        )


def _run[unsigned: Bool, nullable: Bool](
    source: NativeColumn[DType.uint64 if unsigned else DType.int64],
    programs: List[Program], partial_count: Int,
) raises -> List[Int64]:
    var scope = source.scope()
    var lows = scope.scratch[DType.int64](partial_count)
    var highs = scope.scratch[DType.int64](partial_count)
    var counts = scope.scratch[DType.uint32](partial_count)
    var faults = scope.scratch[DType.uint32](partial_count)
    var output = scope.scratch[DType.int64](len(programs) * 3)
    var records = List[Int64](length=len(programs) * 3, fill=0)
    try:
        # The queue orders each final merge before the next output overwrites
        # these same partial buffers. No per-output host observation is needed.
        for index in range(len(programs)):
            _dispatch[unsigned, nullable](
                source, programs[index], partial_count, index,
                lows, highs, counts, faults, output,
            )
        scope.read_control_into(output, records)
        scope.record_download(len(programs) * 24)
    except error:
        try:
            scope.synchronize()
        except:
            pass
        _ = records^
        _ = lows^
        _ = highs^
        _ = counts^
        _ = faults^
        _ = output^
        raise error^
    scope.recycle_scratch(lows^)
    scope.recycle_scratch(highs^)
    scope.recycle_scratch(counts^)
    scope.recycle_scratch(faults^)
    scope.recycle_scratch(output^)
    return records^


def _execute[unsigned: Bool](
    plan: RowPlan, programs: List[Program], scope: Scope, partial_count: Int,
    scratch_bytes: Int,
) raises -> Tuple[DataFrame, Int, Int, Int]:
    var started = Int(perf_counter_ns())
    var source = upload_native[DType.uint64 if unsigned else DType.int64](
        scope, plan.source._columns[0]
    )
    var staging_ns = Int(perf_counter_ns()) - started
    var reservation = _NativeReservation(scope, scratch_bytes)
    started = Int(perf_counter_ns())
    var records = List[Int64]()
    if source.length:
        if source.has_validity():
            records = _run[unsigned, True](source, programs, partial_count)
        else:
            records = _run[unsigned, False](source, programs, partial_count)
    var submit_ns = Int(perf_counter_ns()) - started
    started = Int(perf_counter_ns())
    var fields = List[Series](capacity=len(programs))
    for index in range(len(programs)):
        var value = Int64(0)
        var count = Int64(0)
        if source.length:
            if records[3 * index + 2]:
                raise data_fault("Mojo GPU wide integer SUM overflow")
            value = records[3 * index]
            count = records[3 * index + 1]
        var valid = count >= Int64(plan.outputs[index].min_count)
        var field: Series
        comptime if unsigned:
            field = Series(plan.outputs[index].name, Column[UInt64](
                [bitcast[DType.uint64](value)], [valid]
            ))
        else:
            field = Series(plan.outputs[index].name, Column[Int64]([value], [valid]))
        field._field_metadata = plan.outputs[index].field_metadata
        fields.append(field^)
    var result = DataFrame(fields^)
    var materialize_ns = Int(perf_counter_ns()) - started
    _ = records^
    _ = source^
    _ = reservation^
    return (result^, staging_ns, submit_ns, materialize_ns)


def execute(
    plan: RowPlan, programs: List[Program], scope: Scope, profiling: Bool,
    query_started: Int, preflight_ns: Int,
) raises -> Tuple[DataFrame, DataFrame]:
    var started = Int(perf_counter_ns())
    var rows = plan.source.height()
    var partial_count = min(MAX_PARTIALS, checked_add(rows, 255) // 256)
    var chunks = plan.source._columns[0].chunks()
    var nullable = False
    for part in chunks:
        if part.dtype() == DataType.INT64:
            nullable |= part._data[Column[Int64]].null_count() != 0
        else:
            nullable |= part._data[Column[UInt64]].null_count() != 0
    var bitmap = checked_add(rows, 7) // 8 if nullable else 0
    var source_bytes = checked_add(checked_mul(rows, 8), bitmap)
    if nullable and len(chunks) == 1:
        var offset = 0
        if chunks[0].dtype() == DataType.INT64:
            offset = chunks[0]._data[Column[Int64]].validity_offset() % 8
        else:
            offset = chunks[0]._data[Column[UInt64]].validity_offset() % 8
        source_bytes = checked_add(checked_mul(rows, 8), checked_add(rows + offset, 7) // 8)
    var upload_peak = checked_add(source_bytes, bitmap if len(chunks) > 1 else 0)
    # One shared partial set, a device and host record per output, final scalars.
    var scratch_bytes = checked_add(checked_mul(partial_count, 24), checked_mul(len(programs), 57)) if rows else checked_mul(len(programs), 9)
    _ = scope.check(max(upload_peak, checked_add(source_bytes, scratch_bytes)))
    var prepare_ns = Int(perf_counter_ns()) - started
    var execution: Tuple[DataFrame, Int, Int, Int]
    if plan.source._columns[0].dtype() == DataType.UINT64:
        execution = _execute[True](plan, programs, scope, partial_count, scratch_bytes)
    else:
        execution = _execute[False](plan, programs, scope, partial_count, scratch_bytes)
    var report = ExecutionReport()
    report.record(plan.root, "WIDE INTEGER SUM", "metal", rows, 1,
        algorithm="mojo_gpu_native_wide_integer_sum",
        wall_ns=Int(perf_counter_ns()) - started,
    )
    var fields = report.frame()._columns.copy()
    var traffic = scope.transfers()
    for field in List[Tuple[String, Int]]([
        ("kernel_launches", traffic[2]), ("synchronizations", traffic[3]),
        ("upload_bytes", traffic[0]), ("download_bytes", traffic[1]),
        ("peak_requested_bytes", scope.info()[1]),
        ("host_result_bytes", len(programs) * 9),
        ("resident_input_bytes", 0), ("resident_output_bytes", 0),
        ("reduction_partials", partial_count), ("reduction_outputs", len(programs)),
    ]):
        fields.append(Series(field[0], Column[Int64]([Int64(field[1])])))
    if profiling:
        var timing = scope.timing_ns()
        for field in List[Tuple[String, Int]]([
            ("preflight_ms", preflight_ns), ("host_prepare_ms", prepare_ns),
            ("staging_copy_ms", max(0, execution[1] - timing[0])),
            ("initialization_ms", timing[0]), ("submit_wait_ms", execution[2]),
            ("synchronization_ms", timing[1]), ("result_copy_ms", execution[3]),
        ]):
            fields.append(Series(field[0], Column[Float64]([Float64(field[1]) / 1e6])))
    fields.append(Series("runtime_implementation", Column[String](["mojo"])))
    fields.append(Series("query_wall_ms", Column[Float64]([Float64(Int(perf_counter_ns()) - query_started) / 1e6])))
    return (execution[0].copy(), DataFrame(fields^))
