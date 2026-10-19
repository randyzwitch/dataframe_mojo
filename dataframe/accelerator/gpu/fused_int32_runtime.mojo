"""Terminal scalar execution with bounded native integer reduction state."""

from std.time import perf_counter_ns
from dataframe import Column, DataFrame, Series
from dataframe.accelerator.rows import RowPlan
from dataframe.accelerator.faults import data_fault
from dataframe.execution_report import ExecutionReport
from .query_scope import Scope, checked_add, checked_mul
from .typed_storage import NativeColumn, _NativeReservation, upload_native
from .fused_int32_plan import Program
from . import fused_int32


comptime MAX_PARTIALS = 4096


def _run[
    nullable: Bool, filtered: Bool, arithmetic: Bool
](
    source: NativeColumn[DType.int32], program: Program, partial_count: Int
) raises -> List[Int64]:
    var scope = source.scope()
    var ctx = scope.context()
    var values = source.values()
    var bits = values.create_sub_buffer[DType.uint8](0, 1)
    comptime if nullable:
        bits = source.validity().value().copy()
    var sums = ctx.enqueue_create_buffer[DType.int64](partial_count)
    var counts = ctx.enqueue_create_buffer[DType.uint32](partial_count)
    var faults = ctx.enqueue_create_buffer[DType.uint32](partial_count)
    var output = ctx.enqueue_create_buffer[DType.int64](3)
    var result = List[Int64](length=3, fill=0)
    try:
        ctx.enqueue_function[
            fused_int32.partials[nullable, filtered, arithmetic]
        ](
            values,
            bits,
            UInt32(source.validity_bit_offset),
            UInt32(source.length),
            UInt32(partial_count * 256),
            program.threshold,
            program.addend,
            program.factor,
            sums,
            counts,
            faults,
            grid_dim=partial_count,
            block_dim=256,
        )
        ctx.enqueue_function[fused_int32.finish](
            sums,
            counts,
            faults,
            UInt32(partial_count),
            output,
            grid_dim=1,
            block_dim=256,
        )
        scope.record_launches(2)
        scope.read_control_into(output, result)
        scope.record_download(24)
    except error:
        scope.synchronize()
        _ = result^
        _ = sums^
        _ = counts^
        _ = faults^
        _ = output^
        raise error^
    _ = sums^
    _ = counts^
    _ = faults^
    _ = output^
    return result^


def _dispatch[
    nullable: Bool
](
    source: NativeColumn[DType.int32], program: Program, partial_count: Int
) raises -> List[Int64]:
    if program.filtered:
        if program.arithmetic:
            return _run[nullable, True, True](source, program, partial_count)
        return _run[nullable, True, False](source, program, partial_count)
    if program.arithmetic:
        return _run[nullable, False, True](source, program, partial_count)
    return _run[nullable, False, False](source, program, partial_count)


def execute(
    plan: RowPlan,
    program: Program,
    scope: Scope,
    profiling: Bool,
    query_started: Int,
    preflight_ns: Int,
) raises -> Tuple[DataFrame, DataFrame]:
    var started = Int(perf_counter_ns())
    var rows = plan.source.height()
    var partial_count = min(MAX_PARTIALS, checked_add(rows, 255) // 256)
    var nullable = False
    var chunks = plan.source._columns[0].chunks()
    for part in chunks:
        nullable |= part._data[Column[Int32]].null_count() != 0
    var bitmap = checked_add(rows, 7) // 8 if nullable else 0
    var source_bytes = checked_add(checked_mul(rows, 4), bitmap)
    if nullable and len(chunks) == 1:
        var offset = chunks[0]._data[Column[Int32]].validity_offset() % 8
        source_bytes = checked_add(
            checked_mul(rows, 4), checked_add(rows + offset, 7) // 8
        )
    var upload_peak = checked_add(
        source_bytes, bitmap if len(chunks) > 1 else 0
    )
    # Partial sums/counts/faults, device+host control record, materialized scalar.
    var scratch_bytes = checked_add(
        checked_mul(partial_count, 16), 57
    ) if rows else 9
    _ = scope.check(max(upload_peak, checked_add(source_bytes, scratch_bytes)))
    var prepare_ns = Int(perf_counter_ns()) - started if profiling else 0
    var staging_started = Int(perf_counter_ns()) if profiling else 0
    var source = upload_native[DType.int32](scope, plan.source._columns[0])
    var staging_ns = (
        Int(perf_counter_ns()) - staging_started if profiling else 0
    )
    var scratch = _NativeReservation(scope, scratch_bytes)
    var total = Int64(0)
    var count = Int64(0)
    var submit_started = Int(perf_counter_ns()) if profiling else 0
    if rows:
        var record: List[Int64]
        if source.has_validity():
            record = _dispatch[True](source, program, partial_count)
        else:
            record = _dispatch[False](source, program, partial_count)
        if record[2]:
            raise data_fault("Mojo GPU fused Int32 arithmetic overflow")
        total = record[0]
        count = record[1]
    var submit_ns = Int(perf_counter_ns()) - submit_started if profiling else 0
    var result_started = Int(perf_counter_ns()) if profiling else 0
    var output = Series(
        plan.outputs[0].name,
        Column[Int64]([total], [count >= Int64(plan.outputs[0].min_count)]),
    )
    output._field_metadata = plan.outputs[0].field_metadata
    var result = DataFrame([output^])
    var result_ns = Int(perf_counter_ns()) - result_started if profiling else 0
    _ = source^
    _ = scratch^
    var report = ExecutionReport()
    report.record(
        plan.root,
        "FUSED INTEGER SUM",
        "metal",
        rows,
        1,
        algorithm="mojo_gpu_fused_int32_sum",
        wall_ns=Int(perf_counter_ns()) - started,
    )
    var fields = report.frame()._columns.copy()
    var traffic = scope.transfers()
    for field in List[Tuple[String, Int]](
        [
            ("kernel_launches", traffic[2]),
            ("synchronizations", traffic[3]),
            ("upload_bytes", traffic[0]),
            ("download_bytes", traffic[1]),
            ("peak_requested_bytes", scope.info()[1]),
            ("host_result_bytes", 9),
            ("resident_input_bytes", 0),
            ("resident_output_bytes", 0),
            ("reduction_partials", partial_count),
        ]
    ):
        fields.append(Series(field[0], Column[Int64]([Int64(field[1])])))
    if profiling:
        var timing = scope.timing_ns()
        for field in List[Tuple[String, Int]](
            [
                ("preflight_ms", preflight_ns),
                ("host_prepare_ms", prepare_ns),
                ("staging_copy_ms", max(0, staging_ns - timing[0])),
                ("initialization_ms", timing[0]),
                ("submit_wait_ms", submit_ns),
                ("synchronization_ms", timing[1]),
                ("result_copy_ms", result_ns),
            ]
        ):
            fields.append(
                Series(field[0], Column[Float64]([Float64(field[1]) / 1e6]))
            )
    fields.append(Series("runtime_implementation", Column[String](["mojo"])))
    fields.append(
        Series(
            "query_wall_ms",
            Column[Float64](
                [Float64(Int(perf_counter_ns()) - query_started) / 1e6]
            ),
        )
    )
    return (result^, DataFrame(fields^))
