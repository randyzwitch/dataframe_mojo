"""Execution of certified Float32 programs over native-width buffers."""

from std.time import perf_counter_ns
from std.memory import bitcast
from dataframe import Column, DataFrame, Series
from dataframe.execution_report import ExecutionReport
from dataframe.accelerator.rows import RowPlan
from .query_scope import Scope, checked_add, checked_mul
from .residency import Table
from .typed_storage import NativeColumn, allocate_native_column, upload_float32
from . import typed_rows


struct _Scratch(Movable):
    var scope: Scope
    var bytes: Int

    def __init__(out self, scope: Scope, bytes: Int) raises:
        scope._retain(bytes)
        self.scope = scope.copy()
        self.bytes = bytes

    def __deinit__(deinit self):
        self.scope._release(self.bytes)


def _preflight_bytes(plan: RowPlan, filtered: Bool) raises -> Int:
    var rows = plan.source.height()
    if not rows:
        return 0
    var nullable = False
    var chunks = plan.source._columns[0].chunks()
    for part in chunks:
        nullable |= part._data[Column[Float32]].null_count() != 0
    var bits = checked_add(rows, 7) // 8 if nullable else 0
    # A sliced source bitmap can straddle one more byte than a rebased bitmap.
    var source_bytes = checked_add(
        checked_mul(rows, 4), checked_add(bits, 1) if nullable else 0
    )
    var output_bytes = checked_add(checked_mul(rows, 4), bits)
    var upload_peak = checked_add(source_bytes, bits if len(chunks) > 1 else 0)
    var export_peak = checked_add(
        source_bytes,
        checked_add(
            checked_mul(rows, 4),
            checked_add(checked_mul(bits, 2), 1 if nullable else 0),
        ),
    )
    if not len(plan.steps):
        return max(upload_peak, export_peak)
    var scratch_bytes = checked_add(
        checked_mul(len(plan.code), 8),
        checked_mul(
            checked_add(
                checked_mul(len(plan.literals), 4),
                checked_mul(len(plan.steps), 16),
            ),
            2,
        ),
    )
    if filtered:
        scratch_bytes = checked_add(
            scratch_bytes,
            checked_add(
                checked_mul(rows, 8),
                checked_add(
                    checked_mul(checked_add(rows, 255) // 256, 4),
                    checked_add(checked_add(rows, 7) // 8, 8),
                ),
            ),
        )
    var compute_peak = checked_add(
        source_bytes,
        checked_add(
            scratch_bytes, checked_mul(output_bytes, 2 if filtered else 1)
        ),
    )
    export_peak = checked_add(
        source_bytes,
        checked_add(
            output_bytes,
            checked_add(checked_mul(rows, 4), checked_mul(bits, 2)),
        ),
    )
    return max(upload_peak, max(compute_peak, export_peak))


def _compute[
    nullable: Bool, filtered: Bool
](
    plan: RowPlan, var source: NativeColumn[DType.float32], profiling: Bool
) raises -> Tuple[NativeColumn[DType.float32], Int, Int, Int]:
    var scope = source.scope()
    var rows = source.length
    var bitmap_bytes = checked_add(rows, 7) // 8
    var blocks = checked_add(rows, 255) // 256
    var code_bytes = checked_mul(len(plan.code), 8)
    var literal_bytes = checked_mul(len(plan.literals), 4)
    var step_bytes = checked_mul(len(plan.steps), 16)
    # Device code plus both host and device literals/step descriptors.
    var scratch_bytes = checked_add(
        code_bytes, checked_mul(checked_add(literal_bytes, step_bytes), 2)
    )
    comptime if filtered:
        scratch_bytes = checked_add(
            scratch_bytes,
            checked_add(
                checked_mul(rows, 8),
                checked_add(
                    checked_mul(blocks, 4), checked_add(bitmap_bytes, 8)
                ),
            ),
        )
    var output_bytes = checked_add(
        checked_mul(rows, 4), bitmap_bytes if nullable else 0
    )
    _ = scope.check(
        checked_add(
            scratch_bytes, checked_mul(output_bytes, 2 if filtered else 1)
        )
    )
    var scratch = _Scratch(scope, scratch_bytes)
    var started = Int(perf_counter_ns()) if profiling else 0
    var literals = List[UInt32](capacity=len(plan.literals))
    for value in plan.literals:
        literals.append(bitcast[DType.uint32](Float32(value)))
    var steps = List[Int32](capacity=len(plan.steps) * 4)
    for step in plan.steps:
        steps.append(Int32(step.start))
        steps.append(Int32(step.nodes))
        steps.append(Int32(step.slot))
        steps.append(Int32(step.filter))
    var host_ns = Int(perf_counter_ns()) - started if profiling else 0
    started = Int(perf_counter_ns()) if profiling else 0
    var ctx = scope.context()
    var result = allocate_native_column[DType.float32](
        scope, source.logical, rows, with_validity=nullable
    )
    var input_values = source.values()
    var input_bits = input_values.create_sub_buffer[DType.uint8](0, 1)
    comptime if nullable:
        input_bits = source.validity().value().copy()
    var output_values = result.values()
    var output_bits = output_values.create_sub_buffer[DType.uint8](0, 1)
    comptime if nullable:
        output_bits = result.validity().value().copy()
    var selected = output_values.create_sub_buffer[DType.uint8](0, 1)
    comptime if filtered:
        selected = ctx.enqueue_create_buffer[DType.uint8](bitmap_bytes)
    var code = ctx.enqueue_create_buffer[DType.int64](len(plan.code))
    var device_literals = ctx.enqueue_create_buffer[DType.uint32](len(literals))
    var device_steps = ctx.enqueue_create_buffer[DType.int32](len(steps))
    var positions = output_values.create_sub_buffer[DType.uint32](0, 1)
    var block_offsets = output_values.create_sub_buffer[DType.uint32](0, 1)
    var total = output_values.create_sub_buffer[DType.uint32](0, 1)
    var indices = output_values.create_sub_buffer[DType.uint32](0, 1)
    var compact = Optional[NativeColumn[DType.float32]]()
    comptime if filtered:
        positions = ctx.enqueue_create_buffer[DType.uint32](rows)
        block_offsets = ctx.enqueue_create_buffer[DType.uint32](blocks)
        total = ctx.enqueue_create_buffer[DType.uint32](1)
        indices = ctx.enqueue_create_buffer[DType.uint32](rows)
    var staging_ns = 0
    var submit_ns = 0
    try:
        ctx.enqueue_copy(code, plan.code.unsafe_ptr())
        ctx.enqueue_copy(device_literals, literals.unsafe_ptr())
        ctx.enqueue_copy(device_steps, steps.unsafe_ptr())
        scope.record_upload(
            checked_add(code_bytes, checked_add(literal_bytes, step_bytes))
        )
        staging_ns = Int(perf_counter_ns()) - started if profiling else 0
        started = Int(perf_counter_ns()) if profiling else 0
        var source_values = input_values.create_sub_buffer[DType.float32](
            source.value_offset, rows
        )
        ctx.enqueue_function[typed_rows.evaluate[nullable, filtered]](
            source_values,
            input_bits,
            UInt32(source.validity_bit_offset),
            code,
            device_literals,
            device_steps,
            UInt32(len(plan.steps)),
            UInt32(plan.outputs[0].slot),
            UInt32(rows),
            output_values,
            output_bits,
            selected,
            grid_dim=(bitmap_bytes + 255) // 256,
            block_dim=256,
        )
        scope.record_launches()
        comptime if filtered:
            ctx.enqueue_function[typed_rows.selection_scan](
                selected,
                positions,
                block_offsets,
                UInt32(rows),
                grid_dim=blocks,
                block_dim=256,
            )
            ctx.enqueue_function[typed_rows.selection_offsets](
                block_offsets,
                total,
                UInt32(blocks),
                grid_dim=1,
                block_dim=1,
            )
            scope.record_launches(2)
            var count_words = scope.read_control(total)
            scope.record_download(4)
            var count = Int(count_words[0])
            if count > rows:
                raise Error("Mojo GPU typed selection count exceeds input")
            if plan.limit >= 0:
                count = min(count, plan.limit)
            compact = allocate_native_column[DType.float32](
                scope, source.logical, count, with_validity=nullable
            )
            if count:
                ctx.enqueue_function[typed_rows.selection_indices](
                    selected,
                    positions,
                    block_offsets,
                    indices,
                    UInt32(rows),
                    grid_dim=blocks,
                    block_dim=256,
                )
                var compact_values = compact.value().values()
                var compact_bits = compact_values.create_sub_buffer[
                    DType.uint8
                ](0, 1)
                comptime if nullable:
                    compact_bits = compact.value().validity().value().copy()
                ctx.enqueue_function[typed_rows.gather[nullable]](
                    output_values,
                    output_bits,
                    indices,
                    compact_values,
                    compact_bits,
                    UInt32(count),
                    grid_dim=(count + 2047) // 2048,
                    block_dim=256,
                )
                scope.record_launches(2)
            scope.synchronize()
            result = compact.value().copy()
        else:
            scope.synchronize()
            if plan.limit >= 0:
                result = result.view(0, min(rows, plan.limit))
        submit_ns = Int(perf_counter_ns()) - started if profiling else 0
    except error:
        scope.synchronize()
        _ = positions^
        _ = block_offsets^
        _ = total^
        _ = indices^
        _ = compact^
        _ = code^
        _ = device_literals^
        _ = device_steps^
        _ = selected^
        _ = literals^
        _ = steps^
        _ = source^
        _ = result^
        _ = scratch^
        raise error^
    _ = literals^
    _ = steps^
    _ = positions^
    _ = block_offsets^
    _ = total^
    _ = indices^
    _ = compact^
    _ = code^
    _ = device_literals^
    _ = device_steps^
    _ = selected^
    _ = source^
    _ = scratch^
    return (result^, host_ns, staging_ns, submit_ns)


def execute(
    plan: RowPlan, scope: Scope, profiling: Bool
) raises -> Tuple[Table, DataFrame]:
    var started = Int(perf_counter_ns())
    var before = scope.transfers()
    var timing_before = scope.timing_ns()
    var host_started = Int(perf_counter_ns()) if profiling else 0
    var filtered = False
    for step in plan.steps:
        filtered |= step.filter
    # Reject the complete conservative query peak before creating a context.
    _ = scope.check(_preflight_bytes(plan, filtered))
    var host_ns = Int(perf_counter_ns()) - host_started if profiling else 0
    var stage_started = Int(perf_counter_ns()) if profiling else 0
    var source = upload_float32(scope, plan.source._columns[0])
    var staging_ns = Int(perf_counter_ns()) - stage_started if profiling else 0
    var submit_ns = 0
    var result = source.copy()
    if source.length and len(plan.steps):
        var computed: Tuple[NativeColumn[DType.float32], Int, Int, Int]
        if source.has_validity():
            if filtered:
                computed = _compute[True, True](plan, source.copy(), profiling)
            else:
                computed = _compute[True, False](plan, source.copy(), profiling)
        elif filtered:
            computed = _compute[False, True](plan, source.copy(), profiling)
        else:
            computed = _compute[False, False](plan, source.copy(), profiling)
        result = computed[0].copy()
        host_ns += computed[1]
        staging_ns += computed[2]
        submit_ns = computed[3]
    if plan.limit >= 0:
        result = result.view(0, min(result.length, plan.limit))
    var field = Series(plan.outputs[0].name, Column[Float32](List[Float32]()))
    field._field_metadata = plan.outputs[0].field_metadata
    var schema = DataFrame(List[Series]([field^]))
    var columns = List[NativeColumn[DType.float32]]([result^])
    var table = Table.native_float32(schema^, columns^)
    var report = ExecutionReport()
    report.record(
        plan.root,
        "ROW EXPRESSIONS",
        "metal",
        plan.source.height(),
        table.rows,
        algorithm="mojo_gpu_native_float32_filter" if filtered else "mojo_gpu_native_float32_rows",
        wall_ns=Int(perf_counter_ns()) - started,
    )
    var fields = report.frame()._columns.copy()
    var after = scope.transfers()
    for field in List[Tuple[String, Int]](
        [
            ("kernel_launches", after[2] - before[2]),
            ("synchronizations", after[3] - before[3]),
            ("upload_bytes", after[0] - before[0]),
            ("download_bytes", after[1] - before[1]),
            ("peak_requested_bytes", scope.info()[1]),
            ("host_result_bytes", 0),
            ("resident_input_bytes", 0),
            ("resident_output_bytes", table.retained_bytes()),
        ]
    ):
        fields.append(Series(field[0], Column[Int64]([Int64(field[1])])))
    if profiling:
        var initialization_ns = scope.timing_ns()[0] - timing_before[0]
        for field in List[Tuple[String, Int]](
            [
                ("host_prepare_ms", host_ns),
                ("staging_copy_ms", max(0, staging_ns - initialization_ns)),
                ("initialization_ms", initialization_ns),
                ("submit_wait_ms", submit_ns),
                ("synchronization_ms", scope.timing_ns()[1] - timing_before[1]),
            ]
        ):
            fields.append(
                Series(field[0], Column[Float64]([Float64(field[1]) / 1e6]))
            )
    fields.append(Series("runtime_implementation", Column[String](["mojo"])))
    fields.append(
        Series(
            "query_wall_ms",
            Column[Float64]([Float64(Int(perf_counter_ns()) - started) / 1e6]),
        )
    )
    return (table^, DataFrame(fields^))
