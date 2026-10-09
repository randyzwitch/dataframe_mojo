"""One stream, resident intermediates, and synchronized host row results."""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.time import perf_counter_ns
from dataframe.accel_rows import RowPlan, lower_rows
from dataframe.accel_row_memory import RowMemory, row_memory, bitmap_bytes
from dataframe.accel_memory import ACCEL_THREADS
from dataframe.bool_column import BoolColumn
from dataframe.column import Column
from dataframe.dtype import DataType
from dataframe.expr import COUNT
from dataframe.frame import DataFrame
from dataframe.lazy import LazyFrame
from dataframe.series import Series
from dataframe.execution_report import ExecutionReport
from .nvidia import NvidiaRuntime, _HostDownload, _drain_before_release
from .row_kernels import (
    initialize_column,
    project,
    stable_ranks,
    block_offsets,
    stable_gather,
    pack_output,
    partial_reduce,
)
from .reductions import finish


def _pipeline[
    D: DType
](
    ctx: DeviceContext,
    plan: RowPlan,
    memory: RowMemory,
    values: DeviceBuffer[D],
    valid: DeviceBuffer[DType.uint8],
    alternate: DeviceBuffer[D],
    alternate_valid: DeviceBuffer[DType.uint8],
    code: DeviceBuffer[DType.int64],
    literals: DeviceBuffer[DType.float64],
    gathers: DeviceBuffer[DType.int64],
    rows: DeviceBuffer[DType.int64],
    ranks: DeviceBuffer[DType.int64],
    counts: DeviceBuffer[DType.int64],
    source_bits: List[DeviceBuffer[DType.uint8]],
    bool_bits: List[DeviceBuffer[DType.uint8]],
) raises:
    var current = values.copy()
    var current_valid = valid.copy()
    var other = alternate.copy()
    var other_valid = alternate_valid.copy()
    for i in range(plan.source.width()):
        ctx.enqueue_function[initialize_column[D]](
            current,
            current_valid,
            source_bits[i],
            bool_bits[i],
            Int64(memory.capacity),
            Int64(plan.source.height()),
            Int64(i),
            Int64(memory.offsets[i] % 8),
            Int64(memory.has_validity[i]),
            Int64(plan.source._columns[i].dtype() == DataType.BOOL),
            Int64(memory.blocks),
            grid_dim=memory.blocks,
            block_dim=ACCEL_THREADS,
        )
    for step in plan.steps:
        ctx.enqueue_function[project[D]](
            current,
            current_valid,
            code,
            literals,
            rows,
            Int64(memory.capacity),
            Int64(step.start),
            Int64(step.nodes),
            Int64(step.slot),
            Int64(memory.blocks),
            grid_dim=memory.blocks,
            block_dim=ACCEL_THREADS,
        )
        if step.filter:
            ctx.enqueue_function[stable_ranks[D]](
                current,
                current_valid,
                rows,
                ranks,
                counts,
                Int64(memory.capacity),
                Int64(step.slot),
                Int64(memory.chunk),
                grid_dim=memory.blocks,
                block_dim=ACCEL_THREADS,
            )
            ctx.enqueue_function[block_offsets](
                rows, counts, Int64(memory.blocks), grid_dim=1, block_dim=1
            )
            ctx.enqueue_function[stable_gather[D]](
                current,
                current_valid,
                other,
                other_valid,
                rows,
                ranks,
                counts,
                gathers,
                Int64(memory.capacity),
                Int64(step.slot),
                Int64(memory.chunk),
                Int64(step.gather_start),
                Int64(step.gather_count),
                Int64(memory.blocks),
                grid_dim=memory.blocks,
                block_dim=ACCEL_THREADS,
            )
            var previous = current.copy()
            current = other.copy()
            other = previous^
            var previous_valid = current_valid.copy()
            current_valid = other_valid.copy()
            other_valid = previous_valid^


def _execute[
    D: DType
](
    runtime: NvidiaRuntime, plan: RowPlan, memory: RowMemory, profiling: Bool
) raises -> Tuple[DataFrame, Int, Int, Int, Int]:
    var ctx = runtime._ctx
    # Allocate everything before exposing any raw host pointer to the stream.
    var values = ctx.enqueue_create_buffer[D](memory.matrix)
    var valid = ctx.enqueue_create_buffer[DType.uint8](memory.matrix)
    var alternate = ctx.enqueue_create_buffer[D](memory.alternate)
    var alternate_valid = ctx.enqueue_create_buffer[DType.uint8](
        memory.alternate
    )
    var code = ctx.enqueue_create_buffer[DType.int64](max(1, len(plan.code)))
    var literals = ctx.enqueue_create_buffer[DType.float64](
        max(1, len(plan.literals))
    )
    var gathers = ctx.enqueue_create_buffer[DType.int64](
        max(1, len(plan.gathers))
    )
    var rows = ctx.enqueue_create_buffer[DType.int64](2)
    var ranks = ctx.enqueue_create_buffer[DType.int64](memory.ranks)
    var sums = ctx.enqueue_create_buffer[DType.float64](memory.blocks)
    var counts = ctx.enqueue_create_buffer[DType.int64](memory.blocks)
    var sum_output = ctx.enqueue_create_buffer[DType.float64](1)
    var count_output = ctx.enqueue_create_buffer[DType.int64](1)
    var packed = ctx.enqueue_create_buffer[DType.uint8](memory.packed)
    var packed_bool = ctx.enqueue_create_buffer[DType.uint8](memory.packed)
    var source_bits = List[DeviceBuffer[DType.uint8]]()
    var bool_bits = List[DeviceBuffer[DType.uint8]]()
    for i in range(plan.source.width()):
        source_bits.append(
            ctx.enqueue_create_buffer[DType.uint8](
                max(1, memory.validity_bytes[i])
            )
        )
        bool_bits.append(
            ctx.enqueue_create_buffer[DType.uint8](max(1, memory.bool_bytes[i]))
        )
    var initial_rows = List[Int64]([Int64(plan.source.height()), Int64(0)])
    var host_rows = _HostDownload[DType.int64](ctx, 2, 0)
    var columns = List[Series]()
    var kernel_ns = 0
    var downloads = 16
    var waits = 1
    try:
        if len(plan.code):
            ctx.enqueue_copy(code, plan.code.unsafe_ptr())
            ctx.enqueue_copy(literals, plan.literals.unsafe_ptr())
        if len(plan.gathers):
            ctx.enqueue_copy(gathers, plan.gathers.unsafe_ptr())
        ctx.enqueue_copy(rows, initial_rows.unsafe_ptr())
        for i in range(plan.source.width()):
            var column = plan.source._columns[i].copy()
            if column.dtype() == DataType.BOOL:
                var source = column.bool()
                if memory.bool_bytes[i]:
                    ctx.enqueue_copy(
                        bool_bits[i],
                        source.unsafe_values().unsafe_offset(
                            memory.offsets[i] // 8
                        ),
                    )
                if memory.validity_bytes[i]:
                    ctx.enqueue_copy(
                        source_bits[i],
                        source.unsafe_validity().unsafe_offset(
                            memory.offsets[i] // 8
                        ),
                    )
            else:
                var source = column.numeric[D]()
                if plan.source.height():
                    var destination = values.create_sub_buffer[D](
                        i * memory.capacity, plan.source.height()
                    )
                    ctx.enqueue_copy(destination, source.unsafe_values())
                if memory.validity_bytes[i]:
                    ctx.enqueue_copy(
                        source_bits[i],
                        source.unsafe_validity().unsafe_offset(
                            memory.offsets[i] // 8
                        ),
                    )
        if profiling:

            def kernels(
                timing_ctx: DeviceContext,
            ) raises {
                imm plan,
                imm memory,
                imm values,
                imm valid,
                imm alternate,
                imm alternate_valid,
                imm code,
                imm literals,
                imm gathers,
                imm rows,
                imm ranks,
                imm counts,
                imm source_bits,
                imm bool_bits,
            }:
                _pipeline[D](
                    timing_ctx,
                    plan,
                    memory,
                    values,
                    valid,
                    alternate,
                    alternate_valid,
                    code,
                    literals,
                    gathers,
                    rows,
                    ranks,
                    counts,
                    source_bits,
                    bool_bits,
                )

            kernel_ns += ctx.execution_time(kernels, 1)
            waits += 1
        else:
            _pipeline[D](
                ctx,
                plan,
                memory,
                values,
                valid,
                alternate,
                alternate_valid,
                code,
                literals,
                gathers,
                rows,
                ranks,
                counts,
                source_bits,
                bool_bits,
            )
        var filter_count = 0
        for step in plan.steps:
            filter_count += Int(step.filter)
        var current = alternate.copy() if filter_count % 2 else values.copy()
        var current_valid = (
            alternate_valid.copy() if filter_count % 2 else valid.copy()
        )
        host_rows.pending = True
        ctx.enqueue_copy(host_rows.values.unsafe_ptr(), rows)
        var row_column = host_rows.finish(0)
        var length = Int(row_column._get(0))
        if not plan.reductions and plan.limit >= 0:
            length = min(length, plan.limit)
        for output in plan.outputs:
            if plan.reductions:

                def reduction(
                    timing_ctx: DeviceContext,
                ) raises {
                    imm current,
                    imm current_valid,
                    imm rows,
                    imm sums,
                    imm counts,
                    imm sum_output,
                    imm count_output,
                    imm memory,
                    imm output,
                }:
                    timing_ctx.enqueue_function[partial_reduce[D]](
                        current,
                        current_valid,
                        rows,
                        sums,
                        counts,
                        Int64(memory.capacity),
                        Int64(output.slot),
                        Int64(memory.blocks),
                        grid_dim=memory.blocks,
                        block_dim=ACCEL_THREADS,
                    )
                    timing_ctx.enqueue_function[finish](
                        sums,
                        counts,
                        sum_output,
                        count_output,
                        Int64(memory.blocks),
                        grid_dim=1,
                        block_dim=ACCEL_THREADS,
                    )

                if profiling:
                    kernel_ns += ctx.execution_time(reduction, 1)
                    waits += 1
                else:
                    reduction(ctx)
                var host_sum = _HostDownload[DType.float64](ctx, 1, 0)
                var host_count = _HostDownload[DType.int64](ctx, 1, 0)
                host_sum.pending = True
                ctx.enqueue_copy(host_sum.values.unsafe_ptr(), sum_output)
                host_count.pending = True
                ctx.enqueue_copy(host_count.values.unsafe_ptr(), count_output)
                var sum_column = host_sum.finish(0)
                var count_column = host_count.finish(0)
                waits += 2
                downloads += 16
                if output.reduction == COUNT:
                    columns.append(Series(output.name, count_column^))
                else:
                    columns.append(
                        Series(
                            output.name,
                            Column[Scalar[D]](
                                [sum_column._get(0).cast[D]()],
                                [
                                    count_column._get(0)
                                    >= Int64(output.min_count)
                                ],
                            ),
                        )
                    )
            else:
                var boolean = output.dtype == DataType.BOOL

                def pack(
                    timing_ctx: DeviceContext,
                ) raises {
                    imm current,
                    imm current_valid,
                    imm packed,
                    imm packed_bool,
                    imm memory,
                    imm length,
                    imm output,
                    imm boolean,
                }:
                    timing_ctx.enqueue_function[pack_output[D]](
                        current,
                        current_valid,
                        packed,
                        packed_bool,
                        Int64(memory.capacity),
                        Int64(length),
                        Int64(output.slot),
                        Int64(boolean),
                        Int64(memory.blocks),
                        grid_dim=memory.blocks,
                        block_dim=ACCEL_THREADS,
                    )

                if profiling:
                    kernel_ns += ctx.execution_time(pack, 1)
                    waits += 1
                else:
                    pack(ctx)
                var bytes = bitmap_bytes(length, 0)
                var bitmap = packed.create_sub_buffer[DType.uint8](0, bytes)
                if boolean:
                    var host = _HostDownload[DType.uint8](ctx, bytes, bytes)
                    host.pending = True
                    if bytes:
                        var payload = packed_bool.create_sub_buffer[
                            DType.uint8
                        ](0, bytes)
                        ctx.enqueue_copy(host.values.unsafe_ptr(), payload)
                        ctx.enqueue_copy(host.bits.unsafe_ptr(), bitmap)
                    ctx.synchronize()
                    host.pending = False
                    var bool_values = host.values^
                    host.values = List[UInt8]()
                    var bool_valid = host.bits^
                    host.bits = List[UInt8]()
                    columns.append(
                        Series(
                            output.name,
                            BoolColumn(
                                values=bool_values^,
                                bits=bool_valid^,
                                length=length,
                            ),
                        )
                    )
                    downloads += bytes * 2
                else:
                    var host = _HostDownload[D](ctx, length, bytes)
                    host.pending = True
                    if length:
                        var payload = current.create_sub_buffer[D](
                            output.slot * memory.capacity, length
                        )
                        ctx.enqueue_copy(host.values.unsafe_ptr(), payload)
                    if bytes:
                        ctx.enqueue_copy(host.bits.unsafe_ptr(), bitmap)
                    columns.append(Series(output.name, host.finish(0)))
                    downloads += (
                        length * (4 if D == DType.float32 else 8) + bytes
                    )
                waits += 1
    except error:
        # Source, descriptor lists and device owners are still in scope here.
        _drain_before_release(ctx)
        raise error^
    var free_after = runtime.memory_info()[0]
    var result = DataFrame(columns^)
    if plan.reductions and plan.limit >= 0:
        result = result.head(plan.limit)
    return (result^, kernel_ns, free_after, downloads, waits)


def execute_rows(
    runtime: NvidiaRuntime, query: LazyFrame, profiling: Bool
) raises -> Tuple[DataFrame, DataFrame]:
    var start = perf_counter_ns()
    var plan = lower_rows(query)
    var memory = row_memory(plan)
    var free = runtime.memory_info()[0]
    var budget = runtime.memory_budget(free)
    memory.summary.require_budget(budget)
    var executed: Tuple[DataFrame, Int, Int, Int, Int]
    if plan.dtype == DataType.FLOAT32:
        executed = _execute[DType.float32](runtime, plan, memory, profiling)
    else:
        executed = _execute[DType.float64](runtime, plan, memory, profiling)
    var result = executed[0].copy()
    var report = ExecutionReport()
    report.record(
        plan.root,
        "RESIDENT ROW EXPRESSIONS",
        "nvidia",
        plan.source.height(),
        result.height(),
        algorithm="resident_interpreter_stable_compaction",
        wall_ns=Int(perf_counter_ns() - start),
    )
    var columns = report.frame()._columns.copy()
    columns.append(
        Series("device_id", Column[Int64]([Int64(runtime.device_id())]))
    )
    columns.append(Series("device_name", Column[String]([runtime.name()])))
    columns.append(
        Series("upload_bytes", Column[Int64]([Int64(memory.upload_bytes)]))
    )
    columns.append(
        Series("download_bytes", Column[Int64]([Int64(executed[3])]))
    )
    columns.append(
        Series(
            "workspace_bytes",
            Column[Int64]([Int64(memory.summary.workspace_bytes)]),
        )
    )
    columns.append(
        Series(
            "device_output_bytes",
            Column[Int64]([Int64(memory.summary.output_bytes)]),
        )
    )
    columns.append(
        Series(
            "peak_requested_device_bytes",
            Column[Int64]([Int64(memory.summary.peak_bytes)]),
        )
    )
    columns.append(
        Series("memory_budget_bytes", Column[Int64]([Int64(budget)]))
    )
    columns.append(Series("free_device_bytes", Column[Int64]([Int64(free)])))
    columns.append(
        Series(
            "free_device_after_execution_bytes",
            Column[Int64]([Int64(executed[2])]),
        )
    )
    columns.append(
        Series(
            "kernel_launches", Column[Int64]([Int64(memory.summary.launches)])
        )
    )
    columns.append(
        Series("synchronizations", Column[Int64]([Int64(executed[4])]))
    )
    columns.append(
        Series(
            "kernel_ms",
            Column[Float64]([Float64(executed[1]) / 1e6], [profiling]),
        )
    )
    columns.append(Series("initialization_ms", Column[Float64]([0])))
    columns.append(
        Series(
            "boundaries",
            Column[String](
                [
                    "host upload -> resident expressions and stable compaction -> synchronized host result"
                ]
            ),
        )
    )
    return (result^, DataFrame(columns^))


def describe_rows(runtime: NvidiaRuntime, query: LazyFrame) raises -> String:
    var plan = lower_rows(query)
    var memory = row_memory(plan)
    var free = runtime.memory_info()[0]
    var budget = runtime.memory_budget(free)
    memory.summary.require_budget(budget)
    return (
        "ENGINE accel: NVIDIA supported; resident Float32/Float64 and Bool row expressions\n  device="
        + String(runtime.device_id())
        + " ("
        + runtime.name()
        + "); free="
        + String(free)
        + " B; payload_budget="
        + String(budget)
        + " B\n  "
        + "peak_requested="
        + String(memory.summary.peak_bytes)
        + " B; workspace="
        + String(memory.summary.workspace_bytes)
        + " B; kernels="
        + String(memory.summary.launches)
        + "\n  upload="
        + String(memory.upload_bytes)
        + " B; download depends on stable filter output\n  matrix workspace includes source/intermediate/output slots; descriptors and packed buffers included\n  host upload -> resident expressions and stable compaction -> synchronized host result\n  estimates exclude SDK reservations; allocation/runtime faults propagate without CPU retry\n"
    )
