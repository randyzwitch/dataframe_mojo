"""NVIDIA dispatch with specialized float reductions and resident row regions."""
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.time import perf_counter_ns
from dataframe.accel_plan import (
    AccelPlan,
    AccelReduction,
    AccelScalar,
    lower_accel,
)
from dataframe.accel_memory import (
    AccelMemory,
    plan_memory,
    ACCEL_THREADS,
    ACCEL_MAX_BLOCKS,
)
from dataframe.column import Column
from dataframe.dtype import DataType
from dataframe.expr import COUNT
from dataframe.float_ops import arithmetic, compare
from dataframe.frame import DataFrame
from dataframe.lazy import LazyFrame
from dataframe.series import Series
from dataframe.execution_report import ExecutionReport
from .row_query import execute_rows, describe_rows
from .reductions import reduce_pair as _reduce, finish as _finish
from .nvidia import (
    NvidiaRuntime,
    NvidiaColumn,
    _HostDownload,
    _drain_before_release,
)

comptime THREADS = ACCEL_THREADS
comptime MAX_BLOCKS = ACCEL_MAX_BLOCKS


def _partial[
    D: DType
](
    values: DeviceBuffer[D].device_type,
    bits: DeviceBuffer[DType.uint8].device_type,
    sums: DeviceBuffer[DType.float64].device_type,
    counts: DeviceBuffer[DType.int64].device_type,
    rows: Int64,
    bit_offset: Int64,
    has_bits: Int64,
    blocks: Int64,
    predicate: Int64,
    threshold: Float64,
    predicate_left: Int64,
    operation: Int64,
    literal: Float64,
    literal_left: Int64,
):
    var total = Float64(0)
    var count = Int64(0)
    var i = Int(block_idx.x) * THREADS + Int(thread_idx.x)
    while i < Int(rows):
        var valid = True
        if has_bits:
            var bit = Int(bit_offset) + i
            valid = (
                bits[unsafe_offset=bit // 8] & (UInt8(1) << UInt8(bit % 8))
            ) != 0
        if valid:
            var x = values[unsafe_offset=i]
            var accepted = True
            if predicate >= 0:
                var y = threshold.cast[D]()
                accepted = Bool(
                    compare[D, 1](
                        Int(predicate),
                        y if predicate_left else x,
                        x if predicate_left else y,
                    )
                )
            if accepted:
                var value = x
                if operation >= 0:
                    var y = literal.cast[D]()
                    value = arithmetic[D, 1](
                        Int(operation),
                        y if literal_left else x,
                        x if literal_left else y,
                    )
                # Round row arithmetic in D before the Float64 accumulator.
                total += value.cast[DType.float64]()
                count += 1
        i += Int(blocks) * THREADS
    var reduced = _reduce(total, count)
    if thread_idx.x == 0:
        sums[unsafe_offset=Int(block_idx.x)] = reduced[0]
        counts[unsafe_offset=Int(block_idx.x)] = reduced[1]


def _enqueue[
    D: DType
](
    ctx: DeviceContext,
    source: NvidiaColumn[D],
    reduction: AccelReduction,
    predicate: AccelScalar,
    sums: DeviceBuffer[DType.float64],
    counts: DeviceBuffer[DType.int64],
    output: DeviceBuffer[DType.float64],
    output_count: DeviceBuffer[DType.int64],
    blocks: Int,
) raises:
    ctx.enqueue_function[_partial[D]](
        source._values,
        source._bits,
        sums,
        counts,
        Int64(len(source)),
        Int64(source.validity_offset()),
        Int64(source.has_validity()),
        Int64(blocks),
        Int64(predicate.op),
        predicate.literal,
        Int64(predicate.literal_left),
        Int64(reduction.input.op),
        reduction.input.literal,
        Int64(reduction.input.literal_left),
        grid_dim=blocks,
        block_dim=THREADS,
    )
    ctx.enqueue_function[_finish](
        sums,
        counts,
        output,
        output_count,
        Int64(blocks),
        grid_dim=1,
        block_dim=THREADS,
    )


def _execute[
    D: DType
](
    runtime: NvidiaRuntime,
    plan: AccelPlan,
    memory: AccelMemory,
    profiling: Bool,
) raises -> Tuple[DataFrame, Int, Int]:
    var source = runtime.upload[D](plan.source.numeric[D]())
    var ctx = runtime._ctx
    var blocks = memory.blocks
    var kernel_ns = 0
    var sums = ctx.enqueue_create_buffer[DType.float64](blocks)
    var counts = ctx.enqueue_create_buffer[DType.int64](blocks)
    var output = ctx.enqueue_create_buffer[DType.float64](1)
    var output_count = ctx.enqueue_create_buffer[DType.int64](1)
    var columns = List[Series]()
    for reduction in plan.reductions:
        var host_sum = _HostDownload[DType.float64](ctx, 1, 0)
        var host_count = _HostDownload[DType.int64](ctx, 1, 0)
        try:
            if profiling:

                def kernels(
                    timing_ctx: DeviceContext,
                ) raises {
                    imm source,
                    imm reduction,
                    imm plan,
                    imm sums,
                    imm counts,
                    imm output,
                    imm output_count,
                    imm blocks,
                }:
                    _enqueue[D](
                        timing_ctx,
                        source,
                        reduction,
                        plan.predicate,
                        sums,
                        counts,
                        output,
                        output_count,
                        blocks,
                    )

                kernel_ns += ctx.execution_time(kernels, 1)
            else:
                _enqueue[D](
                    ctx,
                    source,
                    reduction,
                    plan.predicate,
                    sums,
                    counts,
                    output,
                    output_count,
                    blocks,
                )
            host_sum.pending = True
            ctx.enqueue_copy(host_sum.values.unsafe_ptr(), output)
            host_count.pending = True
            ctx.enqueue_copy(host_count.values.unsafe_ptr(), output_count)
            var sum_column = host_sum.finish(0)
            var count_column = host_count.finish(0)
            if reduction.op == COUNT:
                columns.append(Series(reduction.name, count_column^))
            else:
                columns.append(
                    Series(
                        reduction.name,
                        Column[Scalar[D]](
                            [sum_column._get(0).cast[D]()],
                            [
                                count_column._get(0)
                                >= Int64(reduction.min_count)
                            ],
                        ),
                    )
                )
        except error:
            # Drain before scratch/output buffers or host pointers unwind.
            _drain_before_release(ctx)
            raise error^
    source.wait()
    var free_after = runtime.memory_info()[0]
    var result = DataFrame(columns^)
    if plan.limit >= 0:
        result = result.head(plan.limit)
    return (result^, kernel_ns, free_after)


def execute(
    runtime: NvidiaRuntime, query: LazyFrame, *, profiling: Bool = False
) raises -> Tuple[DataFrame, DataFrame]:
    var start = perf_counter_ns()
    var supported = True
    try:
        _ = lower_accel(query)
    except:
        supported = False
    if not supported:
        return execute_rows(runtime, query, profiling)
    var plan = lower_accel(query)
    var memory = plan_memory(plan)
    var free = runtime.memory_info()[0]
    var budget = runtime.memory_budget(free)
    memory.require_budget(budget)
    var executed: Tuple[DataFrame, Int, Int]
    if plan.source.dtype() == DataType.FLOAT32:
        executed = _execute[DType.float32](runtime, plan, memory, profiling)
    else:
        executed = _execute[DType.float64](runtime, plan, memory, profiling)
    var elapsed = Int(perf_counter_ns() - start)
    var result = executed[0].copy()
    var report = ExecutionReport()
    report.record(
        plan.root,
        "FUSED FLOAT REDUCTIONS",
        "nvidia",
        len(plan.source),
        result.height(),
        algorithm="scalar_filter_reductions",
        wall_ns=elapsed,
    )
    var columns = report.frame()._columns.copy()
    columns.append(
        Series("device_id", Column[Int64]([Int64(runtime.device_id())]))
    )
    columns.append(Series("device_name", Column[String]([runtime.name()])))
    columns.append(
        Series("upload_bytes", Column[Int64]([Int64(memory.input_bytes)]))
    )
    columns.append(
        Series("download_bytes", Column[Int64]([Int64(memory.download_bytes)]))
    )
    columns.append(
        Series(
            "workspace_bytes", Column[Int64]([Int64(memory.workspace_bytes)])
        )
    )
    columns.append(
        Series(
            "device_output_bytes", Column[Int64]([Int64(memory.output_bytes)])
        )
    )
    columns.append(
        Series(
            "peak_requested_device_bytes",
            Column[Int64]([Int64(memory.peak_bytes)]),
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
        Series("kernel_launches", Column[Int64]([Int64(memory.launches)]))
    )
    # Each host download finishes with a stream wait; source.wait adds one.
    # Event timing adds one timer wait per projection, only under profile().
    columns.append(
        Series(
            "synchronizations",
            Column[Int64](
                [Int64(len(plan.reductions) * (3 if profiling else 2) + 1)]
            ),
        )
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
                    "host upload -> GPU reductions -> scalar download -> host result"
                ]
            ),
        )
    )
    return (result^, DataFrame(columns^))


def describe(runtime: NvidiaRuntime, query: LazyFrame) -> String:
    try:
        var supported = True
        try:
            _ = lower_accel(query)
        except:
            supported = False
        if not supported:
            return describe_rows(runtime, query)
        var plan = lower_accel(query)
        var memory = plan_memory(plan)
        var free = runtime.memory_info()[0]
        var budget = runtime.memory_budget(free)
        memory.require_budget(budget)
        return (
            "ENGINE accel: NVIDIA supported; one in-memory float reduction region\n"
            + "  device="
            + String(runtime.device_id())
            + " ("
            + runtime.name()
            + "); free="
            + String(free)
            + " B; payload_budget="
            + String(budget)
            + " B\n  "
            + memory.describe()
            + "\n  host upload -> GPU reductions -> scalar download -> synchronized host result\n  estimates exclude SDK reservations; allocation/runtime faults propagate without CPU retry\n"
        )
    except error:
        return "ENGINE accel: " + String(error) + "\n"
