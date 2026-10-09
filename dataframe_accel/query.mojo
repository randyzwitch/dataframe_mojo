"""NVIDIA execution of capability-checked float reduction regions."""
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer
from max.gpu.sync import barrier
from std.memory import AddressSpace, stack_allocation
from std.time import perf_counter_ns
from dataframe.accel_plan import AccelPlan, lower_accel
from dataframe.column import Column
from dataframe.dtype import DataType
from dataframe.expr import COUNT
from dataframe.float_ops import arithmetic, compare
from dataframe.frame import DataFrame
from dataframe.lazy import LazyFrame
from dataframe.series import Series
from dataframe.execution_report import ExecutionReport
from .nvidia import NvidiaRuntime, _HostDownload, _drain_before_release

comptime THREADS = 256
comptime MAX_BLOCKS = 1024


@always_inline
def _reduce(total: Float64, count: Int64) -> Tuple[Float64, Int64]:
    # This pinned SDK does not support Float64 block.sum warp shuffles.
    var sums = stack_allocation[
        THREADS, DType.float64, address_space=AddressSpace.SHARED
    ]()
    var counts = stack_allocation[
        THREADS, DType.int64, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    sums[unsafe_offset=tid] = total
    counts[unsafe_offset=tid] = count
    barrier()
    comptime for step in range(8):
        comptime stride = THREADS >> (step + 1)
        if tid < stride:
            sums[unsafe_offset=tid] += sums[unsafe_offset=tid + stride]
            counts[unsafe_offset=tid] += counts[unsafe_offset=tid + stride]
        barrier()
    return (sums[unsafe_offset=0], counts[unsafe_offset=0])


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


def _finish(
    sums: DeviceBuffer[DType.float64].device_type,
    counts: DeviceBuffer[DType.int64].device_type,
    output: DeviceBuffer[DType.float64].device_type,
    output_count: DeviceBuffer[DType.int64].device_type,
    blocks: Int64,
):
    var total = Float64(0)
    var count = Int64(0)
    var i = Int(thread_idx.x)
    while i < Int(blocks):
        total += sums[unsafe_offset=i]
        count += counts[unsafe_offset=i]
        i += THREADS
    var reduced = _reduce(total, count)
    if thread_idx.x == 0:
        output[unsafe_offset=0] = reduced[0]
        output_count[unsafe_offset=0] = reduced[1]


def _execute[
    D: DType
](runtime: NvidiaRuntime, plan: AccelPlan) raises -> DataFrame:
    var source = runtime.upload[D](plan.source.numeric[D]())
    var ctx = runtime._ctx
    var blocks = max(1, min(MAX_BLOCKS, (len(source) + THREADS - 1) // THREADS))
    var sums = ctx.enqueue_create_buffer[DType.float64](blocks)
    var counts = ctx.enqueue_create_buffer[DType.int64](blocks)
    var output = ctx.enqueue_create_buffer[DType.float64](1)
    var output_count = ctx.enqueue_create_buffer[DType.int64](1)
    var columns = List[Series]()
    for reduction in plan.reductions:
        var host_sum = _HostDownload[DType.float64](ctx, 1, 0)
        var host_count = _HostDownload[DType.int64](ctx, 1, 0)
        try:
            ctx.enqueue_function[_partial[D]](
                source._values,
                source._bits,
                sums,
                counts,
                Int64(len(source)),
                Int64(source.validity_offset()),
                Int64(source.has_validity()),
                Int64(blocks),
                Int64(plan.predicate.op),
                plan.predicate.literal,
                Int64(plan.predicate.literal_left),
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
    var result = DataFrame(columns^)
    if plan.limit >= 0:
        return result.head(plan.limit)
    return result^


def execute(
    runtime: NvidiaRuntime, query: LazyFrame
) raises -> Tuple[DataFrame, DataFrame]:
    var plan = lower_accel(query)
    var start = perf_counter_ns()
    var result: DataFrame
    if plan.source.dtype() == DataType.FLOAT32:
        result = _execute[DType.float32](runtime, plan)
    else:
        result = _execute[DType.float64](runtime, plan)
    var elapsed = Int(perf_counter_ns() - start)
    var report = ExecutionReport()
    # One observed fused region; no invented per-operator row/timing counts.
    report.record(
        plan.root,
        "FUSED FLOAT REDUCTIONS",
        "nvidia",
        len(plan.source),
        result.height(),
        algorithm="scalar_filter_reductions",
        wall_ns=elapsed,
    )
    return (result^, report.frame())


def describe(query: LazyFrame) -> String:
    try:
        var plan = lower_accel(query)
        return (
            "ENGINE accel: NVIDIA supported; one in-memory float reduction region\n"
            + "  "
            + String(len(plan.reductions))
            + " sum/count projections; Float64 accumulation; Int64 counts\n"
        )
    except error:
        return "ENGINE accel: " + String(error) + "\n"
