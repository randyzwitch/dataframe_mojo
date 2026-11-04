"""Regions the CPU executor hands to Metal (`LazyFrame._handoff_region`).

A region arrives as a plan over in-memory columns: one scan, the filters
directly above it, and the projection or aggregation the executor offered.
`supports` decides from the plan and the columns' types alone; `execute`
runs an accepted region and returns its columns. A declined region runs
on the CPU in the same query.

Accepted today: `select(col(c).sum())` over filters that compare `c` with
untyped numeric literals (`>`, `>=`, `<`, `<=`, `==`, joined by `&`), where
`c` is Int32, Int64 or Float32. A one-chunk column is read in place (`wrap_native`): Apple silicon shares
memory, so nothing is copied. Float64 and wider work stays on the CPU:
Apple GPUs have no 64-bit floating point, and emulating it would be slower
than the CPU.
"""
from max.gpu.host import DeviceBuffer
from std.time import perf_counter_ns

from dataframe.storage.column import Column
from dataframe.dtype import DataType
from dataframe.frame import DataFrame
from dataframe.lazy import LazyFrame
from dataframe.placement import load_costs
from dataframe.series import Series
from .filter_plan import FilterSum, plan_region
from .cross_sum import reduce_cross
from .accelerator.gpu.reduction_inputs import (
    HandoffTimings,
    _place_chunks,
    _check_ran,
    _chunk_partials,
)
from .accelerator.faults import data_fault
from .accelerator.gpu.filter_sum import (
    BLOCK_SIZE,
    float32_partials,
    integer_partials,
    split_total,
)
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.typed_storage import (
    NativeColumn,
    upload_native,
    wrap_native,
)


def supports(region: LazyFrame) -> Bool:
    """Whether Metal runs `region`; never raises."""
    comptime if not METAL_TARGET:
        return False
    try:
        return Bool(plan_region(region))
    except:
        return False


def operator_key(plan: FilterSum) -> String:
    """The calibration key of an accepted region (`dataframe.placement`)."""
    if plan.filter_column != plan.column:
        return (
            "cross_filter_sum:"
            + plan.filter_dtype.name()
            + ":"
            + plan.dtype.name()
        )
    return (
        "filter_sum:"
        + plan.dtype.name()
        + (":filtered" if plan.filtered else ":plain")
    )


def estimate(region: LazyFrame, rows: Int) -> Tuple[String, Float64]:
    """This region's calibration key and its measured cost on Metal for
    `rows` input rows, in nanoseconds; -1 when it has no measured cost."""
    try:
        var accepted = plan_region(region)
        if not accepted:
            return (String(), Float64(-1))
        var key = operator_key(accepted.value())
        var costs = load_costs("accel")
        var cost = costs.get(key)
        if not cost:
            return (key^, Float64(-1))
        return (key^, cost.value().ns(rows))
    except:
        return (String(), Float64(-1))


def _enqueue_integer[
    dtype: DType, nullable: Bool, filtered: Bool
](
    plan: FilterSum,
    column: NativeColumn[dtype],
    scope: Scope,
    partials: Int,
    offset: Int,
    high: DeviceBuffer[DType.int64],
    low: DeviceBuffer[DType.uint64],
    counts: DeviceBuffer[DType.uint32],
    empty_bits: DeviceBuffer[DType.uint8],
) raises:
    """Queue one chunk; its owners and output roots survive the final wait."""
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        var bits = (
            column.validity().value().copy() if nullable else empty_bits.copy()
        )
        scope.context().enqueue_function[
            integer_partials[dtype, nullable, filtered]
        ](
            column.values(),
            UInt32(column.value_offset),
            bits,
            UInt32(column.validity_bit_offset),
            UInt32(column.length),
            UInt32(partials * BLOCK_SIZE),
            plan.low,
            plan.high,
            high,
            low,
            counts,
            UInt32(offset),
            grid_dim=partials,
            block_dim=BLOCK_SIZE,
        )
        scope.record_launches(1)


def _integer_launch[
    dtype: DType
](
    plan: FilterSum,
    columns: List[NativeColumn[dtype]],
    scope: Scope,
) raises -> Tuple[
    List[Scalar[DType.int64]], List[Scalar[DType.uint64]], List[UInt32]
]:
    """Queue all chunks into disjoint partial ranges; read back once.

    Keep every input, scratch buffer and host readback owner alive until
    synchronization, including when queueing or synchronization raises.
    """
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        var ctx = scope.context()
        var total_rows = 0
        for column in columns:
            total_rows += column.length
        var total_parts = 0
        for column in columns:
            total_parts += _chunk_partials(column.length, total_rows)
        var high = ctx.enqueue_create_buffer[DType.int64](total_parts)
        var low = ctx.enqueue_create_buffer[DType.uint64](total_parts)
        var counts = ctx.enqueue_create_buffer[DType.uint32](total_parts)
        var empty_bits = ctx.enqueue_create_buffer[DType.uint8](1)
        var first_parts = List[Scalar[DType.int64]](length=total_parts, fill=0)
        var second_parts = List[Scalar[DType.uint64]](
            length=total_parts, fill=0
        )
        var count_parts = List[UInt32](length=total_parts, fill=0)
        try:
            ctx.enqueue_memset(counts, UInt32.MAX)
            var offset = 0
            for column in columns:
                var partials = _chunk_partials(column.length, total_rows)
                if column.has_validity():
                    if plan.filtered:
                        _enqueue_integer[dtype, True, True](
                            plan,
                            column,
                            scope,
                            partials,
                            offset,
                            high,
                            low,
                            counts,
                            empty_bits,
                        )
                    else:
                        _enqueue_integer[dtype, True, False](
                            plan,
                            column,
                            scope,
                            partials,
                            offset,
                            high,
                            low,
                            counts,
                            empty_bits,
                        )
                elif plan.filtered:
                    _enqueue_integer[dtype, False, True](
                        plan,
                        column,
                        scope,
                        partials,
                        offset,
                        high,
                        low,
                        counts,
                        empty_bits,
                    )
                else:
                    _enqueue_integer[dtype, False, False](
                        plan,
                        column,
                        scope,
                        partials,
                        offset,
                        high,
                        low,
                        counts,
                        empty_bits,
                    )
                offset += partials
            ctx.enqueue_copy(first_parts.unsafe_ptr(), high)
            ctx.enqueue_copy(second_parts.unsafe_ptr(), low)
            ctx.enqueue_copy(count_parts.unsafe_ptr(), counts)
            scope.synchronize()
        except error:
            try:
                scope.synchronize()
            except:
                pass
            raise error^
        _check_ran(count_parts)
        return (first_parts^, second_parts^, count_parts^)


def _integer_sum[
    dtype: DType
](
    plan: FilterSum,
    series: Series,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[Int128, Int]:
    """The exact total of the kept values and how many were kept."""
    if len(series) == 0:
        return (Int128(0), 0)
    var started = Int(perf_counter_ns())
    var columns = _place_chunks[dtype](scope, series, zero_copy)
    timings.upload_ns = Int(perf_counter_ns()) - started
    started = Int(perf_counter_ns())
    var parts = _integer_launch[dtype](plan, columns, scope)
    timings.kernel_ns = Int(perf_counter_ns()) - started
    started = Int(perf_counter_ns())
    var total = Int128(0)
    var count = 0
    for p in range(len(parts[0])):
        comptime if split_total[dtype]():
            total += (parts[0][p].cast[DType.int128]() << 32) + parts[1][
                p
            ].cast[DType.int128]()
        else:
            total += parts[0][p].cast[DType.int128]()
        count += Int(parts[2][p])
    _ = columns^
    timings.finish_ns = Int(perf_counter_ns()) - started
    return (total, count)


def _enqueue_float32[
    nullable: Bool, filtered: Bool
](
    plan: FilterSum,
    column: NativeColumn[DType.float32],
    scope: Scope,
    partials: Int,
    offset: Int,
    sums: DeviceBuffer[DType.float32],
    compensations: DeviceBuffer[DType.float32],
    counts: DeviceBuffer[DType.uint32],
    empty_bits: DeviceBuffer[DType.uint8],
) raises:
    """Queue one chunk; its owners and output roots survive the final wait."""
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        var bits = (
            column.validity().value().copy() if nullable else empty_bits.copy()
        )
        scope.context().enqueue_function[float32_partials[nullable, filtered]](
            column.values(),
            UInt32(column.value_offset),
            bits,
            UInt32(column.validity_bit_offset),
            UInt32(column.length),
            UInt32(partials * BLOCK_SIZE),
            plan.low_float,
            plan.high_float,
            plan.inclusive,
            sums,
            compensations,
            counts,
            UInt32(offset),
            grid_dim=partials,
            block_dim=BLOCK_SIZE,
        )
        scope.record_launches(1)


def _float32_launch(
    plan: FilterSum,
    columns: List[NativeColumn[DType.float32]],
    scope: Scope,
) raises -> Tuple[
    List[Scalar[DType.float32]], List[Scalar[DType.float32]], List[UInt32]
]:
    """Queue all chunks into disjoint partial ranges; read back once.

    Keep every input, scratch buffer and host readback owner alive until
    synchronization, including when queueing or synchronization raises.
    """
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        var ctx = scope.context()
        var total_rows = 0
        for column in columns:
            total_rows += column.length
        var total_parts = 0
        for column in columns:
            total_parts += _chunk_partials(column.length, total_rows)
        var sums = ctx.enqueue_create_buffer[DType.float32](total_parts)
        var compensations = ctx.enqueue_create_buffer[DType.float32](
            total_parts
        )
        var counts = ctx.enqueue_create_buffer[DType.uint32](total_parts)
        var empty_bits = ctx.enqueue_create_buffer[DType.uint8](1)
        var first_parts = List[Scalar[DType.float32]](
            length=total_parts, fill=0
        )
        var second_parts = List[Scalar[DType.float32]](
            length=total_parts, fill=0
        )
        var count_parts = List[UInt32](length=total_parts, fill=0)
        try:
            ctx.enqueue_memset(counts, UInt32.MAX)
            var offset = 0
            for column in columns:
                var partials = _chunk_partials(column.length, total_rows)
                if column.has_validity():
                    if plan.filtered:
                        _enqueue_float32[True, True](
                            plan,
                            column,
                            scope,
                            partials,
                            offset,
                            sums,
                            compensations,
                            counts,
                            empty_bits,
                        )
                    else:
                        _enqueue_float32[True, False](
                            plan,
                            column,
                            scope,
                            partials,
                            offset,
                            sums,
                            compensations,
                            counts,
                            empty_bits,
                        )
                elif plan.filtered:
                    _enqueue_float32[False, True](
                        plan,
                        column,
                        scope,
                        partials,
                        offset,
                        sums,
                        compensations,
                        counts,
                        empty_bits,
                    )
                else:
                    _enqueue_float32[False, False](
                        plan,
                        column,
                        scope,
                        partials,
                        offset,
                        sums,
                        compensations,
                        counts,
                        empty_bits,
                    )
                offset += partials
            ctx.enqueue_copy(first_parts.unsafe_ptr(), sums)
            ctx.enqueue_copy(second_parts.unsafe_ptr(), compensations)
            ctx.enqueue_copy(count_parts.unsafe_ptr(), counts)
            scope.synchronize()
        except error:
            try:
                scope.synchronize()
            except:
                pass
            raise error^
        _check_ran(count_parts)
        return (first_parts^, second_parts^, count_parts^)


def _float32_sum(
    plan: FilterSum,
    series: Series,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[Float64, Int]:
    """The compensated total of the kept values and how many were kept."""
    if len(series) == 0:
        return (Float64(0), 0)
    var started = Int(perf_counter_ns())
    var columns = _place_chunks[DType.float32](scope, series, zero_copy)
    timings.upload_ns = Int(perf_counter_ns()) - started
    started = Int(perf_counter_ns())
    var parts = _float32_launch(plan, columns, scope)
    timings.kernel_ns = Int(perf_counter_ns()) - started
    started = Int(perf_counter_ns())
    var total = Float64(0)
    var count = 0
    for p in range(len(parts[0])):
        total += (
            parts[0][p].cast[DType.float64]()
            - parts[1][p].cast[DType.float64]()
        )
        count += Int(parts[2][p])
    _ = columns^
    timings.finish_ns = Int(perf_counter_ns()) - started
    return (total, count)


def execute_timed(
    region: LazyFrame, zero_copy: Bool = True
) raises -> Tuple[DataFrame, HandoffTimings]:
    """Run an accepted region on Metal and report where the time went.
    With `zero_copy` (the default) the GPU reads every native chunk and
    its bitmap in place; without it the column is copied, for comparison."""
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        return _execute_on_device(region, zero_copy)


def _execute_on_device(
    region: LazyFrame, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var accepted = plan_region(region)
    if not accepted:
        raise Error("Metal unsupported [handoff]: region was not accepted")
    var plan = accepted.take()
    var timings = HandoffTimings()
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    var series = frame.column(plan.column)
    if len(series) >= Int(Int32.MAX):
        raise Error("Metal unsupported [handoff]: 2**31 or more rows")
    var started = Int(perf_counter_ns())
    var scope = Scope(service=shared_service())
    _ = scope.context()
    timings.context_ns = Int(perf_counter_ns()) - started
    var integer = Int128(0)
    var floating = Float64(0)
    var count = 0
    if plan.filter_column != plan.column:
        var result = reduce_cross(plan, frame, scope, zero_copy, timings)
        integer = result[0]
        floating = result[1]
        count = result[2]
    elif plan.dtype == DataType.FLOAT32:
        var result = _float32_sum(plan, series, scope, zero_copy, timings)
        floating = result[0]
        count = result[1]
    else:
        var result: Tuple[Int128, Int]
        if plan.dtype == DataType.INT32:
            result = _integer_sum[DType.int32](
                plan, series, scope, zero_copy, timings
            )
        else:
            result = _integer_sum[DType.int64](
                plan, series, scope, zero_copy, timings
            )
        integer = result[0]
        count = result[1]
    var valid = count >= plan.min_count
    var output: Series
    if plan.dtype == DataType.FLOAT32:
        output = Series(
            plan.output,
            Column[Float32]([floating.cast[DType.float32]()], [valid]),
        )
    elif plan.dtype == DataType.INT32:
        if integer > Int128(Int32.MAX) or integer < Int128(Int32.MIN):
            raise data_fault("int32 expression sum overflow")
        output = Series(
            plan.output, Column[Int32]([integer.cast[DType.int32]()], [valid])
        )
    else:
        if integer > Int128(Int64.MAX) or integer < Int128(Int64.MIN):
            raise data_fault("int64 expression sum overflow")
        output = Series(
            plan.output, Column[Int64]([integer.cast[DType.int64]()], [valid])
        )
    scope.mark_success()
    return (DataFrame([output^]), timings^)


def execute(region: LazyFrame) raises -> DataFrame:
    """Run an accepted region on Metal; its columns."""
    return execute_timed(region)[0].copy()
