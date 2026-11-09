"""Regions the CPU executor hands to Metal (`LazyFrame._handoff_region`).

A region arrives as a plan over in-memory columns: one scan, the filters
directly above it, and the projection or aggregation the executor offered.
`supports` decides from the plan and the columns' types alone; `execute`
runs an accepted region and returns its columns. A declined region runs
on the CPU in the same query.

Accepted today: native numeric and Boolean/count reductions over bounded
predicate and value programs, globally or with native integer grouping keys.
Int32, Int64, Float32 and Bool predicate inputs retain their physical widths.
Native chunks are mapped in place on Apple silicon; Float64 input stays on CPU.
"""
from max.gpu.host import DeviceBuffer
from std.time import perf_counter_ns

from dataframe.storage.column import Column
from dataframe.dtype import DataType
from dataframe.frame import DataFrame
from dataframe.lazy import LazyFrame
from .calibration import shape_key, measured_costs
from dataframe.series import Series
from .filter_plan import FilterSum, plan_region
from .region_plan import normalize_region
from .logical_plan import plan_logical, logical_key, finish_logical
from .ordered_plan import plan_ordered, ordered_key
from .ordered_reduce import execute_ordered
from .distinct_plan import plan_distinct, distinct_key
from .distinct_properties import cardinality_key
from .distinct_reduce import execute_distinct
from .cross_sum import reduce_cross
from .multi_plan import plan_bundle, bundle_key
from .multi_reduce import execute_bundle
from .arithmetic_plan import plan_arithmetic, arithmetic_key
from .arithmetic_sum import execute_arithmetic
from .composition_plan import plan_composition, composition_key
from .composition_reduce import execute_composition
from .extrema_plan import plan_extrema, extrema_key
from .extrema_reduce import execute_extrema
from .predicate_plan import plan_predicates, predicate_key
from .group_plan import plan_grouped, grouped_key
from .group_properties import sample_group_tuples
from .group_reduce import execute_grouped
from .predicate_reduce import execute_predicates
from .moment_plan import plan_moments, moment_key
from .logical_reduce import plan_packed_logical, execute_packed_logical
from .moment_reduce import execute_global_moments_math
from .moment_group_reduce import execute_grouped_moments_math
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
        if Bool(plan_distinct(region)) or Bool(plan_ordered(region)):
            return True
        var logical = plan_logical(region)
        if logical:
            return supports(logical.value().region)
        if plan_moments(region):
            return True
        var normalized = normalize_region(region)
        if not normalized:
            return False
        var selected = normalized.take()
        return (
            Bool(plan_region(selected))
            or Bool(plan_bundle(selected))
            or Bool(plan_arithmetic(selected))
            or Bool(plan_composition(selected))
            or Bool(plan_extrema(selected))
            or Bool(plan_predicates(selected))
            or Bool(plan_grouped(selected))
        )
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


def region_key(region: LazyFrame) raises -> String:
    """Schema-only operator identity shared by placement and capture."""
    var distinct = plan_distinct(region)
    if distinct:
        ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
        var key = distinct_key(distinct.value()) + cardinality_key(
            distinct.value(), frame
        )
        if distinct.value().keys:
            key += sample_group_tuples(
                frame, distinct.value().keys.value().names
            ).key()
        return key^
    var ordered = plan_ordered(region)
    if ordered:
        var key = ordered_key(ordered.value())
        if ordered.value().keys:
            ref frame = region._plan._frames[
                region._plan._nodes[0].source_slot()
            ]
            key += sample_group_tuples(
                frame, ordered.value().keys.value().names
            ).key()
        return key^
    var logical = plan_logical(region)
    if logical:
        var underlying = region_key(logical.value().region)
        return (
            logical_key(logical.value())
            + underlying if underlying else String()
        )
    var moments = plan_moments(region)
    if moments:
        var key = moment_key(moments.value())
        if moments.value().grouped:
            ref source = region._plan._frames[
                region._plan._nodes[0].source_slot()
            ]
            key += sample_group_tuples(
                source, moments.value().grouped.value().keys.names
            ).key()
        return key^
    var accepted = plan_region(region)
    if accepted:
        return operator_key(accepted.value())
    var bundle = plan_bundle(region)
    if bundle:
        return bundle_key(bundle.value())
    var arithmetic = plan_arithmetic(region)
    if arithmetic:
        return arithmetic_key(arithmetic.value())
    var composed = plan_composition(region)
    if composed:
        return composition_key(composed.value())
    var extrema = plan_extrema(region)
    if extrema:
        return extrema_key(extrema.value())
    var grouped = plan_grouped(region)
    if grouped:
        ref source = region._plan._frames[region._plan._nodes[0].source_slot()]
        return (
            grouped_key(grouped.value())
            + sample_group_tuples(source, grouped.value().keys.names).key()
        )
    var predicates = plan_predicates(region)
    if predicates:
        return predicate_key(predicates.value())
    return String()


def layout_key(region: LazyFrame) raises -> String:
    """One operator/layout key for placement, capture and execution traces."""
    var moments = plan_moments(region)
    if moments:
        return shape_key(moments.value().matching_region, region_key(region))
    var normalized = normalize_region(region)
    if not normalized:
        return String()
    var selected = normalized.take()
    var operator = region_key(selected)
    return shape_key(selected, operator) if operator else String()


def estimate_costs(
    region: LazyFrame, rows: Int
) -> Tuple[String, Float64, Float64]:
    """Measured Metal cost only for this build, host, threads and layout."""
    try:
        var key = layout_key(region)
        if not key:
            return (String(), Float64(-1), Float64(-1))
        if (
            key.startswith("distinct_reductions:v1:")
            and ":distinct_cardinality=uncertain" in key
        ):
            return (key^, Float64(-1), Float64(-1))
        var costs = measured_costs(key, rows)
        return (key^, costs[0], costs[1])
    except:
        return (String(), Float64(-1), Float64(-1))


def estimate(region: LazyFrame, rows: Int) -> Tuple[String, Float64]:
    var costs = estimate_costs(region, rows)
    return (costs[0].copy(), costs[2])


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
        if plan_distinct(region):
            return execute_distinct(region, zero_copy)
        if plan_ordered(region):
            return execute_ordered(region, zero_copy)
        var logical = plan_logical(region)
        if logical:
            var packed = plan_packed_logical(region)
            if packed:
                return execute_packed_logical(region, packed.value(), zero_copy)
            var result = execute_timed(logical.value().region, zero_copy)
            var started = Int(perf_counter_ns())
            var finished = finish_logical(logical.value(), result[0])
            var timings = result[1].copy()
            timings.finish_ns += Int(perf_counter_ns()) - started
            return (finished^, timings^)
        var moments = plan_moments(region)
        if moments:
            if moments.value().grouped:
                return execute_grouped_moments_math(region, zero_copy)
            return execute_global_moments_math(region, zero_copy)
        var normalized = normalize_region(region)
        if not normalized:
            raise Error(
                "Metal unsupported [handoff]: row expressions cannot be fused"
            )
        var selected = normalized.take()
        var result = _execute_on_device(selected, zero_copy)
        _ = selected^
        return result^


def _execute_on_device(
    region: LazyFrame, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var grouped = plan_grouped(region)
    if grouped:
        return execute_grouped(region, zero_copy)
    var accepted = plan_region(region)
    if not accepted:
        if plan_extrema(region):
            return execute_extrema(region, zero_copy)
        if plan_bundle(region):
            return execute_bundle(region, zero_copy)
        if plan_arithmetic(region):
            return execute_arithmetic(region, zero_copy)
        if plan_composition(region):
            return execute_composition(region, zero_copy)
        if plan_predicates(region):
            return execute_predicates(region, zero_copy)
        return execute_composition(region, zero_copy)
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
