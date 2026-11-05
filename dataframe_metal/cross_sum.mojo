"""Cross-column filtered sums with aligned native chunk views and one wait."""
from std.time import perf_counter_ns
from dataframe import DataFrame
from dataframe.dtype import DataType
from .filter_plan import FilterSum
from .accelerator.gpu.cross_sum import cross_partials
from .accelerator.gpu.filter_sum import BLOCK_SIZE, split_total
from .accelerator.gpu.query_scope import Scope
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.typed_storage import NativeColumn
from .accelerator.gpu.reduction_inputs import (
    HandoffTimings,
    _place_chunks,
    _chunk_partials,
    _check_ran,
)


def _run[
    value_dtype: DType, filter_dtype: DType
](
    plan: FilterSum,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[Int128, Float64, Int]:
    var value_series = frame.column(plan.column)
    var predicate_series = frame.column(plan.filter_column)
    if len(value_series) == 0:
        return (Int128(0), Float64(0), 0)
    var started = Int(perf_counter_ns())
    var values = _place_chunks[value_dtype](scope, value_series, zero_copy)
    var predicates = _place_chunks[filter_dtype](
        scope, predicate_series, zero_copy
    )
    var aligned_values = List[NativeColumn[value_dtype]]()
    var aligned_predicates = List[NativeColumn[filter_dtype]]()
    var vi = 0
    var pi = 0
    var vo = 0
    var po = 0
    # Columns can have unrelated Arrow array boundaries. Intersect their
    # logical row windows; view offsets preserve values and packed validity.
    while vi < len(values) and pi < len(predicates):
        var length = min(values[vi].length - vo, predicates[pi].length - po)
        aligned_values.append(values[vi].view(vo, length))
        aligned_predicates.append(predicates[pi].view(po, length))
        vo += length
        po += length
        if vo == values[vi].length:
            vi += 1
            vo = 0
        if po == predicates[pi].length:
            pi += 1
            po = 0
    if vi != len(values) or pi != len(predicates):
        raise Error("Metal reduction inputs have different lengths")
    timings.upload_ns = Int(perf_counter_ns()) - started
    started = Int(perf_counter_ns())
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        var total_parts = 0
        for value in aligned_values:
            total_parts += _chunk_partials(value.length, len(value_series))
        var ctx = scope.context()
        var high = ctx.enqueue_create_buffer[DType.int64](total_parts)
        var low = ctx.enqueue_create_buffer[DType.uint64](total_parts)
        var sums = ctx.enqueue_create_buffer[DType.float32](total_parts)
        var compensations = ctx.enqueue_create_buffer[DType.float32](
            total_parts
        )
        var counts = ctx.enqueue_create_buffer[DType.uint32](total_parts)
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var hp = List[Int64](length=total_parts, fill=0)
        var lp = List[UInt64](length=total_parts, fill=0)
        var sp = List[Float32](length=total_parts, fill=0)
        var cp = List[Float32](length=total_parts, fill=0)
        var np = List[UInt32](length=total_parts, fill=0)
        try:
            ctx.enqueue_memset(counts, UInt32.MAX)
            var offset = 0
            for i in range(len(aligned_values)):
                var value = aligned_values[i].copy()
                var predicate = aligned_predicates[i].copy()
                var vb = (
                    value.validity()
                    .value()
                    .copy() if value.has_validity() else empty.copy()
                )
                var pb = (
                    predicate.validity()
                    .value()
                    .copy() if predicate.has_validity() else empty.copy()
                )
                var flags = UInt32(value.has_validity()) | (
                    UInt32(predicate.has_validity()) << 1
                )
                var partials = _chunk_partials(value.length, len(value_series))
                ctx.enqueue_function[cross_partials[value_dtype, filter_dtype]](
                    value.values(),
                    UInt32(value.value_offset),
                    vb,
                    UInt32(value.validity_bit_offset),
                    predicate.values(),
                    UInt32(predicate.value_offset),
                    pb,
                    UInt32(predicate.validity_bit_offset),
                    flags,
                    UInt32(value.length),
                    UInt32(partials * BLOCK_SIZE),
                    plan.low,
                    plan.high,
                    plan.low_float,
                    plan.high_float,
                    plan.inclusive,
                    high,
                    low,
                    sums,
                    compensations,
                    counts,
                    UInt32(offset),
                    grid_dim=partials,
                    block_dim=BLOCK_SIZE,
                )
                scope.record_launches(1)
                offset += partials
            ctx.enqueue_copy(hp.unsafe_ptr(), high)
            ctx.enqueue_copy(lp.unsafe_ptr(), low)
            ctx.enqueue_copy(sp.unsafe_ptr(), sums)
            ctx.enqueue_copy(cp.unsafe_ptr(), compensations)
            ctx.enqueue_copy(np.unsafe_ptr(), counts)
            scope.synchronize()
        except error:
            try:
                scope.synchronize()
            except:
                pass
            _ = (hp^, lp^, sp^, cp^, np^)
            _ = (high^, low^, sums^, compensations^, counts^, empty^)
            _ = aligned_values^
            _ = aligned_predicates^
            raise error^
        # Mojo destroys owners after their last use. Pin their reservations
        # through both the normal drain and the exception drain.
        _ = (high^, low^, sums^, compensations^, counts^, empty^)
        _ = aligned_values^
        _ = aligned_predicates^
        _check_ran(np)
        timings.kernel_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        var integer = Int128(0)
        var floating = Float64(0)
        var count = 0
        for p in range(total_parts):
            comptime if value_dtype == DType.float32:
                floating += Float64(sp[p]) - Float64(cp[p])
            else:
                comptime if split_total[value_dtype]():
                    integer += (hp[p].cast[DType.int128]() << 32) + lp[p].cast[
                        DType.int128
                    ]()
                else:
                    integer += hp[p].cast[DType.int128]()
            count += Int(np[p])
        timings.finish_ns = Int(perf_counter_ns()) - started
        return (integer, floating, count)


def _filter[
    value_dtype: DType
](
    plan: FilterSum,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[Int128, Float64, Int]:
    if plan.filter_dtype == DataType.INT32:
        return _run[value_dtype, DType.int32](
            plan, frame, scope, zero_copy, timings
        )
    if plan.filter_dtype == DataType.INT64:
        return _run[value_dtype, DType.int64](
            plan, frame, scope, zero_copy, timings
        )
    return _run[value_dtype, DType.float32](
        plan, frame, scope, zero_copy, timings
    )


def reduce_cross(
    plan: FilterSum,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[Int128, Float64, Int]:
    if plan.dtype == DataType.INT32:
        return _filter[DType.int32](plan, frame, scope, zero_copy, timings)
    if plan.dtype == DataType.INT64:
        return _filter[DType.int64](plan, frame, scope, zero_copy, timings)
    return _filter[DType.float32](plan, frame, scope, zero_copy, timings)
