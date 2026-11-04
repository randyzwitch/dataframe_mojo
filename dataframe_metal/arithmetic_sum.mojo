"""Host batching for one arithmetic expression summed over a common filter."""
from std.time import perf_counter_ns
from std.math import isfinite
from dataframe import Column, DataFrame, DataType, LazyFrame, Series
from .arithmetic_plan import ArithmeticSum, plan_arithmetic
from .accelerator.faults import data_fault
from .accelerator.gpu.arithmetic_sum import arithmetic_partials
from .accelerator.gpu.erased_input import NumericView, numeric_views
from .accelerator.gpu.typed_storage import NativeColumnAny
from .accelerator.gpu.filter_sum import BLOCK_SIZE
from .accelerator.gpu.reduction_inputs import (
    HandoffTimings,
    _chunk_partials,
    _check_ran,
)
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.target import METAL_TARGET


def _run(
    plan: ArithmeticSum,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[Int128, Float64, Int]:
    if frame.height() == 0:
        return (Int128(0), Float64(0), 0)
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        var started = Int(perf_counter_ns())
        var ctx = scope.context()
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var owners = List[NativeColumnAny]()
        var inputs = List[List[NumericView]]()
        var names = List[String]()
        var slots = List[Int]()
        for name in [
            plan.left.copy(),
            plan.right.copy(),
            plan.predicate.filter_column.copy(),
        ]:
            var slot = -1
            for i in range(len(names)):
                if names[i] == name:
                    slot = i
            if slot < 0:
                slot = len(names)
                names.append(name.copy())
                inputs.append(
                    numeric_views(frame, name, scope, zero_copy, empty, owners)
                )
            slots.append(slot)
        var positions = List[Int](length=len(inputs), fill=0)
        var offsets = List[Int](length=len(inputs), fill=0)
        var segments = List[List[NumericView]]()
        var consumed = 0
        var total_parts = 0
        while consumed < frame.height():
            var length = frame.height() - consumed
            for i in range(len(inputs)):
                if positions[i] >= len(inputs[i]):
                    raise Error(
                        "Metal arithmetic columns have different lengths"
                    )
                length = min(length, inputs[i][positions[i]].rows - offsets[i])
            var segment = List[NumericView]()
            for i in range(len(inputs)):
                segment.append(inputs[i][positions[i]].view(offsets[i], length))
                offsets[i] += length
                if offsets[i] == inputs[i][positions[i]].rows:
                    positions[i] += 1
                    offsets[i] = 0
            segments.append(segment^)
            total_parts += _chunk_partials(length, frame.height())
            consumed += length
        timings.upload_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        var high = ctx.enqueue_create_buffer[DType.int64](total_parts)
        var low = ctx.enqueue_create_buffer[DType.uint64](total_parts)
        var sums = ctx.enqueue_create_buffer[DType.float32](total_parts)
        var comp = ctx.enqueue_create_buffer[DType.float32](total_parts)
        var counts = ctx.enqueue_create_buffer[DType.uint32](total_parts)
        var faults = ctx.enqueue_create_buffer[DType.uint32](total_parts)
        var hp = List[Int64](length=total_parts, fill=0)
        var lp = List[UInt64](length=total_parts, fill=0)
        var sp = List[Float32](length=total_parts, fill=0)
        var cp = List[Float32](length=total_parts, fill=0)
        var np = List[UInt32](length=total_parts, fill=0)
        var fp = List[UInt32](length=total_parts, fill=0)
        try:
            ctx.enqueue_memset(counts, UInt32.MAX)
            var output = 0
            for segment in segments:
                var a = segment[slots[0]].copy()
                var b = segment[slots[1]].copy()
                var p = segment[slots[2]].copy()
                var flags = (
                    UInt32(a.nullable)
                    | (UInt32(b.nullable) << 1)
                    | (UInt32(p.nullable) << 2)
                )
                var parts = _chunk_partials(a.rows, frame.height())
                ctx.enqueue_function[arithmetic_partials](
                    a.values,
                    b.values,
                    p.values,
                    a.bits,
                    b.bits,
                    p.bits,
                    UInt32(a.tag)
                    | (UInt32(p.tag) << 2)
                    | (flags << 4)
                    | (UInt32(plan.predicate.filtered) << 7),
                    UInt32(a.bit_offset),
                    UInt32(b.bit_offset),
                    UInt32(p.bit_offset),
                    UInt32(a.rows),
                    UInt32(parts * BLOCK_SIZE),
                    UInt32(output),
                    UInt32(plan.op)
                    | (UInt32(plan.literal_side) << 4)
                    | (plan.predicate.inclusive << 8),
                    plan.predicate.low,
                    plan.predicate.high,
                    plan.predicate.low_float,
                    plan.predicate.high_float,
                    plan.integer,
                    plan.floating,
                    high,
                    low,
                    sums,
                    comp,
                    counts,
                    faults,
                    grid_dim=parts,
                    block_dim=BLOCK_SIZE,
                )
                scope.record_launches(1)
                output += parts
            ctx.enqueue_copy(hp.unsafe_ptr(), high)
            ctx.enqueue_copy(lp.unsafe_ptr(), low)
            ctx.enqueue_copy(sp.unsafe_ptr(), sums)
            ctx.enqueue_copy(cp.unsafe_ptr(), comp)
            ctx.enqueue_copy(np.unsafe_ptr(), counts)
            ctx.enqueue_copy(fp.unsafe_ptr(), faults)
            scope.synchronize()
        except error:
            try:
                scope.synchronize()
            except:
                pass
            _ = owners^
            raise error^
        _ = owners^
        _check_ran(np)
        timings.kernel_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        var integer = Int128(0)
        var floating = Float64(0)
        var count = 0
        for i in range(total_parts):
            if fp[i]:
                raise data_fault("native integer arithmetic overflow")
            count += Int(np[i])
            if plan.predicate.dtype == DataType.FLOAT32:
                if not isfinite(sp[i]) or not isfinite(cp[i]):
                    raise data_fault(
                        "float32 arithmetic sum needs CPU non-finite accumulation"
                    )
                floating += Float64(sp[i]) - Float64(cp[i])
            elif plan.predicate.dtype == DataType.INT64:
                integer += (Int128(hp[i]) << 32) + Int128(lp[i])
            else:
                integer += Int128(hp[i])
        timings.finish_ns = Int(perf_counter_ns()) - started
        return (integer, floating, count)


def execute_arithmetic(
    region: LazyFrame, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var matched = plan_arithmetic(region)
    if not matched:
        raise Error(
            "Metal unsupported [handoff]: arithmetic sum was not accepted"
        )
    var plan = matched.take()
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    if frame.height() >= Int(Int32.MAX):
        raise Error("Metal unsupported [handoff]: 2**31 or more rows")
    var timings = HandoffTimings()
    var scope = Scope(service=shared_service())
    var started = Int(perf_counter_ns())
    _ = scope.context()
    timings.context_ns = Int(perf_counter_ns()) - started
    var result = _run(plan, frame, scope, zero_copy, timings)
    var dtype = plan.predicate.dtype
    var valid = result[2] >= plan.predicate.min_count
    var columns = List[Series]()
    var name = plan.predicate.output.copy()
    if dtype == DataType.FLOAT32:
        columns.append(
            Series(name, Column[Float32]([Float32(result[1])], [valid]))
        )
    elif dtype == DataType.INT64:
        if result[0] < Int128(Int64.MIN) or result[0] > Int128(Int64.MAX):
            raise data_fault("int64 expression sum overflow")
        columns.append(
            Series(
                name, Column[Int64]([result[0].cast[DType.int64]()], [valid])
            )
        )
    else:
        if result[0] < Int128(Int32.MIN) or result[0] > Int128(Int32.MAX):
            raise data_fault("int32 expression sum overflow")
        columns.append(
            Series(
                name, Column[Int32]([result[0].cast[DType.int32]()], [valid])
            )
        )
    scope.mark_success()
    return (DataFrame(columns^), timings^)
