"""Host orchestration for fused heterogeneous SUM, COUNT and MEAN."""
from std.time import perf_counter_ns
from std.math import isfinite
from dataframe import Column, DataFrame, DataType, LazyFrame, Series
from dataframe.expr import COUNT, MEAN, SUM
from .multi_plan import ReductionBundle, plan_bundle
from .exact_float import finish_float_bins
from .accelerator.faults import data_fault
from .accelerator.gpu.erased_input import NumericView, numeric_views
from .accelerator.gpu.typed_storage import NativeColumnAny
from .accelerator.gpu.multi_reduce import multi_partials
from .accelerator.gpu.filter_sum import BLOCK_SIZE
from .accelerator.gpu.reduction_inputs import (
    HandoffTimings,
    _chunk_partials,
    _check_ran,
)
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.target import METAL_TARGET


def _run[
    precise: Bool, wide: Bool, arity: Int
](
    bundle: ReductionBundle,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[List[Int128], List[Float64], List[Int]]:
    comptime FIRST_BIN = 0 if wide else 3
    comptime BIN_COUNT = 9 if wide else 3
    var nf = arity
    var integers = List[Int128](length=nf, fill=0)
    var floating = List[Float64](length=nf, fill=0)
    var counts_out = List[Int](length=nf, fill=0)
    if frame.height() == 0:
        return (integers^, floating^, counts_out^)
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        var started = Int(perf_counter_ns())
        var ctx = scope.context()
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var inputs = List[List[NumericView]]()
        # Byte views retain device buffers; typed owners additionally retain
        # the query's checked allocation reservations until completion.
        var owners = List[NativeColumnAny]()
        var predicate_input = -1
        for i in range(nf):
            inputs.append(
                numeric_views(
                    frame,
                    bundle.fields[i].column,
                    scope,
                    zero_copy,
                    empty,
                    owners,
                )
            )
            if bundle.fields[i].column == bundle.predicate.filter_column:
                predicate_input = i
        if predicate_input < 0:
            predicate_input = len(inputs)
            inputs.append(
                numeric_views(
                    frame,
                    bundle.predicate.filter_column,
                    scope,
                    zero_copy,
                    empty,
                    owners,
                )
            )
        var positions = List[Int](length=len(inputs), fill=0)
        var offsets = List[Int](length=len(inputs), fill=0)
        var segments = List[List[NumericView]]()
        var consumed = 0
        while consumed < frame.height():
            var length = frame.height() - consumed
            for i in range(len(inputs)):
                if positions[i] >= len(inputs[i]):
                    raise Error(
                        "Metal reduction columns have different lengths"
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
            consumed += length
        timings.upload_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        var total_parts = 0
        for segment in segments:
            total_parts += _chunk_partials(segment[0].rows, frame.height())
        var size = total_parts * nf
        var high = ctx.enqueue_create_buffer[DType.int64](size)
        var low = ctx.enqueue_create_buffer[DType.uint64](size)
        var sums = ctx.enqueue_create_buffer[DType.float32](size)
        var compensation = ctx.enqueue_create_buffer[DType.float32](size)
        var counts = ctx.enqueue_create_buffer[DType.uint32](size)
        var bins = ctx.enqueue_create_buffer[DType.int64](
            size * BIN_COUNT if precise else 1
        )
        var specials = ctx.enqueue_create_buffer[DType.uint32](
            size * 6 if precise else 1
        )
        var hp = List[Int64](length=size, fill=0)
        var lp = List[UInt64](length=size, fill=0)
        var sp = List[Float32](length=size, fill=0)
        var cp = List[Float32](length=size, fill=0)
        var np = List[UInt32](length=size, fill=0)
        var bp = List[Int64](length=size * BIN_COUNT if precise else 0, fill=0)
        var xp = List[UInt32](length=size * 6 if precise else 0, fill=0)
        try:
            ctx.enqueue_memset(counts, UInt32.MAX)
            comptime if precise:
                ctx.enqueue_memset(bins, Int64(0))
                ctx.enqueue_memset(specials, UInt32(0))
            var output_offset = 0
            for segment in segments:
                var values = List[NumericView]()
                var tags = SIMD[DType.uint32, 4](0)
                var bit_offsets = SIMD[DType.uint32, 4](0)
                var valid_flags = UInt32(0)
                var sum_flags = UInt32(0)
                for i in range(4):
                    if i < nf:
                        values.append(segment[i].copy())
                        tags[i] = UInt32(segment[i].tag)
                        bit_offsets[i] = UInt32(segment[i].bit_offset)
                        if segment[i].nullable:
                            valid_flags |= UInt32(1) << UInt32(i)
                        if bundle.fields[i].needs_sum:
                            sum_flags |= UInt32(1) << UInt32(i)
                    else:
                        values.append(
                            NumericView(
                                empty.copy(), empty.copy(), 0, 0, 0, False
                            )
                        )
                var predicate = segment[predicate_input].copy()
                var partials = _chunk_partials(segment[0].rows, frame.height())
                ctx.enqueue_function[multi_partials[precise, wide, arity]](
                    values[0].values,
                    values[1].values,
                    values[2].values,
                    values[3].values,
                    values[0].bits,
                    values[1].bits,
                    values[2].bits,
                    values[3].bits,
                    tags[0] | (tags[1] << 2) | (tags[2] << 4) | (tags[3] << 6),
                    UInt64(bit_offsets[0]) | (UInt64(bit_offsets[1]) << 32),
                    UInt64(bit_offsets[2]) | (UInt64(bit_offsets[3]) << 32),
                    valid_flags
                    | (sum_flags << 4)
                    | (UInt32(nf) << 8)
                    | (UInt32(predicate.nullable) << 12)
                    | (bundle.predicate.inclusive << 16),
                    predicate.values,
                    predicate.bits,
                    UInt32(predicate.tag)
                    | (UInt32(bundle.predicate.filtered) << 2)
                    | (
                        UInt32(
                            predicate_input + 1 if predicate_input < nf else 0
                        )
                        << 3
                    ),
                    UInt32(predicate.bit_offset),
                    bundle.predicate.low,
                    bundle.predicate.high,
                    bundle.predicate.low_float,
                    bundle.predicate.high_float,
                    UInt32(segment[0].rows),
                    UInt32(partials * BLOCK_SIZE),
                    UInt32(output_offset),
                    high,
                    low,
                    sums,
                    compensation,
                    counts,
                    bins,
                    specials,
                    grid_dim=partials,
                    block_dim=BLOCK_SIZE,
                )
                scope.record_launches(1)
                output_offset += partials
            ctx.enqueue_copy(hp.unsafe_ptr(), high)
            ctx.enqueue_copy(lp.unsafe_ptr(), low)
            ctx.enqueue_copy(sp.unsafe_ptr(), sums)
            ctx.enqueue_copy(cp.unsafe_ptr(), compensation)
            ctx.enqueue_copy(np.unsafe_ptr(), counts)
            comptime if precise:
                ctx.enqueue_copy(bp.unsafe_ptr(), bins)
                ctx.enqueue_copy(xp.unsafe_ptr(), specials)
            scope.synchronize()
        except error:
            try:
                scope.synchronize()
            except:
                pass
            raise error^
        _check_ran(np)
        timings.kernel_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        for field in range(nf):
            var exact = List[Int128](length=9, fill=0)
            var nan = False
            var positive = False
            var negative = False
            var least = 512
            var greatest = 0
            for part in range(total_parts):
                var i = part * nf + field
                counts_out[field] += Int(np[i])
                if bundle.fields[field].dtype == DataType.FLOAT32:
                    comptime if precise:
                        if xp[i * 6 + 5]:
                            raise Error("Metal exact bins need wide scan")
                        for bin in range(BIN_COUNT):
                            exact[bin + FIRST_BIN] += Int128(
                                bp[i * BIN_COUNT + bin]
                            )
                        nan = nan or xp[i * 6] != 0
                        positive = positive or xp[i * 6 + 1] != 0
                        negative = negative or xp[i * 6 + 2] != 0
                        least = min(least, Int(xp[i * 6 + 3]))
                        greatest = max(greatest, Int(xp[i * 6 + 4]))
                    else:
                        if not isfinite(sp[i]) or not isfinite(cp[i]):
                            raise data_fault(
                                "float32 sum needs CPU non-finite accumulation"
                            )
                        floating[field] += Float64(sp[i]) - Float64(cp[i])
                elif bundle.fields[field].dtype == DataType.INT64:
                    integers[field] += (Int128(hp[i]) << 32) + Int128(lp[i])
                else:
                    integers[field] += Int128(hp[i])
            comptime if precise:
                if bundle.fields[field].dtype == DataType.FLOAT32:
                    floating[field] = finish_float_bins(
                        exact, nan, positive, negative
                    )
                    # Exact bins can expose small residuals that the CPU's
                    # Float64 accumulation loses after extreme cancellation.
                    # Prove the CPU additions exact when possible; otherwise
                    # let it adjudicate ill-conditioned data in its own order.
                    var growth = 0
                    var n = max(1, counts_out[field]) - 1
                    while n:
                        growth += 1
                        n >>= 1
                    if (
                        not nan
                        and not positive
                        and not negative
                        and least < 512
                        and greatest - least + 1 + growth > 53
                    ):
                        raise data_fault(
                            "float32 mean needs CPU accumulation order"
                        )
        timings.finish_ns = Int(perf_counter_ns()) - started
        return (integers^, floating^, counts_out^)


def _dispatch[
    precise: Bool, wide: Bool
](
    bundle: ReductionBundle,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[List[Int128], List[Float64], List[Int]]:
    if len(bundle.fields) == 1:
        return _run[precise, wide, 1](bundle, frame, scope, zero_copy, timings)
    if len(bundle.fields) == 2:
        return _run[precise, wide, 2](bundle, frame, scope, zero_copy, timings)
    if len(bundle.fields) == 3:
        return _run[precise, wide, 3](bundle, frame, scope, zero_copy, timings)
    return _run[precise, wide, 4](bundle, frame, scope, zero_copy, timings)


def execute_bundle(
    region: LazyFrame, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var matched = plan_bundle(region)
    if not matched:
        raise Error(
            "Metal unsupported [handoff]: reduction bundle was not accepted"
        )
    var bundle = matched.take()
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
    var result: Tuple[List[Int128], List[Float64], List[Int]]
    if bundle.precise_float:
        try:
            result = _dispatch[True, False](
                bundle, frame, scope, zero_copy, timings
            )
        except error:
            if String(error) != "Metal exact bins need wide scan":
                raise error^
            result = _dispatch[True, True](
                bundle, frame, scope, zero_copy, timings
            )
    else:
        result = _dispatch[False, False](
            bundle, frame, scope, zero_copy, timings
        )
    var columns = List[Series]()
    for output in bundle.outputs:
        var field = output.field
        var count = result[2][field]
        var dtype = bundle.fields[field].dtype
        if output.op == COUNT:
            columns.append(Series(output.name, Column[Int64]([Int64(count)])))
        elif output.op == MEAN:
            var total = (
                result[1][field] if dtype
                == DataType.FLOAT32 else result[0][field].cast[DType.float64]()
            )
            columns.append(
                Series(
                    output.name,
                    Column[Float64](
                        [total / Float64(count) if count else Float64(0)],
                        [count > 0],
                    ),
                )
            )
        elif dtype == DataType.FLOAT32:
            columns.append(
                Series(
                    output.name,
                    Column[Float32](
                        [Float32(result[1][field])], [count >= output.min_count]
                    ),
                )
            )
        else:
            var total = result[0][field]
            if dtype == DataType.INT32:
                if total > Int128(Int32.MAX) or total < Int128(Int32.MIN):
                    raise data_fault("int32 expression sum overflow")
                columns.append(
                    Series(
                        output.name,
                        Column[Int32](
                            [total.cast[DType.int32]()],
                            [count >= output.min_count],
                        ),
                    )
                )
            else:
                if total > Int128(Int64.MAX) or total < Int128(Int64.MIN):
                    raise data_fault("int64 expression sum overflow")
                columns.append(
                    Series(
                        output.name,
                        Column[Int64](
                            [total.cast[DType.int64]()],
                            [count >= output.min_count],
                        ),
                    )
                )
    scope.mark_success()
    return (DataFrame(columns^), timings^)
