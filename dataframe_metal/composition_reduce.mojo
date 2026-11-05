"""Host orchestration for composed arithmetic and column aggregates."""
from std.time import perf_counter_ns
from std.memory import bitcast
from std.math import isfinite
from dataframe import Column, DataFrame, DataType, LazyFrame, Series
from dataframe.expr import COUNT, MEAN, SUM
from .composition_plan import (
    Composition,
    plan_composition,
    single_float_composition,
)
from .exact_float import finish_float_bins
from .accelerator.faults import data_fault
from .accelerator.gpu.erased_input import (
    NumericView,
    numeric_views,
    single_native_span,
)
from .accelerator.gpu.typed_storage import NativeColumnAny
from .accelerator.gpu.composed_reduce import (
    composed_partials,
    META_WORDS,
    HEADER_WORDS,
    FIELD_WORDS,
)
from .accelerator.gpu.float_composed_reduce import float_composed_partials
from .accelerator.gpu.filter_sum import BLOCK_SIZE
from .accelerator.gpu.reduction_inputs import (
    HandoffTimings,
    _chunk_partials,
    _check_ran,
)
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.target import METAL_TARGET


def _run[
    precise: Bool,
    wide: Bool,
    arity: Int,
    float_fields: Int,
    fast: Bool,
    int32_fields: Int,
](
    bundle: Composition,
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
        var names = bundle.inputs.copy()
        if bundle.predicate.filter_column not in names:
            names.append(bundle.predicate.filter_column.copy())
        var batched = zero_copy
        var fragmented = False
        if batched:
            for name in names:
                var source = frame.column(name)
                fragmented = fragmented or source.is_chunked()
                if not single_native_span(source):
                    batched = False
                    break
        batched = batched and fragmented
        var predicate_input = -1
        for name in names:
            inputs.append(
                numeric_views(
                    frame,
                    name,
                    scope,
                    zero_copy,
                    empty,
                    owners,
                    coalesce=batched,
                )
            )
            if name == bundle.predicate.filter_column:
                predicate_input = len(inputs) - 1
        var lengths = List[List[Int]]()
        for i in range(len(inputs)):
            var sizes = List[Int]()
            if batched:
                for part in frame.column(names[i]).chunks():
                    if len(part):
                        sizes.append(len(part))
            else:
                for part in inputs[i]:
                    sizes.append(part.rows)
            lengths.append(sizes^)
        var positions = List[Int](length=len(inputs), fill=0)
        var offsets = List[Int](length=len(inputs), fill=0)
        var segments = List[List[NumericView]]()
        var starts = List[Int]()
        var consumed = 0
        while consumed < frame.height():
            var length = frame.height() - consumed
            for i in range(len(inputs)):
                if positions[i] >= len(lengths[i]):
                    raise Error(
                        "Metal reduction columns have different lengths"
                    )
                length = min(length, lengths[i][positions[i]] - offsets[i])
            var segment = List[NumericView]()
            for i in range(len(inputs)):
                if batched:
                    var value = inputs[i][0].copy()
                    value.rows = length
                    segment.append(value^)
                else:
                    segment.append(
                        inputs[i][positions[i]].view(offsets[i], length)
                    )
                offsets[i] += length
                if offsets[i] == lengths[i][positions[i]]:
                    positions[i] += 1
                    offsets[i] = 0
            starts.append(consumed)
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
        var faults = ctx.enqueue_create_buffer[DType.uint32](total_parts)
        var fp = List[UInt32](length=total_parts, fill=0)
        # One immutable metadata block per intersected chunk segment. Its host
        # source and device root stay alive until the final queue drain.
        var prefix = 2 if batched else 0
        var words = List[UInt64](
            length=prefix + len(segments) * META_WORDS, fill=0
        )
        if batched:
            words[0] = UInt64(len(segments))
        var output_offset = 0
        for index in range(len(segments)):
            ref segment = segments[index]
            var at = prefix + index * META_WORDS
            var predicate = segment[predicate_input].copy()
            var partials = _chunk_partials(segment[0].rows, frame.height())
            words[at] = UInt64(segment[0].rows)
            words[at + 1] = UInt64(partials * BLOCK_SIZE)
            words[at + 2] = UInt64(output_offset)
            words[at + 4] = UInt64(predicate.tag)
            words[at + 5] = UInt64(predicate.bit_offset)
            words[at + 6] = UInt64(predicate.nullable)
            words[at + 7] = UInt64(bundle.predicate.filtered)
            words[at + 8] = UInt64(bundle.predicate.inclusive)
            words[at + 9] = bitcast[DType.uint64](bundle.predicate.low)
            words[at + 10] = bitcast[DType.uint64](bundle.predicate.high)
            words[at + 11] = UInt64(
                bitcast[DType.uint32](bundle.predicate.low_float)
            )
            words[at + 12] = UInt64(
                bitcast[DType.uint32](bundle.predicate.high_float)
            )
            words[at + 13] = UInt64(starts[index]) if batched else UInt64(
                len(bundle.inputs)
            )
            words[at + 15] = UInt64(
                predicate_input + 1 if predicate_input
                < len(bundle.inputs) else 0
            )
            for field in range(nf):
                ref value = bundle.fields[field]
                var tag = 2 if value.dtype == DataType.FLOAT32 else (
                    1 if value.dtype == DataType.INT64 else 0
                )
                words[at + 3] |= UInt64(tag) << UInt64(2 * field)
                if value.needs_sum:
                    words[at + 14] |= UInt64(1) << UInt64(field)
                var base = at + HEADER_WORDS + field * FIELD_WORDS
                words[base] = UInt64(value.op)
                words[base + 1] = UInt64(value.left_input + 1)
                words[base + 2] = UInt64(value.right_input + 1)
                if value.left_input >= 0:
                    ref left = segment[value.left_input]
                    words[base + 3] |= UInt64(left.nullable)
                    words[base + 4] = UInt64(left.bit_offset)
                if value.right_input >= 0:
                    ref right = segment[value.right_input]
                    words[base + 3] |= UInt64(right.nullable) << 1
                    words[base + 5] = UInt64(right.bit_offset)
                words[base + 6] = UInt64(value.literal_side)
                words[base + 7] = UInt64(
                    bitcast[DType.uint32](value.floating)
                ) if tag == 2 else bitcast[DType.uint64](value.integer)
            output_offset += partials
        var metadata = ctx.enqueue_create_buffer[DType.uint64](len(words))
        try:
            ctx.enqueue_copy(metadata, words.unsafe_ptr())
            ctx.enqueue_memset(counts, UInt32.MAX)
            ctx.enqueue_memset(faults, UInt32.MAX)
            comptime if precise:
                ctx.enqueue_memset(bins, Int64(0))
                ctx.enqueue_memset(specials, UInt32(0))
            for index in range(1 if batched else len(segments)):
                ref segment = segments[index]
                var values = List[NumericView]()
                for i in range(8):
                    if i < len(bundle.inputs):
                        values.append(segment[i].copy())
                    else:
                        values.append(
                            NumericView(
                                empty.copy(), empty.copy(), 0, 0, 0, False
                            )
                        )
                var predicate = segment[predicate_input].copy()
                var partials = total_parts if batched else _chunk_partials(
                    segment[0].rows, frame.height()
                )
                comptime kernel = float_composed_partials[
                    wide, arity, float_fields
                ] if fast else composed_partials[
                    precise, wide, arity, float_fields, int32_fields
                ]
                ctx.enqueue_function[kernel](
                    values[0].values,
                    values[1].values,
                    values[2].values,
                    values[3].values,
                    values[4].values,
                    values[5].values,
                    values[6].values,
                    values[7].values,
                    values[0].bits,
                    values[1].bits,
                    values[2].bits,
                    values[3].bits,
                    values[4].bits,
                    values[5].bits,
                    values[6].bits,
                    values[7].bits,
                    predicate.values,
                    predicate.bits,
                    metadata,
                    UInt32.MAX if batched else UInt32(index * META_WORDS),
                    high,
                    low,
                    sums,
                    compensation,
                    counts,
                    bins,
                    specials,
                    faults,
                    grid_dim=partials,
                    block_dim=BLOCK_SIZE,
                )
                scope.record_launches(1)
            ctx.enqueue_copy(fp.unsafe_ptr(), faults)
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
            _ = (hp^, lp^, sp^, cp^, np^, bp^, xp^, fp^)
            _ = (high^, low^, sums^, compensation^, counts^, bins^, specials^)
            _ = (segments^, empty^, words^, metadata^, faults^)
            _ = owners^
            raise error^
        # Explicit last use keeps typed reservations alive through the drain.
        _ = (high^, low^, sums^, compensation^, counts^, bins^, specials^)
        _ = (segments^, empty^, words^, metadata^, faults^)
        _ = owners^
        _check_ran(np)
        _check_ran(fp)
        for fault in fp:
            if fault & 1:
                raise data_fault("integer arithmetic expression overflow")
            if fault & 2:
                raise data_fault(
                    "float32 arithmetic needs CPU gradual underflow"
                )
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
                        if not fast or field < float_fields:
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
                            floating[field] += _partial_float(sp[i], cp[i])
                    else:
                        floating[field] += _partial_float(sp[i], cp[i])
                elif bundle.fields[field].dtype == DataType.INT64:
                    integers[field] += (Int128(hp[i]) << 32) + Int128(lp[i])
                else:
                    integers[field] += Int128(hp[i])
            comptime if precise:
                if bundle.fields[field].dtype == DataType.FLOAT32 and (
                    not fast or field < float_fields
                ):
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


def _partial_float(s: Float32, c: Float32) raises -> Float64:
    if not isfinite(s) or not isfinite(c):
        raise data_fault("float32 sum needs CPU non-finite accumulation")
    return Float64(s) - Float64(c)


def _float_arity[
    wide: Bool, arity: Int
](
    bundle: Composition,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[List[Int128], List[Float64], List[Int]]:
    var exact_fields = 1 if bundle.precise_float else 0
    if bundle.precise_float:
        for i in range(1, len(bundle.fields)):
            exact_fields += Int(bundle.fields[i].needs_mean)
    comptime for n in range(arity + 1):
        if exact_fields == n:
            return _run[n > 0, wide, arity, n, True, 0](
                bundle, frame, scope, zero_copy, timings
            )
    raise Error("Metal compact float prefix exceeds its field count")


def _by_arity[
    precise: Bool, wide: Bool, arity: Int
](
    bundle: Composition,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[List[Int128], List[Float64], List[Int]]:
    var floats = 0
    var int32s = 0
    for field in bundle.fields:
        floats += Int(field.dtype == DataType.FLOAT32)
        int32s += Int(field.dtype == DataType.INT32)
    comptime for n in range(arity + 1):
        comptime if not precise or n > 0:
            comptime for narrow in range(arity - n + 1):
                if floats == n and int32s == narrow:
                    return _run[precise, wide, arity, n, False, narrow](
                        bundle, frame, scope, zero_copy, timings
                    )
    raise Error("Metal composed field types exceed their arity")


def _dispatch[
    precise: Bool, wide: Bool
](
    bundle: Composition,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[List[Int128], List[Float64], List[Int]]:
    if single_float_composition(bundle):
        if len(bundle.fields) == 1:
            return _float_arity[wide, 1](
                bundle, frame, scope, zero_copy, timings
            )
        if len(bundle.fields) == 2:
            return _float_arity[wide, 2](
                bundle, frame, scope, zero_copy, timings
            )
        if len(bundle.fields) == 3:
            return _float_arity[wide, 3](
                bundle, frame, scope, zero_copy, timings
            )
        return _float_arity[wide, 4](bundle, frame, scope, zero_copy, timings)
    if len(bundle.fields) == 1:
        return _by_arity[precise, wide, 1](
            bundle, frame, scope, zero_copy, timings
        )
    if len(bundle.fields) == 2:
        return _by_arity[precise, wide, 2](
            bundle, frame, scope, zero_copy, timings
        )
    if len(bundle.fields) == 3:
        return _by_arity[precise, wide, 3](
            bundle, frame, scope, zero_copy, timings
        )
    return _by_arity[precise, wide, 4](bundle, frame, scope, zero_copy, timings)


def execute_composition(
    region: LazyFrame, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var matched = plan_composition(region)
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
