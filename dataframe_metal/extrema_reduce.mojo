"""Shared native scan for MIN/MAX with checked owners and queue lifetime."""
from std.time import perf_counter_ns
from std.memory import bitcast
from dataframe import Column, DataFrame, DataType, Series, LazyFrame
from dataframe.expr import MIN
from .extrema_plan import ExtremaBundle, plan_extrema
from .accelerator.faults import data_fault
from .accelerator.gpu.erased_input import NumericView, numeric_views
from .accelerator.gpu.typed_storage import NativeColumnAny
from .accelerator.gpu.extrema_reduce import extrema_partials, EXTREMA_WORDS
from .accelerator.gpu.filter_sum import BLOCK_SIZE
from .accelerator.gpu.reduction_inputs import (
    HandoffTimings,
    _chunk_partials,
    _check_ran,
)
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.target import METAL_TARGET


def _run[
    arity: Int
](
    bundle: ExtremaBundle,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[List[Int64], List[Int64], List[Int], List[UInt32]]:
    var minima = List[Int64](length=arity, fill=Int64.MAX)
    var maxima = List[Int64](length=arity, fill=Int64.MIN)
    var counts_out = List[Int](length=arity, fill=0)
    var specials_out = List[UInt32](length=arity, fill=0)
    if frame.height() == 0:
        return (minima^, maxima^, counts_out^, specials_out^)
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        var started = Int(perf_counter_ns())
        var ctx = scope.context()
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var owners = List[NativeColumnAny]()
        var inputs = List[List[NumericView]]()
        var predicate_input = -1
        for field in bundle.fields:
            inputs.append(
                numeric_views(
                    frame, field.column, scope, zero_copy, empty, owners
                )
            )
            if field.column == bundle.predicate.filter_column:
                predicate_input = len(inputs) - 1
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
        var size = total_parts * arity
        var lows = ctx.enqueue_create_buffer[DType.int64](size)
        var highs = ctx.enqueue_create_buffer[DType.int64](size)
        var counts = ctx.enqueue_create_buffer[DType.uint32](size)
        var specials = ctx.enqueue_create_buffer[DType.uint32](size)
        var lp = List[Int64](length=size, fill=0)
        var hp = List[Int64](length=size, fill=0)
        var np = List[UInt32](length=size, fill=0)
        var xp = List[UInt32](length=size, fill=0)
        var words = List[UInt64](length=len(segments) * EXTREMA_WORDS, fill=0)
        var output_offset = 0
        for index in range(len(segments)):
            ref segment = segments[index]
            var at = index * EXTREMA_WORDS
            var predicate = segment[predicate_input].copy()
            var partials = _chunk_partials(segment[0].rows, frame.height())
            words[at] = UInt64(segment[0].rows)
            words[at + 1] = UInt64(partials * BLOCK_SIZE)
            words[at + 2] = UInt64(output_offset)
            words[at + 3] = UInt64(bundle.predicate.filtered)
            words[at + 4] = UInt64(predicate.tag)
            words[at + 5] = UInt64(predicate.bit_offset)
            words[at + 6] = UInt64(predicate.nullable)
            words[at + 7] = UInt64(bundle.predicate.inclusive)
            words[at + 8] = bitcast[DType.uint64](bundle.predicate.low)
            words[at + 9] = bitcast[DType.uint64](bundle.predicate.high)
            words[at + 10] = UInt64(
                bitcast[DType.uint32](bundle.predicate.low_float)
            )
            words[at + 11] = UInt64(
                bitcast[DType.uint32](bundle.predicate.high_float)
            )
            for field in range(arity):
                var base = at + 16 + field * 3
                words[base] = UInt64(segment[field].tag)
                words[base + 1] = UInt64(segment[field].bit_offset)
                words[base + 2] = UInt64(segment[field].nullable)
            output_offset += partials
        var metadata = ctx.enqueue_create_buffer[DType.uint64](len(words))
        try:
            ctx.enqueue_copy(metadata, words.unsafe_ptr())
            ctx.enqueue_memset(counts, UInt32.MAX)
            for index in range(len(segments)):
                ref segment = segments[index]
                var values = List[NumericView]()
                for field in range(4):
                    if field < arity:
                        values.append(segment[field].copy())
                    else:
                        values.append(
                            NumericView(
                                empty.copy(), empty.copy(), 0, 0, 0, False
                            )
                        )
                var predicate = segment[predicate_input].copy()
                ctx.enqueue_function[extrema_partials[arity]](
                    values[0].values,
                    values[1].values,
                    values[2].values,
                    values[3].values,
                    values[0].bits,
                    values[1].bits,
                    values[2].bits,
                    values[3].bits,
                    predicate.values,
                    predicate.bits,
                    metadata,
                    UInt32(index * EXTREMA_WORDS),
                    lows,
                    highs,
                    counts,
                    specials,
                    grid_dim=_chunk_partials(segment[0].rows, frame.height()),
                    block_dim=BLOCK_SIZE,
                )
                scope.record_launches(1)
            ctx.enqueue_copy(lp.unsafe_ptr(), lows)
            ctx.enqueue_copy(hp.unsafe_ptr(), highs)
            ctx.enqueue_copy(np.unsafe_ptr(), counts)
            ctx.enqueue_copy(xp.unsafe_ptr(), specials)
            scope.synchronize()
        except error:
            try:
                scope.synchronize()
            except:
                pass
            _ = (lp^, hp^, np^, xp^, lows^, highs^, counts^, specials^)
            _ = (segments^, empty^, words^, metadata^, owners^)
            raise error^
        _ = (lows^, highs^, counts^, specials^)
        _ = (segments^, empty^, words^, metadata^, owners^)
        _check_ran(np)
        timings.kernel_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        for part in range(total_parts):
            for field in range(arity):
                var at = part * arity + field
                if np[at]:
                    minima[field] = min(minima[field], lp[at])
                    maxima[field] = max(maxima[field], hp[at])
                counts_out[field] += Int(np[at])
                specials_out[field] |= xp[at]
        timings.finish_ns = Int(perf_counter_ns()) - started
        return (minima^, maxima^, counts_out^, specials_out^)


def extrema_column(
    name: String, dtype: DataType, value: Int64, valid: Bool, flags: UInt32
) raises -> Series:
    if dtype == DataType.FLOAT32:
        if not valid and flags & 1:
            raise data_fault(
                "all-NaN Float32 extrema require CPU payload adjudication"
            )
        var raw = UInt32(0)
        if valid:
            var key = UInt32(value)
            raw = key ^ 0x8000_0000 if key >> 31 else ~key
            if raw & 0x7FFF_FFFF == 0:
                if flags & 6 == 6:
                    raise data_fault(
                        "mixed signed-zero extrema require CPU tie adjudication"
                    )
                if flags & 4:
                    raw = 0x8000_0000
        return Series(
            name, Column[Float32]([bitcast[DType.float32](raw)], [valid])
        )
    if dtype == DataType.INT32:
        return Series(
            name,
            Column[Int32](
                [value.cast[DType.int32]() if valid else Int32(0)], [valid]
            ),
        )
    return Series(name, Column[Int64]([value if valid else Int64(0)], [valid]))


def execute_extrema(
    region: LazyFrame, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var matched = plan_extrema(region)
    if not matched:
        raise Error("Metal unsupported [handoff]: not native extrema")
    var bundle = matched.take()
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    if frame.height() >= Int(Int32.MAX):
        raise Error("Metal unsupported [handoff]: 2**31 or more rows")
    var timings = HandoffTimings()
    var started = Int(perf_counter_ns())
    var scope = Scope(service=shared_service())
    _ = scope.context()
    timings.context_ns = Int(perf_counter_ns()) - started
    var values: Tuple[List[Int64], List[Int64], List[Int], List[UInt32]]
    if len(bundle.fields) == 1:
        values = _run[1](bundle, frame, scope, zero_copy, timings)
    elif len(bundle.fields) == 2:
        values = _run[2](bundle, frame, scope, zero_copy, timings)
    elif len(bundle.fields) == 3:
        values = _run[3](bundle, frame, scope, zero_copy, timings)
    else:
        values = _run[4](bundle, frame, scope, zero_copy, timings)
    var outputs = List[Series]()
    for output in bundle.outputs:
        var field = output.field
        var valid = values[2][field] != 0
        var value = values[0][field] if output.op == MIN else values[1][field]
        var dtype = bundle.fields[field].dtype
        outputs.append(
            extrema_column(
                output.name.copy(), dtype, value, valid, values[3][field]
            )
        )
    scope.mark_success()
    return (DataFrame(outputs^), timings^)
