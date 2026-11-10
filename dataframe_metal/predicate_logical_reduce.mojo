"""Native ballot reductions with result-sized Kleene/count finishing."""
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer
from dataframe import Column, DataFrame, Series, LazyFrame
from dataframe.expr import ANY, ALL, LEN, COUNT, NULL_COUNT
from .predicate_logical_plan import PredicateLogicalPlan
from .value_program_metadata import predicate_words, write_predicate
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.reduction_inputs import HandoffTimings, _check_ran
from .accelerator.gpu.numeric_view import NumericView
from .accelerator.gpu.erased_input import numeric_views, validity_views
from .accelerator.gpu.typed_storage import NativeColumnAny, _NativeReservation
from .accelerator.gpu.logical_predicates import predicate_logical_partials
from .accelerator.gpu.filter_sum import BLOCK_SIZE


def finish_predicate_logical(
    plan: PredicateLogicalPlan,
    rows: Int64,
    true: List[Int64],
    false: List[Int64],
) raises -> DataFrame:
    var columns = List[Series]()
    for output in plan.outputs:
        if output.op == LEN:
            columns.append(Series(output.name, Column[Int64]([rows])))
            continue
        var t = true[output.field]
        var valid = t + false[output.field]
        if valid > rows:
            raise Error("Metal logical true/false masks overlap")
        if output.op == COUNT or output.op == NULL_COUNT:
            columns.append(
                Series(
                    output.name,
                    Column[Int64](
                        [valid if output.op == COUNT else rows - valid]
                    ),
                )
            )
        else:
            var answer = t > 0 if output.op == ANY else t == valid
            var decisive = answer if output.op == ANY else not answer
            var present = output.ignore_nulls or decisive or valid == rows
            columns.append(
                Series(output.name, Column[Bool]([answer], [present]))
            )
    return DataFrame(columns^)


def execute_predicate_logical(
    region: LazyFrame, plan: PredicateLogicalPlan, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    comptime if not METAL_TARGET:
        raise Error(
            "Metal predicate logical reduction requires an Apple target"
        )
    else:
        ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
        var fields = len(plan.positive)
        var nt = List[Int64](length=fields, fill=0)
        var nf = List[Int64](length=fields, fill=0)
        var timings = HandoffTimings()
        if frame.height() == 0:
            return (finish_predicate_logical(plan, 0, nt, nf), timings^)
        if frame.height() >= Int(Int32.MAX):
            raise Error(
                "Metal predicate logical reduction requires fewer than2**31 rows"
            )
        var started = Int(perf_counter_ns())
        var scope = Scope(service=shared_service())
        var ctx = scope.context()
        timings.context_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var owners = List[NativeColumnAny]()
        var sources = List[Series]()
        var inputs = List[List[NumericView]]()
        for i in range(len(plan.inputs)):
            sources.append(frame.column(plan.inputs[i]))
            if not plan.values[i] and not plan.validity[i]:
                inputs.append(
                    [
                        NumericView(
                            empty.copy(),
                            empty.copy(),
                            0,
                            frame.height(),
                            0,
                            False,
                        )
                    ]
                )
            elif zero_copy and not plan.values[i]:
                inputs.append(validity_views(sources[i], scope, empty))
            else:
                inputs.append(
                    numeric_views(
                        frame,
                        plan.inputs[i],
                        scope,
                        zero_copy,
                        empty,
                        owners,
                        coalesce=True,
                    )
                )
        var positions = List[Int](length=len(inputs), fill=0)
        var offsets = List[Int](length=len(inputs), fill=0)
        var segments = List[List[NumericView]]()
        var blocks = List[Int]()
        var consumed = 0
        var partial_words = 0
        var width = 1 + fields * 2
        while consumed < frame.height():
            var length = frame.height() - consumed
            for i in range(len(inputs)):
                if positions[i] >= len(inputs[i]):
                    raise Error("Metal logical inputs have different lengths")
                length = min(length, inputs[i][positions[i]].rows - offsets[i])
            if length <= 0:
                raise Error("Metal logical input window is empty")
            var segment = List[NumericView]()
            for i in range(len(inputs)):
                ref source = inputs[i][positions[i]]
                if plan.values[i]:
                    segment.append(source.view(offsets[i], length))
                else:
                    segment.append(
                        NumericView(
                            empty.copy(),
                            source.bits.copy(),
                            source.bit_offset
                            + offsets[i] if source.nullable else 0,
                            length,
                            source.tag,
                            source.nullable,
                        )
                    )
                offsets[i] += length
                if offsets[i] == source.rows:
                    positions[i] += 1
                    offsets[i] = 0
            var count = min(128, max(1, (length + 8191) // 8192))
            partial_words += count * width
            blocks.append(count)
            segments.append(segment^)
            consumed += length
        if partial_words > Int(UInt32.MAX):
            raise Error("Metal logical partial state exceeds its offset range")
        var metadata_words = 2 + fields * 2 + predicate_words(plan.selection)
        for field in range(fields):
            metadata_words += predicate_words(
                plan.positive[field]
            ) + predicate_words(plan.negative[field])
        var reservation = _NativeReservation(
            scope, partial_words * 4 + len(segments) * metadata_words * 8
        )
        var output = ctx.enqueue_create_buffer[DType.uint32](partial_words)
        var host = List[UInt32](length=partial_words, fill=UInt32.MAX)
        var metadata = List[DeviceBuffer[DType.uint64]]()
        var host_metadata = List[List[UInt64]]()
        var at = 0
        # Complete metadata and input placement before submitting any kernels.
        for segment in segments:
            var words = List[UInt64](length=metadata_words, fill=0)
            words[0] = UInt64(fields)
            var offset = 2 + fields * 2
            words[1] = UInt64(offset)
            write_predicate(words, offset, plan.selection, segment)
            offset += predicate_words(plan.selection)
            for field in range(fields):
                words[2 + field * 2] = UInt64(offset)
                write_predicate(words, offset, plan.positive[field], segment)
                offset += predicate_words(plan.positive[field])
                words[3 + field * 2] = UInt64(offset)
                write_predicate(words, offset, plan.negative[field], segment)
                offset += predicate_words(plan.negative[field])
            host_metadata.append(words^)
            metadata.append(
                ctx.enqueue_create_buffer[DType.uint64](metadata_words)
            )
        timings.upload_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        try:
            for index in range(len(segments)):
                var values = segments[index].copy()
                while len(values) < 9:
                    values.append(
                        NumericView(empty.copy(), empty.copy(), 0, 0, 0, False)
                    )
                ctx.enqueue_copy(
                    metadata[index], host_metadata[index].unsafe_ptr()
                )
                ctx.enqueue_function[predicate_logical_partials](
                    values[0].values,
                    values[1].values,
                    values[2].values,
                    values[3].values,
                    values[4].values,
                    values[5].values,
                    values[6].values,
                    values[7].values,
                    values[8].values,
                    values[0].bits,
                    values[1].bits,
                    values[2].bits,
                    values[3].bits,
                    values[4].bits,
                    values[5].bits,
                    values[6].bits,
                    values[7].bits,
                    values[8].bits,
                    metadata[index],
                    UInt32(segments[index][0].rows),
                    UInt32(blocks[index] * BLOCK_SIZE),
                    UInt32(at),
                    output,
                    grid_dim=blocks[index],
                    block_dim=BLOCK_SIZE,
                )
                scope.record_launches(1)
                at += blocks[index] * width
            ctx.enqueue_copy(host.unsafe_ptr(), output)
            scope.synchronize()
        except error:
            try:
                scope.synchronize()
            except:
                pass
            _ = (
                inputs^,
                segments^,
                sources^,
                owners^,
                metadata^,
                host_metadata^,
                host^,
                output^,
                empty^,
                reservation^,
            )
            raise error^
        _check_ran(host)
        scope.record_download(partial_words * 4)
        timings.kernel_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        var rows = Int64(0)
        for part in range(partial_words // width):
            rows += Int64(host[part * width])
            for field in range(fields):
                nt[field] += Int64(host[part * width + 1 + field * 2])
                nf[field] += Int64(host[part * width + 2 + field * 2])
        var result = finish_predicate_logical(plan, rows, nt, nf)
        scope.mark_success()
        timings.finish_ns = Int(perf_counter_ns()) - started
        _ = (
            inputs^,
            segments^,
            sources^,
            owners^,
            metadata^,
            host_metadata^,
            host^,
            output^,
            empty^,
            reservation^,
        )
        return (result^, timings^)


def merge_predicate_logical(
    plan: PredicateLogicalPlan, logical: DataFrame, residual: DataFrame
) raises -> DataFrame:
    """Restore projection order after result-sized numeric residual work."""
    if logical.height() != residual.height():
        raise Error(
            "Metal logical residual must produce the same global row count"
        )
    var columns = List[Series]()
    for name in plan.order:
        columns.append(
            logical.column(name) if name
            in logical.columns() else residual.column(name)
        )
    return DataFrame(columns^)
