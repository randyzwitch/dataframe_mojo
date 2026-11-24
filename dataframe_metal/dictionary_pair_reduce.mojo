"""Submit dictionary-pair chunks into one bounded table and decode its sums."""
from std.memory import bitcast
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, LazyFrame, Series
from dataframe.expr import SUM
from .accelerator.faults import data_fault
from .accelerator.gpu.dictionary_pair import PAIR_WORDS, dictionary_pair_sum
from .accelerator.gpu.erased_input import numeric_views
from .accelerator.gpu.numeric_view import NumericView, intersect_numeric_views
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.typed_storage import NativeColumnAny, _NativeReservation
from .group_output import GroupOutput, group_order
from .multi_plan import ReductionOutput


def _finish_pair(
    frame: DataFrame,
    output: ReductionOutput,
    left_labels: Int,
    right_labels: Int,
    state: List[UInt32],
) raises -> DataFrame:
    var slots = (left_labels + 1) * (right_labels + 1)
    var first = List[UInt32](capacity=slots)
    for slot in range(slots):
        first.append(~state[slot * PAIR_WORDS])
    var left = List[Int32]()
    var right = List[Int32]()
    var left_valid = List[Bool]()
    var right_valid = List[Bool]()
    var sums = GroupOutput(output, frame._columns[2].dtype(), False)
    for slot in group_order(first):
        var a = slot // (right_labels + 1)
        var b = slot % (right_labels + 1)
        left.append(Int32(a) if a < left_labels else Int32(0))
        right.append(Int32(b) if b < right_labels else Int32(0))
        left_valid.append(a < left_labels)
        right_valid.append(b < right_labels)
        var offset = slot * PAIR_WORDS
        var low = UInt64(state[offset + 2]) | (UInt64(state[offset + 3]) << 32)
        var sum = (
            Int128(bitcast[DType.int32](state[offset + 4])) << 64
        ) + Int128(low)
        sums.append(sum, 0, Int(state[offset + 1]), 0, 0, 0, 0, 0)
    return DataFrame(
        [
            Series(frame._columns[0].name(), Column[Int32](left^, left_valid^)),
            Series(
                frame._columns[1].name(), Column[Int32](right^, right_valid^)
            ),
            sums^.finish(),
        ]
    )


def execute_dictionary_pair(
    region: LazyFrame,
    frame: DataFrame,
    left_labels: Int,
    right_labels: Int,
    zero_copy: Bool,
) raises -> Tuple[DataFrame, HandoffTimings]:
    comptime if not METAL_TARGET:
        raise Error(
            "Metal dictionary grouping requires an Apple compilation target"
        )
    else:
        var timings = HandoffTimings()
        var started = Int(perf_counter_ns())
        var scope = Scope(
            service=shared_service(),
            observation=region._plan._device_observation,
        )
        var ctx = scope.context()
        timings.context_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var owners = List[NativeColumnAny]()
        var inputs = List[List[NumericView]]()
        for name in frame.columns():
            inputs.append(
                numeric_views(
                    frame, name, scope, zero_copy, empty, owners, coalesce=True
                )
            )
        var segments = intersect_numeric_views(inputs, frame.height())
        var launches = len(segments)
        var slots = (left_labels + 1) * (right_labels + 1)
        var words = slots * PAIR_WORDS + 2
        var reservation = _NativeReservation(scope, words * 4)
        var state = ctx.enqueue_create_buffer[DType.uint32](words)
        var host = List[UInt32](length=words, fill=0)
        timings.upload_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        try:
            ctx.enqueue_memset(state, UInt32(0))
            var base = 0
            for segment in segments:
                ref a = segment[0]
                ref b = segment[1]
                ref v = segment[2]
                var flags = (
                    UInt32(a.nullable)
                    | (UInt32(b.nullable) << 1)
                    | (UInt32(v.nullable) << 2)
                )
                ctx.enqueue_function[dictionary_pair_sum](
                    a.values,
                    b.values,
                    v.values,
                    a.bits,
                    b.bits,
                    v.bits,
                    UInt32(a.bit_offset),
                    UInt32(b.bit_offset),
                    UInt32(v.bit_offset),
                    flags,
                    UInt32(left_labels),
                    UInt32(right_labels),
                    UInt32(v.tag),
                    UInt32(base),
                    UInt32(v.rows),
                    state,
                    grid_dim=(v.rows + 255) // 256,
                    block_dim=256,
                )
                scope.record_launches(1)
                base += v.rows
            ctx.enqueue_copy(host.unsafe_ptr(), state)
            scope.synchronize()
            scope.record_download(words * 4)
        except error:
            try:
                scope.synchronize()
            except:
                pass
            _ = (host^, state^, segments^, owners^, reservation^)
            raise error^
        _ = (state^, segments^, owners^, reservation^)
        timings.kernel_ns = Int(perf_counter_ns()) - started
        if host[words - 1]:
            raise data_fault("dictionary code outside its key domain")
        if Int(host[words - 2]) != launches:
            raise Error("Metal dictionary pair scan did not complete")
        started = Int(perf_counter_ns())
        ref expr = region._plan._nodes[1].exprs[0]
        var output = ReductionOutput(
            0, expr._name.copy(), SUM, expr._nodes[1].min_count
        )
        var result = _finish_pair(
            frame, output, left_labels, right_labels, host
        )
        timings.finish_ns = Int(perf_counter_ns()) - started
        scope.mark_success()
        return (result^, timings^)
