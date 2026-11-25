"""Submit dictionary key and value chunks into one bounded table and decode its sums."""
from std.memory import ArcPointer, bitcast
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, Series
from dataframe.execution_observation import DeviceObservation
from .accelerator.faults import data_fault
from .accelerator.gpu.dictionary_pair import PAIR_WORDS, dictionary_pair_sum
from .accelerator.gpu.erased_input import numeric_views
from .accelerator.gpu.numeric_view import NumericView, intersect_numeric_views
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.typed_storage import NativeColumnAny, _NativeReservation
from .group_layout import GENERAL_GROUP_LIMIT
from .group_output import GroupOutput, group_order
from .multi_plan import ReductionOutput


def _finish_dictionary[
    single_key: Bool, sparse: Bool
](
    frame: DataFrame,
    output: ReductionOutput,
    left_labels: Int,
    right_labels: Int,
    slots: Int,
    state: List[UInt32],
) raises -> DataFrame:
    var first = List[UInt32](capacity=slots)
    for slot in range(slots):
        first.append(~state[slot * PAIR_WORDS])
    var left = List[Int32]()
    var right = List[Int32]()
    var left_valid = List[Bool]()
    var right_valid = List[Bool]()
    var sums = GroupOutput(
        output, frame._columns[frame.width() - 1].dtype(), False
    )
    for slot in group_order(first):
        var a = slot // (right_labels + 1)
        var b = slot % (right_labels + 1)
        comptime if sparse:
            var code = state[slots * PAIR_WORDS + slot - 1] if slot else UInt32(
                0
            )
            var ca = Int(code >> 16)
            var cb = Int(code & 0xFFFF)
            a = ca - 1 if ca else left_labels
            b = cb - 1 if cb else right_labels
        left.append(Int32(a) if a < left_labels else Int32(0))
        comptime if not single_key:
            right.append(Int32(b) if b < right_labels else Int32(0))
            right_valid.append(b < right_labels)
        left_valid.append(a < left_labels)
        var offset = slot * PAIR_WORDS
        var low = UInt64(state[offset + 2]) | (UInt64(state[offset + 3]) << 32)
        var sum = (
            Int128(bitcast[DType.int32](state[offset + 4])) << 64
        ) + Int128(low)
        sums.append(sum, 0, Int(state[offset + 1]), 0, 0, 0, 0, 0)
    var columns = List[Series]()
    columns.append(
        Series(frame._columns[0].name(), Column[Int32](left^, left_valid^))
    )
    comptime if not single_key:
        columns.append(
            Series(
                frame._columns[1].name(), Column[Int32](right^, right_valid^)
            )
        )
    columns.append(sums^.finish())
    return DataFrame(columns^)


def execute_dictionary_sum[
    single_key: Bool, sparse: Bool = False
](
    frame: DataFrame,
    output: ReductionOutput,
    observation: Optional[ArcPointer[DeviceObservation]],
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
            observation=observation,
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
        comptime if sparse:
            var capacity = 2
            while capacity < min(frame.height() * 2, GENERAL_GROUP_LIMIT * 2):
                capacity *= 2
            slots = capacity + 1
        var words = slots * PAIR_WORDS + (slots - 1 + 4 if sparse else 2)
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
                ref b = segment[0 if single_key else 1]
                ref v = segment[1 if single_key else 2]
                var flags = (
                    UInt32(a.nullable)
                    | (UInt32(b.nullable) << 1)
                    | (UInt32(v.nullable) << 2)
                )
                ctx.enqueue_function[dictionary_pair_sum[single_key, sparse]](
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
                    UInt32(slots),
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
        var control_at = words - (4 if sparse else 2)
        comptime if sparse:
            if (
                host[control_at + 3]
                or Int(host[control_at + 2]) + Int(host[0] != 0)
                > GENERAL_GROUP_LIMIT
            ):
                raise data_fault("grouped reduction exceeded 65536 groups")
        if host[control_at + 1]:
            raise data_fault("dictionary code outside its key domain")
        if Int(host[control_at]) != launches:
            raise Error("Metal dictionary scan did not complete")
        started = Int(perf_counter_ns())
        var result = _finish_dictionary[single_key, sparse](
            frame, output, left_labels, right_labels, slots, host
        )
        timings.finish_ns = Int(perf_counter_ns()) - started
        scope.mark_success()
        return (result^, timings^)
