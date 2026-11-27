"""Submit dictionary key and value chunks into one bounded table and decode its sums."""
from max.gpu.host import DeviceBuffer
from std.ffi import external_call
from std.memory import ArcPointer, Pointer, bitcast
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, Series
from dataframe.execution_observation import DeviceObservation
from .accelerator.faults import data_fault
from .accelerator.gpu.dictionary_pair import (
    PAIR_WORDS,
    dictionary_pair_sum,
    dictionary_fused,
)
from .accelerator.gpu.erased_input import numeric_views
from .accelerator.gpu.numeric_view import NumericView, intersect_numeric_views
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.shared_pages import SharedPages
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.typed_storage import NativeColumnAny, _NativeReservation
from .dictionary_projection import DictionaryProjection
from .group_layout import GENERAL_GROUP_LIMIT
from .group_output import GroupOutput, group_order
from .multi_plan import ReductionOutput


@always_inline
def _read_sum(state: Span[UInt32, ImmutAnyOrigin], at: Int) -> Int128:
    var low = UInt64(state[at]) | (UInt64(state[at + 1]) << 32)
    return (Int128(bitcast[DType.int32](state[at + 2])) << 64) + Int128(low)


def _descriptor(view: NumericView, flags: UInt64) -> UInt64:
    return (
        (UInt64(view.bit_offset) << 32)
        | (UInt64(view.nullable) << 2)
        | UInt64(view.tag)
        | flags
    )


def _finish_dictionary[
    single_key: Bool, sparse: Bool, fused: Bool
](
    frame: DataFrame,
    output: ReductionOutput,
    left_labels: Int,
    right_labels: Int,
    slots: Int,
    state: Span[UInt32, ImmutAnyOrigin],
    stride: Int,
    projection: Optional[DictionaryProjection],
) raises -> DataFrame:
    var first = List[UInt32](capacity=slots)
    for slot in range(slots):
        first.append(~state[slot * stride])
    var left = List[Int32]()
    var right = List[Int32]()
    var left_valid = List[Bool]()
    var right_valid = List[Bool]()
    var sums = GroupOutput(
        output, frame._columns[frame.width() - 1].dtype(), False
    )
    var fields = List[GroupOutput]()
    comptime if fused:
        for item in projection.value().outputs:
            var dtype = DataType.INT64
            if item.field >= 0:
                dtype = frame._columns[
                    (1 if single_key else 2) + item.field
                ].dtype()
            fields.append(GroupOutput(item, dtype, item.field == -1))
    for slot in group_order(first):
        var a = slot // (right_labels + 1)
        var b = slot % (right_labels + 1)
        comptime if sparse:
            var code = state[slots * stride + slot - 1] if slot else UInt32(0)
            var ca = Int(code >> 16)
            var cb = Int(code & 0xFFFF)
            a = ca - 1 if ca else left_labels
            b = cb - 1 if cb else right_labels
        left.append(Int32(a) if a < left_labels else Int32(0))
        comptime if not single_key:
            right.append(Int32(b) if b < right_labels else Int32(0))
            right_valid.append(b < right_labels)
        left_valid.append(a < left_labels)
        var offset = slot * stride
        comptime if fused:
            ref plan = projection.value()
            var rows = Int(state[offset + 1]) if plan.needs_rows else 0
            for i in range(len(fields)):
                ref item = plan.outputs[i]
                var count = rows
                var total = Int128(0)
                if item.field >= 0:
                    var at = offset + plan.offsets[item.field]
                    count = Int(state[at]) if plan.needs_count[
                        item.field
                    ] else 0
                    if plan.needs_sum[item.field]:
                        total = _read_sum(
                            state, at + Int(plan.needs_count[item.field])
                        )
                elif item.field == -2 and a == left_labels:
                    count = 0
                elif item.field == -3 and b == right_labels:
                    count = 0
                fields[i].append(total, 0, count, 0, 0, 0, 0, rows)
        else:
            sums.append(
                _read_sum(state, offset + 2),
                0,
                Int(state[offset + 1]),
                0,
                0,
                0,
                0,
                0,
            )
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
    comptime if fused:
        while fields:
            columns.append(fields.pop(0).finish())
    else:
        columns.append(sums^.finish())
    return DataFrame(columns^)


def execute_dictionary_sum[
    single_key: Bool, sparse: Bool = False, fused: Bool = False
](
    frame: DataFrame,
    output: ReductionOutput,
    observation: Optional[ArcPointer[DeviceObservation]],
    left_labels: Int,
    right_labels: Int,
    zero_copy: Bool,
    projection: Optional[DictionaryProjection] = None,
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
        var stride = PAIR_WORDS
        var bindings = List[Int]()
        var field_flags = List[UInt64]()
        comptime if fused:
            ref plan = projection.value()
            stride = plan.stride
            for field in range(len(plan.offsets)):
                bindings.append((1 if single_key else 2) + field)
                field_flags.append(
                    (UInt64(plan.offsets[field]) << 8)
                    | (UInt64(plan.needs_sum[field]) << 3)
                    | (UInt64(plan.needs_count[field]) << 4)
                )
            while len(bindings) < 4:
                bindings.append(0)
                field_flags.append(0)
        var words = slots * stride + (slots - 1 + 4 if sparse else 2)
        var shared_output = (sparse or fused) and zero_copy
        var size = words * 4
        var page = 1
        if shared_output:
            page = Int(external_call["getpagesize", Int32]())
            size = ((size + page - 1) // page) * page
        var reservation = _NativeReservation(scope, size)
        var pages = SharedPages(size if shared_output else 0, page)
        var host = List[UInt32]()
        var address = pages.address
        var state: DeviceBuffer[DType.uint32]
        if shared_output:
            state = ctx.wrap_host_memory[DType.uint32](
                Pointer[UInt32, MutAnyOrigin](unsafe_from_address=address),
                size // 4,
            )
        else:
            state = ctx.enqueue_create_buffer[DType.uint32](words)
            host = List[UInt32](length=words, fill=0)
            address = Int(host.unsafe_ptr())
        timings.upload_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        try:
            ctx.enqueue_memset(state, UInt32(0))
            var base = 0
            for segment in segments:
                ref a = segment[0]
                ref b = segment[0 if single_key else 1]
                var key_flags = UInt32(a.nullable) | (UInt32(b.nullable) << 1)
                comptime if fused:
                    ref v0 = segment[bindings[0]]
                    ref v1 = segment[bindings[1]]
                    ref v2 = segment[bindings[2]]
                    ref v3 = segment[bindings[3]]
                    ctx.enqueue_function[dictionary_fused[single_key, sparse]](
                        a.values,
                        b.values,
                        a.bits,
                        b.bits,
                        v0.values,
                        v1.values,
                        v2.values,
                        v3.values,
                        v0.bits,
                        v1.bits,
                        v2.bits,
                        v3.bits,
                        _descriptor(v0, field_flags[0]),
                        _descriptor(v1, field_flags[1]),
                        _descriptor(v2, field_flags[2]),
                        _descriptor(v3, field_flags[3]),
                        UInt32(a.bit_offset),
                        UInt32(b.bit_offset),
                        key_flags
                        | (UInt32(projection.value().needs_rows) << 2),
                        UInt32(left_labels),
                        UInt32(right_labels),
                        UInt32(base),
                        UInt32(a.rows),
                        UInt32(slots),
                        UInt32(stride),
                        state,
                        grid_dim=(a.rows + 255) // 256,
                        block_dim=256,
                    )
                else:
                    ref v = segment[1 if single_key else 2]
                    var flags = key_flags | (UInt32(v.nullable) << 2)
                    ctx.enqueue_function[
                        dictionary_pair_sum[single_key, sparse]
                    ](
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
                base += a.rows
            if not shared_output:
                ctx.enqueue_copy(host.unsafe_ptr(), state)
            scope.synchronize()
            if not shared_output:
                scope.record_download(words * 4)
        except error:
            try:
                scope.synchronize()
            except:
                pass
            _ = (state^, segments^, owners^, pages^, host^, reservation^)
            raise error^
        _ = (state^, segments^, owners^)
        timings.kernel_ns = Int(perf_counter_ns()) - started
        var result_words = Span[UInt32, ImmutAnyOrigin](
            unsafe_ptr=Pointer[UInt32, ImmutAnyOrigin](
                unsafe_from_address=address
            ),
            length=words,
        )
        var control_at = words - (4 if sparse else 2)
        comptime if sparse:
            if (
                result_words[control_at + 3]
                or Int(result_words[control_at + 2]) + Int(result_words[0] != 0)
                > GENERAL_GROUP_LIMIT
            ):
                raise data_fault("grouped reduction exceeded 65536 groups")
        if result_words[control_at + 1]:
            raise data_fault("dictionary code outside its key domain")
        if Int(result_words[control_at]) != launches:
            raise Error("Metal dictionary scan did not complete")
        started = Int(perf_counter_ns())
        var result = _finish_dictionary[single_key, sparse, fused](
            frame,
            output,
            left_labels,
            right_labels,
            slots,
            result_words,
            stride,
            projection,
        )
        timings.finish_ns = Int(perf_counter_ns()) - started
        _ = (host^, pages^, reservation^)
        scope.mark_success()
        return (result^, timings^)
