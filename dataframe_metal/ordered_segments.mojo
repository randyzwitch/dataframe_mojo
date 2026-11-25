"""Segmented ordered reductions over one Int32 key and one Int32 payload."""
from std.time import perf_counter_ns
from dataframe import DataFrame, DataType, LazyFrame
from .accelerator.faults import data_fault
from .accelerator.gpu.erased_input import numeric_views, single_native_span
from .accelerator.gpu.numeric_view import NumericView, intersect_numeric_views
from .accelerator.gpu.ordered_segments import ordered_segment
from .accelerator.gpu.query_scope import Scope
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.typed_storage import NativeColumnAny, _NativeReservation
from .group_layout import GENERAL_TABLE_LIMIT, GROUP_CAPACITY_FAULT
from .group_properties import sample_group_tuples
from .ordered_finish import finish_ordered
from .ordered_plan import OrderedPlan


def segmented_ordered_supported(
    region: LazyFrame, plan: OrderedPlan
) raises -> Bool:
    if not plan.keys or len(region._plan._nodes) != 2 or len(plan.inputs) != 1:
        return False
    ref keys = plan.keys.value()
    if (
        len(keys.names) != 1
        or keys.dtypes[0] != DataType.INT32
        or plan.fields[0].dtype != DataType.INT32
    ):
        return False
    ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
    if frame.height() == 0:
        return False
    for name in [keys.names[0], plan.inputs[0]]:
        var source = frame.column(name)
        if source.is_chunked() and not single_native_span(source):
            return True
    return False


def _finish_tickets(
    mut first: List[UInt32], mut state: List[UInt32], fields: Int
):
    for slot in range(len(first)):
        first[slot] = ~first[slot]
        if first[slot] == UInt32.MAX:
            continue
        for field in range(fields):
            var at = (slot * fields + field) * 8
            # The device stores early winners as complemented absolute rows,
            # so zeroed state is empty. finish_ordered expects row + 1.
            var earliest = state[at]
            state[at] = (~earliest + 1) if earliest else UInt32.MAX
            for op in range(2, 4):
                var winner = state[at + op]
                state[at + op] = (~winner + 1) if winner else UInt32(0)


def execute_segmented_ordered(
    plan: OrderedPlan,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> DataFrame:
    comptime if not METAL_TARGET:
        raise Error(
            "Metal ordered reductions require an Apple compilation target"
        )
    else:
        var started = Int(perf_counter_ns())
        var ctx = scope.context()
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var owners = List[NativeColumnAny]()
        var inputs = List[List[NumericView]]()
        ref key = plan.keys.value().names[0]
        inputs.append(
            numeric_views(
                frame, key, scope, zero_copy, empty, owners, coalesce=True
            )
        )
        var fields = len(plan.fields)
        var policies = UInt32(0)
        var ranks = False
        var requires_payload = False
        for i in range(fields):
            ref field = plan.fields[i]
            policies |= (
                field.operations | (UInt32(field.drop_nulls) << 4)
            ) << UInt32(8 * i)
            ranks = ranks or Bool(field.operations & 12)
            requires_payload = requires_payload or field.drop_nulls or ranks
        # Ordinary FIRST/LAST need only the key scan; finish_ordered gathers
        # the winning rows directly from the original value column.
        var value = 0
        if requires_payload and key != plan.inputs[0]:
            value = 1
            inputs.append(
                numeric_views(
                    frame,
                    plan.inputs[0],
                    scope,
                    zero_copy,
                    empty,
                    owners,
                    coalesce=True,
                )
            )
        var segments = intersect_numeric_views(inputs, frame.height())
        var capacity = sample_group_tuples(
            frame, plan.keys.value().names
        ).capacity()
        timings.upload_ns += Int(perf_counter_ns()) - started
        for attempt in range(2):
            if attempt:
                capacity = GENERAL_TABLE_LIMIT
            started = Int(perf_counter_ns())
            var state_words = capacity * fields * 8
            var reservation = _NativeReservation(
                scope, 4 * (capacity * 2 + state_words + 5)
            )
            var table = ctx.enqueue_create_buffer[DType.uint32](capacity)
            var first_gpu = ctx.enqueue_create_buffer[DType.uint32](capacity)
            var state_gpu = ctx.enqueue_create_buffer[DType.uint32](state_words)
            var control_gpu = ctx.enqueue_create_buffer[DType.uint32](5)
            var first = List[UInt32](length=capacity, fill=0)
            var state = List[UInt32](length=state_words, fill=0)
            var control = List[UInt32](length=5, fill=0)
            try:
                ctx.enqueue_memset(table, UInt32(0))
                ctx.enqueue_memset(first_gpu, UInt32(0))
                ctx.enqueue_memset(state_gpu, UInt32(0))
                ctx.enqueue_memset(control_gpu, UInt32(0))
                comptime for phase in range(3):
                    if phase == 0 or ranks:
                        var base = 0
                        for segment in segments:
                            ref k = segment[0]
                            ref v = segment[value]
                            ctx.enqueue_function[ordered_segment[phase]](
                                k.values,
                                k.bits,
                                v.values,
                                v.bits,
                                UInt32(k.bit_offset),
                                UInt32(v.bit_offset),
                                UInt32(k.nullable) | (UInt32(v.nullable) << 1),
                                UInt32(base),
                                UInt32(k.rows),
                                UInt32(capacity),
                                UInt32(fields),
                                policies,
                                table,
                                first_gpu,
                                state_gpu,
                                control_gpu,
                                grid_dim=(k.rows + 255) // 256,
                                block_dim=256,
                            )
                            scope.record_launches(1)
                            base += k.rows
                ctx.enqueue_copy(first.unsafe_ptr(), first_gpu)
                ctx.enqueue_copy(state.unsafe_ptr(), state_gpu)
                ctx.enqueue_copy(control.unsafe_ptr(), control_gpu)
                scope.synchronize()
                scope.record_download(4 * (capacity + state_words + 5))
            except error:
                try:
                    scope.synchronize()
                except:
                    pass
                _ = (first^, state^, control^, owners^, segments^, reservation^)
                raise error^
            _ = reservation^
            timings.kernel_ns += Int(perf_counter_ns()) - started
            if control[1] & GROUP_CAPACITY_FAULT:
                if capacity == GENERAL_TABLE_LIMIT:
                    raise data_fault("ordered reduction exceeded65536groups")
                continue
            var scans = UInt32(len(segments))
            var rank_scans = scans if ranks else UInt32(0)
            if (
                control[2] != scans
                or control[3] != rank_scans
                or control[4] != rank_scans
            ):
                raise Error("Metal segmented ordered kernels did not complete")
            _ = (owners^, segments^)
            started = Int(perf_counter_ns())
            _finish_tickets(first, state, fields)
            var result = finish_ordered(plan, frame, first, state)
            timings.finish_ns += Int(perf_counter_ns()) - started
            scope.mark_success()
            return result^
        raise Error("Metal ordered capacity retry was not completed")
