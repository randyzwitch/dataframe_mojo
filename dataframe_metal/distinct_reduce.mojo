"""Native exact distinct GPU submission and bounded resource retries."""
from std.time import perf_counter_ns
from dataframe import DataFrame, LazyFrame
from .distinct_plan import plan_distinct
from .distinct_layout import (
    DistinctLayout,
    DISTINCT_CAPACITY_FAULT,
    MAX_DISTINCT_CAPACITY,
)
from .distinct_metadata import distinct_metadata
from .distinct_finish import finish_distinct
from .group_layout import GROUP_CAPACITY_FAULT
from .group_properties import sample_group_tuples
from .accelerator.faults import data_fault
from .accelerator.gpu.erased_input import (
    NumericView,
    numeric_views,
    single_native_span,
)
from .accelerator.gpu.typed_storage import NativeColumnAny, _NativeReservation
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.distinct_reduce import (
    initialize_distinct,
    distinct_partials,
)


def execute_distinct(
    region: LazyFrame, zero_copy: Bool = True
) raises -> Tuple[DataFrame, HandoffTimings]:
    comptime if not METAL_TARGET:
        raise Error("Metal exact distinct requires an Apple compilation target")
    else:
        var offered = plan_distinct(region)
        if not offered:
            raise Error(
                "Metal unsupported [handoff]: distinct region was not accepted"
            )
        var plan = offered.take()
        var frame = region._plan._frames[
            region._plan._nodes[0].source_slot()
        ].copy()
        var timings = HandoffTimings()
        if frame.height() == 0:
            return (
                finish_distinct(
                    plan, frame, List[UInt32](), List[UInt32](), List[UInt32]()
                ),
                timings^,
            )
        var started = Int(perf_counter_ns())
        var scope = Scope(service=shared_service())
        var ctx = scope.context()
        timings.context_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var owners = List[NativeColumnAny]()
        var inputs = List[NumericView]()
        var keys = List[NumericView]()
        for name in plan.inputs:
            var mapped = numeric_views(
                frame,
                name,
                scope,
                zero_copy and single_native_span(frame.column(name)),
                empty,
                owners,
                coalesce=True,
            )
            if len(mapped) != 1:
                raise Error("Metal distinct input did not form one native span")
            inputs.append(mapped[0].copy())
        var capacity = 512
        if plan.keys:
            for name in plan.keys.value().names:
                var existing = -1
                for i in range(len(plan.inputs)):
                    if plan.inputs[i] == name:
                        existing = i
                if existing >= 0:
                    keys.append(inputs[existing].copy())
                else:
                    var mapped = numeric_views(
                        frame,
                        name,
                        scope,
                        zero_copy and single_native_span(frame.column(name)),
                        empty,
                        owners,
                        coalesce=True,
                    )
                    if len(mapped) != 1:
                        raise Error(
                            "Metal distinct tuple key did not form one native span"
                        )
                    keys.append(mapped[0].copy())
            capacity = max(
                512,
                sample_group_tuples(frame, plan.keys.value().names).capacity(),
            )
        timings.upload_ns += Int(perf_counter_ns()) - started
        var blocks = min(128, max(1, (frame.height() + 255) // 256))
        for attempt in range(2):
            var unique_capacity = 16384
            if attempt:
                unique_capacity = MAX_DISTINCT_CAPACITY
                if plan.keys:
                    capacity = 131072
            var layout = DistinctLayout(
                capacity, len(plan.fields), unique_capacity
            )
            layout.check()
            var words = distinct_metadata(
                plan, inputs, keys, layout, frame.height(), blocks
            )
            var reservation = _NativeReservation(
                scope, layout.words() * 4 + len(words) * 8
            )
            var metadata = ctx.enqueue_create_buffer[DType.uint64](len(words))
            var scratch = ctx.enqueue_create_buffer[DType.uint32](
                layout.words()
            )
            var first = List[UInt32](length=capacity, fill=UInt32.MAX)
            var counts = List[UInt32](
                length=capacity * len(plan.fields), fill=0
            )
            var nulls = List[UInt32](length=len(counts), fill=0)
            var control = List[UInt32](length=4 + len(plan.fields), fill=0)
            var original_inputs = len(inputs)
            var original_keys = len(keys)
            while len(inputs) < 9:
                inputs.append(
                    NumericView(empty.copy(), empty.copy(), 0, 0, 0, False)
                )
            while len(keys) < 4:
                keys.append(
                    NumericView(empty.copy(), empty.copy(), 0, 0, 0, False)
                )
            started = Int(perf_counter_ns())
            try:
                ctx.enqueue_copy(metadata, words.unsafe_ptr())
                ctx.enqueue_function[initialize_distinct](
                    metadata,
                    scratch,
                    grid_dim=(layout.words() + 255) // 256,
                    block_dim=256,
                )
                ctx.enqueue_function[distinct_partials](
                    inputs[0].values,
                    inputs[1].values,
                    inputs[2].values,
                    inputs[3].values,
                    inputs[4].values,
                    inputs[5].values,
                    inputs[6].values,
                    inputs[7].values,
                    inputs[0].bits,
                    inputs[1].bits,
                    inputs[2].bits,
                    inputs[3].bits,
                    inputs[4].bits,
                    inputs[5].bits,
                    inputs[6].bits,
                    inputs[7].bits,
                    inputs[8].values,
                    inputs[8].bits,
                    keys[0].values,
                    keys[1].values,
                    keys[2].values,
                    keys[3].values,
                    keys[0].bits,
                    keys[1].bits,
                    keys[2].bits,
                    keys[3].bits,
                    metadata,
                    scratch,
                    grid_dim=blocks,
                    block_dim=256,
                )
                scope.record_launches(2)
                ctx.enqueue_copy(
                    control.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](0, len(control)),
                )
                ctx.enqueue_copy(
                    first.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](
                        layout.first_offset(), capacity
                    ),
                )
                ctx.enqueue_copy(
                    counts.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](
                        layout.counts_offset(), len(counts)
                    ),
                )
                ctx.enqueue_copy(
                    nulls.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](
                        layout.nulls_offset(), len(nulls)
                    ),
                )
                scope.synchronize()
            except error:
                try:
                    scope.synchronize()
                except:
                    pass
                raise error^
            timings.kernel_ns += Int(perf_counter_ns()) - started
            while len(inputs) > original_inputs:
                _ = inputs.pop()
            while len(keys) > original_keys:
                _ = keys.pop()
            if control[1] & (GROUP_CAPACITY_FAULT | DISTINCT_CAPACITY_FAULT):
                if attempt:
                    raise data_fault(
                        "exact distinct exceeded bounded group/value table capacity"
                    )
                continue
            if control[1] or control[2] != UInt32(blocks):
                raise Error("Metal exact distinct kernel did not complete")
            started = Int(perf_counter_ns())
            var result = finish_distinct(plan, frame, first, counts, nulls)
            timings.finish_ns += Int(perf_counter_ns()) - started
            scope.mark_success()
            return (result^, timings^)
        raise Error("Metal exact distinct capacity retry was not completed")
