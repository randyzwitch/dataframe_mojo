"""Native ordered GPU scans, bounded retries and result-sized finishing."""
from std.time import perf_counter_ns
from dataframe import DataFrame, LazyFrame
from .ordered_plan import plan_ordered
from .ordered_metadata import ordered_metadata, ordered_scratch_words
from .ordered_finish import finish_ordered
from .group_layout import GroupLayout, GROUP_CAPACITY_FAULT
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
from .accelerator.gpu.ordered_reduce import (
    initialize_ordered,
    ordered_partials,
    merge_ordered,
    ordered_ranks,
)


def execute_ordered(
    region: LazyFrame, zero_copy: Bool = True
) raises -> Tuple[DataFrame, HandoffTimings]:
    comptime if not METAL_TARGET:
        raise Error(
            "Metal ordered reductions require an Apple compilation target"
        )
    else:
        var matched = plan_ordered(region)
        if not matched:
            raise Error(
                "Metal unsupported [handoff]: ordered reductions were not accepted"
            )
        var plan = matched.take()
        var frame = region._plan._frames[
            region._plan._nodes[0].source_slot()
        ].copy()
        var timings = HandoffTimings()
        if frame.height() == 0:
            return (
                finish_ordered(plan, frame, List[UInt32](), List[UInt32]()),
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
                raise Error("Metal ordered input did not form one native span")
            inputs.append(mapped[0].copy())
        var capacity = 512
        var global_state = not Bool(plan.keys)
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
                            "Metal ordered tuple key did not form one native span"
                        )
                    keys.append(mapped[0].copy())
            var properties = sample_group_tuples(frame, plan.keys.value().names)
            global_state = properties.global_state()
            capacity = max(1024, properties.capacity()) if global_state else 512
        timings.upload_ns += Int(perf_counter_ns()) - started
        var blocks = min(128, max(1, (frame.height() + 255) // 256))
        var need_rank = False
        for field in plan.fields:
            need_rank = need_rank or Bool(field.operations & 12)
        for attempt in range(2):
            if attempt:
                capacity = 131072
                global_state = True
            var layout = GroupLayout(
                capacity, len(plan.fields), blocks, global_state
            )
            var words = ordered_metadata(
                plan, inputs, keys, layout, frame.height()
            )
            var scratch_words = ordered_scratch_words(layout)
            var reservation = _NativeReservation(
                scope, scratch_words * 4 + len(words) * 8
            )
            var metadata = ctx.enqueue_create_buffer[DType.uint64](len(words))
            var scratch = ctx.enqueue_create_buffer[DType.uint32](scratch_words)
            var first = List[UInt32](length=capacity, fill=UInt32.MAX)
            var final = List[UInt32](
                length=capacity * len(plan.fields) * 8, fill=0
            )
            var control = List[UInt32](length=4, fill=0)
            var rank_launches = List[UInt32](length=1, fill=0)
            # Pad only after descriptor packing, preserving its input checks.
            var unpadded_inputs = len(inputs)
            var unpadded_keys = len(keys)
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
                ctx.enqueue_function[initialize_ordered](
                    metadata,
                    scratch,
                    grid_dim=(scratch_words + 255) // 256,
                    block_dim=256,
                )
                ctx.enqueue_function[ordered_partials](
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
                ctx.enqueue_function[merge_ordered](
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
                    grid_dim=(capacity * len(plan.fields) + 255) // 256,
                    block_dim=256,
                )
                if need_rank:
                    ctx.enqueue_function[ordered_ranks](
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
                    ctx.enqueue_function[merge_ordered](
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
                        grid_dim=(capacity * len(plan.fields) + 255) // 256,
                        block_dim=256,
                    )
                scope.record_launches(5 if need_rank else 3)
                ctx.enqueue_copy(
                    control.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](0, 4),
                )
                ctx.enqueue_copy(
                    first.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](
                        layout.first_offset(), capacity
                    ),
                )
                ctx.enqueue_copy(
                    final.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](
                        layout.words(), len(final)
                    ),
                )
                ctx.enqueue_copy(
                    rank_launches.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](
                        layout.rows_offset(), 1
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
            while len(inputs) > unpadded_inputs:
                _ = inputs.pop()
            while len(keys) > unpadded_keys:
                _ = keys.pop()
            if control[1] & GROUP_CAPACITY_FAULT:
                if capacity == 131072 or attempt:
                    raise data_fault("ordered reduction exceeded65536groups")
                continue
            var merge_blocks = UInt32(
                (capacity * len(plan.fields) + 255) // 256
            )
            if (
                control[2] != UInt32(blocks)
                or control[3] != merge_blocks * UInt32(2 if need_rank else 1)
                or rank_launches[0] != UInt32(blocks if need_rank else 0)
            ):
                raise Error("Metal ordered kernels did not complete")
            started = Int(perf_counter_ns())
            var result = finish_ordered(plan, frame, first, final)
            timings.finish_ns += Int(perf_counter_ns()) - started
            scope.mark_success()
            return (result^, timings^)
        raise Error("Metal ordered capacity retry was not completed")
