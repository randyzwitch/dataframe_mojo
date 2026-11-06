"""Host ownership and submission for bounded GPU hash aggregation."""
from std.time import perf_counter_ns
from dataframe import DataFrame, DataType, LazyFrame
from .group_plan import GroupedComposition, plan_grouped
from .group_metadata import group_metadata
from .group_finish import finish_grouped
from .accelerator.faults import data_fault
from .accelerator.gpu.erased_input import (
    NumericView,
    numeric_views,
    single_native_span,
)
from .accelerator.gpu.typed_storage import NativeColumnAny, _NativeReservation
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.group_hash import GROUP_SLOTS, GROUP_LIMIT
from .accelerator.gpu.group_state import (
    GROUP_STATE_WORDS,
    initialize_group_states,
)
from .accelerator.gpu.group_reduce import (
    grouped_partials,
    resolve_int64_extrema,
)


@fieldwise_init
struct GroupedPartials(Movable):
    var first: List[UInt32]
    var rows: List[UInt32]
    var states: List[UInt32]
    var extrema: List[Int64]
    var blocks: Int


def _scan_grouped[
    arity: Int, floats: Int, narrow: Int
](
    plan: GroupedComposition,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> GroupedPartials:
    comptime if not METAL_TARGET:
        raise Error(
            "Metal grouped hand-off requires an Apple compilation target"
        )
    else:
        var started = Int(perf_counter_ns())
        var ctx = scope.context()
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var owners = List[NativeColumnAny]()
        var inputs = List[NumericView]()
        ref bundle = plan.selected.bundle
        var names = bundle.inputs.copy()
        if bundle.predicate.filter_column not in names:
            names.append(bundle.predicate.filter_column.copy())
        var predicate_input = -1
        for name in names:
            # Row-ticket extrema require random access to one logical buffer.
            # Adjacent shared-owner chunks map in place; independent chunks
            # use the existing explicit upload path with pinned ownership.
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
                raise Error("Metal grouped input did not form one native span")
            inputs.append(mapped[0].copy())
            if name == bundle.predicate.filter_column:
                predicate_input = len(inputs) - 1
        var key_input = -1
        for i in range(len(names)):
            if names[i] == plan.key:
                key_input = i
        var mapped_keys = List[NumericView]()
        if key_input >= 0:
            # The key may also be a value or predicate input. Reuse its
            # mapping, upload and byte reservation instead of charging twice.
            mapped_keys.append(inputs[key_input].copy())
        else:
            mapped_keys = numeric_views(
                frame,
                plan.key,
                scope,
                zero_copy and single_native_span(frame.column(plan.key)),
                empty,
                owners,
                coalesce=True,
            )
        if len(mapped_keys) != 1:
            raise Error("Metal grouped key did not form one native span")
        var keys = mapped_keys[0].copy()
        var predicate = inputs[predicate_input].copy()
        var blocks = min(128, max(1, (frame.height() + 255) // 256))
        var words = group_metadata(plan, inputs, keys, blocks)
        while len(inputs) < 8:
            inputs.append(
                NumericView(empty.copy(), empty.copy(), 0, 0, 0, False)
            )
        var entries = GROUP_SLOTS * blocks * arity
        var state_words = entries * GROUP_STATE_WORDS
        var count_words = GROUP_SLOTS * blocks
        # Reserve bounded scratch before allocating. No per-input-row group
        # IDs, gathered payload columns or CPU key-discovery table exist.
        var bytes = (
            (state_words + count_words + GROUP_SLOTS * 2 + 4) * 4
            + entries * 2 * 8
            + len(words) * 8
        )
        var reservation = _NativeReservation(scope, bytes)
        var table = ctx.enqueue_create_buffer[DType.uint32](GROUP_SLOTS)
        var control = ctx.enqueue_create_buffer[DType.uint32](4)
        var first = ctx.enqueue_create_buffer[DType.uint32](GROUP_SLOTS)
        var rows = ctx.enqueue_create_buffer[DType.uint32](count_words)
        var states = ctx.enqueue_create_buffer[DType.uint32](state_words)
        var extrema = ctx.enqueue_create_buffer[DType.int64](entries * 2)
        var metadata = ctx.enqueue_create_buffer[DType.uint64](len(words))
        var hp = List[UInt32](length=GROUP_SLOTS, fill=UInt32.MAX)
        var rp = List[UInt32](length=count_words, fill=0)
        var sp = List[UInt32](length=state_words, fill=0)
        var xp = List[Int64](length=entries * 2, fill=0)
        var cp = List[UInt32](length=4, fill=0)
        timings.upload_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        try:
            ctx.enqueue_copy(metadata, words.unsafe_ptr())
            ctx.enqueue_memset(table, UInt32(0))
            ctx.enqueue_memset(control, UInt32(0))
            ctx.enqueue_memset(first, UInt32.MAX)
            ctx.enqueue_memset(rows, UInt32(0))
            ctx.enqueue_memset(extrema, Int64(0))
            ctx.enqueue_function[initialize_group_states](
                states,
                UInt32(state_words),
                grid_dim=(state_words + 255) // 256,
                block_dim=256,
            )
            ctx.enqueue_function[grouped_partials[arity, floats, narrow]](
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
                predicate.values,
                predicate.bits,
                keys.values,
                keys.bits,
                metadata,
                table,
                control,
                rows,
                first,
                states,
                grid_dim=blocks,
                block_dim=256,
            )
            scope.record_launches(2)
            comptime if floats + narrow < arity:
                ctx.enqueue_function[
                    resolve_int64_extrema[arity, floats, narrow]
                ](
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
                    metadata,
                    states,
                    extrema,
                    control,
                    grid_dim=(entries + 255) // 256,
                    block_dim=256,
                )
                scope.record_launches(1)
            ctx.enqueue_copy(hp.unsafe_ptr(), first)
            ctx.enqueue_copy(rp.unsafe_ptr(), rows)
            ctx.enqueue_copy(sp.unsafe_ptr(), states)
            ctx.enqueue_copy(xp.unsafe_ptr(), extrema)
            ctx.enqueue_copy(cp.unsafe_ptr(), control)
            scope.synchronize()
        except error:
            try:
                scope.synchronize()
            except:
                pass
            _ = (
                words^,
                hp^,
                rp^,
                sp^,
                xp^,
                cp^,
                inputs^,
                keys^,
                predicate^,
                mapped_keys^,
                owners^,
                reservation^,
            )
            _ = (
                table^,
                control^,
                first^,
                rows^,
                states^,
                extrema^,
                metadata^,
                empty^,
            )
            raise error^
        _ = (
            table^,
            control^,
            first^,
            rows^,
            states^,
            extrema^,
            metadata^,
            empty^,
        )
        _ = (
            words^,
            inputs^,
            keys^,
            predicate^,
            mapped_keys^,
            owners^,
            reservation^,
        )
        if cp[1]:
            raise data_fault(
                "bounded grouped reduction exceeded its cardinality or checked-value limits"
            )
        if cp[0] > GROUP_LIMIT or cp[2] != UInt32(blocks):
            raise Error("Metal grouped kernel did not complete")
        comptime if floats + narrow < arity:
            if cp[3] != UInt32((entries + 255) // 256):
                raise Error("Metal grouped extrema resolver did not complete")
        timings.kernel_ns = Int(perf_counter_ns()) - started
        return GroupedPartials(hp^, rp^, sp^, xp^, blocks)


def execute_grouped(
    region: LazyFrame, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var matched = plan_grouped(region)
    if not matched:
        raise Error(
            "Metal unsupported [handoff]: grouped reductions were not accepted"
        )
    var plan = matched.take()
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    if frame.height() >= Int(Int32.MAX):
        raise Error("Metal unsupported [handoff]: 2**31 or more rows")
    var timings = HandoffTimings()
    if frame.height() == 0:
        return (
            finish_grouped(
                plan,
                frame,
                0,
                List[UInt32](),
                List[UInt32](),
                List[UInt32](),
                List[Int64](),
            ),
            timings^,
        )
    var scope = Scope(service=shared_service())
    var started = Int(perf_counter_ns())
    _ = scope.context()
    timings.context_ns = Int(perf_counter_ns()) - started
    var partials = _dispatch_grouped(plan, frame, scope, zero_copy, timings)
    started = Int(perf_counter_ns())
    var result = finish_grouped(
        plan,
        frame,
        partials.blocks,
        partials.first,
        partials.rows,
        partials.states,
        partials.extrema,
    )
    timings.finish_ns = Int(perf_counter_ns()) - started
    scope.mark_success()
    return (result^, timings^)


def _dispatch_grouped(
    plan: GroupedComposition,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> GroupedPartials:
    var floats = 0
    var narrow = 0
    for field in plan.selected.bundle.fields:
        floats += Int(field.dtype == DataType.FLOAT32)
        narrow += Int(field.dtype == DataType.INT32)
    comptime for arity in range(1, 5):
        comptime for nf in range(arity + 1):
            comptime for ni in range(arity - nf + 1):
                if (
                    len(plan.selected.bundle.fields) == arity
                    and floats == nf
                    and narrow == ni
                ):
                    return _scan_grouped[arity, nf, ni](
                        plan, frame, scope, zero_copy, timings
                    )
    raise Error("Metal grouped value arity was not dispatched")
