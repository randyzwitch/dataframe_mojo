"""Mapped-input ownership and packed scratch for general grouped scans."""
from std.time import perf_counter_ns
from dataframe import DataFrame, DataType
from dataframe.expr import MIN, MAX
from .composition_plan import (
    Composition,
    needs_general_values,
    simple_conditional_trees,
    has_extrema,
)
from .group_properties import sample_group_tuples
from .group_types import group_value_types
from .group_plan import GroupedComposition
from .group_partials import GroupedPartials
from .group_layout import GroupLayout, GROUP_CAPACITY_RETRY, GROUP_MOMENT_WORDS
from .group_metadata import general_group_metadata
from .accelerator.faults import data_fault
from .accelerator.gpu.erased_input import (
    NumericView,
    numeric_views,
    single_native_span,
)
from .accelerator.gpu.typed_storage import NativeColumnAny, _NativeReservation
from .accelerator.gpu.query_scope import Scope
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.group_state import GROUP_STATE_WORDS
from .accelerator.gpu.group_cached import cached_grouped_partials
from .accelerator.gpu.group_tuple_hash import GROUP_CAPACITY_FAULT
from .accelerator.gpu.group_general import (
    initialize_general_groups,
    general_grouped_partials,
    resolve_general_int64,
)


def scan_general_groups[
    cases: Bool = False,
    simple_cases: Bool = False,
    global_state: Bool = True,
    value_types: Int = 7,
    moments: Bool = False,
](
    plan: GroupedComposition,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
    capacity: Int,
    moment_mask: UInt32 = 0,
    cached: Bool = False,
) raises -> GroupedPartials:
    comptime if not METAL_TARGET:
        raise Error(
            "Metal grouped hand-off requires an Apple compilation target"
        )
    else:
        comptime if moments:
            if (
                moment_mask == 0
                or (moment_mask >> UInt32(len(plan.selected.bundle.fields)))
                != 0
            ):
                raise Error(
                    "Metal grouped moment mask does not match its schema"
                )
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
        var keys = List[NumericView]()
        for name in plan.keys.names:
            var key_input = -1
            for i in range(len(names)):
                if names[i] == name:
                    key_input = i
            if key_input >= 0:
                keys.append(inputs[key_input].copy())
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
                    raise Error("Metal tuple key did not form one native span")
                keys.append(mapped[0].copy())
        var predicate = inputs[predicate_input].copy()
        var blocks = min(128, max(1, (frame.height() + 255) // 256))
        var layout = GroupLayout(
            capacity,
            len(bundle.fields),
            blocks,
            global_state,
            GROUP_MOMENT_WORDS if moments else GROUP_STATE_WORDS,
        )
        layout.check()
        var words = general_group_metadata(plan, inputs, keys, layout)
        comptime if moments:
            words[9] |= UInt64(moment_mask) << 24
        var needs_resolver = False
        for output in bundle.outputs:
            if output.op == MIN or output.op == MAX:
                needs_resolver = (
                    needs_resolver
                    or bundle.fields[output.field].dtype == DataType.INT64
                )
        while len(keys) < 4:
            keys.append(NumericView(empty.copy(), empty.copy(), 0, 0, 0, False))
        while len(inputs) < 8:
            inputs.append(
                NumericView(empty.copy(), empty.copy(), 0, 0, 0, False)
            )
        var entries = layout.entries()
        var state_words = entries * layout.value_words
        var count_words = capacity * layout.state_blocks()
        var resolved_values = entries * 2 if needs_resolver else 0
        var reservation = _NativeReservation(
            scope,
            layout.words() * 4 + max(1, resolved_values) * 8 + len(words) * 8,
        )
        var scratch = ctx.enqueue_create_buffer[DType.uint32](layout.words())
        var extrema = ctx.enqueue_create_buffer[DType.int64](
            max(1, resolved_values)
        )
        var metadata = ctx.enqueue_create_buffer[DType.uint64](len(words))
        var hp = List[UInt32](length=capacity, fill=UInt32.MAX)
        var rp = List[UInt32](length=count_words, fill=0)
        var sp = List[UInt32](length=state_words, fill=0)
        var xp = List[Int64](length=resolved_values, fill=0)
        var cp = List[UInt32](length=4, fill=0)
        timings.upload_ns += Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        try:
            ctx.enqueue_copy(metadata, words.unsafe_ptr())
            if needs_resolver:
                ctx.enqueue_memset(extrema, Int64(0))
            ctx.enqueue_function[
                initialize_general_groups[global_state, moments]
            ](
                metadata,
                scratch,
                grid_dim=(layout.words() + 255) // 256,
                block_dim=256,
            )
            # Int64 extrema reevaluate winning expressions under a ticket lock.
            # Compile those paths only when a moment query requests MIN/MAX.
            # Ordinary grouped kernels retain their existing specialization.
            comptime for mode in range(2):
                comptime WITH_EXTREMA = mode == 0
                if (
                    moments
                    and has_extrema(plan.selected.bundle) == WITH_EXTREMA
                ) or (not moments and cached == Bool(mode)):
                    comptime kernel = cached_grouped_partials[
                        cases, simple_cases, global_state
                    ] if not moments and mode == 1 else general_grouped_partials[
                        cases,
                        simple_cases,
                        global_state,
                        value_types,
                        moments,
                        WITH_EXTREMA,
                    ]
                    ctx.enqueue_function[kernel](
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
            if needs_resolver:
                ctx.enqueue_function[
                    resolve_general_int64[
                        cases, simple_cases, global_state, moments
                    ]
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
                    predicate.values,
                    predicate.bits,
                    metadata,
                    scratch,
                    extrema,
                    scratch,
                    grid_dim=(entries + 255) // 256,
                    block_dim=256,
                )
                scope.record_launches(1)
            ctx.enqueue_copy(
                hp.unsafe_ptr(),
                scratch.create_sub_buffer[DType.uint32](
                    layout.first_offset(), capacity
                ),
            )
            ctx.enqueue_copy(
                rp.unsafe_ptr(),
                scratch.create_sub_buffer[DType.uint32](
                    layout.rows_offset(), count_words
                ),
            )
            ctx.enqueue_copy(
                sp.unsafe_ptr(),
                scratch.create_sub_buffer[DType.uint32](
                    layout.states_offset(), state_words
                ),
            )
            if needs_resolver:
                ctx.enqueue_copy(xp.unsafe_ptr(), extrema)
            ctx.enqueue_copy(
                cp.unsafe_ptr(), scratch.create_sub_buffer[DType.uint32](0, 4)
            )
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
                owners^,
                reservation^,
            )
            _ = (
                scratch^,
                extrema^,
                metadata^,
                empty^,
            )
            raise error^
        _ = (
            scratch^,
            extrema^,
            metadata^,
            empty^,
        )
        _ = (
            words^,
            inputs^,
            keys^,
            predicate^,
            owners^,
            reservation^,
        )
        timings.kernel_ns += Int(perf_counter_ns()) - started
        if cp[1] & ~GROUP_CAPACITY_FAULT:
            raise data_fault(
                "general grouped reduction exceeded checked-value limits"
            )
        if cp[1] & GROUP_CAPACITY_FAULT:
            raise Error(GROUP_CAPACITY_RETRY)
        if cp[0] > UInt32(capacity // 2) or cp[2] != UInt32(blocks):
            raise Error("Metal general grouped kernel did not complete")
        if needs_resolver and cp[3] != UInt32((entries + 255) // 256):
            raise Error(
                "Metal general grouped extrema resolver did not complete"
            )
        return GroupedPartials(hp^, rp^, sp^, xp^, layout.state_blocks())


def general_case_mode(bundle: Composition) -> Int:
    """0 ordinary,1 general CASE,2 host-proven register-only CASE."""
    if simple_conditional_trees(bundle):
        return 2
    return Int(needs_general_values(bundle))


def _dispatch_general[
    global_state: Bool, moments: Bool = False
](
    plan: GroupedComposition,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
    capacity: Int,
    moment_mask: UInt32 = 0,
    cached: Bool = False,
) raises -> GroupedPartials:
    var dtypes = List[DataType]()
    for field in plan.selected.bundle.fields:
        dtypes.append(field.dtype)
    var selected = group_value_types(dtypes)
    var case_mode = general_case_mode(plan.selected.bundle)
    comptime for mask in range(1, 8):
        if selected == mask:
            if case_mode == 2:
                return scan_general_groups[
                    True, True, global_state, mask, moments
                ](
                    plan,
                    frame,
                    scope,
                    zero_copy,
                    timings,
                    capacity,
                    moment_mask,
                    cached,
                )
            if case_mode == 1:
                return scan_general_groups[
                    True, False, global_state, mask, moments
                ](
                    plan,
                    frame,
                    scope,
                    zero_copy,
                    timings,
                    capacity,
                    moment_mask,
                    cached,
                )
            return scan_general_groups[
                False, False, global_state, mask, moments
            ](
                plan,
                frame,
                scope,
                zero_copy,
                timings,
                capacity,
                moment_mask,
                cached,
            )
    raise Error("Metal grouped value dtype mask was not dispatched")


def general_partials[
    moments: Bool = False
](
    plan: GroupedComposition,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
    force_global: Bool = False,
    moment_mask: UInt32 = 0,
    cached: Bool = False,
) raises -> GroupedPartials:
    var sample = sample_group_tuples(frame, plan.keys.names)
    var capacity = sample.capacity()
    try:
        if force_global or sample.global_state():
            return _dispatch_general[True, moments](
                plan,
                frame,
                scope,
                zero_copy,
                timings,
                max(1024, capacity),
                moment_mask,
                cached,
            )
        return _dispatch_general[False, moments](
            plan, frame, scope, zero_copy, timings, 512, moment_mask, cached
        )
    except error:
        if String(error) != GROUP_CAPACITY_RETRY:
            raise error^
        if (force_global or sample.global_state()) and max(
            1024, capacity
        ) == 131072:
            raise data_fault("grouped reduction exceeded 65536 groups")
    # Sampling describes cost, not coverage. Capacity overflow always retries
    # the complete query on GPU with the full supported table.
    try:
        return _dispatch_general[True, moments](
            plan, frame, scope, zero_copy, timings, 131072, moment_mask, cached
        )
    except error:
        if String(error) != GROUP_CAPACITY_RETRY:
            raise error^
        raise data_fault("grouped reduction exceeded 65536 groups")
