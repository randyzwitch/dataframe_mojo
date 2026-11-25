"""Submit unfiltered Int32-key aggregates over native numeric spans."""
from max.gpu.host import DeviceBuffer
from std.ffi import external_call
from std.memory import unsafe_memcpy
from std.time import perf_counter_ns
from dataframe import DataFrame, DataType, LazyFrame
from dataframe.expr import COL, MIN, MAX
from .accelerator.faults import data_fault
from .accelerator.gpu.chunked_numeric import (
    FIELD_EXTREMA,
    FIELD_NULLABLE,
    FIELD_SUM,
    chunked_numeric_partials,
)
from .accelerator.gpu.erased_input import numeric_views, single_native_span
from .accelerator.gpu.group_hash import HASH_CAPACITY
from .accelerator.gpu.group_state import initialize_group_states
from .accelerator.gpu.numeric_view import NumericView, intersect_numeric_views
from .accelerator.gpu.query_scope import Scope
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.typed_storage import NativeColumnAny, _NativeReservation
from .group_layout import (
    GENERAL_TABLE_LIMIT,
    GROUP_CAPACITY_FAULT,
    GROUP_CAPACITY_RETRY,
    GROUP_CONTROL_WORDS,
    GroupLayout,
)
from .group_partials import GroupedPartials
from .group_plan import GroupedComposition
from .group_properties import sample_group_tuples


comptime NUMERIC_DEFAULT = 0
comptime NUMERIC_SEGMENTED = 1
comptime NUMERIC_SINGLE_SPAN = 2


def native_numeric_layout(
    region: LazyFrame, plan: GroupedComposition, frame: DataFrame
) raises -> Int:
    if (
        len(region._plan._nodes) != 2
        or len(plan.keys.names) != 1
        or plan.key_dtype != DataType.INT32
        or frame.height() == 0
        or frame.height() >= Int(Int32.MAX)
    ):
        return NUMERIC_DEFAULT
    ref fields = plan.selected.bundle.fields
    for field in fields:
        if field.op != COL or field.dtype not in [
            DataType.INT32,
            DataType.INT64,
            DataType.FLOAT32,
        ]:
            return NUMERIC_DEFAULT
    for output in plan.outputs:
        if (
            output.op in [MIN, MAX]
            and fields[output.field].dtype == DataType.INT64
        ):
            # The existing resolver can reread Int64 winners from a single span.
            return NUMERIC_DEFAULT
    var key = frame.column(plan.key)
    # The composition planner merges aggregates over the same direct column.
    var multifield_span = len(fields) > 1
    if (multifield_span or key.is_chunked()) and not single_native_span(key):
        if key.is_chunked():
            return NUMERIC_SEGMENTED
        multifield_span = False
    for field in fields:
        var source = frame.column(field.left)
        if (multifield_span or source.is_chunked()) and not single_native_span(
            source
        ):
            if source.is_chunked():
                return NUMERIC_SEGMENTED
            multifield_span = False
    return NUMERIC_SINGLE_SPAN if multifield_span else NUMERIC_DEFAULT


def _descriptor(view: NumericView, flags: UInt64) -> UInt64:
    return (
        (UInt64(view.bit_offset) << 32)
        | UInt64(view.tag)
        | (FIELD_NULLABLE if view.nullable else UInt64(0))
        | flags
    )


def _read_words(storage: List[UInt32], start: Int, count: Int) -> List[UInt32]:
    # The caller has synchronized all device writes to this owned allocation.
    var result = List[UInt32](unsafe_uninit_length=count)
    unsafe_memcpy(
        dest=result.unsafe_ptr().unsafe_bitcast[UInt8](),
        src=storage.unsafe_ptr().unsafe_offset(start).unsafe_bitcast[UInt8](),
        count=count * 4,
    )
    return result^


def _scan(
    segments: List[List[NumericView]],
    bindings: List[Int],
    flags: List[UInt64],
    layout: GroupLayout,
    scope: Scope,
    shared_output: Bool,
    mut timings: HandoffTimings,
) raises -> GroupedPartials:
    comptime if not METAL_TARGET:
        raise Error("Metal chunked numeric groups require an Apple GPU")
    else:
        var started = Int(perf_counter_ns())
        layout.check()
        var ctx = scope.context()
        var page_words = (
            Int(external_call["getpagesize", Int32]())
            // 4 if shared_output else 1
        )
        var mapped_words = (
            (layout.words() + page_words - 1) // page_words
        ) * page_words
        var reservation = _NativeReservation(scope, mapped_words * 4)
        var backing = List[UInt32]()
        var offset = 0
        var scratch: DeviceBuffer[DType.uint32]
        if shared_output:
            # Metal requires a page-aligned start and a page-multiple byte size.
            # Reserve enough padding that the entire mapping is inside the List.
            backing = List[UInt32](
                unsafe_uninit_length=mapped_words + page_words - 1
            )
            var address_words = Int(backing.unsafe_ptr()) // 4
            offset = (page_words - address_words % page_words) % page_words
            scratch = ctx.wrap_host_memory[DType.uint32](
                backing.unsafe_ptr()
                .unsafe_offset(offset)
                .unsafe_origin_cast[MutAnyOrigin](),
                mapped_words,
            )
        else:
            scratch = ctx.enqueue_create_buffer[DType.uint32](layout.words())
        var state_words = layout.entries() * layout.value_words
        var states = scratch.create_sub_buffer[DType.uint32](
            layout.states_offset(), state_words
        )
        var first = scratch.create_sub_buffer[DType.uint32](
            layout.first_offset(), layout.capacity
        )
        var host_first = List[UInt32]()
        var host_rows = List[UInt32]()
        var host_states = List[UInt32]()
        if not shared_output:
            host_first = List[UInt32](length=layout.capacity, fill=UInt32.MAX)
            host_rows = List[UInt32](length=layout.capacity, fill=0)
            host_states = List[UInt32](length=state_words, fill=0)
        var control = List[UInt32](length=GROUP_CONTROL_WORDS, fill=0)
        timings.upload_ns += Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        try:
            ctx.enqueue_memset(
                scratch.create_sub_buffer[DType.uint32](
                    0, layout.states_offset()
                ),
                UInt32(0),
            )
            ctx.enqueue_memset(first, UInt32.MAX)
            ctx.enqueue_function[initialize_group_states](
                states,
                UInt32(state_words),
                grid_dim=(state_words + 255) // 256,
                block_dim=256,
            )
            scope.record_launches(1)
            var base = 0
            for segment in segments:
                ref key = segment[0]
                ref a = segment[bindings[0]]
                ref b = segment[bindings[1]]
                ref c = segment[bindings[2]]
                ref d = segment[bindings[3]]
                ctx.enqueue_function[chunked_numeric_partials](
                    key.values,
                    key.bits,
                    a.values,
                    b.values,
                    c.values,
                    d.values,
                    a.bits,
                    b.bits,
                    c.bits,
                    d.bits,
                    _descriptor(key, 0),
                    _descriptor(a, flags[0]),
                    _descriptor(b, flags[1]),
                    _descriptor(c, flags[2]),
                    _descriptor(d, flags[3]),
                    UInt32(base),
                    UInt32(key.rows),
                    UInt32(layout.capacity),
                    UInt32(layout.fields),
                    scratch,
                    grid_dim=(key.rows + 255) // 256,
                    block_dim=256,
                )
                scope.record_launches(1)
                base += key.rows
            if not shared_output:
                ctx.enqueue_copy(host_first.unsafe_ptr(), first)
                ctx.enqueue_copy(
                    host_rows.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](
                        layout.rows_offset(), layout.capacity
                    ),
                )
                ctx.enqueue_copy(host_states.unsafe_ptr(), states)
                ctx.enqueue_copy(
                    control.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](0, len(control)),
                )
            scope.synchronize()
            if shared_output:
                for i in range(len(control)):
                    control[i] = backing[offset + i]
            else:
                scope.record_download(
                    4 * (2 * layout.capacity + state_words + len(control))
                )
        except error:
            try:
                scope.synchronize()
            except:
                pass
            _ = (
                backing^,
                scratch^,
                states^,
                first^,
                reservation^,
                host_first^,
                host_rows^,
                host_states^,
                control^,
            )
            raise error^
        _ = (scratch^, states^, first^)
        if shared_output and not (control[1] & GROUP_CAPACITY_FAULT):
            host_first = _read_words(
                backing, offset + layout.first_offset(), layout.capacity
            )
            host_rows = _read_words(
                backing, offset + layout.rows_offset(), layout.capacity
            )
            host_states = _read_words(
                backing, offset + layout.states_offset(), state_words
            )
        _ = (backing^, reservation^)
        timings.kernel_ns += Int(perf_counter_ns()) - started
        if control[1] & GROUP_CAPACITY_FAULT:
            raise Error(GROUP_CAPACITY_RETRY)
        if control[2] != UInt32(len(segments)):
            raise Error("Metal chunked numeric group scan did not complete")
        return GroupedPartials(
            host_first^, host_rows^, host_states^, List[Int64](), 1
        )


def scan_chunked_numeric(
    plan: GroupedComposition,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> GroupedPartials:
    comptime if not METAL_TARGET:
        raise Error("Metal chunked numeric groups require an Apple GPU")
    else:
        var started = Int(perf_counter_ns())
        var owners = List[NativeColumnAny]()
        var ctx = scope.context()
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var names: List[String] = [plan.key.copy()]
        var inputs = List[List[NumericView]]()
        var bindings = List[Int]()
        ref fields = plan.selected.bundle.fields
        var flags = List[UInt64](length=4, fill=0)
        for i in range(len(fields)):
            ref field = fields[i]
            if field.left not in names:
                names.append(field.left.copy())
            for j in range(len(names)):
                if names[j] == field.left:
                    bindings.append(j)
                    break
            if field.needs_sum:
                flags[i] |= FIELD_SUM
        for output in plan.outputs:
            if output.op in [MIN, MAX]:
                flags[output.field] |= FIELD_EXTREMA
        while len(bindings) < 4:
            bindings.append(0)
        for name in names:
            inputs.append(
                numeric_views(
                    frame, name, scope, zero_copy, empty, owners, coalesce=True
                )
            )
        var segments = intersect_numeric_views(inputs, frame.height())
        var capacity = sample_group_tuples(frame, plan.keys.names).capacity()
        timings.upload_ns += Int(perf_counter_ns()) - started
        var answer: GroupedPartials
        for attempt in range(2):
            var layout = GroupLayout(capacity, len(fields), 1, True)
            try:
                answer = _scan(
                    segments,
                    bindings,
                    flags,
                    layout,
                    scope,
                    zero_copy and capacity > Int(HASH_CAPACITY),
                    timings,
                )
                _ = (owners^, segments^, empty^)
                return answer^
            except error:
                if String(error) != GROUP_CAPACITY_RETRY:
                    raise error^
                if capacity == GENERAL_TABLE_LIMIT:
                    raise data_fault("grouped reduction exceeded 65536 groups")
                capacity = GENERAL_TABLE_LIMIT
        raise Error(GROUP_CAPACITY_RETRY)
