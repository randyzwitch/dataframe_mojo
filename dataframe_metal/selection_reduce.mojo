"""One queue of native reductions over the CPU schedule's retained batches."""
from std.ffi import external_call
from std.memory import bitcast, ArcPointer, Pointer
from dataframe.execution_observation import DeviceObservation
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer
from dataframe.dtype import DataType
from dataframe.expr import COUNT, MAX, MEAN, MIN
from dataframe.frame import DataFrame
from dataframe.planning.context import QueryContext
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.vector.selection_batches import SelectionBatches
from .selection_plan import SelectionReduction, plan_selection
from .exact_float import finish_float_bins
from .extrema_value import float_extrema_bits
from .accelerator.faults import data_fault
from .accelerator.gpu.numeric_view import NumericView
from .accelerator.gpu.selection_inputs import (
    SelectionGroup,
    map_selection_owner,
    _validated_selection_inputs,
    selection_inputs_supported,
)
from .accelerator.gpu.selection_reduce import (
    SELECTION_STATE_WORDS,
    selection_partials,
)
from .accelerator.gpu.filter_sum import BLOCK_SIZE
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.query_scope import (
    Scope,
    shared_service,
    checked_add,
    checked_mul,
)
from .accelerator.gpu.typed_storage import _NativeReservation
from .accelerator.gpu.target import METAL_TARGET


struct _IndexPages(Movable):
    # Metal borrows these pages until the queued reductions have finished.
    var address: Int

    def __init__(out self, size: Int, alignment: Int) raises:
        var address = 0
        if (
            size
            and external_call["posix_memalign", Int32](
                Pointer(to=address), alignment, size
            )
            != 0
        ):
            raise Error("Could not allocate shared Metal selection indices")
        self.address = address

    def __deinit__(deinit self):
        if self.address:
            external_call["free", NoneType](self.address)


struct _Total(Movable):
    var integer: Int128
    var count: Int
    var numeric: Int
    var minimum: Int64
    var maximum: Int64
    var flags: UInt32
    var bins: List[Int128]
    var nan: Bool
    var positive: Bool
    var negative: Bool
    var least: Int
    var greatest: Int

    def __init__(out self):
        self.integer = 0
        self.count = 0
        self.numeric = 0
        self.minimum = Int64.MAX
        self.maximum = Int64.MIN
        self.flags = 0
        self.bins = List[Int128](length=9, fill=0)
        self.nan = False
        self.positive = False
        self.negative = False
        self.least = 512
        self.greatest = 0

    def floating(self) raises -> Float64:
        var growth = 0
        var n = max(1, self.count) - 1
        while n:
            growth += 1
            n >>= 1
        # The CPU accumulates Float32 inputs in Float64. Exact bins can be
        # rounded once only when those additions are exact in any row order.
        if (
            not self.nan
            and not self.positive
            and not self.negative
            and self.least < 512
            and self.greatest - self.least + 1 + growth > 53
        ):
            raise data_fault(
                "selected float32 sum needs CPU accumulation order"
            )
        return finish_float_bins(
            self.bins, self.nan, self.positive, self.negative
        )


def selection_supports(region: QueryContext) -> Bool:
    comptime if not METAL_TARGET:
        return False
    try:
        return Bool(plan_selection(region))
    except:
        return False


def _finish(plan: SelectionReduction, totals: List[_Total]) raises -> DataFrame:
    var columns = List[Series]()
    for output in plan.outputs:
        ref total = totals[output.field]
        var dtype = plan.fields[output.field].dtype
        if output.op == COUNT:
            columns.append(
                Series(output.name, Column[Int64]([Int64(total.count)]))
            )
        elif output.op == MEAN:
            var sum = (
                total.floating() if dtype
                == DataType.FLOAT32 else Float64(total.integer)
            )
            columns.append(
                Series(
                    output.name,
                    Column[Float64](
                        [
                            sum
                            / Float64(total.count) if total.count else Float64(
                                0
                            )
                        ],
                        [total.count > 0],
                    ),
                )
            )
        elif output.op == MIN or output.op == MAX:
            var value = total.minimum if output.op == MIN else total.maximum
            var valid = total.numeric > 0
            if dtype == DataType.FLOAT32:
                var raw = float_extrema_bits(value, valid, total.flags)
                columns.append(
                    Series(
                        output.name,
                        Column[Float32]([bitcast[DType.float32](raw)], [valid]),
                    )
                )
            elif dtype == DataType.INT32:
                columns.append(
                    Series(
                        output.name,
                        Column[Int32](
                            [Int32(value) if valid else Int32(0)], [valid]
                        ),
                    )
                )
            else:
                columns.append(
                    Series(
                        output.name,
                        Column[Int64]([value if valid else Int64(0)], [valid]),
                    )
                )
        elif dtype == DataType.FLOAT32:
            columns.append(
                Series(
                    output.name,
                    Column[Float32](
                        [Float32(total.floating())],
                        [total.count >= output.min_count],
                    ),
                )
            )
        elif dtype == DataType.INT32:
            if total.integer > Int128(Int32.MAX) or total.integer < Int128(
                Int32.MIN
            ):
                raise data_fault("int32 selected sum overflow")
            columns.append(
                Series(
                    output.name,
                    Column[Int32](
                        [Int32(total.integer)],
                        [total.count >= output.min_count],
                    ),
                )
            )
        else:
            if total.integer > Int128(Int64.MAX) or total.integer < Int128(
                Int64.MIN
            ):
                raise data_fault("int64 selected sum overflow")
            columns.append(
                Series(
                    output.name,
                    Column[Int64](
                        [Int64(total.integer)],
                        [total.count >= output.min_count],
                    ),
                )
            )
    return DataFrame(columns^)


def _launch[
    D: DType, sums: Bool, extrema: Bool
](
    scope: Scope,
    view: NumericView,
    group: SelectionGroup,
    selections: DeviceBuffer[DType.uint32],
    descriptors: DeviceBuffer[DType.uint64],
    states: DeviceBuffer[DType.int64],
    bins: DeviceBuffer[DType.int64],
    specials: DeviceBuffer[DType.uint32],
) raises:
    scope.context().enqueue_function[selection_partials[D, sums, extrema]](
        view.values,
        view.bits,
        UInt32(view.bit_offset),
        UInt32(view.nullable),
        selections,
        descriptors,
        UInt32(group.first),
        UInt32(group.count),
        states,
        bins,
        specials,
        grid_dim=group.grid,
        block_dim=BLOCK_SIZE,
    )
    scope.record_launches(1)


def _field[
    D: DType
](
    plan: SelectionReduction,
    scope: Scope,
    view: NumericView,
    group: SelectionGroup,
    selections: DeviceBuffer[DType.uint32],
    descriptors: DeviceBuffer[DType.uint64],
    states: DeviceBuffer[DType.int64],
    bins: DeviceBuffer[DType.int64],
    specials: DeviceBuffer[DType.uint32],
) raises:
    ref field = plan.fields[group.field]
    if field.needs_sum:
        if field.needs_extrema:
            _launch[D, True, True](
                scope,
                view,
                group,
                selections,
                descriptors,
                states,
                bins,
                specials,
            )
        else:
            _launch[D, True, False](
                scope,
                view,
                group,
                selections,
                descriptors,
                states,
                bins,
                specials,
            )
    elif field.needs_extrema:
        _launch[D, False, True](
            scope, view, group, selections, descriptors, states, bins, specials
        )
    else:
        _launch[D, False, False](
            scope, view, group, selections, descriptors, states, bins, specials
        )


def execute_selection_timed(
    region: QueryContext, batches: SelectionBatches
) raises -> Tuple[DataFrame, HandoffTimings]:
    var matched = plan_selection(region)
    if not matched:
        raise Error("Metal selection reduction was not accepted")
    var plan = matched.take()
    var started = Int(perf_counter_ns())
    batches.validate()
    var validation_ns = Int(perf_counter_ns()) - started
    return _execute_selection_validated(
        plan, batches, validation_ns, region._device_observation
    )


def _execute_selection_validated(
    plan: SelectionReduction,
    batches: SelectionBatches,
    validation_ns: Int,
    observation: Optional[ArcPointer[DeviceObservation]],
) raises -> Tuple[DataFrame, HandoffTimings]:
    # Both public entry points validate once before reaching this path.
    var timings = HandoffTimings()
    var totals = List[_Total]()
    for _ in plan.fields:
        totals.append(_Total())
    var started = Int(perf_counter_ns())
    var inputs = _validated_selection_inputs(plan, batches)
    timings.upload_ns = validation_ns + Int(perf_counter_ns()) - started
    if inputs.partials == 0:
        started = Int(perf_counter_ns())
        var result = _finish(plan, totals)
        timings.finish_ns = Int(perf_counter_ns()) - started
        return (result^, timings^)
    comptime if not METAL_TARGET:
        raise Error("Metal selection reduction requires an Apple target")
    else:
        var scope = Scope(service=shared_service(), observation=observation)
        started = Int(perf_counter_ns())
        var ctx = scope.context()
        timings.context_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        var state_count = checked_mul(inputs.partials, SELECTION_STATE_WORDS)
        var bin_count = checked_mul(inputs.partials, 9)
        var special_count = checked_mul(inputs.partials, 6)
        var state_bytes = checked_add(
            checked_mul(state_count + bin_count, 8),
            checked_mul(special_count, 4),
        )
        var page = Int(external_call["getpagesize", Int32]())
        var index_bytes = (
            checked_add(checked_mul(inputs.selection_count, 4), page - 1)
            // page
            * page
        )
        var input_bytes = checked_add(
            max(index_bytes, 4),
            checked_mul(len(inputs.descriptors), 8),
        )
        var host_input_bytes = checked_add(
            inputs.selection_layout_bytes(),
            checked_mul(inputs.descriptors.capacity(), 8),
        )
        var reservations = List[_NativeReservation]()
        # Host result arrays and device scratch coexist through the drain.
        reservations.append(
            _NativeReservation(
                scope,
                checked_add(
                    checked_mul(state_bytes, 2),
                    checked_add(checked_add(input_bytes, host_input_bytes), 1),
                ),
            )
        )
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var views = List[NumericView]()
        for o in range(len(inputs.owners)):
            comptime for t in range(3):
                comptime D = DType.int32 if t == 0 else (
                    DType.int64 if t == 1 else DType.float32
                )
                if inputs.tags[o] == t:
                    views.append(
                        map_selection_owner[D](
                            inputs.owners[o], scope, empty, reservations
                        )
                    )
        var pages = _IndexPages(index_bytes, page)
        var selections: DeviceBuffer[DType.uint32]
        if index_bytes:
            var dest = Pointer[UInt32, MutAnyOrigin](
                unsafe_from_address=pages.address
            )
            inputs.fill_selections(batches, dest)
            selections = ctx.wrap_host_memory[DType.uint32](
                dest, index_bytes // 4
            )
        else:
            # Identity and constant segments never dereference this argument.
            selections = ctx.enqueue_create_buffer[DType.uint32](1)
        var descriptors = ctx.enqueue_create_buffer[DType.uint64](
            len(inputs.descriptors)
        )
        var states = ctx.enqueue_create_buffer[DType.int64](state_count)
        var bins = ctx.enqueue_create_buffer[DType.int64](bin_count)
        var specials = ctx.enqueue_create_buffer[DType.uint32](special_count)
        var host = List[Int64](length=state_count, fill=0)
        var host_bins = List[Int64](length=bin_count, fill=0)
        var host_specials = List[UInt32](length=special_count, fill=0)
        timings.upload_ns += Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        try:
            ctx.enqueue_copy(descriptors, inputs.descriptors.unsafe_ptr())
            scope.record_upload(len(inputs.descriptors) * 8)
            ctx.enqueue_memset(states, Int64(-1))
            for group_index in range(len(inputs.groups)):
                ref group = inputs.groups[group_index]
                comptime for t in range(3):
                    comptime D = DType.int32 if t == 0 else (
                        DType.int64 if t == 1 else DType.float32
                    )
                    if inputs.tags[group.owner] == t:
                        _field[D](
                            plan,
                            scope,
                            views[group.owner],
                            group,
                            selections,
                            descriptors,
                            states,
                            bins,
                            specials,
                        )
            ctx.enqueue_copy(host.unsafe_ptr(), states)
            ctx.enqueue_copy(host_bins.unsafe_ptr(), bins)
            ctx.enqueue_copy(host_specials.unsafe_ptr(), specials)
            scope.record_download(state_bytes)
            scope.synchronize()
        except error:
            try:
                scope.synchronize()
            except:
                pass
            _ = (host^, host_bins^, host_specials^)
            _ = (selections^, descriptors^, states^, bins^, specials^)
            _ = (pages^, views^, empty^, inputs^, reservations^)
            raise error^
        timings.kernel_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        for group_index in range(len(inputs.groups)):
            ref group = inputs.groups[group_index]
            ref field = plan.fields[group.field]
            ref total = totals[group.field]
            for p in range(group.partial, group.partial + group.grid):
                var at = p * SELECTION_STATE_WORDS
                if host[at + 2] < 0:
                    raise Error(
                        "Metal selected kernel did not complete on the device"
                    )
                total.count += Int(host[at + 2])
                total.numeric += Int(host[at + 5])
                total.minimum = min(total.minimum, host[at + 3])
                total.maximum = max(total.maximum, host[at + 4])
                total.flags |= UInt32(host[at + 6])
                if field.dtype == DataType.INT64:
                    total.integer += (Int128(host[at]) << 32) + Int128(
                        host[at + 1]
                    )
                elif field.dtype == DataType.INT32:
                    total.integer += Int128(host[at])
                elif field.needs_sum:
                    for b in range(9):
                        total.bins[b] += Int128(host_bins[p * 9 + b])
                    total.nan |= host_specials[p * 6] != 0
                    total.positive |= host_specials[p * 6 + 1] != 0
                    total.negative |= host_specials[p * 6 + 2] != 0
                    total.least = min(
                        total.least, Int(host_specials[p * 6 + 3])
                    )
                    total.greatest = max(
                        total.greatest, Int(host_specials[p * 6 + 4])
                    )
        # The source columns, selection arrays and reservations outlive the
        # last device read on both the normal and failed paths.
        _ = (selections^, descriptors^, states^, bins^, specials^)
        _ = (pages^, views^, empty^, inputs^, reservations^)
        var result = _finish(plan, totals)
        timings.finish_ns = Int(perf_counter_ns()) - started
        scope.mark_success()
        return (result^, timings^)


def selection_execute(
    region: QueryContext, batches: SelectionBatches
) raises -> Optional[DataFrame]:
    var started = Int(perf_counter_ns())
    batches.validate()
    var validation_ns = Int(perf_counter_ns()) - started
    var matched = plan_selection(region)
    if not matched or not selection_inputs_supported(matched.value(), batches):
        return None
    if region._device_observation:
        region._device_observation.value()[].provider = "metal"
    var result = _execute_selection_validated(
        matched.value(), batches, validation_ns, region._device_observation
    )
    result[1].record(region._device_observation)
    return result[0].copy()
