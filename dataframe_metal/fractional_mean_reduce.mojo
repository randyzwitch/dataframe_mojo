"""Preserve per-key CPU input order while accumulating IEEE binary64 on Metal."""
from std.memory import bitcast
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, LazyFrame, Series
from .accelerator.faults import data_fault
from .accelerator.gpu.erased_input import numeric_views
from .accelerator.gpu.fractional_mean import (
    bucket_mean_rows,
    reduce_mean_buckets,
    MEAN_BUCKETS,
    MEAN_TILE,
    MEAN_WINDOW,
    MEAN_KEY_WORDS,
    MEAN_FIELD_WORDS,
)
from .accelerator.gpu.numeric_view import NumericView, intersect_numeric_views
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.typed_storage import NativeColumnAny, _NativeReservation
from .fractional_mean_plan import FractionalMeans
from .group_layout import GENERAL_TABLE_LIMIT
from .group_output import group_order
from .group_properties import sample_group_tuples


def _descriptor(view: NumericView) -> UInt64:
    return (
        (UInt64(view.bit_offset) << 32)
        | UInt64(view.nullable) << 3
        | UInt64(view.tag)
    )


def _finish(
    plan: FractionalMeans, state: List[UInt32], slots: Int
) raises -> DataFrame:
    var fields = len(plan.values)
    var stride = MEAN_KEY_WORDS + fields * MEAN_FIELD_WORDS
    var first = List[UInt32](length=slots, fill=UInt32.MAX)
    for slot in range(slots):
        if state[slot * stride]:
            first[slot] = state[slot * stride + 1] - 1
    var order = group_order(first)
    var narrow = List[Int32]()
    var wide = List[Int64]()
    var valid = List[Bool]()
    for slot in order:
        var start = slot * stride
        valid.append(Bool(state[start + 4]))
        if plan.key_dtype == DataType.INT32:
            narrow.append(bitcast[DType.int32](state[start + 2]))
        else:
            wide.append(
                bitcast[DType.int64](
                    UInt64(state[start + 2]) | (UInt64(state[start + 3]) << 32)
                )
            )
    var columns = List[Series]()
    if plan.key_dtype == DataType.INT32:
        columns.append(Series(plan.key, Column[Int32](narrow^, valid^)))
    else:
        columns.append(Series(plan.key, Column[Int64](wide^, valid^)))
    for i in range(len(plan.fields)):
        var field = plan.fields[i]
        var dtype = plan.dtypes[field]
        var values = List[Float64](capacity=len(order))
        var present = List[Bool](capacity=len(order))
        for slot in order:
            var at = slot * stride + MEAN_KEY_WORDS + field * MEAN_FIELD_WORDS
            var n = state[at]
            var low = UInt64(state[at + 1]) | (UInt64(state[at + 2]) << 32)
            var total: Float64
            if dtype == DataType.FLOAT32 or dtype == DataType.FLOAT64:
                total = bitcast[DType.float64](low)
            else:
                var integer = (
                    Int128(bitcast[DType.int32](state[at + 3])) << 64
                ) + Int128(low)
                total = integer.cast[DType.float64]()
            values.append(total / Float64(n) if n else Float64(0))
            present.append(n != 0)
        columns.append(
            Series(plan.names[i], Column[Float64](values^, present^))
        )
    return DataFrame(columns^)


def scan_fractional_means(
    frame: DataFrame,
    plan: FractionalMeans,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> DataFrame:
    comptime if not METAL_TARGET:
        raise Error("Grouped fractional means require Metal")
    else:
        if not frame.height():
            return _finish(plan, List[UInt32](), 0)
        var started = Int(perf_counter_ns())
        var ctx = scope.context()
        timings.context_ns += Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var owners = List[NativeColumnAny]()
        var views = List[List[NumericView]]()
        views.append(
            numeric_views(
                frame, plan.key, scope, zero_copy, empty, owners, coalesce=True
            )
        )
        var bindings = List[Int]()
        for name in plan.values:
            if name == plan.key:
                bindings.append(0)
            else:
                bindings.append(len(views))
                views.append(
                    numeric_views(
                        frame,
                        name,
                        scope,
                        zero_copy,
                        empty,
                        owners,
                        coalesce=True,
                    )
                )
        while len(bindings) < 4:
            bindings.append(0)
        var segments = intersect_numeric_views(views, frame.height())
        var windows = List[List[NumericView]]()
        var largest = 0
        for segment in segments:
            for offset in range(0, segment[0].rows, MEAN_WINDOW):
                var rows = min(MEAN_WINDOW, segment[0].rows - offset)
                var window = List[NumericView]()
                for v in segment:
                    window.append(v.view(offset, rows))
                windows.append(window^)
                largest = max(largest, rows)
        var head_words = ((largest + MEAN_TILE - 1) // MEAN_TILE) * MEAN_BUCKETS
        var index_memory = _NativeReservation(
            scope, (head_words + largest + 4) * 4
        )
        var heads = ctx.enqueue_create_buffer[DType.uint32](head_words)
        var links = ctx.enqueue_create_buffer[DType.uint32](largest)
        var control = ctx.enqueue_create_buffer[DType.uint32](4)
        var keys: List[String] = [plan.key.copy()]
        var capacity = sample_group_tuples(frame, keys).capacity()
        var fields = len(plan.values)
        var stride = MEAN_KEY_WORDS + fields * MEAN_FIELD_WORDS
        timings.upload_ns += Int(perf_counter_ns()) - started
        for attempt in range(2):
            started = Int(perf_counter_ns())
            var reservation = _NativeReservation(scope, 4 * capacity * stride)
            var table = ctx.enqueue_create_buffer[DType.uint32](
                capacity * stride
            )
            var state = List[UInt32](length=capacity * stride, fill=0)
            var status = List[UInt32](length=4, fill=0)
            try:
                ctx.enqueue_memset(table, UInt32(0))
                ctx.enqueue_memset(control, UInt32(0))
                var base = 0
                for window in windows:
                    ref k = window[0]
                    ref a = window[bindings[0]]
                    ref b = window[bindings[1]]
                    ref c = window[bindings[2]]
                    ref d = window[bindings[3]]
                    var tiles = (k.rows + MEAN_TILE - 1) // MEAN_TILE
                    ctx.enqueue_function[bucket_mean_rows](
                        k.values,
                        k.bits,
                        _descriptor(k),
                        k.rows,
                        heads,
                        links,
                        control,
                        grid_dim=(tiles + 63) // 64,
                        block_dim=64,
                    )
                    scope.record_launches(1)
                    ctx.enqueue_function[reduce_mean_buckets](
                        k.values,
                        k.bits,
                        _descriptor(k),
                        a.values,
                        a.bits,
                        _descriptor(a),
                        b.values,
                        b.bits,
                        _descriptor(b),
                        c.values,
                        c.bits,
                        _descriptor(c),
                        d.values,
                        d.bits,
                        _descriptor(d),
                        UInt32(base),
                        k.rows,
                        fields,
                        UInt32(capacity),
                        heads,
                        links,
                        table,
                        control,
                        grid_dim=MEAN_BUCKETS // 32,
                        block_dim=32,
                    )
                    scope.record_launches(1)
                    base += k.rows
                ctx.enqueue_copy(state.unsafe_ptr(), table)
                ctx.enqueue_copy(status.unsafe_ptr(), control)
                scope.synchronize()
                scope.record_download(4 * (len(state) + len(status)))
            except error:
                try:
                    scope.synchronize()
                except:
                    pass
                _ = (
                    state^,
                    status^,
                    table^,
                    reservation^,
                    heads^,
                    links^,
                    control^,
                    index_memory^,
                    windows^,
                    owners^,
                )
                raise error^
            _ = (table^, reservation^)
            timings.kernel_ns += Int(perf_counter_ns()) - started
            if status[1]:
                if capacity == GENERAL_TABLE_LIMIT:
                    if status[1] == 2:
                        raise Error(
                            "Metal grouped mean exhausted a complete table probe cycle"
                        )
                    raise data_fault("grouped reduction exceeded 65536 groups")
                capacity = GENERAL_TABLE_LIMIT
                continue
            if status[2] != UInt32(len(windows)) or status[3] != UInt32(
                len(windows)
            ):
                raise Error("Metal fractional mean scans did not complete")
            _ = (heads^, links^, control^, index_memory^, windows^, owners^)
            started = Int(perf_counter_ns())
            var result = _finish(plan, state, capacity)
            timings.finish_ns += Int(perf_counter_ns()) - started
            return result^
        raise Error("Metal fractional mean capacity retry did not complete")


def execute_fractional_means(
    region: LazyFrame, plan: FractionalMeans, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var timings = HandoffTimings()
    var scope = Scope(
        service=shared_service(), observation=region._plan._device_observation
    )
    ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
    var result = scan_fractional_means(frame, plan, scope, zero_copy, timings)
    scope.mark_success()
    return (result^, timings^)
