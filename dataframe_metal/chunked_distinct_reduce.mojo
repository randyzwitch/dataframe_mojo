"""Submit exact grouped distinct scans while retaining each native chunk owner."""
from std.time import perf_counter_ns
from dataframe import DataFrame, DataType
from .accelerator.faults import data_fault
from .accelerator.gpu.chunked_distinct import chunked_distinct_partials
from .accelerator.gpu.erased_input import numeric_views
from .accelerator.gpu.numeric_view import NumericView, intersect_numeric_views
from .accelerator.gpu.query_scope import Scope
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.typed_storage import NativeColumnAny, _NativeReservation
from .distinct_finish import finish_distinct
from .distinct_layout import DISTINCT_CAPACITY_FAULT, MAX_DISTINCT_CAPACITY
from .distinct_metadata import distinct_descriptor
from .distinct_plan import DistinctPlan
from .group_layout import GENERAL_TABLE_LIMIT, GROUP_CAPACITY_FAULT
from .group_properties import sample_group_tuples


def execute_chunked_distinct(
    frame: DataFrame,
    plan: DistinctPlan,
    scope: Scope,
    mut timings: HandoffTimings,
) raises -> DataFrame:
    comptime if not METAL_TARGET:
        raise Error("Metal exact distinct requires an Apple compilation target")
    else:
        var started = Int(perf_counter_ns())
        var ctx = scope.context()
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var owners = List[NativeColumnAny]()
        var columns = List[List[NumericView]]()
        ref key = plan.keys.value().names[0]
        ref field = plan.fields[0]
        columns.append(
            numeric_views(frame, key, scope, True, empty, owners, coalesce=True)
        )
        if field.column == key:
            columns.append(columns[0].copy())
        else:
            columns.append(
                numeric_views(
                    frame,
                    field.column,
                    scope,
                    True,
                    empty,
                    owners,
                    coalesce=True,
                )
            )
        var spans = intersect_numeric_views(columns, frame.height())
        var groups = sample_group_tuples(
            frame, plan.keys.value().names
        ).capacity()
        var values = 16384
        var record_words = 3 + Int(field.dtype == DataType.INT64)
        timings.upload_ns += Int(perf_counter_ns()) - started

        for attempt in range(2):
            started = Int(perf_counter_ns())
            var words = 4 + 4 * groups + record_words * values
            var reservation = _NativeReservation(scope, words * 4)
            var scratch = ctx.enqueue_create_buffer[DType.uint32](words)
            var control = List[UInt32](length=4, fill=0)
            var first = List[UInt32](length=groups, fill=0)
            var counts = List[UInt32](length=groups, fill=0)
            var nulls = List[UInt32](length=groups, fill=0)
            timings.upload_ns += Int(perf_counter_ns()) - started
            started = Int(perf_counter_ns())
            try:
                ctx.enqueue_memset(scratch, UInt32(0))
                var base = 0
                for span in spans:
                    ref k = span[0]
                    ref v = span[1]
                    ctx.enqueue_function[chunked_distinct_partials](
                        k.values,
                        k.bits,
                        v.values,
                        v.bits,
                        distinct_descriptor(k),
                        distinct_descriptor(v),
                        UInt32(base),
                        UInt32(k.rows),
                        UInt32(groups),
                        UInt32(values),
                        scratch,
                        grid_dim=(k.rows + 255) // 256,
                        block_dim=256,
                    )
                    scope.record_launches(1)
                    base += k.rows
                ctx.enqueue_copy(
                    control.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](0, 4),
                )
                ctx.enqueue_copy(
                    first.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](4, groups),
                )
                ctx.enqueue_copy(
                    counts.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](4 + groups, groups),
                )
                ctx.enqueue_copy(
                    nulls.unsafe_ptr(),
                    scratch.create_sub_buffer[DType.uint32](
                        4 + 2 * groups, groups
                    ),
                )
                scope.synchronize()
                scope.record_download((4 + 3 * groups) * 4)
            except error:
                try:
                    scope.synchronize()
                except:
                    pass
                _ = (
                    control^,
                    first^,
                    counts^,
                    nulls^,
                    scratch^,
                    reservation^,
                    owners^,
                    spans^,
                )
                raise error^
            _ = (scratch^, reservation^)
            timings.kernel_ns += Int(perf_counter_ns()) - started
            if control[1] & (GROUP_CAPACITY_FAULT | DISTINCT_CAPACITY_FAULT):
                if attempt:
                    raise data_fault(
                        "exact distinct exceeded bounded group/value table capacity"
                    )
                groups = GENERAL_TABLE_LIMIT
                values = MAX_DISTINCT_CAPACITY
                continue
            if control[1] or control[2] != UInt32(len(spans)):
                raise Error(
                    "Metal segmented exact distinct scan did not complete"
                )
            started = Int(perf_counter_ns())
            for i in range(len(first)):
                first[i] = ~first[i]
            var result = finish_distinct(plan, frame, first, counts, nulls)
            timings.finish_ns += Int(perf_counter_ns()) - started
            _ = (owners^, spans^)
            return result^
        raise Error("Metal segmented exact distinct retry was not completed")
