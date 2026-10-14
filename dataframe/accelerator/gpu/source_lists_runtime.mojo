"""Resident fixed-width source List consumers followed by ordinary row_program."""
from max.gpu import global_idx
from std.collections import Dict
from std.memory import ArcPointer, Pointer
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, Series
from dataframe.expr import LIST_LEN
from dataframe.lazy import LazyFrame
from dataframe.accelerator.source_lists import (
    SourceLists,
    lower_source_lists,
    attach_metadata,
)
from . import runtime as rows
from .list_storage import stage_child, list_ref, consume
from .residency import (
    Scope,
    Matrix,
    Table,
    CELL_BYTES,
    checked_add,
    checked_mul,
)


def _copy_segment(
    source_words: Pointer[UInt64, ImmutAnyOrigin],
    source_valid: Pointer[UInt8, ImmutAnyOrigin],
    source_errors: Pointer[UInt32, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_valid: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    source_capacity: UInt32,
    source_slots: UInt32,
    source_column: UInt32,
    source_offset: UInt32,
    count: UInt32,
    target_capacity: UInt32,
    target_slots: UInt32,
    target_column: UInt32,
    target_offset: UInt32,
):
    var row = global_idx.x
    if row >= Int(count):
        return
    var source = (
        Int(source_column) * Int(source_capacity) + Int(source_offset) + row
    )
    var target = (
        Int(target_column) * Int(target_capacity) + Int(target_offset) + row
    )
    output_words[unsafe_offset=target] = source_words[unsafe_offset=source]
    output_words[
        unsafe_offset=Int(target_capacity) * Int(target_slots) + target
    ] = source_words[
        unsafe_offset=Int(source_capacity) * Int(source_slots) + source
    ]
    output_valid[unsafe_offset=target] = source_valid[unsafe_offset=source]
    output_errors[unsafe_offset=target] = source_errors[unsafe_offset=source]


def _copy(
    source: Table, target: Table, scope: Scope, column: Int, start: Int
) raises:
    if not source.rows:
        return
    var owner = source.matrix()
    var destination = target.matrix()
    if not owner[].scope.same_query(
        scope
    ) or not destination[].scope.same_query(scope):
        raise Error("Source List merge belongs to another query context")
    if (
        source.schema.width() != 1
        or len(source.columns) != 1
        or column < 0
        or column >= destination[].slots
        or start < 0
        or source.rows > target.rows - start
    ):
        raise Error("Source List merge segment shape differs")
    var ctx = scope.context()
    try:
        ctx.enqueue_function[_copy_segment](
            owner[].words.unsafe_ptr(),
            owner[].validity.unsafe_ptr(),
            owner[].errors.unsafe_ptr(),
            destination[].words.unsafe_ptr(),
            destination[].validity.unsafe_ptr(),
            destination[].errors.unsafe_ptr(),
            UInt32(owner[].capacity),
            UInt32(owner[].slots),
            UInt32(source.columns[0]),
            UInt32(source.offset),
            UInt32(source.rows),
            UInt32(destination[].capacity),
            UInt32(destination[].slots),
            UInt32(column),
            UInt32(start),
            grid_dim=(source.rows + 255) // 256,
            block_dim=256,
        )
        scope.record_launches()
        scope.synchronize()
        _ = owner^
        _ = destination^
    except error:
        scope.synchronize()
        _ = owner^
        _ = destination^
        raise error^


def _preprocess_peak(plan: SourceLists) raises -> Int:
    var count = plan.source.height()
    var slots = len(plan.ordinary) + len(plan.consumers)
    if count > Int(Int32.MAX) or slots > 256:
        raise Error("Metal source List exceeds fixed-width resident dimensions")
    var matrix = checked_mul(checked_mul(count, slots), CELL_BYTES)
    var scratch = 0
    if count and len(plan.ordinary):
        scratch = checked_mul(count, CELL_BYTES * 2)
    var seen = Dict[String, Bool]()
    if count:
        for consumer in plan.consumers:
            if consumer.source in seen:
                continue
            seen[consumer.source] = True
            for part in plan.source.column(consumer.source).chunks():
                var parent_rows = len(part)
                var child_rows = len(part.list_column().child())
                if child_rows > Int(Int32.MAX):
                    raise Error(
                        "Metal source List child exceeds resident dimensions"
                    )
                var child_bytes = checked_mul(child_rows, CELL_BYTES)
                var descriptors = checked_add(
                    checked_mul(parent_rows + 1, 8), parent_rows
                )
                scratch = max(scratch, checked_mul(child_bytes, 2))
                scratch = max(
                    scratch,
                    checked_add(
                        child_bytes,
                        checked_add(
                            checked_mul(descriptors, 2),
                            checked_mul(parent_rows, CELL_BYTES),
                        ),
                    ),
                )
    return checked_add(matrix, scratch)


def _prepare(plan: SourceLists, scope: Scope) raises -> Table:
    var schema = plan.query._frames[plan.query._nodes[0].offset].copy()
    var count = plan.source.height()
    var slots = schema.width()
    if not count:
        return Table.empty(schema^)
    if not slots:
        return Table.rows_only(scope, count)
    var cells = checked_mul(count, slots)
    _ = scope.check(checked_mul(cells, CELL_BYTES))
    var ctx = scope.context()
    var words = ctx.enqueue_create_buffer[DType.uint64](2 * cells)
    var validity = ctx.enqueue_create_buffer[DType.uint8](cells)
    var errors = ctx.enqueue_create_buffer[DType.uint32](cells)
    # Private builder: no immutable Table consumer sees these buffers until
    # every segment is filled and fenced. One retained owner accounts all live
    # destination cells while temporary source/consumer matrices are allocated.
    var matrix = ArcPointer(
        Matrix(scope, count, slots, words^, validity^, errors^)
    )
    var merged = Table.empty(schema^)
    merged.storage = matrix.copy()
    merged.columns = List[Int64](capacity=slots)
    for slot in range(slots):
        merged.columns.append(Int64(slot))
    merged.rows = count
    for slot in range(len(plan.ordinary)):
        var field = stage_child(plan.source.column(plan.ordinary[slot]), scope)
        _copy(field, merged, scope, slot, 0)
    var seen = Dict[String, Bool]()
    for consumer in plan.consumers:
        if consumer.source in seen:
            continue
        seen[consumer.source] = True
        var start = 0
        for part in plan.source.column(consumer.source).chunks():
            if not len(part):
                continue
            var child = stage_child(part.list_column().child(), scope)
            var reference = list_ref(part, child, scope)
            for index in range(len(plan.consumers)):
                ref operation = plan.consumers[index]
                if operation.source != consumer.source:
                    continue
                var value: Table
                if operation.op == LIST_LEN:
                    value = consume[True](reference, scope, operation.name)
                else:
                    value = consume[False](
                        reference, scope, operation.name, operation.index
                    )
                _copy(value, merged, scope, len(plan.ordinary) + index, start)
            start += len(part)
        if start != count:
            raise Error("Metal source List parent chunk lengths differ")
    _ = matrix^
    return merged^


def _column(report: DataFrame, name: String, value: Int) raises -> DataFrame:
    return report.with_column(
        Series(
            name,
            Column[Int64](
                List[Int64](length=report.height(), fill=Int64(value))
            ),
        )
    )


def execute(
    query: LazyFrame,
    device: Int,
    budget: Int,
    profiling: Bool,
    *,
    observation_only: Bool = False,
) raises -> Tuple[DataFrame, DataFrame]:
    var start = Int(perf_counter_ns())
    # On the admitted direct path CPU materializes the scan before binding its
    # projection. Preserve a failed source's precedence over invalid operands.
    if len(query._nodes) == 2 and query._nodes[0].source_failure:
        raise Error(query._nodes[0].source_failure)
    var plan = lower_source_lists(query)
    if plan.source_failure:
        raise Error(plan.source_failure)
    # Complete row capability/provenance and resource admission precedes all
    # source staging or context creation, including rejected temporal aliases.
    var row_plan = rows._preflight(
        plan.query, source_rows=plan.source.height(), terminal_boundary=True
    )
    var scope = Scope(device, budget)
    var facts = rows.estimate_resources(
        row_plan, plan.source.height(), True, True, not observation_only
    )
    var source_bytes = checked_mul(
        checked_mul(
            plan.source.height(), len(plan.ordinary) + len(plan.consumers)
        ),
        CELL_BYTES,
    )
    _ = scope.check(
        max(_preprocess_peak(plan), checked_add(source_bytes, facts.peak_bytes))
    )
    plan.query._nodes[0]._record_execution()
    var resident = _prepare(plan, scope)
    var before = scope.transfers()
    var executed = rows.execute_plan(
        row_plan,
        scope,
        Optional(resident.copy()),
        profiling,
        terminal_boundary=True,
        export_values=not observation_only,
    )
    var result: DataFrame
    if observation_only:
        executed[0].observe_errors()
        result = plan.schema.copy()
    else:
        result = attach_metadata(rows.materialize(executed[0]), plan)
    var traffic = scope.transfers()
    var report = executed[1].copy()
    for field in List[Tuple[String, Int]](
        [
            ("source_list_consumers", len(plan.consumers)),
            ("source_list_matrix_bytes", source_bytes),
            ("source_list_preprocess_upload_bytes", before[0]),
            ("source_list_preprocess_download_bytes", before[1]),
            ("source_list_preprocess_launches", before[2]),
            ("source_list_preprocess_synchronizations", before[3]),
            ("query_peak_requested_bytes", scope.info()[1]),
            ("query_upload_bytes", traffic[0]),
            ("query_download_bytes", traffic[1]),
            ("query_kernel_launches", traffic[2]),
            ("query_synchronizations", traffic[3]),
        ]
    ):
        report = _column(report, field[0], field[1])
    var output_bytes = 0
    if not observation_only:
        for column in result._columns:
            output_bytes = checked_add(
                output_bytes,
                checked_add(
                    checked_mul(result.height(), rows._width(column.dtype())),
                    (result.height() + 7) // 8,
                ),
            )
    report = _column(report, "host_result_bytes", output_bytes)
    report = report.with_column(
        Series(
            "source_list_backend",
            Column[String](
                List[String](
                    length=report.height(), fill="mojo_gpu_resident_list"
                )
            ),
        )
    )
    report = report.with_column(
        Series(
            "query_wall_ms",
            Column[Float64](
                List[Float64](
                    length=report.height(),
                    fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                )
            ),
        )
    )
    _ = resident^
    return (result^, report^)
