"""Resident observed-pair execution for dictionary-backed string concat."""
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, Series
from dataframe.binding import bind, ROWS
from dataframe.expr import LIT_INT
from dataframe.lazy import LazyFrame, SCAN_FRAME
from dataframe.execution_report import ExecutionReport
from dataframe.accelerator.observed_concat import (
    controls,
    Controls,
    is_marker,
)
from dataframe.accelerator.code_domains import TextField
from .residency import (
    Scope,
    TextDomain,
    Table,
    Matrix,
    CELL_BYTES,
    checked_add,
    checked_mul,
)
from . import observed_text_pairs as kernels
from . import observed_concat_labels
from .text_domains import (
    MapPlan,
    Submission,
    source_maps,
    submit as submit_text,
)
from .runtime import _width


@fieldwise_init
struct Layout(Movable):
    var controls: Controls
    var schema: DataFrame
    var left_slot: Int
    var right_slot: Int
    var input_columns: Int
    var output_columns: Int


def preflight(query: LazyFrame, source_rows: Int = -1) raises -> Layout:
    if not is_marker(query) or len(query._nodes) != 2:
        raise Error(
            "Observed concat requires one isolated resident control step"
        )
    ref operation = query._nodes[1]
    if (
        operation.left != 0
        or query._nodes[0].kind != SCAN_FRAME
        or query._nodes[0].offset != 0
    ):
        raise Error("Observed concat resident source metadata differs")
    if source_rows < -1 or source_rows > Int(Int32.MAX):
        raise Error(
            "Observed concat source cardinality exceeds Int32 device range"
        )
    var spec = controls(query)
    var input = query._frames[0].clear()
    var left = input._index(spec.left)
    var right = input._index(spec.right)
    if left == right:
        # A single column is legal, but the two bindings still have to name
        # the same original string field.
        pass
    for slot in [left, right]:
        var dtype = input._columns[slot].dtype()
        if dtype != DataType.UINT32:
            raise Error(
                "Observed concat operands must be encoded resident text"
            )
        var found_text = False
        for field in query._nodes[0].text_fields:
            if field.name == input._columns[slot].name():
                found_text = (
                    field.dtype == DataType.STRING
                    or field.dtype.is_categorical()
                )
                break
        if not found_text:
            raise Error("Observed concat operand lost text dictionary metadata")
    if (
        not query._nodes[0].code_domain
        or not query._nodes[0].code_domain.value().is_categorical()
    ):
        raise Error("Observed concat source text dictionary is missing")
    var columns = input._columns.copy()
    if len(operation.exprs) != 1:
        raise Error("Observed concat requires one output placeholder")
    ref expression = operation.exprs[0]
    var bound = bind(expression, input._columns)
    if (
        bound.shape() != ROWS
        or bound.dtypes[len(bound.dtypes) - 1] != DataType.UINT32
    ):
        raise Error("Observed concat placeholder output metadata differs")
    for column in columns:
        if column.name() == expression._name:
            raise Error("Observed concat output collides with source field")
    columns.append(Series.full_null(expression._name, DataType.UINT32, 0))
    if len(columns) > 256:
        raise Error("Observed concat output exceeds resident slot capacity")
    return Layout(
        spec^,
        DataFrame(columns^),
        left,
        right,
        input.width(),
        input.width() + 1,
    )


def estimate_bytes(
    layout: Layout, rows: Int, export_values: Bool
) raises -> Int:
    # Pair controls deliberately scale with rows, never with the Cartesian
    # product of source dictionaries.
    var scratch = checked_add(
        kernels.estimate_bytes(rows), checked_mul(rows, 8)
    )
    var output = checked_mul(
        checked_mul(rows, layout.output_columns), CELL_BYTES
    )
    var total = checked_add(
        output,
        checked_add(
            checked_add(scratch, checked_mul(layout.input_columns, 56)),
            8,
        ),
    )
    if export_values:
        total = checked_add(total, output)
        var bitmap = checked_add(rows, 7) // 8
        for column in layout.schema._columns:
            total = checked_add(
                total,
                checked_add(checked_mul(rows, _width(column.dtype())), bitmap),
            )
    return total


def _report(
    scope: Scope,
    source: Table,
    result: Table,
    before: Tuple[Int, Int, Int, Int],
    started: Int,
    peak: Int,
) raises -> DataFrame:
    var traffic = scope.transfers()
    var report = ExecutionReport()
    report.record(
        1,
        "STRING_CONCAT",
        "metal",
        source.rows,
        result.rows,
        algorithm="mojo_gpu_observed_text_pairs",
        wall_ns=Int(perf_counter_ns()) - started,
    )
    var fields = report.frame()._columns.copy()
    var values: List[Tuple[String, Int64]] = [
        ("kernel_launches", Int64(traffic[2] - before[2])),
        ("synchronizations", Int64(traffic[3] - before[3])),
        ("upload_bytes", Int64(traffic[0] - before[0])),
        ("download_bytes", Int64(traffic[1] - before[1])),
        ("peak_requested_bytes", Int64(peak)),
        ("host_result_bytes", Int64(0)),
        ("resident_input_bytes", Int64(source.retained_bytes())),
        ("resident_output_bytes", Int64(result.retained_bytes())),
    ]
    for pair in values:
        fields.append(Series(pair[0], Column[Int64]([pair[1]])))
    fields.append(Series("runtime_implementation", Column[String](["mojo"])))
    fields.append(
        Series(
            "query_wall_ms",
            Column[Float64]([Float64(Int(perf_counter_ns()) - started) / 1e6]),
        )
    )
    return DataFrame(fields^)


def execute_plan(
    query: LazyFrame,
    input: Table,
    scope: Scope,
    *,
    export_values: Bool = True,
    text_domain: Optional[ArcPointer[TextDomain]] = None,
) raises -> Tuple[Table, DataFrame]:
    var started = Int(perf_counter_ns())
    var before = scope.transfers()
    var layout = preflight(query, source_rows=input.rows)
    if (
        input.schema.columns() != query._frames[0].columns()
        or input.schema.dtypes() != query._frames[0].dtypes()
    ):
        raise Error("Observed concat resident schema differs from bound source")
    if not text_domain:
        raise Error("Observed concat requires remapped resident text ownership")
    var rows = input.rows
    var source = input.copy()
    if rows and source.identity_storage:
        if not source.identity_storage.value()[].scope.same_query(scope):
            raise Error(
                "Observed concat input belongs to another query context"
            )
    elif rows:
        var matrix = source.matrix()
        if not matrix[].scope.same_query(scope):
            raise Error(
                "Observed concat input belongs to another query context"
            )
        if (
            input.offset < 0
            or input.offset > matrix[].capacity
            or rows > matrix[].capacity - input.offset
        ):
            raise Error("Observed concat resident row range is outside matrix")
    var reservation = estimate_bytes(layout, rows, export_values)
    var base_domain = ArcPointer(
        TextDomain(scope, query._nodes[0].code_domain.value(), False)
    )
    var remap_plan = MapPlan(
        text_domain.value(),
        source_maps(
            input.schema,
            query._nodes[0].text_fields,
            base_domain,
            Optional[Table](input.copy()),
            rows,
        ),
    )
    reservation = checked_add(reservation, remap_plan.bytes())
    var peak = scope.check(reservation)
    if not rows:
        var result = Table.empty(layout.schema.copy())
        result.attach_text(query._nodes[0].text_fields, text_domain.value())
        var empty_labels = observed_concat_labels.build(
            scope,
            List[UInt32](),
            List[UInt32](),
            text_domain.value()[].dtype,
            layout.controls.separator,
            reservation,
        )
        result.attach_text(
            [TextField(layout.controls.output, DataType.STRING)],
            empty_labels[0],
        )
        peak = max(peak, empty_labels[2])
        return (
            result.copy(),
            _report(scope, input, result, before, started, peak),
        )
    var cells = checked_mul(rows, layout.output_columns)
    var mapping = List[Int64](capacity=layout.input_columns)
    for column in range(layout.input_columns):
        mapping.append(Int64(column))
    var context = scope.context()
    var words = context.enqueue_create_buffer[DType.uint64](2 * cells)
    var valid = context.enqueue_create_buffer[DType.uint8](cells)
    var errors = context.enqueue_create_buffer[DType.uint32](cells)
    var held_words = words.copy()
    var held_valid = valid.copy()
    var held_errors = errors.copy()
    var remap_owner = Optional[Submission]()
    var pair_owner = Optional[kernels.ObservedTextPairs]()
    var result_owner = Optional[Table]()
    var control = List[UInt32](length=2, fill=0)
    var host_left = List[UInt32]()
    var host_right = List[UInt32]()
    var host_pair_codes = List[UInt32]()
    var dynamic = Optional[ArcPointer[TextDomain]]()
    var packed_left = Optional[DeviceBuffer[DType.uint32]]()
    var packed_right = Optional[DeviceBuffer[DType.uint32]]()
    var pair_codes = Optional[DeviceBuffer[DType.uint32]]()
    try:
        _ = input.copy_to(
            scope, words, valid, errors, rows, layout.output_columns, mapping
        )
        remap_owner = submit_text(
            scope, remap_plan^, words, valid, errors, rows
        )
        packed_left = context.enqueue_create_buffer[DType.uint32](rows)
        packed_right = context.enqueue_create_buffer[DType.uint32](rows)
        context.enqueue_function[kernels.pack_codes](
            words.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            packed_left.value().unsafe_ptr(),
            UInt32(rows),
            UInt32(layout.left_slot),
            UInt32(rows),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        context.enqueue_function[kernels.pack_codes](
            words.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            packed_right.value().unsafe_ptr(),
            UInt32(rows),
            UInt32(layout.right_slot),
            UInt32(rows),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        scope.record_launches(2)
        var left_valid = valid.create_sub_buffer[DType.uint8](
            layout.left_slot * rows, rows
        )
        var right_valid = valid.create_sub_buffer[DType.uint8](
            layout.right_slot * rows, rows
        )
        var left_errors = errors.create_sub_buffer[DType.uint32](
            layout.left_slot * rows, rows
        )
        var right_errors = errors.create_sub_buffer[DType.uint32](
            layout.right_slot * rows, rows
        )
        pair_owner = kernels.run(
            context,
            packed_left.value(),
            left_valid,
            left_errors,
            packed_right.value(),
            right_valid,
            right_errors,
            rows,
        )
        scope.record_launches(4)
        scope.read_control_into(pair_owner.value().metadata, control)
        scope.record_download(8)
        if control[1]:
            raise Error("Observed concat pair table capacity/protocol failure")
        var pair_count = Int(control[0])
        if pair_count < 0 or pair_count > rows:
            raise Error("Observed concat pair count exceeds row cardinality")
        var compact_host = checked_add(
            checked_mul(pair_count, 8),
            checked_add(
                checked_mul(pair_count, 4),
                checked_add(
                    checked_mul(pair_count, 4),
                    checked_add(checked_mul(pair_count, 48), 132),
                ),
            ),
        )
        peak = max(peak, scope.check(checked_add(reservation, compact_host)))
        host_left = List[UInt32](length=pair_count, fill=0)
        host_right = List[UInt32](length=pair_count, fill=0)
        if pair_count:
            context.enqueue_copy(
                host_left.unsafe_ptr(),
                pair_owner.value().pair_left.create_sub_buffer[DType.uint32](
                    0, pair_count
                ),
            )
            context.enqueue_copy(
                host_right.unsafe_ptr(),
                pair_owner.value().pair_right.create_sub_buffer[DType.uint32](
                    0, pair_count
                ),
            )
            scope.synchronize()
            scope.record_download(pair_count * 8)
        var built = observed_concat_labels.build(
            scope,
            host_left,
            host_right,
            text_domain.value()[].dtype,
            layout.controls.separator,
            checked_add(reservation, compact_host),
        )
        dynamic = built[0].copy()
        peak = max(peak, built[2])
        host_pair_codes = built[1].copy()
        pair_codes = context.enqueue_create_buffer[DType.uint32](
            max(1, pair_count)
        )
        if pair_count:
            context.enqueue_copy(
                pair_codes.value().create_sub_buffer[DType.uint32](
                    0, pair_count
                ),
                host_pair_codes.unsafe_ptr(),
            )
        context.enqueue_function[kernels.write_output](
            pair_owner.value()
            .row_ids.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            pair_codes.value()
            .unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            pair_owner.value()
            .row_errors.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            words.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            valid.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            errors.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            UInt32(rows),
            Int64(layout.input_columns * rows),
            Int64(cells),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        scope.record_upload(pair_count * 4)
        scope.record_launches()
        scope.synchronize()
        var output_columns = List[Int64](capacity=layout.output_columns)
        for column in range(layout.output_columns):
            output_columns.append(Int64(column))
        var result = Table(
            Matrix(scope, rows, layout.output_columns, words^, valid^, errors^),
            layout.schema.copy(),
            output_columns^,
            rows,
        )
        result.attach_text(query._nodes[0].text_fields, text_domain.value())
        result.attach_text(
            [TextField(layout.controls.output, DataType.STRING)],
            dynamic.value(),
        )
        result_owner = result.copy()
        var report = _report(scope, input, result, before, started, peak)
        _ = layout^
        _ = source^
        _ = mapping^
        _ = remap_owner^
        _ = pair_owner^
        _ = control^
        _ = host_left^
        _ = host_right^
        _ = host_pair_codes^
        _ = packed_left^
        _ = packed_right^
        _ = pair_codes^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        return (result^, report^)
    except error:
        scope.synchronize()
        _ = layout^
        _ = source^
        _ = mapping^
        _ = remap_owner^
        _ = pair_owner^
        _ = result_owner^
        _ = control^
        _ = host_left^
        _ = host_right^
        _ = host_pair_codes^
        _ = dynamic^
        _ = packed_left^
        _ = packed_right^
        _ = pair_codes^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        raise error^
