"""Isolated resident custom-label QCUT executor with CPU-proven rounding.

Dynamic edges use separate software IEEE operations; an index permutation
correction preserves the pinned CPU selected-zero representative.
No source values or categories are read back before the final materializer.
"""
from max.gpu.host import DeviceBuffer
from std.memory import bitcast, ArcPointer
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, Series
from dataframe.binding import bind, ROWS
from dataframe.expr import LIT_INT
from dataframe.lazy import LazyFrame, SCAN_FRAME
from dataframe.execution_report import ExecutionReport
from dataframe.accelerator.qcuts import controls, Controls, is_qcut_marker
from .residency import (
    Scope,
    TextDomain,
    Table,
    Matrix,
    CELL_BYTES,
    checked_add,
    checked_mul,
)
from . import qcuts as kernels
from . import qcut_labels
from .text_domains import (
    MapPlan,
    Submission,
    source_maps,
    submit as submit_text,
)
from dataframe.accelerator.code_domains import TextField
from .runtime import _width


@fieldwise_init
struct Layout(Movable):
    var controls: Controls
    var label_codes: List[UInt32]
    var schema: DataFrame
    var operand: Int
    var input_columns: Int
    var output_columns: Int


def preflight(query: LazyFrame, source_rows: Int = -1) raises -> Layout:
    if not is_qcut_marker(query) or len(query._nodes) != 2:
        raise Error("QCUT requires one isolated resident control step")
    ref operation = query._nodes[1]
    if (
        operation.left != 0
        or query._nodes[0].kind != SCAN_FRAME
        or query._nodes[0].offset != 0
    ):
        raise Error("QCUT resident source metadata differs")
    if source_rows < -1 or source_rows > Int(Int32.MAX):
        raise Error("QCUT source cardinality exceeds Int32 device range")
    var spec = controls(query)
    var input = query._frames[0].clear()
    var operand = input._index(spec.operand)
    if input._columns[operand].dtype() != DataType.FLOAT64:
        raise Error("QCUT operand must be observed resident Float64")
    var codes = List[UInt32](capacity=len(spec.label_nodes))
    if spec.defaults:
        codes = List[UInt32](length=len(spec.probabilities) + 1, fill=0)
    for index in spec.label_nodes:
        ref node = operation.exprs[0]._nodes[index]
        if (
            node.op != LIT_INT
            or node.integer < 0
            or node.integer > Int64(UInt32.MAX)
        ):
            raise Error(
                "QCUT custom labels require dictionary-recoded UInt32 literals"
            )
        if (
            not len(node.dtypes)
            or not node.dtypes[0]
            or node.dtypes[0].value() != DataType.UINT32
        ):
            raise Error("QCUT custom label literal type differs")
        codes.append(UInt32(node.integer))
    var columns = input._columns.copy()
    for index in range(len(operation.exprs)):
        ref expression = operation.exprs[index]
        var bound = bind(expression, input._columns)
        var expected = DataType.UINT32 if index == 0 else DataType.FLOAT64
        if (
            bound.shape() != ROWS
            or bound.dtypes[len(bound.dtypes) - 1] != expected
        ):
            raise Error("QCUT placeholder output metadata differs")
        for column in columns:
            if column.name() == expression._name:
                raise Error("QCUT private output collides with source field")
        columns.append(Series.full_null(expression._name, expected, 0))
    if len(columns) > 256:
        raise Error("QCUT output exceeds resident slot capacity")
    return Layout(
        spec^,
        codes^,
        DataFrame(columns^),
        operand,
        input.width(),
        input.width() + len(operation.exprs),
    )


def estimate_bytes(
    layout: Layout, rows: Int, export_values: Bool
) raises -> Int:
    """Complete reservation beyond existing live owners, before device allocation.

    Includes scratch/host controls, output Matrix, resident copy descriptors,
    and final compact host/typed-column export when requested. Keeping phases
    summed is a conservative reservation; no pre-existing source is duplicated.
    """
    var count = len(layout.controls.probabilities)
    var entries = max(1, count)
    var device_scratch = checked_add(
        checked_mul(rows, 5 if count else 1),
        checked_add(
            checked_mul(entries, 28), checked_add(checked_mul(count + 1, 8), 8)
        ),
    )
    var host_controls = checked_add(
        checked_mul(count, 8), checked_add(checked_mul(count + 1, 12), 8)
    )
    var output = checked_mul(
        checked_mul(rows, layout.output_columns), CELL_BYTES
    )
    var total = checked_add(
        output,
        checked_add(
            device_scratch,
            checked_add(host_controls, checked_mul(layout.input_columns, 56)),
        ),
    )
    if export_values:
        total = checked_add(total, output)  # compact HostMatrix readback
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
        "QCUT",
        "metal",
        source.rows,
        result.rows,
        algorithm="mojo_gpu_qcut_controls",
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
    """Input must be the already-observed operand producer's immutable table."""
    var started = Int(perf_counter_ns())
    var before = scope.transfers()
    var layout = preflight(query, source_rows=input.rows)
    if (
        input.schema.columns() != query._frames[0].columns()
        or input.schema.dtypes() != query._frames[0].dtypes()
    ):
        raise Error("QCUT resident schema differs from bound source")
    var rows = input.rows
    var dynamic = Optional[ArcPointer[TextDomain]]()
    if layout.controls.defaults and not text_domain:
        raise Error(
            "QCUT default labels require explicit dynamic text ownership"
        )
    if not rows:
        var empty = kernels.empty_controls(layout.controls.probabilities)
        if empty[1]:
            raise Error(kernels.failure_message(empty[1]))
        var peak = scope.check(
            checked_add(
                checked_mul(len(layout.controls.probabilities), 8),
                checked_mul(len(layout.label_codes), 12),
            )
        )
        var result = Table.empty(layout.schema.copy())
        if text_domain:
            result.attach_text(query._nodes[1].text_fields, text_domain.value())
        if layout.controls.defaults:
            var labels = qcut_labels.build(
                scope, List[UInt64](), 0, Bool(layout.controls.flags & 1), 0
            )
            result.attach_text(
                [TextField(layout.controls.category, DataType.STRING)],
                labels[0],
            )
            peak = max(peak, labels[2])
        return (
            result.copy(),
            _report(scope, input, result, before, started, peak),
        )
    var source = input.copy()
    if source.identity_storage:
        if not source.identity_storage.value()[].scope.same_query(scope):
            raise Error("QCUT resident input belongs to another query context")
    else:
        var matrix = source.matrix()
        if not matrix[].scope.same_query(scope):
            raise Error("QCUT resident input belongs to another query context")
        if (
            input.offset < 0
            or input.offset > matrix[].capacity
            or rows > matrix[].capacity - input.offset
        ):
            raise Error("QCUT resident row range is outside matrix")
    # The immutable append creates one ordinary output Matrix. Its copied
    # input slots also supply quantile words; no additional source plane copy
    # is needed, including for a metadata-finalizer HostMatrix identity.
    var base = checked_mul(layout.operand, rows)
    var reservation = estimate_bytes(layout, rows, export_values)
    var remap_plan = Optional[MapPlan]()
    var remap_owner = Optional[Submission]()
    if text_domain:
        var base_domain = ArcPointer(
            TextDomain(scope, query._nodes[0].code_domain.value(), False)
        )
        remap_plan = MapPlan(
            text_domain.value(),
            source_maps(
                input.schema,
                query._nodes[0].text_fields,
                base_domain,
                Optional[Table](input.copy()),
                rows,
            ),
        )
        reservation = checked_add(reservation, remap_plan.value().bytes())
    if layout.controls.defaults:
        reservation = checked_add(
            reservation, checked_mul(len(layout.controls.probabilities), 8)
        )
    var peak = scope.check(reservation)
    var count = len(layout.controls.probabilities)
    var entries = max(1, count)
    var cells = checked_mul(rows, layout.output_columns)
    var mapping = List[Int64](capacity=layout.input_columns)
    for column in range(layout.input_columns):
        mapping.append(Int64(column))
    var control = List[UInt32](length=2, fill=0)
    var edge_control = List[UInt64](
        length=count if layout.controls.defaults else 0, fill=0
    )
    var context = scope.context()
    var probabilities = context.enqueue_create_buffer[DType.uint64](entries)
    var codes = context.enqueue_create_buffer[DType.uint32](
        len(layout.label_codes)
    )
    var mask = context.enqueue_create_buffer[DType.uint8](rows)
    var permutation = Optional[DeviceBuffer[DType.uint32]]()
    if count:
        permutation = context.enqueue_create_buffer[DType.uint32](rows)
    var raw = context.enqueue_create_buffer[DType.uint64](entries)
    var status = context.enqueue_create_buffer[DType.uint32](entries)
    var edges = context.enqueue_create_buffer[DType.uint64](entries)
    var labels = context.enqueue_create_buffer[DType.uint32](count + 1)
    var device_control = context.enqueue_create_buffer[DType.uint32](2)
    var words = context.enqueue_create_buffer[DType.uint64](2 * cells)
    var valid = context.enqueue_create_buffer[DType.uint8](cells)
    var errors = context.enqueue_create_buffer[DType.uint32](cells)
    var held_words = words.copy()
    var held_valid = valid.copy()
    var held_errors = errors.copy()
    try:
        _ = input.copy_to(
            scope, words, valid, errors, rows, layout.output_columns, mapping
        )
        if remap_plan:
            remap_owner = submit_text(
                scope, remap_plan.take(), words, valid, errors, rows
            )
        if count:
            context.enqueue_copy(
                probabilities, layout.controls.probabilities.unsafe_ptr()
            )
        if not layout.controls.defaults:
            context.enqueue_copy(codes, layout.label_codes.unsafe_ptr())
        scope.record_upload(
            count * 8
            + (0 if layout.controls.defaults else len(layout.label_codes) * 4)
        )
        context.enqueue_function[kernels.sample_mask](
            words.unsafe_ptr(),
            valid.unsafe_ptr(),
            mask.unsafe_ptr(),
            UInt64(base),
            UInt32(rows),
            grid_dim=(rows + 127) // 128,
            block_dim=128,
        )
        if count:
            context.enqueue_function[kernels.raw_edges](
                words.unsafe_ptr(),
                mask.unsafe_ptr(),
                probabilities.unsafe_ptr(),
                raw.unsafe_ptr(),
                status.unsafe_ptr(),
                UInt64(base),
                UInt32(rows),
                UInt32(count),
                grid_dim=(count + 31) // 32,
                block_dim=32,
            )
        if count:
            context.enqueue_function[kernels.correct_zero_edges](
                words.unsafe_ptr(),
                mask.unsafe_ptr(),
                probabilities.unsafe_ptr(),
                raw.unsafe_ptr(),
                status.unsafe_ptr(),
                permutation.value().unsafe_ptr(),
                UInt64(base),
                UInt32(rows),
                UInt32(count),
                grid_dim=1,
                block_dim=1,
            )
        context.enqueue_function[kernels.compress_edges](
            raw.unsafe_ptr(),
            status.unsafe_ptr(),
            edges.unsafe_ptr(),
            labels.unsafe_ptr(),
            device_control.unsafe_ptr(),
            UInt32(count),
            UInt32((layout.controls.flags & 4) != 0),
            grid_dim=1,
            block_dim=1,
        )
        scope.record_launches(4 if count else 2)
        if layout.controls.defaults:
            context.enqueue_copy(control.unsafe_ptr(), device_control)
            if count:
                context.enqueue_copy(edge_control.unsafe_ptr(), edges)
            scope.synchronize()
            scope.record_download(8 + count * 8)
        else:
            scope.read_control_into(device_control, control)
            scope.record_download(8)
        if control[1]:
            raise Error(kernels.failure_message(control[1]))
        if layout.controls.defaults:
            var formatted = qcut_labels.build(
                scope,
                edge_control,
                Int(control[0]),
                Bool(layout.controls.flags & 1),
                reservation,
            )
            dynamic = formatted[0].copy()
            peak = max(peak, formatted[2])
            layout.label_codes = formatted[1].copy()
            context.enqueue_copy(
                codes.create_sub_buffer[DType.uint32](
                    0, len(layout.label_codes)
                ),
                layout.label_codes.unsafe_ptr(),
            )
            context.enqueue_function[kernels.default_label_indices](
                labels.unsafe_ptr(),
                control[0],
                grid_dim=(Int(control[0]) + 256) // 256,
                block_dim=256,
            )
            scope.record_upload(len(layout.label_codes) * 4)
            scope.record_launches()
        # Source operand and newly appended output slots are disjoint. Erase
        # submission origins explicitly so readonly source views do not share
        # a mutable argument origin with the output matrix aliases.
        context.enqueue_function[kernels.classify](
            words.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            errors.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            mask.unsafe_ptr(),
            edges.unsafe_ptr(),
            labels.unsafe_ptr(),
            codes.unsafe_ptr(),
            words.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            valid.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            errors.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            UInt64(base),
            UInt32(rows),
            control[0],
            UInt32((layout.controls.flags & 1) != 0),
            UInt32((layout.controls.flags & 2) != 0),
            UInt32(rows),
            UInt32(layout.output_columns),
            UInt32(layout.input_columns),
            UInt32(layout.input_columns + 1),
            grid_dim=(rows + 127) // 128,
            block_dim=128,
        )
        scope.record_launches()
        scope.synchronize()
        var columns = List[Int64](capacity=layout.output_columns)
        for column in range(layout.output_columns):
            columns.append(Int64(column))
        var result = Table(
            Matrix(scope, rows, layout.output_columns, words^, valid^, errors^),
            layout.schema.copy(),
            columns^,
            rows,
        )
        if text_domain:
            result.attach_text(query._nodes[1].text_fields, text_domain.value())
        if dynamic:
            result.attach_text(
                [TextField(layout.controls.category, DataType.STRING)],
                dynamic.value(),
            )
        var report = _report(scope, input, result, before, started, peak)
        _ = layout^
        _ = source^
        _ = mapping^
        _ = control^
        _ = edge_control^
        _ = remap_owner^
        _ = probabilities^
        _ = codes^
        _ = mask^
        _ = permutation^
        _ = raw^
        _ = status^
        _ = edges^
        _ = labels^
        _ = device_control^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        return (result^, report^)
    except error:
        scope.synchronize()
        _ = layout^
        _ = source^
        _ = mapping^
        _ = control^
        _ = edge_control^
        _ = remap_owner^
        _ = probabilities^
        _ = codes^
        _ = mask^
        _ = permutation^
        _ = raw^
        _ = status^
        _ = edges^
        _ = labels^
        _ = device_control^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        raise error^
