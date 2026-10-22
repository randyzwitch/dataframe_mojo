"""Direct Mojo host orchestration for shared fixed-width join tasks.

The first submission learns limited cardinality. The complete requested peak
is checked before output allocation, then a second submission gathers typed
values. Inputs and outputs are ordinary dataframe columns; no CPU join, native
Metal library, shader source, or external dataframe ABI executes data logic.
"""
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, Series, Expr, col
from dataframe.accelerator.joins import DagTask, DAG_JOIN
from dataframe.execution_report import ExecutionReport
from .runtime import (
    _kind,
    _width,
    _stage,
    _checked_add,
    _checked_mul,
    materialize,
    CELL_BYTES,
    _preflight as preflight_rows,
    execute_plan as execute_rows,
)
from .residency import Scope, Matrix, Table
from .text_domains import MapPlan, Submission, submit as submit_text
from dataframe.accelerator.code_domains import TextField
from .runtime import _stage_category_column
from . import category_keys
from . import joins as j
from . import native_hash_join


@fieldwise_init
struct Layout(Movable):
    var columns: List[Int64]
    var keys: List[Int64]
    var outputs: List[Int64]
    var cells: Int
    var capacity: Int
    var probe: Int
    var build: Int
    var side: Int
    var key_count: Int


def preflight(
    task: DagTask,
    left: DataFrame,
    right: DataFrame,
    left_rows: Int = -1,
    right_rows: Int = -1,
) raises -> Layout:
    """Validate schemas and descriptor bounds without accessing row values."""
    if task.kind != DAG_JOIN or task.how < 0 or task.how > 7 or task.limit < -1:
        raise Error("Mojo GPU invalid join task metadata")
    if left_rows < -1 or right_rows < -1:
        raise Error("Mojo GPU invalid join source row override")
    var sizes: List[Int] = [
        left.height() if left_rows < 0 else left_rows,
        right.height() if right_rows < 0 else right_rows,
    ]
    if sizes[0] > Int(Int32.MAX) or sizes[1] > Int(Int32.MAX):
        raise Error("Mojo GPU join input exceeds supported device index range")
    if (
        len(task.fields) < 1
        or len(task.fields) > 256
        or left.width() + right.width() > 256
    ):
        raise Error("Mojo GPU join requires at most 256 fixed-width columns")
    if (
        len(task.keys) % 2
        or len(task.keys) > 512
        or ((task.how == 6) != (len(task.keys) == 0))
    ):
        raise Error("Mojo GPU invalid join key metadata")
    var columns = List[Int64]()
    var cells = 0
    var capacity = max(1, max(sizes[0], sizes[1]))
    for side in range(2):
        var frame = right.copy() if side else left.copy()
        for column in frame._columns:
            columns.extend(
                [Int64(cells), Int64(sizes[side]), _kind(column.dtype())]
            )
            cells = _checked_add(cells, capacity)
    for key in range(0, len(task.keys), 2):
        var l = Int(task.keys[key])
        var r = Int(task.keys[key + 1]) - left.width()
        if l < 0 or l >= left.width() or r < 0 or r >= right.width():
            raise Error("Mojo GPU join key source is outside its input")
        if left._columns[l].dtype() != right._columns[r].dtype():
            raise Error("Mojo GPU join key dtypes differ")
    var outputs = List[Int64]()
    for slot in range(len(task.fields)):
        ref field = task.fields[slot]
        _ = _kind(field.dtype)
        if (
            field.left < -1
            or field.right < -1
            or (field.left < 0 and field.right < 0)
        ):
            raise Error("Mojo GPU invalid join output source")
        if field.left >= 0:
            if (
                field.left >= left.width()
                or left._columns[field.left].dtype() != field.dtype
            ):
                raise Error(
                    "Mojo GPU join left output dtype or source mismatch"
                )
        if field.right >= 0:
            var r = field.right - left.width()
            if (
                r < 0
                or r >= right.width()
                or right._columns[r].dtype() != field.dtype
            ):
                raise Error(
                    "Mojo GPU join right output dtype or source mismatch"
                )
        if field.prefer_right and field.right < 0:
            raise Error("Mojo GPU preferred right join output has no source")
        if (task.how == 4 or task.how == 5) and field.right >= 0:
            raise Error("Mojo GPU semi/anti output must come from the left")
        outputs.extend(
            [
                Int64(field.left),
                Int64(field.right),
                Int64(field.prefer_right),
                Int64(slot),
            ]
        )
    var keys = task.keys.copy()
    var key_count = len(keys) // 2
    if len(keys) == 0:
        keys.append(0)  # Device buffers have at least one typed element.
    return Layout(
        columns^,
        keys^,
        outputs^,
        cells,
        capacity,
        sizes[1] if task.how == 2 else sizes[0],
        sizes[0] if task.how == 2 else sizes[1],
        Int(task.how == 2),
        key_count,
    )


def execute_plan(
    task: DagTask,
    left: DataFrame,
    right: DataFrame,
    scope: Scope,
    profiling: Bool = False,
    left_resident: Optional[Table] = None,
    right_resident: Optional[Table] = None,
    export_values: Bool = True,
    var text_remap: Optional[MapPlan] = None,
    source_text: List[TextField] = List[TextField](),
) raises -> Tuple[Table, DataFrame]:
    var started = Int(perf_counter_ns())
    var before = scope.transfers()
    var text_submission = Optional[Submission]()
    var category_submission = category_keys.Submission(task)
    var layout = preflight(
        task,
        left,
        right,
        left_resident.value().rows if left_resident else -1,
        right_resident.value().rows if right_resident else -1,
    )
    if layout.probe == 0 and layout.build == 0:
        # Both producers have already been observed by the DAG. Every join
        # style (including validated/full/cross) has exactly zero rows here;
        # no match/count/gather data or device context is required. Reuse the
        # context-free empty row executor for typed schema and report fields.
        if text_remap:
            text_remap.value().validate_shape(scope, 0, 0, 0, 0)
        var columns = List[Expr]()
        for name in task.schema.columns():
            columns.append(col(name))
        var empty_plan = preflight_rows(
            task.schema.clear().lazy().select_exprs(columns)
        )
        var empty = execute_rows(
            empty_plan, scope, profiling=profiling, export_values=export_values
        )
        var report = (
            empty[1]
            .with_column(Series("node", Column[Int64]([Int64(task.root)])))
            .with_column(Series("operator", Column[String](["JOIN"])))
            .with_column(
                Series(
                    "algorithm",
                    Column[String](
                        [
                            "mojo_gpu_cross_join" if task.how
                            == 6 else "mojo_gpu_sorted_join"
                        ]
                    ),
                )
            )
        )
        return (empty[0].copy(), report^)
    var native_hash = native_hash_join.try_execute(
        task,
        left,
        right,
        scope,
        profiling,
        left_resident,
        right_resident,
        Bool(text_remap) or len(source_text) > 0,
        layout.probe,
        layout.build,
        layout.side,
    )
    if native_hash.used:
        return (native_hash.table.take(), native_hash.report.take())
    var resident_workspace = _checked_mul(
        (left.width() if left_resident else 0)
        + (right.width() if right_resident else 0),
        48,
    )
    var descriptors = _checked_mul(
        len(layout.columns) + len(layout.keys) + len(layout.outputs), 8
    )
    var matrices = _checked_mul(layout.cells, CELL_BYTES)
    var shared = _checked_add(
        _checked_add(matrices, descriptors),
        j.workspace_bytes(layout.probe, layout.build, task.how),
    )
    var host = _checked_add(
        _checked_add(matrices, descriptors), j.METADATA_WORDS * 8
    )
    var peak = _checked_add(_checked_add(shared, host), resident_workspace)
    if text_remap:
        peak = _checked_add(peak, text_remap.value().bytes())
    _ = scope.check(peak)
    var host_cells = (
        layout.cells if not left_resident or not right_resident else 0
    )
    var words = List[UInt64](length=2 * host_cells, fill=0)
    var validity = List[UInt8](length=host_cells, fill=0)
    var errors = List[UInt32](length=host_cells, fill=0)
    var native_input_conversion_bytes = 0
    var column = 0
    for side in range(2):
        var frame = right.copy() if side else left.copy()
        var resident = right_resident.copy() if side else left_resident.copy()
        for source in frame._columns:
            if not resident:
                _stage(
                    source,
                    layout.cells,
                    0,
                    Int(layout.columns[column * j.COLUMN_WORDS]),
                    words,
                    validity,
                )
            if not resident:
                for field in source_text:
                    if field.name == String(column):
                        _stage_category_column(
                            field,
                            len(source),
                            Int(layout.columns[column * j.COLUMN_WORDS]),
                            words,
                        )
                        break
            column += 1
    var ctx = scope.context()
    # Host staging and layout owners survive every failed submission.
    try:
        var device_words = ctx.enqueue_create_buffer[DType.uint64](
            2 * layout.cells
        )
        var device_validity = ctx.enqueue_create_buffer[DType.uint8](
            layout.cells
        )
        var device_errors = ctx.enqueue_create_buffer[DType.uint32](
            layout.cells
        )
        var device_columns = ctx.enqueue_create_buffer[DType.int64](
            len(layout.columns)
        )
        var device_keys = ctx.enqueue_create_buffer[DType.int64](
            len(layout.keys)
        )
        var device_outputs = ctx.enqueue_create_buffer[DType.int64](
            len(layout.outputs)
        )
        var index_count = max(1, layout.build if task.how != 6 else 0)
        var index = ctx.enqueue_create_buffer[DType.uint32](index_count)
        var scratch = ctx.enqueue_create_buffer[DType.uint32](index_count)
        var ranges = ctx.enqueue_create_buffer[DType.uint64](
            max(1, layout.probe)
        )
        var prefix = ctx.enqueue_create_buffer[DType.uint64](
            max(1, layout.probe)
        )
        var pb = max(1, (layout.probe + 255) // 256)
        var bb = max(1, (layout.build + 255) // 256)
        var blocks = ctx.enqueue_create_buffer[DType.uint64](pb)
        var full = task.how == 3
        var differences = ctx.enqueue_create_buffer[DType.int32](
            max(1, layout.build + 1 if full else 0)
        )
        var build_prefix = ctx.enqueue_create_buffer[DType.uint64](
            max(1, layout.build if full else 0)
        )
        var build_blocks = ctx.enqueue_create_buffer[DType.uint64](
            bb if full else 1
        )
        var flags = ctx.enqueue_create_buffer[DType.uint32](
            max(1, layout.build if full else 0)
        )
        var unmatched = ctx.enqueue_create_buffer[DType.uint32](
            max(1, layout.build if full else 0)
        )
        var metadata = ctx.enqueue_create_buffer[DType.uint64](j.METADATA_WORDS)
        if host_cells:
            ctx.enqueue_copy(device_words, words.unsafe_ptr())
            ctx.enqueue_copy(device_validity, validity.unsafe_ptr())
            ctx.enqueue_copy(device_errors, errors.unsafe_ptr())
            scope.record_upload(host_cells * CELL_BYTES)
        for side in range(2):
            var resident = (
                right_resident.copy() if side else left_resident.copy()
            )
            if resident:
                var frame = right.copy() if side else left.copy()
                var mapping = List[Int64](capacity=frame.width())
                for column in frame._columns:
                    var found = -1
                    for i in range(resident.value().schema.width()):
                        ref field = resident.value().schema._columns[i]
                        if (
                            field.name() == column.name()
                            and field.dtype() == column.dtype()
                        ):
                            found = i
                            break
                    if found < 0:
                        raise Error(
                            "Mojo GPU resident join source field mismatch"
                        )
                    mapping.append(Int64(found))
                var copied = resident.value().copy_to(
                    scope,
                    device_words,
                    device_validity,
                    device_errors,
                    layout.capacity,
                    left.width() + right.width(),
                    mapping,
                    left.width() if side else 0,
                )
                if len(resident.value().native_columns):
                    native_input_conversion_bytes = _checked_add(
                        native_input_conversion_bytes, copied
                    )
        if text_remap:
            text_submission = submit_text(
                scope,
                text_remap.take(),
                device_words,
                device_validity,
                device_errors,
                layout.capacity,
            )
        if text_submission:
            category_submission.prepare(
                scope,
                task,
                source_text,
                text_submission.value().host.target,
                layout.columns,
                device_words,
                device_validity,
                False,
                peak,
            )
        ctx.enqueue_copy(device_columns, layout.columns.unsafe_ptr())
        ctx.enqueue_copy(device_keys, layout.keys.unsafe_ptr())
        ctx.enqueue_copy(device_outputs, layout.outputs.unsafe_ptr())
        scope.record_upload(descriptors)
        var launches = 0
        if task.how != 6:
            ctx.enqueue_function[j.join_init](
                index.unsafe_ptr(),
                UInt32(layout.build),
                grid_dim=bb,
                block_dim=256,
            )
            launches += 1
            var width = 1
            while width < layout.build:
                ctx.enqueue_function[j.join_merge](
                    device_words.unsafe_ptr(),
                    device_validity.unsafe_ptr(),
                    index.unsafe_ptr(),
                    scratch.unsafe_ptr(),
                    device_columns.unsafe_ptr(),
                    device_keys.unsafe_ptr(),
                    UInt32(layout.key_count),
                    UInt32(layout.side),
                    UInt32(layout.build),
                    UInt32(width),
                    UInt64(layout.cells),
                    grid_dim=bb,
                    block_dim=256,
                )
                var previous = index^
                index = scratch^
                scratch = previous^
                width *= 2
                launches += 1
        if full:
            ctx.enqueue_function[j.join_zero](
                differences.unsafe_ptr(),
                UInt32(layout.build),
                grid_dim=(layout.build + 256) // 256,
                block_dim=256,
            )
            launches += 1
        ctx.enqueue_function[j.join_match](
            device_words.unsafe_ptr(),
            device_validity.unsafe_ptr(),
            index.unsafe_ptr(),
            device_columns.unsafe_ptr(),
            device_keys.unsafe_ptr(),
            ranges.unsafe_ptr(),
            prefix.unsafe_ptr(),
            differences.unsafe_ptr(),
            UInt32(layout.key_count),
            UInt32(layout.side),
            UInt32(layout.probe),
            UInt32(layout.build),
            UInt32(task.how),
            UInt32(task.nulls_equal or task.how == 7),
            UInt64(layout.cells),
            grid_dim=pb,
            block_dim=256,
        )
        ctx.enqueue_function[j.join_scan](
            prefix.unsafe_ptr(),
            blocks.unsafe_ptr(),
            differences.unsafe_ptr(),
            flags.unsafe_ptr(),
            UInt32(layout.probe),
            UInt32(0),
            grid_dim=pb,
            block_dim=256,
        )
        var limit = UInt64(task.limit) if task.limit >= 0 else UInt64.MAX
        ctx.enqueue_function[j.join_offsets](
            blocks.unsafe_ptr(),
            metadata.unsafe_ptr(),
            UInt32(pb),
            UInt32(0),
            limit,
            grid_dim=1,
            block_dim=1,
        )
        launches += 3
        if full:
            ctx.enqueue_function[j.join_scan](
                build_prefix.unsafe_ptr(),
                build_blocks.unsafe_ptr(),
                differences.unsafe_ptr(),
                flags.unsafe_ptr(),
                UInt32(layout.build),
                UInt32(1),
                grid_dim=bb,
                block_dim=256,
            )
            ctx.enqueue_function[j.join_offsets](
                build_blocks.unsafe_ptr(),
                metadata.unsafe_ptr(),
                UInt32(bb),
                UInt32(1),
                limit,
                grid_dim=1,
                block_dim=1,
            )
            ctx.enqueue_function[j.join_coverage](
                build_prefix.unsafe_ptr(),
                build_blocks.unsafe_ptr(),
                differences.unsafe_ptr(),
                index.unsafe_ptr(),
                flags.unsafe_ptr(),
                UInt32(layout.build),
                grid_dim=bb,
                block_dim=256,
            )
            ctx.enqueue_function[j.join_scan](
                build_prefix.unsafe_ptr(),
                build_blocks.unsafe_ptr(),
                differences.unsafe_ptr(),
                flags.unsafe_ptr(),
                UInt32(layout.build),
                UInt32(2),
                grid_dim=bb,
                block_dim=256,
            )
            ctx.enqueue_function[j.join_offsets](
                build_blocks.unsafe_ptr(),
                metadata.unsafe_ptr(),
                UInt32(bb),
                UInt32(2),
                limit,
                grid_dim=1,
                block_dim=1,
            )
            ctx.enqueue_function[j.join_unmatched](
                build_prefix.unsafe_ptr(),
                build_blocks.unsafe_ptr(),
                flags.unsafe_ptr(),
                unmatched.unsafe_ptr(),
                UInt32(layout.build),
                grid_dim=bb,
                block_dim=256,
            )
            launches += 6
        var shape = scope.read_control(metadata)
        scope.record_download(j.METADATA_WORDS * 8)
        # Only the requested prefix must fit device indices. Full fanout may
        # exceed Int32.MAX when a head bounds the observable output.
        if shape[0] > UInt64(Int32.MAX):
            raise Error(
                "Mojo GPU join output exceeds supported device index range"
            )
        var rows = Int(shape[0])
        var capacity = max(1, rows)
        var output_cells = _checked_mul(capacity, len(task.fields))
        var matrix_result_bytes = _checked_mul(output_cells, CELL_BYTES)
        shared = _checked_add(shared, matrix_result_bytes)
        var typed = 0
        if export_values:
            host = _checked_add(host, matrix_result_bytes)
            for field in task.fields:
                typed = _checked_add(
                    typed, _checked_mul(rows, _width(field.dtype))
                )
                typed = _checked_add(typed, (rows + 7) // 8)
        # Error-only observers download every result error, but allocate no
        # host value/validity matrix or typed output columns.
        var materialization_scratch = _checked_mul(
            rows, 4
        ) if export_values else _checked_mul(
            _checked_mul(rows, len(task.fields)), 4
        )
        peak = _checked_add(
            _checked_add(shared, host),
            _checked_add(typed, materialization_scratch),
        )
        peak = _checked_add(peak, resident_workspace)
        peak = _checked_add(peak, category_submission.bytes())
        _ = scope.check(peak)
        var result_words = ctx.enqueue_create_buffer[DType.uint64](
            2 * output_cells
        )
        var result_validity = ctx.enqueue_create_buffer[DType.uint8](
            output_cells
        )
        var result_errors = ctx.enqueue_create_buffer[DType.uint32](
            output_cells
        )
        var waits = 1
        if rows:
            category_submission.enqueue(
                scope, device_words, device_validity, True
            )
            ctx.enqueue_function[j.join_gather](
                device_words.unsafe_ptr(),
                device_validity.unsafe_ptr(),
                device_errors.unsafe_ptr(),
                device_columns.unsafe_ptr(),
                device_outputs.unsafe_ptr(),
                index.unsafe_ptr(),
                ranges.unsafe_ptr(),
                prefix.unsafe_ptr(),
                blocks.unsafe_ptr(),
                unmatched.unsafe_ptr(),
                metadata.unsafe_ptr(),
                result_words.unsafe_ptr(),
                result_validity.unsafe_ptr(),
                result_errors.unsafe_ptr(),
                UInt32(layout.probe),
                UInt32(layout.side),
                UInt32(task.how),
                UInt32(len(task.fields)),
                UInt64(layout.cells),
                grid_dim=(rows + 255) // 256,
                block_dim=256,
            )
            scope.synchronize()
            launches += 1
            waits += 1
        var schema_columns = List[Series](capacity=len(task.fields))
        var output_slots = List[Int64](capacity=len(task.fields))
        for slot in range(len(task.fields)):
            schema_columns.append(task.schema._columns[slot].copy())
            output_slots.append(Int64(slot))
        var result = Table(
            Matrix(
                scope,
                capacity,
                len(task.fields),
                result_words^,
                result_validity^,
                result_errors^,
            ),
            DataFrame(schema_columns^),
            output_slots^,
            rows,
        )
        category_submission.attach(result)
        if left_resident:
            result.category_leases.extend(
                left_resident.value().category_leases.copy()
            )
        if right_resident:
            result.category_leases.extend(
                right_resident.value().category_leases.copy()
            )
        # Non-key fields and ASOF outputs retain their exact source owner.
        for output in range(len(task.fields)):
            if result.categories[output]:
                continue
            ref field = task.fields[output]
            var source = (
                field.right if field.prefer_right
                or field.left < 0 else field.left
            )
            for metadata in source_text:
                if metadata.name == String(source):
                    if (
                        metadata.category
                        and metadata.category.value().string_codes
                        and task.how == 3
                        and field.left >= 0
                        and field.right >= 0
                    ):
                        break
                    result.categories[output] = metadata.category.copy()
                    break
        scope.record_launches(launches)
        var traffic = scope.transfers()
        launches = traffic[2] - before[2]
        waits = traffic[3] - before[3]
        var report = ExecutionReport()
        report.record(
            task.root,
            "JOIN",
            "metal",
            layout.probe,
            rows,
            build_rows=layout.build,
            builds=Int(task.how != 6),
            build_side="left" if layout.side else "right",
            algorithm="mojo_gpu_cross_join" if task.how
            == 6 else "mojo_gpu_sorted_join",
            wall_ns=Int(perf_counter_ns()) - started,
        )
        var fields = report.frame()._columns.copy()
        fields.append(
            Series("kernel_launches", Column[Int64]([Int64(launches)]))
        )
        fields.append(Series("synchronizations", Column[Int64]([Int64(waits)])))
        fields.append(
            Series(
                "peak_requested_bytes",
                Column[Int64]([Int64(max(peak, scope.info()[1]))]),
            )
        )
        fields.append(
            Series("runtime_implementation", Column[String](["mojo"]))
        )
        fields.append(
            Series(
                "upload_bytes", Column[Int64]([Int64(traffic[0] - before[0])])
            )
        )
        fields.append(
            Series(
                "download_bytes", Column[Int64]([Int64(traffic[1] - before[1])])
            )
        )
        fields.append(Series("host_result_bytes", Column[Int64]([Int64(0)])))
        fields.append(
            Series(
                "resident_input_bytes",
                Column[Int64](
                    [
                        Int64(
                            (
                                left_resident.value().retained_bytes() if left_resident else 0
                            )
                            + (
                                right_resident.value().retained_bytes() if right_resident else 0
                            )
                        )
                    ]
                ),
            )
        )
        fields.append(
            Series(
                "resident_output_bytes",
                Column[Int64]([Int64(result.retained_bytes())]),
            )
        )
        fields.append(
            Series(
                "native_input_conversion_bytes",
                Column[Int64]([Int64(native_input_conversion_bytes)]),
            )
        )
        fields.append(
            Series(
                "query_wall_ms",
                Column[Float64](
                    [Float64(Int(perf_counter_ns()) - started) / 1e6]
                ),
            )
        )
        var report_frame = DataFrame(fields^)
        # Keep raw-pointer upload owners alive until all queued reads finish.
        _ = words^
        _ = validity^
        _ = errors^
        _ = layout^
        _ = text_submission^
        _ = category_submission^
        return (result^, report_frame^)
    except error:
        scope.synchronize()
        _ = text_submission^
        _ = category_submission^
        _ = words^
        _ = validity^
        _ = errors^
        _ = layout^
        raise error^


def execute(
    task: DagTask,
    left: DataFrame,
    right: DataFrame,
    device_id: Int,
    budget: Int,
    profiling: Bool,
) raises -> Tuple[DataFrame, DataFrame]:
    var started = Int(perf_counter_ns())
    _ = preflight(task, left, right)
    var scope = Scope(device_id, budget)
    var executed = execute_plan(task, left, right, scope, profiling)
    var result = materialize(executed[0])
    var traffic = scope.transfers()
    var report = executed[1].copy()
    report = report.with_column(
        Series("kernel_launches", Column[Int64]([Int64(traffic[2])]))
    )
    report = report.with_column(
        Series("synchronizations", Column[Int64]([Int64(traffic[3])]))
    )
    report = report.with_column(
        Series("upload_bytes", Column[Int64]([Int64(traffic[0])]))
    )
    report = report.with_column(
        Series("download_bytes", Column[Int64]([Int64(traffic[1])]))
    )
    var host_result_bytes = 0
    for column in result._columns:
        host_result_bytes = _checked_add(
            host_result_bytes,
            _checked_add(
                _checked_mul(result.height(), _width(column.dtype())),
                (result.height() + 7) // 8,
            ),
        )
    report = report.with_column(
        Series("host_result_bytes", Column[Int64]([Int64(host_result_bytes)]))
    )
    report = report.with_column(
        Series("resident_input_bytes", Column[Int64]([Int64(0)]))
    )
    report = report.with_column(
        Series("resident_output_bytes", Column[Int64]([Int64(0)]))
    )
    report = report.with_column(
        Series(
            "query_wall_ms",
            Column[Float64]([Float64(Int(perf_counter_ns()) - started) / 1e6]),
        )
    )
    return (result^, report^)
