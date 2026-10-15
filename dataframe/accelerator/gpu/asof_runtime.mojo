"""Resident Mojo as-of tasks; only control metadata crosses the host boundary."""
from std.memory import bitcast
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, Series
from dataframe.lazy import ASOF
from dataframe.asof import _tolerance
from dataframe.accelerator.joins import DagTask, DAG_JOIN, DAG_ASOF
from dataframe.execution_report import ExecutionReport
from . import asof as a
from . import joins as j
from . import join_runtime as common
from .runtime import _width, _stage, _checked_add, _checked_mul, CELL_BYTES
from .residency import Scope, Matrix, Table, raise_value_error
from .text_domains import MapPlan, Submission, submit as submit_text
from dataframe.accelerator.code_domains import TextField
from .runtime import _stage_category_column
from . import category_keys


@fieldwise_init
struct Layout(Movable):
    var data: common.Layout
    var ordered_left: Int64
    var ordered_right: Int64
    var exact: UInt32
    var bounded: UInt32
    var tolerance_integer: UInt64
    var tolerance_float: UInt64


def preflight(
    task: DagTask,
    left: DataFrame,
    right: DataFrame,
    left_rows: Int = -1,
    right_rows: Int = -1,
) raises -> Layout:
    if task.kind != DAG_ASOF or task.how < 0 or task.how > 2 or task.equal_rows:
        raise Error("Mojo GPU invalid as-of task metadata")
    if not len(task.query._nodes):
        raise Error("Mojo GPU as-of options are missing")
    ref node = task.query._nodes[len(task.query._nodes) - 1]
    if (
        node.kind != ASOF
        or len(node.names) != 1
        or len(node.right_keys) != 1
        or not len(node.flags)
    ):
        raise Error("Mojo GPU invalid as-of ordered-key metadata")
    var ordered_left = left._index(node.names[0])
    var ordered_right = right._index(node.right_keys[0])
    var dtype = left._columns[ordered_left].dtype()
    if dtype != right._columns[ordered_right].dtype():
        raise Error("join_asof key dtypes must match")
    if not (
        dtype.is_integer()
        or dtype.is_float()
        or dtype.is_date()
        or dtype.is_datetime()
        or dtype.is_duration()
    ):
        raise Error(
            "join_asof requires integer, float, Date, Datetime or Duration keys"
        )
    var bound = _tolerance(node.asof_tolerance, dtype)
    # Share the already verified fixed-width descriptor/schema validation.
    # ASOF without BY is a single ordered group, not a Cartesian operation.
    var checked = task.copy()
    checked.kind = DAG_JOIN
    checked.how = 0 if len(task.keys) else 6
    var layout = common.preflight(checked, left, right, left_rows, right_rows)
    return Layout(
        layout^,
        Int64(ordered_left),
        Int64(left.width() + ordered_right),
        UInt32(node.flags[0]),
        UInt32(bound[0]),
        min(bound[1], Int128(UInt64.MAX)).cast[DType.uint64](),
        bitcast[DType.uint64](bound[2]),
    )


def _report(
    task: DagTask,
    scope: Scope,
    before: Tuple[Int, Int, Int, Int],
    started: Int,
    left: Int,
    right: Int,
    rows: Int,
    peak: Int,
    resident_input: Int,
    resident_output: Int,
) raises -> DataFrame:
    var traffic = scope.transfers()
    var report = ExecutionReport()
    report.record(
        task.root,
        "ASOF",
        "metal",
        left,
        rows,
        build_rows=right,
        builds=2 if left or right else 0,
        build_side="both",
        algorithm="mojo_gpu_asof_backward" if task.how
        == 0 else "mojo_gpu_asof_forward" if task.how
        == 1 else "mojo_gpu_asof_nearest",
        wall_ns=Int(perf_counter_ns()) - started,
    )
    var fields = report.frame()._columns.copy()
    fields.append(
        Series(
            "kernel_launches", Column[Int64]([Int64(traffic[2] - before[2])])
        )
    )
    fields.append(
        Series(
            "synchronizations", Column[Int64]([Int64(traffic[3] - before[3])])
        )
    )
    fields.append(
        Series(
            "peak_requested_bytes",
            Column[Int64]([Int64(max(peak, scope.info()[1]))]),
        )
    )
    fields.append(Series("runtime_implementation", Column[String](["mojo"])))
    fields.append(
        Series("upload_bytes", Column[Int64]([Int64(traffic[0] - before[0])]))
    )
    fields.append(
        Series("download_bytes", Column[Int64]([Int64(traffic[1] - before[1])]))
    )
    fields.append(Series("host_result_bytes", Column[Int64]([Int64(0)])))
    fields.append(
        Series("resident_input_bytes", Column[Int64]([Int64(resident_input)]))
    )
    fields.append(
        Series("resident_output_bytes", Column[Int64]([Int64(resident_output)]))
    )
    fields.append(
        Series(
            "query_wall_ms",
            Column[Float64]([Float64(Int(perf_counter_ns()) - started) / 1e6]),
        )
    )
    return DataFrame(fields^)


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
    var prepared = preflight(
        task,
        left,
        right,
        left_resident.value().rows if left_resident else -1,
        right_resident.value().rows if right_resident else -1,
    )
    ref layout = prepared.data
    var left_rows = layout.probe
    var right_rows = layout.build
    var rows = left_rows if task.limit < 0 else min(left_rows, task.limit)
    var resident_bytes = (
        left_resident.value().retained_bytes() if left_resident else 0
    ) + (right_resident.value().retained_bytes() if right_resident else 0)
    if left_rows == 0 and right_rows == 0:
        var result = Table(task.schema.copy())
        var report = _report(
            task, scope, before, started, 0, 0, 0, 0, resident_bytes, 0
        )
        return (result^, report^)
    var capacity = max(1, rows)
    var output_cells = _checked_mul(capacity, len(task.fields))
    var output_bytes = _checked_mul(output_cells, CELL_BYTES)
    var descriptors = _checked_mul(
        len(layout.columns) + len(layout.keys) + len(layout.outputs), 8
    )
    var host_cells = (
        layout.cells if not left_resident or not right_resident else 0
    )
    var input_bytes = _checked_mul(layout.cells, CELL_BYTES)
    var peak = _checked_add(input_bytes, _checked_mul(host_cells, CELL_BYTES))
    peak = _checked_add(peak, _checked_mul(descriptors, 2))
    peak = _checked_add(peak, a.workspace_bytes(left_rows, right_rows))
    peak = _checked_add(peak, a.CONTROL_WORDS * 4)
    peak = _checked_add(
        peak, _checked_mul(output_bytes, 2 if export_values else 1)
    )
    var resident_workspace = _checked_mul(
        (left.width() if left_resident else 0)
        + (right.width() if right_resident else 0),
        48,
    )
    peak = _checked_add(peak, resident_workspace)
    var typed = _checked_mul(rows, 4) if export_values else _checked_mul(
        _checked_mul(rows, len(task.fields)), 4
    )
    if export_values:
        for field in task.fields:
            typed = _checked_add(
                typed,
                _checked_add(
                    _checked_mul(rows, _width(field.dtype)), (rows + 7) // 8
                ),
            )
    peak = _checked_add(peak, typed)
    if text_remap:
        peak = _checked_add(peak, text_remap.value().bytes())
    _ = scope.check(peak)
    var words = List[UInt64](length=2 * host_cells, fill=0)
    var validity = List[UInt8](length=host_cells, fill=0)
    var errors = List[UInt32](length=host_cells, fill=0)
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
    # Allocate owners before submissions, so catch synchronizes before any
    # input, descriptor or result buffer can unwind on an enqueue failure.
    var dw = ctx.enqueue_create_buffer[DType.uint64](2 * layout.cells)
    var dv = ctx.enqueue_create_buffer[DType.uint8](layout.cells)
    var de = ctx.enqueue_create_buffer[DType.uint32](layout.cells)
    var dc = ctx.enqueue_create_buffer[DType.int64](len(layout.columns))
    var dk = ctx.enqueue_create_buffer[DType.int64](len(layout.keys))
    var dout = ctx.enqueue_create_buffer[DType.int64](len(layout.outputs))
    var li = ctx.enqueue_create_buffer[DType.uint32](max(1, left_rows))
    var ln = ctx.enqueue_create_buffer[DType.uint32](max(1, left_rows))
    var ri = ctx.enqueue_create_buffer[DType.uint32](max(1, right_rows))
    var rn = ctx.enqueue_create_buffer[DType.uint32](max(1, right_rows))
    var lc = ctx.enqueue_create_buffer[DType.uint64](max(1, left_rows))
    var rc = ctx.enqueue_create_buffer[DType.uint64](max(1, right_rows))
    var ends = ctx.enqueue_create_buffer[DType.uint32](max(1, right_rows))
    var pairs = ctx.enqueue_create_buffer[DType.uint64](max(1, left_rows))
    var control = ctx.enqueue_create_buffer[DType.uint32](a.CONTROL_WORDS)
    var ow = ctx.enqueue_create_buffer[DType.uint64](2 * output_cells)
    var ov = ctx.enqueue_create_buffer[DType.uint8](output_cells)
    var oe = ctx.enqueue_create_buffer[DType.uint32](output_cells)
    # Matrix adoption moves the output handles later in this block. Separate
    # references keep their allocations alive on every exception edge too.
    var output_words_owner = ow.copy()
    var output_validity_owner = ov.copy()
    var output_errors_owner = oe.copy()
    try:
        if host_cells:
            ctx.enqueue_copy(dw, words.unsafe_ptr())
            ctx.enqueue_copy(dv, validity.unsafe_ptr())
            ctx.enqueue_copy(de, errors.unsafe_ptr())
            scope.record_upload(host_cells * CELL_BYTES)
        for side in range(2):
            var resident = (
                right_resident.copy() if side else left_resident.copy()
            )
            if resident:
                var frame = right.copy() if side else left.copy()
                var mapping = List[Int64](capacity=frame.width())
                for field in frame._columns:
                    var found = -1
                    for index in range(resident.value().schema.width()):
                        ref candidate = resident.value().schema._columns[index]
                        if (
                            candidate.name() == field.name()
                            and candidate.dtype() == field.dtype()
                        ):
                            found = index
                            break
                    if found < 0:
                        raise Error("Mojo GPU resident as-of field mismatch")
                    mapping.append(Int64(found))
                _ = resident.value().copy_to(
                    scope,
                    dw,
                    dv,
                    de,
                    layout.capacity,
                    left.width() + right.width(),
                    mapping,
                    left.width() if side else 0,
                )
        if text_remap:
            text_submission = submit_text(
                scope,
                text_remap.take(),
                dw,
                dv,
                de,
                layout.capacity,
            )
        if text_submission:
            category_submission.prepare(
                scope,
                task,
                source_text,
                text_submission.value().host.target,
                layout.columns,
                dw,
                dv,
                True,
                peak,
            )
        ctx.enqueue_copy(dc, layout.columns.unsafe_ptr())
        ctx.enqueue_copy(dk, layout.keys.unsafe_ptr())
        ctx.enqueue_copy(dout, layout.outputs.unsafe_ptr())
        scope.record_upload(descriptors)
        var lg = max(1, (left_rows + 255) // 256)
        var rg = max(1, (right_rows + 255) // 256)
        ctx.enqueue_function[j.join_init](
            li.unsafe_ptr(), UInt32(left_rows), grid_dim=lg, block_dim=256
        )
        ctx.enqueue_function[j.join_init](
            ri.unsafe_ptr(), UInt32(right_rows), grid_dim=rg, block_dim=256
        )
        var launches = 2
        var width = 1
        while layout.key_count and width < max(left_rows, right_rows):
            if width < left_rows:
                ctx.enqueue_function[j.join_merge](
                    dw.unsafe_ptr(),
                    dv.unsafe_ptr(),
                    li.unsafe_ptr(),
                    ln.unsafe_ptr(),
                    dc.unsafe_ptr(),
                    dk.unsafe_ptr(),
                    UInt32(layout.key_count),
                    UInt32(1),
                    UInt32(left_rows),
                    UInt32(width),
                    UInt64(layout.cells),
                    grid_dim=lg,
                    block_dim=256,
                )
                var previous = li^
                li = ln^
                ln = previous^
                launches += 1
            if width < right_rows:
                ctx.enqueue_function[j.join_merge](
                    dw.unsafe_ptr(),
                    dv.unsafe_ptr(),
                    ri.unsafe_ptr(),
                    rn.unsafe_ptr(),
                    dc.unsafe_ptr(),
                    dk.unsafe_ptr(),
                    UInt32(layout.key_count),
                    UInt32(0),
                    UInt32(right_rows),
                    UInt32(width),
                    UInt64(layout.cells),
                    grid_dim=rg,
                    block_dim=256,
                )
                var previous = ri^
                ri = rn^
                rn = previous^
                launches += 1
            width *= 2
        ctx.enqueue_function[a.asof_validate](
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            de.unsafe_ptr(),
            li.unsafe_ptr(),
            dc.unsafe_ptr(),
            dk.unsafe_ptr(),
            lc.unsafe_ptr(),
            prepared.ordered_left,
            UInt32(layout.key_count),
            UInt32(1),
            UInt32(left_rows),
            UInt64(layout.cells),
            grid_dim=lg,
            block_dim=256,
        )
        ctx.enqueue_function[a.asof_validate](
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            de.unsafe_ptr(),
            ri.unsafe_ptr(),
            dc.unsafe_ptr(),
            dk.unsafe_ptr(),
            rc.unsafe_ptr(),
            prepared.ordered_right,
            UInt32(layout.key_count),
            UInt32(0),
            UInt32(right_rows),
            UInt64(layout.cells),
            grid_dim=rg,
            block_dim=256,
        )
        ctx.enqueue_function[a.asof_control](
            lc.unsafe_ptr(),
            rc.unsafe_ptr(),
            control.unsafe_ptr(),
            UInt32(left_rows),
            UInt32(right_rows),
            grid_dim=1,
            block_dim=1,
        )
        launches += 3
        scope.record_launches(launches)
        var checked = scope.read_control(control)
        scope.record_download(a.CONTROL_WORDS * 4)
        if checked[2]:
            raise_value_error(
                checked[2],
                "Mojo GPU arithmetic overflow or strict cast failed in as-of input",
            )
        ref node = task.query._nodes[len(task.query._nodes) - 1]
        if checked[0] != j.MISSING:
            raise Error(
                "join_asof left column '"
                + node.names[0]
                + "' is not sorted within its group (row "
                + String(checked[0])
                + ")"
            )
        if checked[1] != j.MISSING:
            raise Error(
                "join_asof right column '"
                + node.right_keys[0]
                + "' is not sorted within its group (row "
                + String(checked[1])
                + ")"
            )
        launches = 0
        if rows:
            if task.how == 2:
                ctx.enqueue_function[a.asof_right_ends](
                    dw.unsafe_ptr(),
                    dv.unsafe_ptr(),
                    ri.unsafe_ptr(),
                    dc.unsafe_ptr(),
                    dk.unsafe_ptr(),
                    ends.unsafe_ptr(),
                    prepared.ordered_right,
                    UInt32(layout.key_count),
                    UInt32(right_rows),
                    UInt64(layout.cells),
                    grid_dim=rg,
                    block_dim=256,
                )
                launches += 1
            ctx.enqueue_function[a.asof_match](
                dw.unsafe_ptr(),
                dv.unsafe_ptr(),
                li.unsafe_ptr(),
                ri.unsafe_ptr(),
                ends.unsafe_ptr(),
                dc.unsafe_ptr(),
                dk.unsafe_ptr(),
                pairs.unsafe_ptr(),
                prepared.ordered_left,
                prepared.ordered_right,
                UInt32(layout.key_count),
                UInt32(left_rows),
                UInt32(right_rows),
                UInt32(task.how),
                prepared.exact,
                prepared.bounded,
                prepared.tolerance_integer,
                prepared.tolerance_float,
                UInt64(layout.cells),
                grid_dim=lg,
                block_dim=256,
            )
            category_submission.enqueue(scope, dw, dv, True)
            ctx.enqueue_function[a.asof_gather](
                dw.unsafe_ptr(),
                dv.unsafe_ptr(),
                de.unsafe_ptr(),
                dc.unsafe_ptr(),
                dout.unsafe_ptr(),
                pairs.unsafe_ptr(),
                ow.unsafe_ptr(),
                ov.unsafe_ptr(),
                oe.unsafe_ptr(),
                UInt32(rows),
                UInt32(len(task.fields)),
                UInt64(layout.cells),
                grid_dim=(rows + 255) // 256,
                block_dim=256,
            )
            launches += 2
            scope.synchronize()
        var slots = List[Int64]()
        for slot in range(len(task.fields)):
            slots.append(Int64(slot))
        var result = Table(
            Matrix(scope, capacity, len(task.fields), ow^, ov^, oe^),
            task.schema.copy(),
            slots^,
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
                    result.categories[output] = metadata.category.copy()
                    break
        scope.record_launches(launches)
        var report = _report(
            task,
            scope,
            before,
            started,
            left_rows,
            right_rows,
            rows,
            peak,
            resident_bytes,
            result.retained_bytes(),
        )
        _ = words^
        _ = validity^
        _ = errors^
        _ = prepared^
        _ = dw^
        _ = dv^
        _ = de^
        _ = dc^
        _ = dk^
        _ = dout^
        _ = li^
        _ = ln^
        _ = ri^
        _ = rn^
        _ = lc^
        _ = rc^
        _ = ends^
        _ = pairs^
        _ = control^
        _ = output_words_owner^
        _ = output_validity_owner^
        _ = output_errors_owner^
        _ = text_submission^
        _ = category_submission^
        return (result^, report^)
    except error:
        scope.synchronize()
        _ = text_submission^
        _ = category_submission^
        _ = words^
        _ = validity^
        _ = errors^
        _ = prepared^
        _ = dw^
        _ = dv^
        _ = de^
        _ = dc^
        _ = dk^
        _ = dout^
        _ = li^
        _ = ln^
        _ = ri^
        _ = rn^
        _ = lc^
        _ = rc^
        _ = ends^
        _ = pairs^
        _ = control^
        _ = output_words_owner^
        _ = output_validity_owner^
        _ = output_errors_owner^
        raise error^
