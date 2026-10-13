"""Direct Mojo host orchestration for shared fixed-width join tasks.

The first submission learns limited cardinality. The complete requested peak
is checked before output allocation, then a second submission gathers typed
values. Inputs and outputs are ordinary dataframe columns; no CPU join, native
Metal library, shader source, or external dataframe ABI executes data logic.
"""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, Series
from dataframe.accelerator.joins import DagTask, DAG_JOIN
from dataframe.execution_report import ExecutionReport
from .runtime import _kind, _width, _stage, _output, _checked_add, _checked_mul
from . import joins as j


@fieldwise_init
struct Layout(Movable):
    var columns: List[Int64]
    var keys: List[Int64]
    var outputs: List[Int64]
    var cells: Int
    var probe: Int
    var build: Int
    var side: Int
    var key_count: Int


def preflight(
    task: DagTask, left: DataFrame, right: DataFrame
) raises -> Layout:
    """Validate schemas and descriptor bounds without accessing row values."""
    if task.kind != DAG_JOIN or task.how < 0 or task.how > 7 or task.limit < -1:
        raise Error("Mojo GPU invalid join task metadata")
    if left.height() > Int(Int32.MAX) or right.height() > Int(Int32.MAX):
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
    for side in range(2):
        var frame = right.copy() if side else left.copy()
        for column in frame._columns:
            columns.extend(
                [Int64(cells), Int64(frame.height()), _kind(column.dtype())]
            )
            cells = _checked_add(cells, max(1, frame.height()))
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
        right.height() if task.how == 2 else left.height(),
        left.height() if task.how == 2 else right.height(),
        Int(task.how == 2),
        key_count,
    )


def _check_budget(peak: Int, budget: Int) raises:
    if budget >= 0 and peak > budget:
        raise Error("Mojo GPU join exceeds memory budget before submission")


def execute(
    task: DagTask,
    left: DataFrame,
    right: DataFrame,
    device_id: Int,
    budget: Int,
    profiling: Bool,
) raises -> Tuple[DataFrame, DataFrame]:
    var started = Int(perf_counter_ns())
    var layout = preflight(task, left, right)
    var descriptors = _checked_mul(
        len(layout.columns) + len(layout.keys) + len(layout.outputs), 8
    )
    var matrices = _checked_mul(layout.cells, 13)
    var shared = _checked_add(
        _checked_add(matrices, descriptors),
        j.workspace_bytes(layout.probe, layout.build, task.how),
    )
    var host = _checked_add(
        _checked_add(matrices, descriptors), j.METADATA_WORDS * 8
    )
    var peak = _checked_add(shared, host)
    _check_budget(peak, budget)
    var words = List[UInt64](length=layout.cells, fill=0)
    var validity = List[UInt8](length=layout.cells, fill=0)
    var errors = List[UInt32](length=layout.cells, fill=0)
    var column = 0
    for side in range(2):
        var frame = right.copy() if side else left.copy()
        for source in frame._columns:
            _stage(
                source,
                layout.cells,
                0,
                Int(layout.columns[column * j.COLUMN_WORDS]),
                words,
                validity,
            )
            column += 1
    var ctx = DeviceContext(device_id)
    var device_words = ctx.enqueue_create_buffer[DType.uint64](layout.cells)
    var device_validity = ctx.enqueue_create_buffer[DType.uint8](layout.cells)
    var device_errors = ctx.enqueue_create_buffer[DType.uint32](layout.cells)
    var device_columns = ctx.enqueue_create_buffer[DType.int64](
        len(layout.columns)
    )
    var device_keys = ctx.enqueue_create_buffer[DType.int64](len(layout.keys))
    var device_outputs = ctx.enqueue_create_buffer[DType.int64](
        len(layout.outputs)
    )
    var index_count = max(1, layout.build if task.how != 6 else 0)
    var index = ctx.enqueue_create_buffer[DType.uint32](index_count)
    var scratch = ctx.enqueue_create_buffer[DType.uint32](index_count)
    var ranges = ctx.enqueue_create_buffer[DType.uint64](max(1, layout.probe))
    var prefix = ctx.enqueue_create_buffer[DType.uint64](max(1, layout.probe))
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
    ctx.enqueue_copy(device_words, words.unsafe_ptr())
    ctx.enqueue_copy(device_validity, validity.unsafe_ptr())
    ctx.enqueue_copy(device_errors, errors.unsafe_ptr())
    ctx.enqueue_copy(device_columns, layout.columns.unsafe_ptr())
    ctx.enqueue_copy(device_keys, layout.keys.unsafe_ptr())
    ctx.enqueue_copy(device_outputs, layout.outputs.unsafe_ptr())
    var launches = 0
    if task.how != 6:
        ctx.enqueue_function[j.join_init](
            index.unsafe_ptr(), UInt32(layout.build), grid_dim=bb, block_dim=256
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
        UInt32(task.how == 7),
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
    var shape = List[UInt64](length=j.METADATA_WORDS, fill=0)
    ctx.enqueue_copy(shape.unsafe_ptr(), metadata)
    ctx.synchronize()
    # Only the requested prefix must fit device indices. Full fanout may
    # exceed Int32.MAX when a head bounds the observable output.
    if shape[0] > UInt64(Int32.MAX):
        raise Error("Mojo GPU join output exceeds supported device index range")
    var rows = Int(shape[0])
    var capacity = max(1, rows)
    var output_cells = _checked_mul(capacity, len(task.fields))
    var matrix_result_bytes = _checked_mul(output_cells, 13)
    shared = _checked_add(shared, matrix_result_bytes)
    host = _checked_add(host, matrix_result_bytes)
    var typed = 0
    for field in task.fields:
        typed = _checked_add(typed, _checked_mul(rows, _width(field.dtype)))
        typed = _checked_add(typed, (rows + 7) // 8)
    peak = _checked_add(
        _checked_add(shared, host), _checked_add(typed, _checked_mul(rows, 4))
    )
    _check_budget(peak, budget)
    var output_words = List[UInt64](length=output_cells, fill=0)
    var output_validity = List[UInt8](length=output_cells, fill=0)
    var output_errors = List[UInt32](length=output_cells, fill=0)
    var result_words = ctx.enqueue_create_buffer[DType.uint64](output_cells)
    var result_validity = ctx.enqueue_create_buffer[DType.uint8](output_cells)
    var result_errors = ctx.enqueue_create_buffer[DType.uint32](output_cells)
    var waits = 1
    if rows:
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
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_copy(output_words.unsafe_ptr(), result_words)
        ctx.enqueue_copy(output_validity.unsafe_ptr(), result_validity)
        ctx.enqueue_copy(output_errors.unsafe_ptr(), result_errors)
        ctx.synchronize()
        launches += 1
        waits += 1
    var columns = List[Series](capacity=len(task.fields))
    for slot in range(len(task.fields)):
        for row in range(rows):
            if output_errors[slot * capacity + row]:
                raise Error(
                    "Mojo GPU join input expression failed checked arithmetic or cast"
                )
        ref field = task.fields[slot]
        columns.append(
            _output(
                field.name,
                field.dtype,
                slot,
                capacity,
                rows,
                output_words,
                output_validity,
            )
        )
    var result = DataFrame(columns^, height=rows)
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
    fields.append(Series("kernel_launches", Column[Int64]([Int64(launches)])))
    fields.append(Series("synchronizations", Column[Int64]([Int64(waits)])))
    fields.append(Series("peak_requested_bytes", Column[Int64]([Int64(peak)])))
    fields.append(Series("runtime_implementation", Column[String](["mojo"])))
    return (result^, DataFrame(fields^))
