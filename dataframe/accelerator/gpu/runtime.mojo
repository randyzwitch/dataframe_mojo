"""Direct Mojo execution of shared bound dataframe row plans.

No native dataframe library, C ABI descriptors, or shader source is involved.
The compiler emits GPU code from `rows.mojo`; DeviceContext owns the queue,
buffers, transfers and synchronization. Unsupported regions reject before
initializing a context while the remaining native operations are migrated.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.time import perf_counter_ns
from std.sys import size_of
from dataframe import Column, DataFrame, DataType, Series
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.expr import (
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_NULL,
    ADD,
    SUB,
    MUL,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    SQRT,
    EXP,
    LOG,
    FLOOR,
    CEIL,
    ROUND,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    FILL_NAN,
    CLIP_LOW,
    CLIP_HIGH,
    CAST,
    GT,
    EQ,
    LT,
    GE,
    LE,
    NE,
    AND,
    OR,
    XOR,
    FILL_NULL,
    NEG,
    ABS,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    WHEN,
)
from dataframe.lazy import LazyFrame
from dataframe.execution_report import ExecutionReport
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import RowPlan, lower_rows, ROW_SLICE
from .rows import row_program
from .movement import filter_scan, filter_offsets, filter_gather


def _kind(dtype: DataType) raises -> Int64:
    if dtype == DataType.BOOL:
        return 4
    var physical = dtype.physical()
    var types: List[DataType] = [
        DataType.FLOAT32,
        DataType.INT32,
        DataType.INT64,
        DataType.BOOL,
        DataType.INT8,
        DataType.INT16,
        DataType.UINT8,
        DataType.UINT16,
        DataType.UINT32,
        DataType.UINT64,
        DataType.FLOAT64,
    ]
    for i in range(len(types)):
        if physical == types[i]:
            return Int64(i + 1)
    raise Error("Mojo GPU storage is not implemented for " + dtype.name())


def _preflight(query: LazyFrame) raises -> RowPlan:
    var capabilities = RowCapabilities(
        "Mojo GPU",
        float64=False,
        float64_storage=True,
        float64_emulated=True,
        float_division=True,
        float_modulo=True,
        float_transcendentals=True,
        float_power=True,
        float_sqrt=True,
        float_rounding=True,
        casts=True,
        wide_integer=False,
        int64_arithmetic=True,
        extended_integers=True,
        mixed_types=True,
        row_extras=True,
        slices=True,
        fixed_logical=True,
        chunked_inputs=True,
        empty_inputs=True,
        empty_outputs=True,
        max_nodes=128,
        max_slots=256,
        max_steps=128,
    )
    var plan = lower_rows(query, capabilities)
    if plan.source.height() > Int(Int32.MAX):
        raise Error(
            "Mojo GPU row count exceeds the supported device index range"
        )
    if plan.reductions:
        raise Error("Mojo GPU reductions are not implemented yet")
    for dtype in plan.slot_dtypes:
        _ = _kind(dtype)
    var supported: List[Int] = [
        ROW_SLICE,
        COL,
        LIT_INT,
        LIT_FLOAT,
        LIT_BOOL,
        LIT_NULL,
        ADD,
        SUB,
        MUL,
        DIV,
        FLOORDIV,
        MOD,
        POW,
        SQRT,
        EXP,
        LOG,
        FLOOR,
        CEIL,
        ROUND,
        IS_NAN,
        IS_NOT_NAN,
        IS_FINITE,
        IS_INFINITE,
        FILL_NAN,
        CLIP_LOW,
        CLIP_HIGH,
        CAST,
        GT,
        EQ,
        LT,
        GE,
        LE,
        NE,
        AND,
        OR,
        XOR,
        FILL_NULL,
        NEG,
        ABS,
        NOT,
        IS_NULL,
        IS_NOT_NULL,
        WHEN,
    ]
    for step in plan.steps:
        if step.order or step.grouped:
            raise Error("Mojo GPU ordering and groups are not implemented yet")
        for node in range(step.start, step.start + step.nodes):
            var op = Int(plan.code[4 * node])
            if op not in supported:
                raise Error(
                    "Mojo GPU operation is not implemented: " + String(op)
                )
            if op == ADD or op == SUB or op == MUL or op == NEG or op == ABS:
                if not (
                    plan.node_dtypes[node].is_integer()
                    or plan.node_dtypes[node].is_float()
                ):
                    raise Error(
                        "Mojo GPU logical arithmetic is not implemented yet"
                    )
            if (
                op == DIV
                or op == FLOORDIV
                or op == MOD
                or op == POW
                or op == SQRT
                or op == EXP
                or op == LOG
                or op == FLOOR
                or op == CEIL
                or op == ROUND
            ):
                if not plan.node_dtypes[node].is_float():
                    raise Error(
                        "Mojo GPU integer division and rounding are not implemented yet"
                    )
            if op == CAST:
                var source = step.start + Int(plan.code[4 * node + 1])
                if not (
                    plan.node_dtypes[source].is_numeric()
                    or plan.node_dtypes[source] == DataType.BOOL
                ) or not (
                    plan.node_dtypes[node].is_numeric()
                    or plan.node_dtypes[node] == DataType.BOOL
                ):
                    raise Error(
                        "Mojo GPU logical casts are not implemented yet"
                    )
    return plan^


def _checked_add(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0 or a > Int.MAX - b:
        raise Error("Mojo GPU allocation size overflow")
    return a + b


def _checked_mul(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0 or (b and a > Int.MAX // b):
        raise Error("Mojo GPU allocation size overflow")
    return a * b


def _width(dtype: DataType) raises -> Int:
    if dtype == DataType.BOOL:
        return 1
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if dtype.physical() == DataType.of(D):
            return size_of[Scalar[D]]()
    raise Error("Mojo GPU unsupported output width")


def _stage(
    column: Series,
    rows: Int,
    slot: Int,
    begin: Int,
    mut words: List[UInt64],
    mut validity: List[UInt8],
) raises:
    if column.is_chunked():
        var offset = begin
        for part in column.chunks():
            _stage(part, rows, slot, offset, words, validity)
            offset += len(part)
        return
    var base = slot * rows + begin
    if column.dtype() == DataType.BOOL:
        var input = column.bool()
        for row in range(len(column)):
            var valid = input._valid(row)
            validity[base + row] = UInt8(valid)
            if valid:
                words[base + row] = UInt64(input._get(row))
        return
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if column.dtype().physical() == DataType.of(D):
            ref input = column._data[Column[Scalar[D]]]
            for row in range(len(column)):
                var valid = input._valid(row)
                validity[base + row] = UInt8(valid)
                if valid:
                    comptime if D == DType.float64:
                        words[base + row] = bitcast[DType.uint64](
                            input._get(row)
                        )
                    elif D == DType.float32:
                        words[base + row] = UInt64(
                            bitcast[DType.uint32](input._get(row))
                        )
                    else:
                        words[base + row] = input._get(row).cast[DType.uint64]()
            return
    raise Error("Mojo GPU unsupported input storage")


def _output(
    name: String,
    dtype: DataType,
    slot: Int,
    rows: Int,
    count: Int,
    words: List[UInt64],
    validity: List[UInt8],
) raises -> Series:
    var base = slot * rows
    var valid = List[Bool](capacity=count)
    for row in range(count):
        valid.append(validity[base + row] != 0)
    if dtype == DataType.BOOL:
        var values = List[Bool](capacity=count)
        for row in range(count):
            values.append(valid[row] and words[base + row] != 0)
        return Series(name, Column[Bool](values^, valid^))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if dtype.physical() == DataType.of(D):
            var values = List[Scalar[D]](capacity=count)
            for row in range(count):
                var value = words[base + row] if valid[row] else UInt64(0)
                comptime if D == DType.float64:
                    values.append(bitcast[DType.float64](value).cast[D]())
                elif D == DType.float32:
                    values.append(
                        bitcast[DType.float32](UInt32(value)).cast[D]()
                    )
                else:
                    values.append(value.cast[D]())
            return Series(name, Column[Scalar[D]](values^, valid^)).with_dtype(
                dtype
            )
    raise Error("Mojo GPU unsupported output storage")


def execute(
    query: LazyFrame,
    device_id: Int,
    budget: Int,
    profiling: Bool,
) raises -> Tuple[DataFrame, DataFrame]:
    var started = Int(perf_counter_ns())
    var plan = _preflight(query)
    var rows = plan.source.height()
    var count = rows if plan.limit < 0 else min(rows, plan.limit)
    var cells = _checked_mul(rows, plan.slots)
    var current_rows = rows
    var movement = False
    for step in plan.steps:
        movement = movement or step.filter
    # GPU words, byte validity and per-value errors; one host staging matrix
    # is reused for readback after the stream consumes its upload.
    var matrix_bytes = _checked_mul(cells, 13)
    var program_bytes = _checked_mul(len(plan.code) + 2 * len(plan.literals), 8)
    var shared = _checked_add(matrix_bytes, program_bytes) if rows else 0
    if rows and movement:
        # A second matrix prevents overlapping reads/writes during compaction.
        # Scans use row positions, two block arrays and a two-word count/error
        # readback. Gather metadata is bounded by the full plan's gather list.
        shared = _checked_add(shared, matrix_bytes)
        shared = _checked_add(shared, _checked_mul(rows, 4))
        shared = _checked_add(shared, _checked_mul((rows + 255) // 256, 8))
        shared = _checked_add(shared, 16)
        shared = _checked_add(shared, _checked_mul(len(plan.gathers), 16))
    var result_bytes = _checked_add(matrix_bytes, program_bytes) if rows else 0
    for output in plan.outputs:
        result_bytes = _checked_add(
            result_bytes, _checked_mul(count, _width(output.dtype))
        )
        result_bytes = _checked_add(result_bytes, (count + 7) // 8)
    # Column construction temporarily holds unpacked Boolean validity and,
    # for Boolean outputs, both byte and packed representations. Charge a
    # conservative upper bound for the largest one-column materialization.
    var scratch = _checked_mul(count, 4) if len(plan.outputs) else 0
    var peak = _checked_add(_checked_add(shared, result_bytes), scratch)
    if budget >= 0 and peak > budget:
        raise Error("Mojo GPU request exceeds memory budget before submission")
    var words = List[UInt64](length=cells, fill=0)
    var validity = List[UInt8](length=cells, fill=0)
    var errors = List[UInt32](length=cells, fill=0)
    var code = plan.code.copy()
    var literals = List[UInt64](capacity=len(plan.literals))
    var types = List[Int64](capacity=len(plan.node_dtypes))
    for node in range(len(plan.literals)):
        var op = Int(code[4 * node])
        var dtype = plan.node_dtypes[node]
        var value = bitcast[DType.uint64](plan.literals[node])
        if op == LIT_BOOL:
            value = UInt64(plan.literals[node] != 0)
        elif op == LIT_INT and dtype.is_float():
            var integer = bitcast[DType.int64](plan.literals[node])
            if dtype == DataType.FLOAT32:
                value = UInt64(bitcast[DType.uint32](Float32(integer)))
            else:
                value = bitcast[DType.uint64](Float64(integer))
        elif op == LIT_FLOAT and dtype == DataType.FLOAT32:
            value = UInt64(bitcast[DType.uint32](Float32(plan.literals[node])))
        literals.append(value)
        types.append(_kind(dtype))
    for slot in range(plan.source.width()):
        _stage(plan.source._columns[slot], rows, slot, 0, words, validity)
    var submissions = 0
    var waits = 0
    if rows and cells:
        var ctx = DeviceContext(device_id)
        var device_words = ctx.enqueue_create_buffer[DType.uint64](cells)
        var device_valid = ctx.enqueue_create_buffer[DType.uint8](cells)
        var device_errors = ctx.enqueue_create_buffer[DType.uint32](cells)
        # Zero-step projections need no program buffers or kernel launches.
        if len(plan.steps):
            var device_code = ctx.enqueue_create_buffer[DType.int64](len(code))
            var device_literals = ctx.enqueue_create_buffer[DType.uint64](
                len(literals)
            )
            var device_types = ctx.enqueue_create_buffer[DType.int64](
                len(types)
            )
            ctx.enqueue_copy(device_words, words.unsafe_ptr())
            ctx.enqueue_copy(device_valid, validity.unsafe_ptr())
            ctx.enqueue_copy(device_errors, errors.unsafe_ptr())
            ctx.enqueue_copy(device_code, code.unsafe_ptr())
            ctx.enqueue_copy(device_literals, literals.unsafe_ptr())
            ctx.enqueue_copy(device_types, types.unsafe_ptr())
            for step in plan.steps:
                if current_rows == 0:
                    break
                ctx.enqueue_function[row_program](
                    device_words.unsafe_ptr(),
                    device_valid.unsafe_ptr(),
                    device_errors.unsafe_ptr(),
                    device_code.unsafe_ptr(),
                    device_literals.unsafe_ptr(),
                    device_types.unsafe_ptr(),
                    UInt32(step.start),
                    UInt32(step.nodes),
                    UInt32(step.slot),
                    UInt32(current_rows),
                    UInt32(rows),
                    grid_dim=(current_rows + 255) // 256,
                    block_dim=256,
                )
                submissions += 1
                if step.filter:
                    var blocks = (current_rows + 255) // 256
                    var positions = ctx.enqueue_create_buffer[DType.uint32](
                        current_rows
                    )
                    var offsets = ctx.enqueue_create_buffer[DType.uint32](
                        blocks * 2
                    )
                    var metadata = ctx.enqueue_create_buffer[DType.uint32](2)
                    var gathered_words = ctx.enqueue_create_buffer[
                        DType.uint64
                    ](cells)
                    var gathered_valid = ctx.enqueue_create_buffer[DType.uint8](
                        cells
                    )
                    var gathered_errors = ctx.enqueue_create_buffer[
                        DType.uint32
                    ](cells)
                    ctx.enqueue_function[filter_scan](
                        device_words.unsafe_ptr(),
                        device_valid.unsafe_ptr(),
                        device_errors.unsafe_ptr(),
                        positions.unsafe_ptr(),
                        offsets.unsafe_ptr(),
                        UInt32(step.slot),
                        UInt32(current_rows),
                        UInt32(rows),
                        grid_dim=blocks,
                        block_dim=256,
                    )
                    ctx.enqueue_function[filter_offsets](
                        offsets.unsafe_ptr(),
                        metadata.unsafe_ptr(),
                        UInt32(blocks),
                        grid_dim=1,
                        block_dim=1,
                    )
                    submissions += 2
                    var shape = List[UInt32](length=2, fill=0)
                    ctx.enqueue_copy(shape.unsafe_ptr(), metadata)
                    ctx.synchronize()
                    waits += 1
                    if shape[1]:
                        raise Error(
                            "Mojo GPU filter predicate failed checked arithmetic or cast"
                        )
                    if shape[0] and step.gather_count:
                        var gathers = List[Int64](capacity=step.gather_count)
                        for column in range(step.gather_count):
                            gathers.append(
                                plan.gathers[step.gather_start + column]
                            )
                        var device_gathers = ctx.enqueue_create_buffer[
                            DType.int64
                        ](len(gathers))
                        ctx.enqueue_copy(device_gathers, gathers.unsafe_ptr())
                        ctx.enqueue_function[filter_gather](
                            device_words.unsafe_ptr(),
                            device_valid.unsafe_ptr(),
                            device_errors.unsafe_ptr(),
                            gathered_words.unsafe_ptr(),
                            gathered_valid.unsafe_ptr(),
                            gathered_errors.unsafe_ptr(),
                            positions.unsafe_ptr(),
                            offsets.unsafe_ptr(),
                            device_gathers.unsafe_ptr(),
                            UInt32(step.gather_count),
                            UInt32(step.slot),
                            UInt32(current_rows),
                            UInt32(rows),
                            grid_dim=blocks,
                            block_dim=256,
                        )
                        submissions += 1
                        # Complete gathers before replacing their source owners.
                        ctx.synchronize()
                        waits += 1
                    device_words = gathered_words^
                    device_valid = gathered_valid^
                    device_errors = gathered_errors^
                    current_rows = Int(shape[0])
            ctx.enqueue_copy(words.unsafe_ptr(), device_words)
            ctx.enqueue_copy(validity.unsafe_ptr(), device_valid)
            ctx.enqueue_copy(errors.unsafe_ptr(), device_errors)
            ctx.synchronize()
            waits += 1
        else:
            ctx.enqueue_copy(device_words, words.unsafe_ptr())
            ctx.enqueue_copy(device_valid, validity.unsafe_ptr())
            ctx.enqueue_copy(words.unsafe_ptr(), device_words)
            ctx.enqueue_copy(validity.unsafe_ptr(), device_valid)
            ctx.synchronize()
            waits += 1
    count = current_rows if plan.limit < 0 else min(current_rows, plan.limit)
    for output in plan.outputs:
        for row in range(count):
            if errors[output.slot * rows + row]:
                raise Error(
                    "Mojo GPU arithmetic overflow or strict cast failure"
                )
    var columns = List[Series](capacity=len(plan.outputs))
    for output in plan.outputs:
        columns.append(
            _output(
                output.name,
                output.dtype,
                output.slot,
                rows,
                count,
                words,
                validity,
            )
        )
    var result = DataFrame(columns^, height=count)
    var report = ExecutionReport()
    report.record(
        plan.root,
        "ROW EXPRESSIONS",
        "metal",
        rows,
        count,
        algorithm="mojo_gpu_row_program",
        wall_ns=Int(perf_counter_ns()) - started,
    )
    var fields = report.frame()._columns.copy()
    fields.append(
        Series("kernel_launches", Column[Int64]([Int64(submissions)]))
    )
    fields.append(Series("synchronizations", Column[Int64]([Int64(waits)])))
    fields.append(Series("peak_requested_bytes", Column[Int64]([Int64(peak)])))
    fields.append(Series("runtime_implementation", Column[String](["mojo"])))
    return (result^, DataFrame(fields^))


def describe(query: LazyFrame) -> String:
    try:
        var plan = _preflight(query)
        return (
            "ENGINE accel: Mojo GPU row expressions; "
            + String(len(plan.steps))
            + " steps; kernels and runtime execute directly in the dataframe library\n"
        )
    except error:
        return "ENGINE accel: " + String(error) + "\n"
