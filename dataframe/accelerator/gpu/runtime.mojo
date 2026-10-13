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
    SUM,
    MEAN,
    MIN,
    MAX,
    COUNT,
    LEN,
    NULL_COUNT,
    FIRST,
    LAST,
    ANY,
    ALL,
    ARG_MIN,
    ARG_MAX,
    VAR,
    STD,
    SKEW,
    KURTOSIS,
    COV,
    CORR,
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
    DT_YEAR,
    DT_MONTH,
    DT_DAY,
    DT_HOUR,
    DT_MINUTE,
    DT_SECOND,
    DT_NANOSECOND,
    DT_WEEKDAY,
    DT_ORDINAL_DAY,
    DT_DATE,
    DT_TIME,
    DT_TRUNCATE,
    DT_OFFSET_BY,
    DT_TOTAL,
    DT_REPLACE_TZ,
    DT_CONVERT_TZ,
)
from dataframe.lazy import LazyFrame
from dataframe.execution_report import ExecutionReport
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import (
    RowPlan,
    lower_rows,
    ROW_SLICE,
    ROW_SORT_KEY,
    ROW_RANK_KEY,
    ROW_CUMULATIVE_KEY,
    ROW_DURATION_KEY,
    ROW_DURATION_BOUNDS,
    ROW_DURATION_CALENDAR,
    ROW_DURATION_UTC,
    ROW_GROUP_HEADER,
    ROW_GROUP_HEADER_LAST,
    ROW_GROUP_KEY,
    ROW_GROUP_OUTPUT,
    ROW_CALENDAR_OFFSET,
    ROW_CALENDAR_TRUNCATE,
    ROW_QUANTILE_LINEAR,
    ROW_QUANTILE_LOWER,
    ROW_QUANTILE_HIGHER,
    ROW_QUANTILE_NEAREST,
    ROW_QUANTILE_MIDPOINT,
)
from .rows import row_program
from .movement import filter_scan, filter_offsets, filter_gather
from . import statistics, quantiles
from .grouping import group_mark, group_offsets, group_starts, group_reduce
from .ordering import order_init, order_merge, order_key_errors, order_gather
from .ranking import rank_order_merge, rank_prepare, rank_offsets, rank_assign
from .windows import (
    window_assign,
    window_check_errors,
    window_input_errors,
    duration_assign,
    duration_check_errors,
    duration_headers,
)
from .reductions import (
    State,
    STATE_WORDS,
    DESCRIPTOR_WORDS,
    reduce_partials,
    reduce_finish,
)


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


def capabilities() -> RowCapabilities:
    return RowCapabilities(
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
        bounded_integer_sum=True,
        extra_reductions=True,
        extrema=True,
        moments=True,
        higher_moments=True,
        paired_moments=True,
        quantiles=True,
        order=True,
        ranks=True,
        cumulative_windows=True,
        shift_fill_windows=True,
        rolling_windows=True,
        rolling_moments=True,
        interpolation_windows=True,
        coordinate_interpolation=True,
        duration_rolling=True,
        calendar_duration_rolling=True,
        utc_duration_rolling=True,
        fixed_zone_duration_rolling=True,
        grouped=True,
        int64_arithmetic=True,
        deferred_integer_errors=True,
        extended_integers=True,
        mixed_types=True,
        row_extras=True,
        slices=True,
        fixed_logical=True,
        temporal_fields=True,
        utc_temporal_conversions=True,
        duration_totals=True,
        temporal_offsets=True,
        calendar_offsets=True,
        calendar_truncation=True,
        chunked_inputs=True,
        empty_inputs=True,
        empty_outputs=True,
        max_nodes=128,
        max_slots=256,
        max_steps=128,
    )


def _preflight(
    query: LazyFrame, *, terminal_boundary: Bool = True
) raises -> RowPlan:
    var plan = lower_rows(
        query, capabilities(), terminal_boundary=terminal_boundary
    )
    if plan.source.height() > Int(Int32.MAX):
        raise Error(
            "Mojo GPU row count exceeds the supported device index range"
        )
    var supported_reductions: List[Int] = [
        SUM,
        MEAN,
        MIN,
        MAX,
        COUNT,
        LEN,
        NULL_COUNT,
        FIRST,
        LAST,
        ANY,
        ALL,
        ARG_MIN,
        ARG_MAX,
        VAR,
        STD,
        SKEW,
        KURTOSIS,
        COV,
        CORR,
        ROW_QUANTILE_LINEAR,
        ROW_QUANTILE_LOWER,
        ROW_QUANTILE_HIGHER,
        ROW_QUANTILE_NEAREST,
        ROW_QUANTILE_MIDPOINT,
    ]
    for output in plan.outputs:
        _ = _kind(output.dtype)
        if plan.reductions and output.reduction not in supported_reductions:
            raise Error(
                "Mojo GPU reduction is not implemented: "
                + String(output.reduction)
            )
    for dtype in plan.slot_dtypes:
        _ = _kind(dtype)
    var supported: List[Int] = [
        ROW_CALENDAR_OFFSET,
        ROW_CALENDAR_TRUNCATE,
        DT_YEAR,
        DT_MONTH,
        DT_DAY,
        DT_HOUR,
        DT_MINUTE,
        DT_SECOND,
        DT_NANOSECOND,
        DT_WEEKDAY,
        DT_ORDINAL_DAY,
        DT_DATE,
        DT_TIME,
        DT_TRUNCATE,
        DT_OFFSET_BY,
        DT_TOTAL,
        DT_REPLACE_TZ,
        DT_CONVERT_TZ,
        ROW_SLICE,
        ROW_SORT_KEY,
        ROW_RANK_KEY,
        ROW_CUMULATIVE_KEY,
        ROW_DURATION_KEY,
        ROW_DURATION_BOUNDS,
        ROW_DURATION_CALENDAR,
        ROW_DURATION_UTC,
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
        if step.grouped:
            var header = Int(plan.code[4 * step.start])
            if header != ROW_GROUP_HEADER and header != ROW_GROUP_HEADER_LAST:
                raise Error("Mojo GPU grouped windows are not implemented yet")
            var keys = Int(plan.code[4 * step.start + 1])
            if keys < 1 or keys >= step.nodes:
                raise Error("Mojo GPU grouping requires fixed-width keys")
            for offset in range(1, step.nodes):
                var node = step.start + offset
                var op = Int(plan.code[4 * node])
                if offset <= keys:
                    if op != ROW_GROUP_KEY:
                        raise Error("Mojo GPU invalid group key descriptor")
                elif (
                    op != ROW_GROUP_OUTPUT
                    or Int(plan.code[4 * node + 3]) not in supported_reductions
                ):
                    raise Error("Mojo GPU grouped reduction is not implemented")
            continue
        if step.order:
            var header = Int(plan.code[4 * step.start])
            var method = Int(plan.code[4 * step.start + 3])
            if header == ROW_RANK_KEY:
                if method < 0 or method > 4 or step.gather_count:
                    raise Error(
                        "Mojo GPU invalid rank method or gather metadata"
                    )
            elif header == ROW_CUMULATIVE_KEY:
                if method < 0 or method > 15 or step.gather_count:
                    raise Error("Mojo GPU window method is not implemented")
            elif header == ROW_DURATION_KEY:
                if (
                    method < 0
                    or method > 5
                    or step.gather_count
                    or step.nodes < 2
                ):
                    raise Error(
                        "Mojo GPU duration window descriptor is not implemented"
                    )
                if plan.code[4 * (step.start + 1)] != ROW_DURATION_BOUNDS:
                    raise Error(
                        "Mojo GPU duration window bounds metadata is invalid"
                    )
            elif header != ROW_SORT_KEY:
                raise Error("Mojo GPU ordering descriptor is not implemented")
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
    *,
    terminal_boundary: Bool = True,
) raises -> Tuple[DataFrame, DataFrame]:
    var started = Int(perf_counter_ns())
    var plan = _preflight(query, terminal_boundary=terminal_boundary)
    var rows = plan.source.height()
    var count = rows if plan.limit < 0 else min(rows, plan.limit)
    if plan.reductions:
        count = 0 if plan.limit == 0 else 1
    var cells = _checked_mul(rows, plan.slots)
    var current_rows = rows
    var movement = False
    var ordering = False
    var grouping = False
    var ranking = False
    var windows = False
    var duration_windows = False
    for step in plan.steps:
        ordering = ordering or step.order or step.grouped
        grouping = grouping or step.grouped
        if step.order and not step.grouped:
            var header = Int(plan.code[4 * step.start])
            ranking = ranking or header == ROW_RANK_KEY
            windows = (
                windows
                or header == ROW_CUMULATIVE_KEY
                or header == ROW_DURATION_KEY
            )
            duration_windows = duration_windows or header == ROW_DURATION_KEY
        movement = movement or step.filter or step.order or step.grouped
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
        shared = _checked_add(shared, _checked_mul(rows, 8 if ordering else 4))
        shared = _checked_add(shared, _checked_mul((rows + 255) // 256, 8))
        shared = _checked_add(shared, 16)
        shared = _checked_add(shared, _checked_mul(len(plan.gathers), 16))
        if duration_windows:
            # Duration errors preserve four distinct fault categories on both
            # host and device, replacing the ordinary one-word block summary.
            shared = _checked_add(shared, _checked_mul((rows + 255) // 256, 24))
        if ranking:
            # The idle sort-index owner supplies dense prefixes. Only the
            # additional per-block dense counts need separate device storage.
            shared = _checked_add(shared, _checked_mul((rows + 255) // 256, 4))
        if grouping:
            # Group starts and block-local positions accompany both sort indices.
            shared = _checked_add(shared, _checked_mul(rows, 8))
            shared = _checked_add(shared, _checked_mul(plan.slots, 8))
    var result_bytes = _checked_add(matrix_bytes, program_bytes) if rows else 0
    if rows and grouping:
        result_bytes = _checked_add(result_bytes, _checked_mul(plan.slots, 8))
    if plan.reductions:
        var outputs = len(plan.outputs)
        # Descriptors and scalar results have both host and device storage.
        # Partials are bounded independently of the input row count.
        var descriptors = _checked_mul(outputs, DESCRIPTOR_WORDS * 8)
        var scalar_results = _checked_mul(outputs, 13)
        result_bytes = _checked_add(
            result_bytes, _checked_add(descriptors, scalar_results)
        )
        if rows:
            var groups = min(1024, max(1, (rows + 255) // 256))
            shared = _checked_add(
                shared,
                _checked_mul(_checked_mul(outputs, groups), STATE_WORDS * 8),
            )
            shared = _checked_add(
                shared, _checked_add(descriptors, scalar_results)
            )
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
    var reduction_code = List[Int64]()
    var has_statistics = False
    var has_quantiles = False
    if plan.reductions:
        for output in plan.outputs:
            has_statistics = has_statistics or statistics.supported(
                Int64(output.reduction)
            )
            has_quantiles = has_quantiles or quantiles.supported(
                Int64(output.reduction)
            )
            reduction_code.extend(
                [
                    Int64(output.slot),
                    _kind(plan.slot_dtypes[output.slot]),
                    _kind(output.dtype),
                    Int64(output.reduction),
                    Int64(output.min_count),
                ]
            )
    var scalar_count = len(plan.outputs) if plan.reductions else 0
    var reduced_words = List[UInt64](length=scalar_count, fill=0)
    var reduced_validity = List[UInt8](length=scalar_count, fill=0)
    var reduced_errors = List[UInt32](length=scalar_count, fill=0)
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
    var slot_types = List[Int64]()
    if grouping:
        for dtype in plan.slot_dtypes:
            slot_types.append(_kind(dtype))
    for slot in range(plan.source.width()):
        _stage(plan.source._columns[slot], rows, slot, 0, words, validity)
    var submissions = 0
    var waits = 0
    if rows and cells:
        var ctx = DeviceContext(device_id)
        var device_words = ctx.enqueue_create_buffer[DType.uint64](cells)
        var device_valid = ctx.enqueue_create_buffer[DType.uint8](cells)
        var device_errors = ctx.enqueue_create_buffer[DType.uint32](cells)
        ctx.enqueue_copy(device_words, words.unsafe_ptr())
        ctx.enqueue_copy(device_valid, validity.unsafe_ptr())
        ctx.enqueue_copy(device_errors, errors.unsafe_ptr())
        # Zero-step projections need no program buffers or kernel launches.
        if len(plan.steps):
            var device_code = ctx.enqueue_create_buffer[DType.int64](len(code))
            var device_literals = ctx.enqueue_create_buffer[DType.uint64](
                len(literals)
            )
            var device_types = ctx.enqueue_create_buffer[DType.int64](
                len(types)
            )
            ctx.enqueue_copy(device_code, code.unsafe_ptr())
            ctx.enqueue_copy(device_literals, literals.unsafe_ptr())
            ctx.enqueue_copy(device_types, types.unsafe_ptr())
            for step in plan.steps:
                if current_rows == 0:
                    break
                if step.grouped:
                    var blocks = (current_rows + 255) // 256
                    var keys = Int(code[4 * step.start + 1])
                    var index = ctx.enqueue_create_buffer[DType.uint32](
                        current_rows
                    )
                    var next_index = ctx.enqueue_create_buffer[DType.uint32](
                        current_rows
                    )
                    var positions = ctx.enqueue_create_buffer[DType.uint32](
                        current_rows
                    )
                    var starts = ctx.enqueue_create_buffer[DType.uint32](
                        current_rows
                    )
                    var offsets = ctx.enqueue_create_buffer[DType.uint32](
                        blocks * 2
                    )
                    var metadata = ctx.enqueue_create_buffer[DType.uint32](2)
                    var device_slot_types = ctx.enqueue_create_buffer[
                        DType.int64
                    ](len(slot_types))
                    ctx.enqueue_copy(device_slot_types, slot_types.unsafe_ptr())
                    ctx.enqueue_function[order_init](
                        index.unsafe_ptr(),
                        UInt32(current_rows),
                        grid_dim=blocks,
                        block_dim=256,
                    )
                    submissions += 1
                    var run_width = 1
                    while run_width < current_rows:
                        ctx.enqueue_function[order_merge](
                            device_words.unsafe_ptr(),
                            device_valid.unsafe_ptr(),
                            index.unsafe_ptr(),
                            next_index.unsafe_ptr(),
                            device_code.unsafe_ptr(),
                            device_types.unsafe_ptr(),
                            UInt32(step.start + 1),
                            UInt32(keys),
                            UInt32(current_rows),
                            UInt32(rows),
                            UInt32(run_width),
                            grid_dim=blocks,
                            block_dim=256,
                        )
                        var previous = index^
                        index = next_index^
                        next_index = previous^
                        run_width *= 2
                        submissions += 1
                    ctx.enqueue_function[group_mark](
                        device_words.unsafe_ptr(),
                        device_valid.unsafe_ptr(),
                        device_errors.unsafe_ptr(),
                        index.unsafe_ptr(),
                        positions.unsafe_ptr(),
                        offsets.unsafe_ptr(),
                        device_code.unsafe_ptr(),
                        device_types.unsafe_ptr(),
                        UInt32(step.start),
                        UInt32(keys),
                        UInt32(current_rows),
                        UInt32(rows),
                        grid_dim=blocks,
                        block_dim=256,
                    )
                    ctx.enqueue_function[group_offsets](
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
                            "Mojo GPU group key failed checked arithmetic or cast"
                        )
                    var gathered_words = ctx.enqueue_create_buffer[
                        DType.uint64
                    ](cells)
                    var gathered_valid = ctx.enqueue_create_buffer[DType.uint8](
                        cells
                    )
                    var gathered_errors = ctx.enqueue_create_buffer[
                        DType.uint32
                    ](cells)
                    if shape[0]:
                        ctx.enqueue_function[group_starts](
                            device_words.unsafe_ptr(),
                            device_valid.unsafe_ptr(),
                            index.unsafe_ptr(),
                            positions.unsafe_ptr(),
                            offsets.unsafe_ptr(),
                            starts.unsafe_ptr(),
                            device_code.unsafe_ptr(),
                            device_types.unsafe_ptr(),
                            UInt32(step.start),
                            UInt32(keys),
                            UInt32(current_rows),
                            UInt32(rows),
                            grid_dim=blocks,
                            block_dim=256,
                        )
                        ctx.enqueue_function[group_reduce](
                            device_words.unsafe_ptr(),
                            device_valid.unsafe_ptr(),
                            device_errors.unsafe_ptr(),
                            gathered_words.unsafe_ptr(),
                            gathered_valid.unsafe_ptr(),
                            gathered_errors.unsafe_ptr(),
                            index.unsafe_ptr(),
                            starts.unsafe_ptr(),
                            device_code.unsafe_ptr(),
                            device_literals.unsafe_ptr(),
                            device_slot_types.unsafe_ptr(),
                            UInt32(step.start),
                            UInt32(step.nodes),
                            shape[0],
                            UInt32(current_rows),
                            UInt32(rows),
                            grid_dim=(Int(shape[0]) * step.nodes + 255) // 256,
                            block_dim=256,
                        )
                        submissions += 2
                        # Starts, sort indices, and descriptor owners outlive all queued reads.
                        ctx.synchronize()
                        waits += 1
                    current_rows = Int(shape[0])
                    device_words = gathered_words^
                    device_valid = gathered_valid^
                    device_errors = gathered_errors^
                    continue
                var header = Int(code[4 * step.start])
                if step.order and (
                    header == ROW_RANK_KEY
                    or header == ROW_CUMULATIVE_KEY
                    or header == ROW_DURATION_KEY
                ):
                    var blocks = (current_rows + 255) // 256
                    var index = ctx.enqueue_create_buffer[DType.uint32](
                        current_rows
                    )
                    var next_index = ctx.enqueue_create_buffer[DType.uint32](
                        current_rows
                    )
                    var is_duration = header == ROW_DURATION_KEY
                    var headers = Int(
                        duration_headers(
                            code.unsafe_ptr()
                            .unsafe_mut_cast[False]()
                            .unsafe_origin_cast[ImmutAnyOrigin](),
                            UInt32(step.start),
                            UInt32(step.nodes),
                        )
                    ) if is_duration else 1
                    var failure_count = blocks * (4 if is_duration else 1)
                    var failures = ctx.enqueue_create_buffer[DType.uint32](
                        failure_count
                    )
                    var host_failures = List[UInt32](
                        length=failure_count, fill=0
                    )
                    ctx.enqueue_function[window_input_errors](
                        device_errors.unsafe_ptr(),
                        device_code.unsafe_ptr(),
                        failures.unsafe_ptr(),
                        UInt32(step.start),
                        UInt32(step.nodes),
                        UInt32(current_rows),
                        UInt32(rows),
                        grid_dim=blocks,
                        block_dim=256,
                    )
                    ctx.enqueue_function[order_init](
                        index.unsafe_ptr(),
                        UInt32(current_rows),
                        grid_dim=blocks,
                        block_dim=256,
                    )
                    submissions += 2
                    var run_width = 1
                    while run_width < current_rows:
                        if header == ROW_RANK_KEY:
                            ctx.enqueue_function[rank_order_merge](
                                device_words.unsafe_ptr(),
                                device_valid.unsafe_ptr(),
                                index.unsafe_ptr(),
                                next_index.unsafe_ptr(),
                                device_code.unsafe_ptr(),
                                device_types.unsafe_ptr(),
                                UInt32(step.start),
                                UInt32(step.nodes),
                                UInt32(current_rows),
                                UInt32(rows),
                                UInt32(run_width),
                                grid_dim=blocks,
                                block_dim=256,
                            )
                        else:
                            ctx.enqueue_function[order_merge](
                                device_words.unsafe_ptr(),
                                device_valid.unsafe_ptr(),
                                index.unsafe_ptr(),
                                next_index.unsafe_ptr(),
                                device_code.unsafe_ptr(),
                                device_types.unsafe_ptr(),
                                UInt32(step.start + headers),
                                UInt32(step.nodes - headers),
                                UInt32(current_rows),
                                UInt32(rows),
                                UInt32(run_width),
                                grid_dim=blocks,
                                block_dim=256,
                            )
                        var previous = index^
                        index = next_index^
                        next_index = previous^
                        run_width *= 2
                        submissions += 1
                    ctx.enqueue_copy(host_failures.unsafe_ptr(), failures)
                    ctx.synchronize()
                    waits += 1
                    for failure in host_failures:
                        if failure:
                            raise Error(
                                "Mojo GPU window key failed checked arithmetic or cast"
                            )
                    if header == ROW_RANK_KEY:
                        var dense_blocks = ctx.enqueue_create_buffer[
                            DType.uint32
                        ](blocks)
                        if code[4 * step.start + 3] == 3:
                            ctx.enqueue_function[rank_prepare](
                                device_words.unsafe_ptr(),
                                device_valid.unsafe_ptr(),
                                index.unsafe_ptr(),
                                next_index.unsafe_ptr(),
                                dense_blocks.unsafe_ptr(),
                                device_code.unsafe_ptr(),
                                device_types.unsafe_ptr(),
                                UInt32(step.start),
                                UInt32(step.nodes),
                                UInt32(current_rows),
                                UInt32(rows),
                                grid_dim=blocks,
                                block_dim=256,
                            )
                            ctx.enqueue_function[rank_offsets](
                                dense_blocks.unsafe_ptr(),
                                UInt32(blocks),
                                grid_dim=1,
                                block_dim=1,
                            )
                            submissions += 2
                        ctx.enqueue_function[rank_assign](
                            device_words.unsafe_ptr(),
                            device_valid.unsafe_ptr(),
                            device_errors.unsafe_ptr(),
                            index.unsafe_ptr(),
                            next_index.unsafe_ptr(),
                            dense_blocks.unsafe_ptr(),
                            device_code.unsafe_ptr(),
                            device_types.unsafe_ptr(),
                            UInt32(step.start),
                            UInt32(step.nodes),
                            UInt32(step.slot),
                            UInt32(current_rows),
                            UInt32(rows),
                            grid_dim=blocks,
                            block_dim=256,
                        )
                        submissions += 1
                        # Permutation, dense prefixes and block owners outlive
                        # their last queued reads before the next plan step.
                        ctx.synchronize()
                        waits += 1
                    else:
                        if is_duration:
                            ctx.enqueue_function[duration_assign](
                                device_words.unsafe_ptr(),
                                device_valid.unsafe_ptr(),
                                device_errors.unsafe_ptr(),
                                index.unsafe_ptr(),
                                device_code.unsafe_ptr(),
                                device_types.unsafe_ptr(),
                                device_literals.unsafe_ptr(),
                                UInt32(step.start),
                                UInt32(step.nodes),
                                UInt32(step.slot),
                                UInt32(_kind(plan.slot_dtypes[step.slot])),
                                UInt32(current_rows),
                                UInt32(rows),
                                grid_dim=blocks,
                                block_dim=256,
                            )
                            ctx.enqueue_function[duration_check_errors](
                                device_errors.unsafe_ptr(),
                                failures.unsafe_ptr(),
                                UInt32(step.slot),
                                UInt32(current_rows),
                                UInt32(rows),
                                grid_dim=blocks,
                                block_dim=256,
                            )
                        else:
                            ctx.enqueue_function[window_assign](
                                device_words.unsafe_ptr(),
                                device_valid.unsafe_ptr(),
                                device_errors.unsafe_ptr(),
                                index.unsafe_ptr(),
                                device_code.unsafe_ptr(),
                                device_types.unsafe_ptr(),
                                device_literals.unsafe_ptr(),
                                UInt32(step.start),
                                UInt32(step.nodes),
                                UInt32(step.slot),
                                UInt32(_kind(plan.slot_dtypes[step.slot])),
                                UInt32(current_rows),
                                UInt32(rows),
                                grid_dim=blocks,
                                block_dim=256,
                            )
                            ctx.enqueue_function[window_check_errors](
                                device_errors.unsafe_ptr(),
                                failures.unsafe_ptr(),
                                UInt32(step.slot),
                                UInt32(current_rows),
                                UInt32(rows),
                                grid_dim=blocks,
                                block_dim=256,
                            )
                        submissions += 2
                        ctx.enqueue_copy(host_failures.unsafe_ptr(), failures)
                        ctx.synchronize()
                        waits += 1
                        if is_duration:
                            for block in range(blocks):
                                if host_failures[4 * block]:
                                    raise Error(
                                        "Mojo GPU time window index contains null"
                                    )
                                if host_failures[4 * block + 1]:
                                    raise Error(
                                        "Mojo GPU time window index is not sorted within its group"
                                    )
                                if host_failures[4 * block + 2]:
                                    raise Error(
                                        "Mojo GPU time window bounds overflow"
                                    )
                                if host_failures[4 * block + 3]:
                                    raise Error(
                                        "Mojo GPU duration window arithmetic overflow"
                                    )
                        else:
                            for failure in host_failures:
                                if failure:
                                    raise Error(
                                        "Mojo GPU cumulative or rolling window overflow or input expression failure"
                                    )
                    continue
                if step.order:
                    var blocks = (current_rows + 255) // 256
                    var index = ctx.enqueue_create_buffer[DType.uint32](
                        current_rows
                    )
                    var next_index = ctx.enqueue_create_buffer[DType.uint32](
                        current_rows
                    )
                    var failures = ctx.enqueue_create_buffer[DType.uint32](
                        blocks
                    )
                    var host_failures = List[UInt32](length=blocks, fill=0)
                    ctx.enqueue_function[order_key_errors](
                        device_errors.unsafe_ptr(),
                        device_code.unsafe_ptr(),
                        failures.unsafe_ptr(),
                        UInt32(step.start),
                        UInt32(step.nodes),
                        UInt32(current_rows),
                        UInt32(rows),
                        grid_dim=blocks,
                        block_dim=256,
                    )
                    ctx.enqueue_function[order_init](
                        index.unsafe_ptr(),
                        UInt32(current_rows),
                        grid_dim=blocks,
                        block_dim=256,
                    )
                    submissions += 2
                    var run_width = 1
                    while run_width < current_rows:
                        ctx.enqueue_function[order_merge](
                            device_words.unsafe_ptr(),
                            device_valid.unsafe_ptr(),
                            index.unsafe_ptr(),
                            next_index.unsafe_ptr(),
                            device_code.unsafe_ptr(),
                            device_types.unsafe_ptr(),
                            UInt32(step.start),
                            UInt32(step.nodes),
                            UInt32(current_rows),
                            UInt32(rows),
                            UInt32(run_width),
                            grid_dim=blocks,
                            block_dim=256,
                        )
                        # Swapping owners keeps both buffers alive across queued passes.
                        var previous = index^
                        index = next_index^
                        next_index = previous^
                        run_width *= 2
                        submissions += 1
                    var gathered_words = ctx.enqueue_create_buffer[
                        DType.uint64
                    ](cells)
                    var gathered_valid = ctx.enqueue_create_buffer[DType.uint8](
                        cells
                    )
                    var gathered_errors = ctx.enqueue_create_buffer[
                        DType.uint32
                    ](cells)
                    if step.gather_count:
                        var gathers = List[Int64](capacity=step.gather_count)
                        for column in range(step.gather_count):
                            gathers.append(
                                plan.gathers[step.gather_start + column]
                            )
                        var device_gathers = ctx.enqueue_create_buffer[
                            DType.int64
                        ](len(gathers))
                        ctx.enqueue_copy(device_gathers, gathers.unsafe_ptr())
                        ctx.enqueue_function[order_gather](
                            device_words.unsafe_ptr(),
                            device_valid.unsafe_ptr(),
                            device_errors.unsafe_ptr(),
                            gathered_words.unsafe_ptr(),
                            gathered_valid.unsafe_ptr(),
                            gathered_errors.unsafe_ptr(),
                            index.unsafe_ptr(),
                            device_gathers.unsafe_ptr(),
                            UInt32(step.gather_count),
                            UInt32(current_rows),
                            UInt32(rows),
                            grid_dim=blocks,
                            block_dim=256,
                        )
                        submissions += 1
                        # The gather metadata and source owners outlive all queued reads.
                        ctx.synchronize()
                        waits += 1
                    ctx.enqueue_copy(host_failures.unsafe_ptr(), failures)
                    ctx.synchronize()
                    waits += 1
                    for failure in host_failures:
                        if failure:
                            raise Error(
                                "Mojo GPU sort key failed checked arithmetic or cast"
                            )
                    device_words = gathered_words^
                    device_valid = gathered_valid^
                    device_errors = gathered_errors^
                    continue
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
        if plan.reductions:
            var outputs = len(plan.outputs)
            var groups = min(1024, max(1, (current_rows + 255) // 256))
            var descriptors = ctx.enqueue_create_buffer[DType.int64](
                len(reduction_code)
            )
            var partials = ctx.enqueue_create_buffer[DType.uint64](
                outputs * groups * STATE_WORDS
            )
            var results = ctx.enqueue_create_buffer[DType.uint64](outputs)
            var result_validity = ctx.enqueue_create_buffer[DType.uint8](
                outputs
            )
            var result_errors = ctx.enqueue_create_buffer[DType.uint32](outputs)
            ctx.enqueue_copy(descriptors, reduction_code.unsafe_ptr())
            ctx.enqueue_function[reduce_partials](
                device_words.unsafe_ptr(),
                device_valid.unsafe_ptr(),
                device_errors.unsafe_ptr(),
                descriptors.unsafe_ptr(),
                partials.unsafe_ptr(),
                UInt32(outputs),
                UInt32(groups),
                UInt32(current_rows),
                UInt32(rows),
                grid_dim=(outputs * groups + 127) // 128,
                block_dim=128,
            )
            ctx.enqueue_function[reduce_finish](
                partials.unsafe_ptr(),
                descriptors.unsafe_ptr(),
                results.unsafe_ptr(),
                result_validity.unsafe_ptr(),
                result_errors.unsafe_ptr(),
                UInt32(outputs),
                UInt32(groups),
                UInt32(current_rows),
                grid_dim=(outputs + 127) // 128,
                block_dim=128,
            )
            submissions += 2
            if has_statistics:
                ctx.enqueue_function[statistics.reduce_partials](
                    device_words.unsafe_ptr(),
                    device_valid.unsafe_ptr(),
                    device_errors.unsafe_ptr(),
                    descriptors.unsafe_ptr(),
                    partials.unsafe_ptr(),
                    UInt32(outputs),
                    UInt32(groups),
                    UInt32(current_rows),
                    UInt32(rows),
                    grid_dim=(outputs * groups + 127) // 128,
                    block_dim=128,
                )
                ctx.enqueue_function[statistics.reduce_finish](
                    partials.unsafe_ptr(),
                    descriptors.unsafe_ptr(),
                    results.unsafe_ptr(),
                    result_validity.unsafe_ptr(),
                    result_errors.unsafe_ptr(),
                    UInt32(outputs),
                    UInt32(groups),
                    UInt32(current_rows),
                    grid_dim=(outputs + 127) // 128,
                    block_dim=128,
                )
                submissions += 2
            if has_quantiles:
                ctx.enqueue_function[quantiles.reduce_partials](
                    device_words.unsafe_ptr(),
                    device_valid.unsafe_ptr(),
                    device_errors.unsafe_ptr(),
                    descriptors.unsafe_ptr(),
                    partials.unsafe_ptr(),
                    UInt32(outputs),
                    UInt32(groups),
                    UInt32(current_rows),
                    UInt32(rows),
                    grid_dim=(outputs * groups + 127) // 128,
                    block_dim=128,
                )
                ctx.enqueue_function[quantiles.reduce_finish](
                    partials.unsafe_ptr(),
                    descriptors.unsafe_ptr(),
                    results.unsafe_ptr(),
                    result_validity.unsafe_ptr(),
                    result_errors.unsafe_ptr(),
                    UInt32(outputs),
                    UInt32(groups),
                    UInt32(current_rows),
                    grid_dim=(outputs + 127) // 128,
                    block_dim=128,
                )
                submissions += 2
            ctx.enqueue_copy(reduced_words.unsafe_ptr(), results)
            ctx.enqueue_copy(reduced_validity.unsafe_ptr(), result_validity)
            ctx.enqueue_copy(reduced_errors.unsafe_ptr(), result_errors)
        else:
            ctx.enqueue_copy(words.unsafe_ptr(), device_words)
            ctx.enqueue_copy(validity.unsafe_ptr(), device_valid)
            ctx.enqueue_copy(errors.unsafe_ptr(), device_errors)
        ctx.synchronize()
        waits += 1
    elif plan.reductions:
        # Empty input has no data to submit. The same Mojo state finalizer
        # supplies its scalar identities/nulls without initializing a device.
        for i in range(len(plan.outputs)):
            var base = i * DESCRIPTOR_WORDS
            var empty = State().finish(
                reduction_code[base + 3],
                reduction_code[base + 1],
                reduction_code[base + 2],
                reduction_code[base + 4],
                0,
            )
            if statistics.supported(reduction_code[base + 3]):
                empty = statistics.empty_result(
                    reduction_code[base + 3], reduction_code[base + 4]
                )
            if quantiles.supported(reduction_code[base + 3]):
                empty = quantiles.empty_result(
                    reduction_code[base + 3], reduction_code[base + 4]
                )
            reduced_words[i] = empty[0]
            reduced_validity[i] = empty[1]
            reduced_errors[i] = empty[2]
    if not plan.reductions:
        count = current_rows if plan.limit < 0 else min(
            current_rows, plan.limit
        )
    var columns = List[Series](capacity=len(plan.outputs))
    for i in range(len(plan.outputs)):
        ref output = plan.outputs[i]
        if plan.reductions:
            if count and reduced_errors[i]:
                raise Error(
                    "Mojo GPU reduction overflow or input expression failure"
                )
            columns.append(
                _output(
                    output.name,
                    output.dtype,
                    i,
                    1,
                    count,
                    reduced_words,
                    reduced_validity,
                )
            )
        else:
            for row in range(count):
                if errors[output.slot * rows + row]:
                    raise Error(
                        "Mojo GPU arithmetic overflow or strict cast failure"
                    )
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
        algorithm="mojo_gpu_grouped_rows" if grouping else "mojo_gpu_reductions" if plan.reductions else "mojo_gpu_windows" if ranking
        or windows else "mojo_gpu_ordered_rows" if ordering else "mojo_gpu_row_program",
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
