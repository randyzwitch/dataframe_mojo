"""Direct Mojo execution of shared bound dataframe row plans.

No native dataframe library, C ABI descriptors, or shader source is involved.
The compiler emits GPU code from `rows.mojo`; DeviceContext owns the queue,
buffers, transfers and synchronization. Unsupported regions reject before
initializing a context while the remaining native operations are migrated.
"""
from dataframe_metal.accelerator import decimal_sum_receipts

from .residency import TextDomain
from .text_domains import (
    MapPlan,
    Submission,
    source_maps,
    submit as submit_text,
)
from std.memory import bitcast, ArcPointer, Pointer
from std.time import perf_counter_ns
from std.sys import size_of
from max.gpu.host import DeviceBuffer
from dataframe.parallel import configured_workers
from dataframe import Column, DataFrame, DataType, Series
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.binding import temporal_result
from dataframe_metal.accelerator.decimal_group_feed import infallible_sum
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
    KEEP_NULLS,
    INT_IS_IN,
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
from dataframe_metal.accelerator.capabilities import RowCapabilities
from dataframe_metal.accelerator.rows import (
    RowPlan,
    RowStep,
    lower_rows,
    ROW_SLICE,
    ROW_IANA_ZONE_FIELDS,
    ROW_SORT_KEY,
    ROW_RANK_KEY,
    ROW_CUMULATIVE_KEY,
    ROW_DURATION_KEY,
    ROW_DURATION_BOUNDS,
    ROW_DURATION_CALENDAR,
    ROW_DURATION_UTC,
    ROW_GROUP_HEADER,
    ROW_GROUP_HEADER_LAST,
    ROW_GROUP_HEADER_WINDOW,
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
from dataframe_metal.accelerator.timezone_tables import (
    validate as validate_zone_table,
)
from .residency import (
    DeviceService,
    Scope,
    Matrix,
    Table,
    HostMatrix,
    raise_value_error,
)
from dataframe_metal.accelerator.faults import data_fault
from dataframe_metal.accelerator.code_domains import TextField, CategoryField
from dataframe.categorical import decode
from dataframe.string_column import StringCodes, StringColumn
from .rows import (
    integer_width,
    row_program,
    row_program_ordered,
    row_program_iana,
)
from . import iana_scalar_scopes
from dataframe_metal.accelerator import iana_certificate, checked_rank
from dataframe_metal.accelerator.iana_mutations import (
    ROW_IANA_MUTATION,
    validate_descriptor,
)
from dataframe.temporal import _naive_text
from . import ordered_faults
from .temporal_descriptors import (
    ROW_TEMPORAL_BINARY,
    ROW_TEMPORAL_CAST,
    binary_literal,
    cast_literal,
)
from .movement import (
    filter_scan,
    filter_offsets,
    filter_gather,
    observe_slot,
)
from . import statistics, quantiles, decimal_descriptors, hash_grouping
from dataframe.trace import trace_path
from .program import Program, kind as _kind
from .grouping import (
    group_mark,
    group_offsets,
    group_starts,
    group_window_prepare,
    group_window_merge,
    group_window_schedule,
    group_reduce,
    group_integer_partials,
    group_integer_finish,
    integer_partial_supported,
)
from .ordering import (
    order_init,
    order_merge,
    order_key_errors,
    order_gather,
    merge_grid,
    radix_histogram,
    radix_scan_level,
    radix_apply_carries,
    radix_bucket_offsets,
    radix_scatter,
    order_compact_runs,
    order_merge_topk,
    order_gather_topk,
)
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
    reduce_float_partials,
    reduce_finish,
)


comptime CELL_BYTES = 21
comptime TOPK_MAX_LIMIT = 4096
comptime TOPK_BASE_RUN_ROWS = 256


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
        observed_rank_inputs=True,
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
        asof_joins=True,
        primitive_list_explode=True,
        contiguous_window_reductions=True,
        int64_arithmetic=True,
        deferred_integer_errors=True,
        adjudicated_faults=True,
        extended_integers=True,
        mixed_types=True,
        row_extras=True,
        slices=True,
        fixed_logical=True,
        decimal128=True,
        decimal128_reductions=True,
        decimal_mean=True,
        decimal_float_cast=True,
        decimal_casts=True,
        decimal_comparisons=True,
        decimal_arithmetic=True,
        decimal_multiplicative=True,
        temporal_fields=True,
        temporal_binary_arithmetic=True,
        temporal_physical_casts=True,
        utc_temporal_conversions=True,
        fixed_zone_temporal_fields=True,
        iana_zone_temporal_fields=True,
        iana_zone_mutations=True,
        duration_totals=True,
        temporal_offsets=True,
        calendar_offsets=True,
        calendar_truncation=True,
        chunked_inputs=True,
        empty_inputs=True,
        empty_outputs=True,
        reuse_column_slots=True,
        proven_integer_operations=True,
        max_join_columns=256,
        max_nodes=128,
        max_slots=256,
        max_steps=128,
    )


def _preflight(
    query: LazyFrame,
    *,
    terminal_boundary: Bool = True,
    source_rows: Int = -1,
    sum_receipt: Optional[ArcPointer[decimal_sum_receipts.Receipt]] = None,
    ordered_source: Bool = False,
) raises -> RowPlan:
    if source_rows < -1 or source_rows > Int(Int32.MAX):
        raise Error(
            "Mojo GPU source row override exceeds the supported device index"
            " range"
        )
    var plan = lower_rows(
        query,
        capabilities(),
        terminal_boundary=terminal_boundary,
        source_rows=source_rows,
        sum_receipt=sum_receipt,
        ordered_source=ordered_source,
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
        Int(ROW_IANA_MUTATION),
        Int(ROW_TEMPORAL_BINARY),
        Int(ROW_TEMPORAL_CAST),
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
        KEEP_NULLS,
        INT_IS_IN,
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
            if (
                header != ROW_GROUP_HEADER
                and header != ROW_GROUP_HEADER_LAST
                and header != ROW_GROUP_HEADER_WINDOW
            ):
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
            if op == Int(ROW_TEMPORAL_BINARY) or op == Int(ROW_TEMPORAL_CAST):
                var left = Int(plan.code[4 * node + 1])
                var right = Int(plan.code[4 * node + 2])
                if (
                    left < 0
                    or left >= node - step.start
                    or plan.code[4 * node + 3] != -1
                ):
                    raise Error("Mojo GPU invalid temporal operand descriptor")
                var payload = bitcast[DType.uint64](plan.literals[node])
                var expected: UInt64
                if op == Int(ROW_TEMPORAL_BINARY):
                    if right < 0 or right >= node - step.start:
                        raise Error("Mojo GPU invalid temporal right operand")
                    expected = binary_literal(
                        Int64(payload >> 12),
                        plan.node_dtypes[step.start + left],
                        plan.node_dtypes[step.start + right],
                    )
                    if (
                        temporal_result(
                            Int(payload >> 12),
                            plan.node_dtypes[step.start + left],
                            plan.node_dtypes[step.start + right],
                        )
                        != plan.node_dtypes[node]
                    ):
                        raise Error("Mojo GPU invalid temporal result dtype")
                else:
                    if right != -1:
                        raise Error("Mojo GPU invalid temporal cast descriptor")
                    expected = cast_literal(
                        plan.node_dtypes[step.start + left],
                        plan.node_dtypes[node],
                        (payload & (UInt64(1) << 12)) != 0,
                    )
                if payload != expected:
                    raise Error("Mojo GPU invalid temporal literal descriptor")
                continue
            if op == ADD or op == SUB or op == MUL or op == NEG or op == ABS:
                if not (
                    plan.node_dtypes[node].is_integer()
                    or plan.node_dtypes[node].is_float()
                    or plan.node_dtypes[node].is_decimal()
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
                if not (
                    plan.node_dtypes[node].is_float()
                    or (op == DIV and plan.node_dtypes[node].is_decimal())
                    or (
                        plan.node_dtypes[node].is_integer()
                        and (
                            op == FLOORDIV
                            or op == MOD
                            or op == POW
                            or op == FLOOR
                            or op == CEIL
                            or op == ROUND
                        )
                    )
                ):
                    raise Error(
                        "Mojo GPU integer division and rounding are not"
                        " implemented yet"
                    )
            if (
                op >= DT_YEAR
                and op <= DT_TIME
                and (plan.code[4 * node + 3] & ROW_IANA_ZONE_FIELDS)
            ):
                var unit = plan.code[4 * node + 3]
                var local_source = Int(plan.code[4 * node + 1])
                if (
                    local_source < 0
                    or local_source >= node - step.start
                    or (unit & 7) < 1
                    or (unit & 7) > 3
                    or (unit & ~(ROW_IANA_ZONE_FIELDS | 7))
                ):
                    raise Error("Mojo GPU IANA field descriptor is invalid")
                var source_dtype = plan.node_dtypes[step.start + local_source]
                if (
                    not source_dtype.is_datetime()
                    or not source_dtype.time_zone().byte_length()
                    or source_dtype.time_zone() == "UTC"
                ):
                    raise Error(
                        "Mojo GPU IANA field source metadata is invalid"
                    )
                var expected_unit = Int64(
                    1 if source_dtype.per_second()
                    == 1000 else 2 if source_dtype.per_second()
                    == 1000000 else 3
                )
                if (unit & 7) != expected_unit:
                    raise Error("Mojo GPU IANA field source unit differs")
                validate_zone_table(
                    plan.zone_words, Int(plan.code[4 * node + 2])
                )
            if op == Int(ROW_IANA_MUTATION):
                if (
                    plan.ordered_mode != 5 and plan.ordered_mode != 6
                ) or plan.code[4 * node + 2] < 0:
                    raise Error(
                        "Mojo GPU IANA mutation requires an"
                        " occurrence-certified phase"
                    )
                validate_descriptor(
                    plan.zone_words, Int(plan.code[4 * node + 2])
                )
            if op == CAST:
                var source = step.start + Int(plan.code[4 * node + 1])
                if not (
                    plan.node_dtypes[source].is_numeric()
                    or plan.node_dtypes[source].is_decimal()
                    or plan.node_dtypes[source] == DataType.BOOL
                ) or not (
                    plan.node_dtypes[node].is_numeric()
                    or plan.node_dtypes[node].is_decimal()
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
    if dtype.is_decimal() and dtype.decimal_width() == 128:
        return 16
    if dtype == DataType.BOOL:
        return 1
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if dtype.physical() == DataType.of(D):
            return size_of[Scalar[D]]()
    raise Error("Mojo GPU unsupported output width")


def _host_column_bytes(column: Series) raises -> Int:
    """Count the published host storage after logical dictionary restoration."""
    if column.is_chunked():
        var total = 0
        for chunk in column.chunks():
            total = _checked_add(total, _host_column_bytes(chunk))
        return total
    var rows = len(column)
    var bitmap = (rows + 7) // 8
    if column.dtype() != DataType.STRING:
        return _checked_add(_checked_mul(rows, _width(column.dtype())), bitmap)
    from dataframe.string_column import StringColumn
    from dataframe.string_view import StringView

    ref strings = column._data[StringColumn]
    var total = bitmap
    if strings._view_storage:
        total = _checked_add(total, _checked_mul(rows, size_of[StringView]()))
        total = _checked_add(
            total, strings._view_storage.value().total_buffer_bytes()
        )
    else:
        total = _checked_add(total, _checked_mul(rows + 1, 8))
        total = _checked_add(total, len(strings._bytes[]))
    if strings._codes:
        total = _checked_add(
            total, _checked_mul(len(strings._codes.value()[].codes[]), 4)
        )
        if strings._codes.value()[].rows:
            total = _checked_add(
                total,
                _checked_mul(
                    len(strings._codes.value()[].rows.value()[]), size_of[Int]()
                ),
            )
    return total


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
    if column.dtype().is_decimal() and column.dtype().decimal_width() == 128:
        ref input = column._data[Column[Int128]]
        var high_base = len(words) // 2
        for row in range(len(column)):
            var valid = input._valid(row)
            validity[base + row] = UInt8(valid)
            if valid:
                var value = input._get(row).cast[DType.uint128]()
                words[base + row] = value.cast[DType.uint64]()
                words[high_base + base + row] = (value >> 64).cast[
                    DType.uint64
                ]()
        return
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


def _stage_categories(
    fields: List[TextField],
    schema: DataFrame,
    rows: Int,
    mut words: List[UInt64],
) raises:
    """Stage certified original codes in HIGH without changing text LOW."""
    for field in fields:
        if not field.category:
            continue
        var slot = -1
        for index in range(schema.width()):
            if schema._columns[index].name() == field.name:
                slot = index
        if slot < 0:
            continue
        _stage_category_column(field, rows, slot * rows, words)


def _stage_category_column(
    field: TextField,
    rows: Int,
    base: Int,
    mut words: List[UInt64],
) raises:
    if not field.category:
        return
    var cells = len(words) // 2
    var row = 0
    for chunk in field.category.value().owner[].codes:
        for at in range(len(chunk)):
            if row >= rows:
                raise Error("Original categorical scan row range mismatch")
            if chunk._valid(at):
                words[cells + base + row] = UInt64(chunk._get(at)) + 1
            row += 1
    if row != rows:
        raise Error("Original categorical scan owner is missing")


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
    if dtype.is_decimal() and dtype.decimal_width() == 128:
        var values = List[Int128](capacity=count)
        var high_base = len(words) // 2
        for row in range(count):
            var value = UInt128(words[base + row]) | (
                UInt128(words[high_base + base + row]) << 64
            )
            values.append(
                value.cast[DType.int128]() if valid[row] else Int128(0)
            )
        return Series(name, Column[Int128](values^, valid)).with_dtype(dtype)
    if dtype == DataType.BOOL:
        var values = List[Bool](capacity=count)
        for row in range(count):
            values.append(valid[row] and words[base + row] != 0)
        return Series(name, Column[Bool](values^, valid))
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
            return Series(name, Column[Scalar[D]](values^, valid)).with_dtype(
                dtype
            )
    raise Error("Mojo GPU unsupported output storage")


@fieldwise_init
struct ResourceFacts(Copyable):
    var peak_bytes: Int
    var matrix_bytes: Int
    var program_bytes: Int
    var scalar_bytes: Int
    var upload_bytes: Int
    var download_bytes: Int
    var kernel_launches: Int
    var synchronizations: Int


def _group_quantiles(plan: RowPlan, start: Int, nodes: Int) -> Bool:
    var keys = Int(plan.code[4 * start + 1])
    for column in range(keys + 1, nodes):
        if quantiles.supported(plan.code[4 * (start + column) + 3]):
            return True
    return False


def _grouped_integer_partial_nodes(
    plan: RowPlan, start: Int, nodes: Int
) raises -> List[UInt32]:
    """Metadata-only output certificate shared by sizing and dispatch."""
    var result = List[UInt32]()
    var header = plan.code[4 * start]
    if header != ROW_GROUP_HEADER and header != ROW_GROUP_HEADER_LAST:
        return result^
    var keys = Int(plan.code[4 * start + 1])
    var unsupported_physical_candidate = False
    for column in range(keys + 1, nodes):
        var node = start + column
        var source_slot = Int(plan.code[4 * node + 1])
        var target_slot = Int(plan.code[4 * node + 2])
        var op = plan.code[4 * node + 3]
        var source_dtype = plan.slot_dtypes[source_slot]
        var target_dtype = plan.slot_dtypes[target_slot]
        if (
            source_dtype.physical().is_integer()
            and target_dtype.physical().is_integer()
        ):
            if integer_partial_supported(
                header, op, _kind(source_dtype), _kind(target_dtype)
            ):
                if source_dtype.is_integer() and target_dtype.is_integer():
                    result.append(UInt32(node))
                else:
                    # Device slot kinds are physical. Avoid skipping a temporal
                    # logical aggregate if another output admits this step.
                    unsupported_physical_candidate = True
    if unsupported_physical_candidate:
        return List[UInt32]()
    return result^


def _quantile_indices(plan: RowPlan) -> Bool:
    if plan.reductions:
        for output in plan.outputs:
            if quantiles.supported(Int64(output.reduction)):
                return True
    for step in plan.steps:
        if step.grouped and _group_quantiles(plan, step.start, step.nodes):
            return True
    return False


def _radix_sort_passes(plan: RowPlan, step: RowStep) raises -> Int:
    """Admit only one direct, untransformed fixed-width integer sort key."""
    if (
        not step.order
        or step.grouped
        or step.nodes != 1
        or plan.code[4 * step.start] != ROW_SORT_KEY
    ):
        return -1
    var node = step.start
    # A categorical key sorts by the rank of its original code (the Program
    # sets that flag, the plan does not); radix passes would sort its words.
    for descriptor in plan.category_sorts:
        if descriptor[0] == node:
            return -1
    var dtype = plan.node_dtypes[node]
    if (
        not dtype.is_integer()
        or (plan.code[4 * node + 2] != 0 and plan.code[4 * node + 2] != 1)
        or (plan.code[4 * node + 3] & ~Int64(1)) != 0
    ):
        return -1
    return Int(integer_width(_kind(dtype)) // 4) + 1


def _topk_limit(
    plan: RowPlan, rows: Int, terminal_boundary: Bool
) raises -> Int:
    """Certify only a terminal ORDER BY over raw input columns followed by head."""
    if (
        not terminal_boundary
        or plan.ordered_mode != 0
        or plan.reductions
        or plan.limit <= 0
        or plan.limit > TOPK_MAX_LIMIT
        or plan.limit >= rows
        or len(plan.steps) != 1
        or len(plan.outputs) != plan.source.width()
    ):
        return -1
    var step = plan.steps[0].copy()
    if (
        not step.order
        or step.filter
        or step.grouped
        or step.nodes <= 0
        or step.gather_count != plan.source.width()
    ):
        return -1
    for column in range(step.gather_count):
        if plan.gathers[step.gather_start + column] != Int64(column):
            return -1
    for node in range(step.start, step.start + step.nodes):
        var slot = Int(plan.code[4 * node + 1])
        if (
            plan.code[4 * node] != ROW_SORT_KEY
            or slot < 0
            or slot >= plan.source.width()
            or plan.node_dtypes[node] != plan.source._columns[slot].dtype()
            or (plan.code[4 * node + 2] != 0 and plan.code[4 * node + 2] != 1)
            or (plan.code[4 * node + 3] & ~Int64(3)) != 0
        ):
            return -1
    for output in range(len(plan.outputs)):
        if (
            plan.outputs[output].slot != output
            or plan.outputs[output].reduction >= 0
            or plan.outputs[output].dtype
            != plan.source._columns[output].dtype()
        ):
            return -1
    return plan.limit


def _topk_retained_rows(rows: Int, run_width: Int, limit: Int) raises -> Int:
    var full_runs = rows // run_width
    var tail = rows % run_width
    var keep = min(limit, run_width)
    return _checked_add(_checked_mul(full_runs, keep), min(limit, tail))


def _iana_scalar_slots(plan: RowPlan) raises -> Int:
    var slots = 0
    if (
        plan.ordered_mode == 5
        or plan.ordered_mode == 6
        or plan.ordered_mode == checked_rank.ORDERED_RANK_INPUT
    ):
        if len(plan.iana_mask_bases) != len(plan.steps):
            raise Error("Certified row mask step geometry is invalid")
        for at in range(len(plan.steps)):
            if plan.iana_mask_bases[at] < 0:
                continue
            slots = max(
                slots,
                iana_certificate.scalar_count(
                    plan.zone_words,
                    plan.iana_mask_bases[at],
                    plan.steps[at].nodes,
                ),
            )
    return slots


def _source_high_words(plan: RowPlan) -> Bool:
    """Only wide decimals and original category codes carry source high words."""
    for column in plan.source._columns:
        var dtype = column.dtype()
        if dtype.is_decimal() and dtype.decimal_width() == 128:
            return True
    for field in plan.source_text:
        if field.category:
            return True
    return False


def estimate_resources(
    plan: RowPlan,
    rows_override: Int = -1,
    resident_input: Bool = False,
    terminal_boundary: Bool = True,
    export_values: Bool = True,
) raises -> ResourceFacts:
    """Bound requested payload/traffic using metadata alone.

    Peak deliberately retains the conservative standalone allocation envelope
    for resident calls too. Traffic omits an already resident input and a
    nonterminal or observation-only result; shape/error observations remain
    charged. Filters and
    groups use the source-row upper bound, without reading selectivity/nulls.
    """
    from . import typed_rows

    if rows_override < -1:
        raise Error("Mojo GPU resource row override must be >= -1")
    var rows = plan.source.height() if rows_override < 0 else rows_override
    var count = rows if plan.limit < 0 else min(rows, plan.limit)
    if plan.reductions:
        count = 0 if plan.limit == 0 else 1
    if rows > Int(Int32.MAX):
        raise Error(
            "Mojo GPU resource rows exceed the supported device index range"
        )
    var cells = _checked_mul(rows, plan.slots)
    var movement = False
    var ordering = False
    var grouping = False
    var ranking = False
    var windows = False
    var duration_windows = False
    var radix_ordering = False
    var topk_limit = _topk_limit(plan, rows, terminal_boundary)
    for step in plan.steps:
        ordering = ordering or step.order or step.grouped
        grouping = grouping or step.grouped
        radix_ordering = radix_ordering or _radix_sort_passes(plan, step) > 0
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
    if topk_limit > 0:
        radix_ordering = False
    # GPU words, byte validity and per-value errors; one host staging matrix
    # is reused for readback after the stream consumes its upload.
    var matrix_bytes = _checked_mul(cells, CELL_BYTES)
    var result_matrix_bytes = matrix_bytes
    if topk_limit > 0:
        result_matrix_bytes = _checked_mul(
            _checked_mul(topk_limit, plan.slots), CELL_BYTES
        )
    var program_bytes = _checked_mul(
        len(plan.code)
        + 2 * len(plan.literals)
        + len(plan.key_values)
        + len(plan.zone_words),
        8,
    )
    # One shared host rank owner and one device rank buffer. Both existing
    # program envelopes charge this term once; upload sends only the device copy.
    program_bytes = _checked_add(
        program_bytes, _checked_mul(len(plan.ordered_ranks[]), 4)
    )
    var shared = _checked_add(matrix_bytes, program_bytes) if rows else 0
    if rows and movement:
        # A second matrix prevents overlapping reads/writes during compaction.
        # Scans use row positions, two block arrays and a two-word count/error
        # readback. Gather metadata is bounded by the full plan's gather list.
        shared = _checked_add(shared, result_matrix_bytes)
        shared = _checked_add(shared, _checked_mul(rows, 8 if ordering else 4))
        shared = _checked_add(shared, _checked_mul((rows + 255) // 256, 8))
        shared = _checked_add(shared, 16)
        shared = _checked_add(shared, _checked_mul(len(plan.gathers), 16))
        if duration_windows:
            # Duration errors preserve four distinct fault categories on both
            # host and device, replacing the ordinary one-word block summary.
            shared = _checked_add(shared, _checked_mul((rows + 255) // 256, 24))
        if radix_ordering:
            # Sixteen per-tile UInt32 histograms plus every recursive summary
            # level and sixteen global bucket offsets; scratch is reused by
            # all digit passes and released after the sort's error drain.
            var level_count = (rows + 255) // 256
            var summary_count = 0
            while level_count > 0:
                summary_count = _checked_add(summary_count, level_count)
                if level_count == 1:
                    break
                level_count = (level_count + 255) // 256
            shared = _checked_add(
                shared,
                _checked_add(_checked_mul(summary_count, 16 * 4), 16 * 4),
            )
        if ranking:
            # The idle sort-index owner supplies dense prefixes. Only the
            # additional per-block dense counts need separate device storage.
            shared = _checked_add(shared, _checked_mul((rows + 255) // 256, 4))
        if grouping:
            # Group starts and block-local positions accompany both sort indices.
            shared = _checked_add(shared, _checked_mul(rows, 8))
            shared = _checked_add(shared, _checked_mul(plan.slots, 8))
            # Group shape/bounds control expands from two to four UInt32
            # words, retained on both host and device during this region.
            shared = _checked_add(shared, 16)
    if rows >= 256:
        for step in plan.steps:
            if hash_grouping.eligible(plan, step):
                shared = _checked_add(
                    shared, _checked_mul(hash_grouping.capacity(rows), 8)
                )
                break
    if rows and _quantile_indices(plan):
        # One device UInt32 source-index plane is reused across scalar outputs
        # and grouped phases; no host copy, download or extra wait is needed.
        shared = _checked_add(shared, _checked_mul(rows, 4))
    var result_bytes = _checked_add(
        result_matrix_bytes, program_bytes
    ) if rows else 0
    if rows and grouping:
        result_bytes = _checked_add(result_bytes, _checked_mul(plan.slots, 8))
    if plan.reductions:
        var outputs = len(plan.outputs)
        # Descriptors and scalar results have both host and device storage.
        # Partials are bounded independently of the input row count.
        var descriptors = _checked_mul(outputs, DESCRIPTOR_WORDS * 8)
        var scalar_results = _checked_mul(outputs, CELL_BYTES)
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
    if export_values:
        for output in plan.outputs:
            result_bytes = _checked_add(
                result_bytes, _checked_mul(count, _width(output.dtype))
            )
            result_bytes = _checked_add(result_bytes, (count + 7) // 8)
    # Column construction temporarily holds unpacked Boolean validity and,
    # for Boolean outputs, both byte and packed representations. Charge a
    # conservative upper bound for the largest one-column materialization.
    var scratch = (
        _checked_mul(count, 4) if export_values and len(plan.outputs) else 0
    )
    var peak = _checked_add(_checked_add(shared, result_bytes), scratch)
    # Program host tail and device tail are in program_bytes above. Keep the
    # captured host zone owner too; empty execution still creates two host tails.
    var captured_zone_bytes = _checked_mul(
        plan.zone_words.capacity() if plan.ordered_mode == 5
        or plan.ordered_mode == 6
        or plan.ordered_mode
        == checked_rank.ORDERED_RANK_INPUT else len(plan.zone_words),
        8,
    )
    if (
        plan.ordered_mode == 5
        or plan.ordered_mode == 6
        or plan.ordered_mode == checked_rank.ORDERED_RANK_INPUT
    ):
        var scalar_workspace = iana_scalar_scopes.resources(
            rows,
            _iana_scalar_slots(plan),
            checked_rank.BATCH_ROWS if plan.ordered_mode
            == checked_rank.ORDERED_RANK_INPUT else 8192,
        )
        peak = _checked_add(peak, scalar_workspace.workspace_bytes)
        peak = _checked_add(peak, 48 if rows else 0)
        peak = _checked_add(peak, _checked_mul(len(plan.iana_mask_bases), 16))
    peak = _checked_add(
        peak, _checked_mul(captured_zone_bytes, 1 if rows else 2)
    )
    # Empty consumers never enter the device-copy branch, so they allocate
    # neither host nor device column-mapping descriptors.
    if rows and resident_input:
        peak = _checked_add(peak, _checked_mul(plan.source.width(), 48))
    if rows and len(plan.ordered_ranks[]):
        # Two-word control record has simultaneous host/device owners and is
        # reused for input-zero proof and ranked output observation.
        peak = _checked_add(peak, 16)
        if (
            plan.ordered_mode == 2
            or plan.ordered_mode == 3
            or plan.ordered_mode == 4
            or plan.ordered_mode == 5
            or plan.ordered_mode == 6
            or plan.ordered_mode == 7
            or plan.ordered_mode == 8
            or plan.ordered_mode == checked_rank.ORDERED_RANK_INPUT
        ):
            peak = _checked_add(
                peak, _checked_mul(min(1024, max(1, (rows + 255) // 256)), 8)
            )
    # The grouped integer path reuses one bounded 4096-state device scratch
    # buffer across outputs and grouped steps. The group count is only known
    # after sorting, so charge the full cap whenever metadata and row count
    # could admit P >= 2; resident calls keep this same conservative envelope.
    var grouped_integer_possible = False
    if rows > 256:
        for step in plan.steps:
            if step.grouped and len(
                _grouped_integer_partial_nodes(plan, step.start, step.nodes)
            ):
                grouped_integer_possible = True
                break
    if grouped_integer_possible:
        peak = _checked_add(peak, 4096 * STATE_WORDS * 8)
    var upload = 0
    var download = 0
    var launches = 0
    var waits = 0
    var scalar_bytes = _checked_mul(
        len(plan.outputs), CELL_BYTES
    ) if plan.reductions else 0
    if rows and cells:
        if not resident_input:
            upload = _checked_mul(
                _checked_mul(rows, plan.source.width()),
                17 if _source_high_words(plan) else 9,
            )
        else:
            # GPU-resident copies upload only their three-word column mapping.
            upload = _checked_mul(plan.source.width(), 24)
            launches = 1
        if len(plan.steps):
            upload = _checked_add(upload, program_bytes)
        if terminal_boundary and export_values:
            download = scalar_bytes if plan.reductions else matrix_bytes
        else:
            # Nonterminal regions observe result errors before any consumer.
            download = _checked_mul(_checked_mul(count, len(plan.outputs)), 4)
        # Buffer adoption and terminal readback/nonterminal error observation
        # each complete a stream phase. Resident source-copy descriptors also
        # survive a separate synchronized gather into this region's slots.
        waits = 2 + Int(resident_input)
        var passes = 0
        var run_width = 1
        while run_width < rows:
            passes += 1
            run_width *= 2
        # Fault observation (accelerator/faults.mojo), as the executor does
        # it: one launch per plain-region step or grouped output that can
        # fault and one for reduction states, then a 4-byte zero upload, one
        # wait and a 4-byte read.
        var observed = False
        if plan.ordered_mode == 0:
            var kinds = List[Int64](capacity=len(plan.node_dtypes))
            for dtype in plan.node_dtypes:
                try:
                    kinds.append(_kind(dtype))
                except:
                    kinds.append(0)
            for step in plan.steps:
                if step.grouped:
                    for node in range(step.start, step.start + step.nodes):
                        if plan.code[
                            4 * node
                        ] == ROW_GROUP_OUTPUT and _can_fault(
                            plan.slot_dtypes[Int(plan.code[4 * node + 2])]
                        ):
                            launches = _checked_add(launches, 1)
                            observed = True
                elif not step.order and _step_can_fault(
                    plan.code, kinds, step.start, step.nodes
                ):
                    launches = _checked_add(launches, 1)
                    observed = True
            if plan.reductions:
                for output in plan.outputs:
                    observed |= _can_fault(plan.slot_dtypes[output.slot])
                launches = _checked_add(launches, Int(observed))
        if observed:
            upload = _checked_add(upload, 4)
            waits = _checked_add(waits, 1)
            download = _checked_add(download, 4)
        for step in plan.steps:
            if step.order or step.grouped:
                # Includes all sort, dense/group scans, scatter and checked
                # key/window observation phases; bounds every supported mode.
                var order_launch_bound = passes + 16
                var radix_pass_count = _radix_sort_passes(plan, step)
                var topk_limit = _topk_limit(plan, rows, terminal_boundary)
                if topk_limit > 0:
                    var base_passes = 0
                    var run_width = 1
                    while run_width < min(rows, TOPK_BASE_RUN_ROWS):
                        base_passes += 1
                        run_width *= 2
                    var merge_levels = 0
                    var span = TOPK_BASE_RUN_ROWS
                    while span < rows:
                        merge_levels += 1
                        span *= 2
                    order_launch_bound = 4 + base_passes + merge_levels
                elif radix_pass_count > 0:
                    var level_count = (rows + 255) // 256
                    var level_total = 1
                    var down_carries = 0
                    while level_count > 1:
                        level_total += 1
                        if level_count > 256:
                            down_carries += 1
                        level_count = (level_count + 255) // 256
                    var per_digit_launches = 3 + level_total - 1 + down_carries
                    order_launch_bound = max(
                        order_launch_bound,
                        2
                        + radix_pass_count * per_digit_launches
                        + Int(step.gather_count > 0),
                    )
                launches = _checked_add(launches, order_launch_bound)
                waits = _checked_add(waits, 8)
                download = _checked_add(
                    download, _checked_mul((rows + 255) // 256, 64)
                )
                download = _checked_add(download, 16)
                if step.grouped:
                    if _group_quantiles(plan, step.start, step.nodes):
                        launches = _checked_add(launches, 1)
                    upload = _checked_add(upload, _checked_mul(plan.slots, 8))
                    download = _checked_add(download, 8)
                    if plan.code[
                        4 * step.start
                    ] == ROW_GROUP_HEADER_WINDOW and bitcast[DType.uint64](
                        plan.literals[step.start]
                    ) & (
                        UInt64(1) << 32
                    ):
                        # Exact CPU group ordering adds a bounded merge pass
                        # for every source-row sort pass, plus preparation.
                        launches = _checked_add(launches, passes + 2)
            else:
                launches = _checked_add(launches, 1)
            if step.filter:
                launches = _checked_add(
                    launches, 2 + Int(step.gather_count > 0)
                )
                waits = _checked_add(waits, 1 + Int(step.gather_count > 0))
                download = _checked_add(download, 8)
            upload = _checked_add(upload, _checked_mul(step.gather_count, 8))
        if plan.reductions:
            upload = _checked_add(
                upload, _checked_mul(len(plan.outputs), DESCRIPTOR_WORDS * 8)
            )
            launches = _checked_add(launches, 2)
            var has_statistics = False
            var has_quantiles = False
            var has_float_sums = False
            for output in plan.outputs:
                has_float_sums |= (
                    output.reduction == SUM or output.reduction == MEAN
                ) and plan.slot_dtypes[output.slot].is_float()
                has_statistics |= statistics.supported(Int64(output.reduction))
                has_quantiles |= quantiles.supported(Int64(output.reduction))
            launches = _checked_add(
                launches,
                2 * Int(has_statistics)
                + 3 * Int(has_quantiles)
                + Int(has_float_sums),
            )
    if rows and (
        plan.ordered_mode == 5
        or plan.ordered_mode == 6
        or plan.ordered_mode == checked_rank.ORDERED_RANK_INPUT
    ):
        for at in range(len(plan.steps)):
            if plan.iana_mask_bases[at] < 0:
                continue
            launches = _checked_add(
                launches,
                _checked_mul(
                    2,
                    iana_certificate.scalar_count(
                        plan.zone_words,
                        plan.iana_mask_bases[at],
                        plan.steps[at].nodes,
                    ),
                ),
            )
    if rows and len(plan.ordered_ranks[]):
        var observations = 2 if plan.ordered_mode == 1 else len(
            plan.steps
        ) + Int(
            plan.ordered_mode == 3
            or plan.ordered_mode == 4
            or plan.ordered_mode == 6
            or plan.ordered_mode == 7
            or plan.ordered_mode == 8
            or plan.ordered_mode == checked_rank.ORDERED_RANK_INPUT
        )
        if plan.ordered_mode == 7 or plan.ordered_mode == 8:
            observations = 1  # Input-zero proof precedes all original operands.
            for step in plan.steps:
                observations += Int(not step.grouped and not step.order)
        launches = _checked_add(
            launches,
            observations
            * (
                2 if plan.ordered_mode == 2
                or plan.ordered_mode == 3
                or plan.ordered_mode == 4
                or plan.ordered_mode == 5
                or plan.ordered_mode == 6
                or plan.ordered_mode == 7
                or plan.ordered_mode == 8
                or plan.ordered_mode == checked_rank.ORDERED_RANK_INPUT else 1
            ),
        )
        waits = _checked_add(waits, observations)
        download = _checked_add(download, _checked_mul(observations, 8))
    # A reusable standalone service drains its stream once more before the
    # context can return to the idle pool. Shared-region and uncached calls
    # may omit this drain; the metadata estimate remains an upper bound.
    # Zero-column inputs carry row metadata without creating a device context.
    if rows and cells:
        waits = _checked_add(waits, 1)
        # Native computation drains before its temporary owners leave scope;
        # terminal readback is a separate phase from that ownership boundary.
        if (
            terminal_boundary
            and export_values
            and not resident_input
            and len(plan.steps)
            and typed_rows.supports(plan)
        ):
            waits = _checked_add(waits, 1)
    peak = max(peak, plan.iana_construction_peak_bytes)
    return ResourceFacts(
        peak,
        matrix_bytes,
        program_bytes,
        scalar_bytes,
        upload,
        download,
        launches,
        waits,
    )


def materialize(table: Table) raises -> DataFrame:
    """Export one compact resident result through the existing typed builder."""
    if len(table.native_columns):
        return _materialize_native_host(table, _readback_native(table))
    var host = table.readback()
    return _materialize_host(table, host)


def _readback_native(table: Table) raises -> List[Series]:
    var additional = 0
    for output in range(table.schema.width()):
        var column = table.native_columns[Int(table.columns[output])].view(
            table.offset, table.rows
        )
        additional = _checked_add(additional, column.readback_bytes())
    _ = table.native_columns[0].scope().check(additional)
    var columns = List[Series](capacity=table.schema.width())
    for output in range(table.schema.width()):
        var index = Int(table.columns[output])
        columns.append(
            table.native_columns[index]
            .view(table.offset, table.rows)
            .readback_series(table.schema._columns[output])
        )
    return columns^


def _materialize_native_host(
    table: Table, var host: List[Series]
) raises -> DataFrame:
    var columns = List[Series](capacity=table.schema.width())
    for output in host:
        columns.append(output.copy())
    return DataFrame(columns^, height=table.rows)


def _materialize_host(table: Table, host: HostMatrix) raises -> DataFrame:
    """Construct typed columns from an already completed host readback."""
    var additional = 0
    for category in table.categories:
        if category and category.value().string_codes:
            var dtype = category.value().owner[].dtype
            # Numeric transport coexists with decoded String views, original
            # code retention, copied dictionary bytes and decoder lookup views.
            additional = _checked_add(
                additional,
                _checked_add(
                    _checked_mul(host.rows, 32),
                    _checked_add(
                        _checked_mul(len(dtype.dictionary()[].bytes), 2),
                        _checked_mul(len(dtype.dictionary()[]), 32),
                    ),
                ),
            )
    if additional:
        var envelope = _checked_mul(host.rows * host.schema.width(), CELL_BYTES)
        for field in host.schema._columns:
            envelope = _checked_add(
                envelope,
                _checked_add(
                    _checked_mul(host.rows, _width(field.dtype())),
                    (host.rows + 7) // 8,
                ),
            )
        if table.storage:
            _ = table.storage.value()[].scope.check(
                _checked_add(envelope, additional)
            )
        elif table.identity_storage:
            _ = table.identity_storage.value()[].scope.check(
                _checked_add(envelope, additional)
            )
    var columns = List[Series](capacity=host.schema.width())
    for column in range(host.schema.width()):
        ref field = host.schema._columns[column]
        var output = _output(
            field.name(),
            field.dtype(),
            column,
            host.rows,
            host.rows,
            host.words,
            host.validity,
        )
        if table.categories[column]:
            var owner = table.categories[column].value().owner
            var codes = List[UInt32](capacity=host.rows)
            var valid = List[Bool](capacity=host.rows)
            for row in range(host.rows):
                var at = column * host.rows + row
                var present = host.validity[at] != 0
                var original = host.words[host.rows * host.schema.width() + at]
                if present and (
                    original == 0
                    or original > UInt64(len(owner[].dtype.dictionary()[]))
                ):
                    raise Error(
                        "Original categorical output code provenance is missing"
                    )
                codes.append(UInt32(original - 1) if present else UInt32(0))
                valid.append(present)
            output = Series(
                field.name(), Column[UInt32](codes^, valid)
            ).with_dtype(owner[].dtype)
            if table.categories[column].value().string_codes:
                # A source coded String keeps its own codes when transported.
                # CPU grouped String publication has no category certificate
                # and therefore follows ordinary decoded String restoration.
                var raw = output._data[Column[UInt32]].copy()
                var retained = List[UInt32](capacity=host.rows)
                for row in range(host.rows):
                    retained.append(
                        raw._get(row) if raw._valid(row) else UInt32(0)
                    )
                output = decode(output)
                output._data[StringColumn]._codes = ArcPointer(
                    StringCodes(ArcPointer(retained^), owner[].dtype)
                )
        output._field_metadata = field._field_metadata
        columns.append(output^)
    return DataFrame(columns^, height=host.rows)


def _execute_preflighted(
    plan: RowPlan,
    scope: Scope,
    profiling: Bool,
    started: Int,
    preflight_ns: Int,
    terminal_boundary: Bool,
    observation_only: Bool,
) raises -> Tuple[DataFrame, DataFrame]:
    from . import fused_int32_plan, fused_int32_runtime
    from . import wide_integer_plan, wide_integer_runtime

    if terminal_boundary and not observation_only:
        var program = fused_int32_plan.bind(plan)
        if program:
            return fused_int32_runtime.execute(
                plan, program.value(), scope, profiling, started, preflight_ns
            )
        var wide_programs = wide_integer_plan.bind(plan)
        if wide_programs:
            return wide_integer_runtime.execute(
                plan,
                wide_programs.value(),
                scope,
                profiling,
                started,
                preflight_ns,
            )
    var executed = execute_plan(
        plan,
        scope,
        None,
        profiling,
        terminal_boundary=terminal_boundary,
        export_values=not observation_only,
        native_output=not observation_only,
    )
    var result: DataFrame
    var readback_ns = 0
    var result_copy_ns = 0
    if observation_only:
        if terminal_boundary:
            executed[0].observe_errors()
        result = executed[0].schema.copy()
    elif profiling:
        var readback_started = Int(perf_counter_ns())
        if len(executed[0].native_columns):
            var host = _readback_native(executed[0])
            readback_ns = Int(perf_counter_ns()) - readback_started
            var result_started = Int(perf_counter_ns())
            result = _materialize_native_host(executed[0], host^)
            result_copy_ns = Int(perf_counter_ns()) - result_started
        else:
            var host = executed[0].readback()
            readback_ns = Int(perf_counter_ns()) - readback_started
            var result_started = Int(perf_counter_ns())
            result = _materialize_host(executed[0], host)
            result_copy_ns = Int(perf_counter_ns()) - result_started
    else:
        result = materialize(executed[0])
    var traffic = scope.transfers()
    var report = executed[1].copy()
    if profiling:
        for field in List[Tuple[String, Int]](
            [
                ("preflight_ms", preflight_ns),
                ("readback_ms", readback_ns),
                ("result_copy_ms", result_copy_ns),
            ]
        ):
            report = report.with_column(
                Series(field[0], Column[Float64]([Float64(field[1]) / 1e6]))
            )
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
            _host_column_bytes(column),
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


def execute(
    query: LazyFrame,
    device_id: Int,
    budget: Int,
    profiling: Bool,
    *,
    terminal_boundary: Bool = True,
    ordered_source: Bool = False,
    observation_only: Bool = False,
    device_service: Optional[DeviceService] = None,
) raises -> Tuple[DataFrame, DataFrame]:
    var started = Int(perf_counter_ns())
    var plan = _preflight(
        query,
        terminal_boundary=terminal_boundary,
        ordered_source=ordered_source,
    )
    var preflight_ns = Int(perf_counter_ns()) - started if profiling else 0
    var scope = Scope(
        device_id, budget, profiling=profiling, service=device_service
    )
    try:
        var executed = _execute_preflighted(
            plan,
            scope,
            profiling,
            started,
            preflight_ns,
            terminal_boundary,
            observation_only,
        )
        if not device_service:
            return executed^
        # Runtime-owned services require a final drain before caching.
        scope.synchronize()
        var traffic = scope.transfers()
        var timing = scope.timing_ns()
        var report = executed[1].copy()
        report = report.with_column(
            Series("synchronizations", Column[Int64]([Int64(traffic[3])]))
        )
        report = report.with_column(
            Series(
                "query_wall_ms",
                Column[Float64](
                    [Float64(Int(perf_counter_ns()) - started) / 1e6]
                ),
            )
        )
        if profiling:
            var scratch_cache = device_service.value().scratch_info()
            var pipeline_cache = device_service.value().pipeline_info()
            for field in List[Tuple[String, Int]](
                [
                    (
                        "service_context_generations",
                        device_service.value().cached_generations(),
                    ),
                    ("service_idle_scratch_bytes", scratch_cache[0]),
                    ("service_idle_scratch_entries", scratch_cache[1]),
                    ("service_scratch_hits", scratch_cache[2]),
                    ("service_scratch_misses", scratch_cache[3]),
                    ("service_pipeline_entries", pipeline_cache[0]),
                    ("service_pipeline_hits", pipeline_cache[1]),
                    ("service_pipeline_misses", pipeline_cache[2]),
                    ("service_pipeline_clears", pipeline_cache[3]),
                    ("service_pipeline_capacity", 9),
                ]
            ):
                report = report.with_column(
                    Series(field[0], Column[Int64]([Int64(field[1])]))
                )
            report = report.with_column(
                Series(
                    "synchronization_ms",
                    Column[Float64]([Float64(timing[1]) / 1e6]),
                )
            )
        var result = executed[0].copy()
        scope.mark_success()
        return (result^, report^)
    except error:
        scope.mark_failed()
        # Inner helpers must drain while submitted-buffer owners are alive.
        # The helper stack has unwound here, so this only drains remaining scope
        # work; preserve the original error if cleanup also fails.
        if device_service:
            try:
                scope.synchronize()
            except cleanup_error:
                _ = String(cleanup_error)
        raise error^


def execute_plan(
    plan: RowPlan,
    scope: Scope,
    resident_input: Optional[Table] = None,
    profiling: Bool = False,
    *,
    terminal_boundary: Bool = True,
    export_values: Bool = True,
    native_output: Bool = False,
    text_domain: Optional[ArcPointer[TextDomain]] = None,
) raises -> Tuple[Table, DataFrame]:
    """Execute one preflighted region on the query's shared Metal context.

    Resident inputs are immutable. Each consumer owns a fresh matrix, and only
    a final materialization exports values. Intermediate error observation is
    eager, matching the previous typed-column boundary.
    """
    if plan.source_failure:
        raise Error(plan.source_failure)
    # Native output is selected by the DAG's preflighted representation plan,
    # or by the existing direct terminal route. Other families retain MATRIX.
    from . import typed_rows, typed_runtime

    if (
        native_output
        and export_values
        and not resident_input
        and not text_domain
        and typed_rows.supports(plan)
    ):
        return typed_runtime.execute(plan, scope, profiling)
    var started = Int(perf_counter_ns())
    var before = scope.transfers()
    var timing_before = scope.timing_ns()
    var rows = (
        resident_input.value().rows if resident_input else plan.source.height()
    )
    var ordered_scalar = plan.ordered_mode == 1
    if plan.decimal_feed_batch_rows:
        if (
            plan.decimal_feed_batch_rows != 8192
            or configured_workers() > 65535
            or len(plan.outputs) != 1
            or plan.outputs[0].reduction != SUM
            or not plan.slot_dtypes[plan.outputs[0].slot].is_decimal()
            or not plan.reductions
            or plan.limit >= 0
            or not terminal_boundary
        ):
            raise Error("Mojo GPU checked Decimal feed certificate changed")
        for step in plan.steps:
            if step.filter or step.order or step.grouped:
                raise Error("Mojo GPU checked Decimal feed cannot move rows")
    var observed_producer = plan.ordered_mode == 6
    var observed_rank = plan.ordered_mode == checked_rank.ORDERED_RANK_INPUT
    var ordered_batch_rows = checked_rank.BATCH_ROWS if observed_rank else 8192
    var iana_certified = (
        plan.ordered_mode == 5 or observed_producer or observed_rank
    )
    var ordered_terminal = plan.ordered_mode == 2 or plan.ordered_mode == 5
    var ordered_private = plan.ordered_mode == 3
    var ordered_decimal = plan.ordered_mode == 4
    var ordered_rows = (
        ordered_terminal
        or ordered_private
        or ordered_decimal
        or observed_producer
        or observed_rank
    )
    var ordered_group = plan.ordered_mode == 7
    var precision_rows = plan.ordered_mode == 8
    var ordered_precision = ordered_group or precision_rows
    var ordered = ordered_scalar or ordered_rows or ordered_precision
    if observed_rank:
        if len(plan.iana_mask_bases) != len(plan.steps):
            raise Error("Checked rank certificate step geometry changed")
        var certified_inputs = 0
        for at in range(len(plan.steps)):
            if plan.iana_mask_bases[at] >= 0:
                ref input_step = plan.steps[at]
                if input_step.filter or input_step.order or input_step.grouped:
                    raise Error("Checked rank input cannot move rows")
                iana_certificate.validate(
                    plan.zone_words, plan.iana_mask_bases[at], input_step.nodes
                )
                certified_inputs += 1
        if certified_inputs != 1:
            raise Error("Checked rank requires one isolated input phase")
    if ordered != (len(plan.ordered_ranks[]) != 0):
        raise Error("Mojo GPU ordered error mode does not match its ranks")
    if ordered_group:
        var groups = 0
        for step in plan.steps:
            if step.filter or (step.order and (plan.reductions or not groups)):
                raise Error(
                    "Mojo GPU narrow Decimal group cannot move source rows"
                )
            groups += Int(step.grouped)
        if (
            not terminal_boundary
            or plan.limit >= 0
            or groups != (0 if plan.reductions else 1)
            or rows > Int(UInt32.MAX)
            or len(plan.ordered_ranks[]) != len(plan.literals)
        ):
            raise Error("Mojo GPU narrow Decimal group certificate changed")
        if plan.reductions:
            var floating_sum = False
            for output in plan.outputs:
                var input_type = plan.slot_dtypes[output.slot]
                if output.reduction != SUM:
                    raise Error(
                        "Mojo GPU narrow Decimal scalar requires SUM states"
                    )
                if input_type == DataType.FLOAT64:
                    floating_sum = True
                elif not infallible_sum(input_type, rows):
                    raise Error(
                        "Mojo GPU narrow Decimal scalar state bound changed"
                    )
            if not floating_sum:
                raise Error("Mojo GPU narrow Decimal scalar lacks Float64 SUM")
    if ordered_rows and not observed_rank:
        for step in plan.steps:
            if step.filter or step.order or step.grouped:
                raise Error(
                    "Mojo GPU ordered ROWS cannot feed row movement or state"
                    " phases"
                )
    if precision_rows:
        if (
            not terminal_boundary
            or plan.limit >= 0
            or plan.reductions
            or rows > Int(UInt32.MAX)
            or len(plan.ordered_ranks[]) != len(plan.literals)
        ):
            raise Error("Mojo GPU narrow Decimal ROW certificate changed")
        for step in plan.steps:
            if step.filter or step.order or step.grouped:
                raise Error(
                    "Mojo GPU narrow Decimal ROW producer cannot move rows"
                )
    if ordered_terminal and (
        Bool(resident_input)
        or plan.limit >= 0
        or plan.reductions
        or len(plan.steps) < 1
        or len(plan.ordered_ranks[]) != len(plan.literals)
    ):
        raise Error(
            "Mojo GPU ordered ROWS requires an original raw scan and terminal"
            " projections"
        )
    if ordered_private and (
        not terminal_boundary
        or plan.limit >= 0
        or plan.reductions
        or len(plan.steps) != 1
        or len(plan.outputs) != plan.source.width() + 1
        or len(plan.ordered_ranks[]) != len(plan.literals)
    ):
        raise Error(
            "Mojo GPU ordered IMPLODE requires one certified observed row"
            " producer"
        )
    if ordered_decimal and (
        not terminal_boundary
        or plan.limit >= 0
        or plan.reductions
        or len(plan.steps) < 1
        or len(plan.ordered_ranks[]) != len(plan.literals)
    ):
        raise Error(
            "Mojo GPU ordered decimal requires an observed row producer"
        )
    if observed_producer and (
        not terminal_boundary
        or plan.limit >= 0
        or plan.reductions
        or not plan.producer_region
        or len(plan.steps) < 1
    ):
        raise Error("Mojo GPU observed temporal producer proof is missing")
    if observed_producer and Bool(resident_input) != (
        plan.producer_region.value()[].source_task >= 0
    ):
        raise Error("Mojo GPU observed temporal producer input lineage differs")
    if ordered_scalar and (
        rows != 1
        or plan.reductions
        or plan.limit >= 0
        or len(plan.steps) != 1
        or len(plan.outputs) != 1
        or plan.steps[0].filter
        or plan.steps[0].order
        or plan.steps[0].grouped
        or len(plan.ordered_ranks[]) != len(plan.literals)
    ):
        raise Error(
            "Mojo GPU ordered scalar requires exactly one proven anchor row"
        )
    var count = rows if plan.limit < 0 else min(rows, plan.limit)
    if plan.reductions:
        count = 0 if plan.limit == 0 else 1
    var cells = _checked_mul(rows, plan.slots)
    var current_rows = rows
    var result_capacity = rows
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
    var program = Program(plan, resident_input, Optional[Scope](scope.copy()))
    var resources = estimate_resources(
        plan, rows, Bool(resident_input), terminal_boundary, export_values
    )
    var remap_plan = Optional[MapPlan]()
    var remap_owner = Optional[Submission]()
    var peak = _checked_add(
        resources.peak_bytes,
        _checked_mul(
            len(program.literals) - len(plan.literals) - len(plan.zone_words),
            16,
        ),
    )
    if text_domain:
        var base_domain = ArcPointer(
            TextDomain(scope, plan.source_domain.value(), False)
        )
        remap_plan = MapPlan(
            text_domain.value(),
            source_maps(
                plan.source, plan.source_text, base_domain, resident_input, rows
            ),
        )
        peak = _checked_add(peak, remap_plan.value().bytes())
    _ = scope.check(peak)
    # Host staging owns only immutable source columns. Derived slots and error
    # words start at zero on the device, rather than crossing the host boundary.
    var host_cells = 0 if resident_input else _checked_mul(
        rows, plan.source.width()
    )
    var words = List[UInt64](length=2 * host_cells, fill=0)
    var validity = List[UInt8](length=host_cells, fill=0)
    var errors = List[UInt32]()
    var schema_columns = List[Series](capacity=len(plan.outputs))
    var output_slots = List[Int64](capacity=len(plan.outputs))
    for i in range(len(plan.outputs)):
        var field = Series.full_null(
            plan.outputs[i].name, plan.outputs[i].dtype, 0
        )
        field._field_metadata = plan.outputs[i].field_metadata
        schema_columns.append(field^)
        output_slots.append(
            Int64(i if plan.reductions else plan.outputs[i].slot)
        )
    var schema = DataFrame(schema_columns^)
    var resident_result: Optional[Table] = None
    var projection_view = False
    if (
        resident_input
        and (
            resident_input.value().storage
            or resident_input.value().identity_storage
        )
        and not len(plan.steps)
        and not plan.reductions
        and plan.ordered_mode == 0
        and (not remap_plan or not len(remap_plan.value().descriptors))
    ):
        var source_mapping = List[Int](capacity=plan.source.width())
        for column in plan.source._columns:
            var found = -1
            for index in range(resident_input.value().schema.width()):
                ref field = resident_input.value().schema._columns[index]
                if (
                    field.name() == column.name()
                    and field.dtype() == column.dtype()
                ):
                    found = index
                    break
            if found < 0:
                raise Error(
                    "Mojo GPU resident source field does not match the row plan"
                )
            source_mapping.append(found)
        var mapping = List[Int](capacity=len(plan.outputs))
        for output in plan.outputs:
            if output.slot < 0 or output.slot >= len(source_mapping):
                raise Error(
                    "Mojo GPU resident projection source slot is invalid"
                )
            mapping.append(source_mapping[output.slot])
        resident_result = resident_input.value().project_view(
            scope, schema, mapping, count
        )
        projection_view = True
    ref reduction_code = program.outputs
    var has_statistics = program.has_statistics
    var has_quantiles = program.has_quantiles
    var has_float_sums = False
    for output in plan.outputs:
        has_float_sums |= (
            plan.reductions
            and (output.reduction == SUM or output.reduction == MEAN)
            and plan.slot_dtypes[output.slot].is_float()
        )
    var scalar_count = len(plan.outputs) if plan.reductions else 0
    var reduced_words = List[UInt64](length=2 * scalar_count, fill=0)
    var reduced_validity = List[UInt8](length=scalar_count, fill=0)
    var reduced_errors = List[UInt32](length=scalar_count, fill=0)
    ref code = program.code
    ref literals = program.literals
    ref types = program.node_types
    var slot_types = List[Int64]()
    if grouping:
        for dtype in plan.slot_dtypes:
            slot_types.append(_kind(dtype))
    var staging_started = Int(perf_counter_ns()) if profiling else 0
    var host_prepare_ns = staging_started - started if profiling else 0
    if not resident_input:
        for slot in range(plan.source.width()):
            _stage(plan.source._columns[slot], rows, slot, 0, words, validity)
        _stage_categories(plan.source_text, plan.source, rows, words)
    var staging_ns = (
        Int(perf_counter_ns()) - staging_started if profiling else 0
    )
    var submissions = 0
    var waits = 0
    var uploaded = 0
    var downloaded = 0
    var device_key_values = Optional[DeviceBuffer[DType.uint64]]()
    var quantile_indices = Optional[DeviceBuffer[DType.uint32]]()
    var device_ranks = Optional[DeviceBuffer[DType.uint32]]()
    var iana_replay = Optional[DeviceBuffer[DType.uint64]]()
    var iana_scalar_cache = Optional[DeviceBuffer[DType.uint8]]()
    var iana_active_scratch = Optional[DeviceBuffer[DType.uint8]]()
    var iana_scalar_resources = iana_scalar_scopes.resources(
        rows,
        _iana_scalar_slots(plan),
        checked_rank.BATCH_ROWS if plan.ordered_mode
        == checked_rank.ORDERED_RANK_INPUT else 8192,
    )
    var ordered_control = Optional[DeviceBuffer[DType.uint32]]()
    var ordered_partials = Optional[DeviceBuffer[DType.uint32]]()
    var grouped_integer_scratch = Optional[DeviceBuffer[DType.uint64]]()
    var grouped_int_parallel_outputs = 0
    var grouped_int_max_partial_states = 0
    var grouped_int_max_partials_per_group = 0
    var grouped_int_max_group_batch = 0
    var grouped_int_max_qualifying_hash_group_rows = 0
    var grouped_int_partial_tiles = 0
    var grouped_int_scratch_bytes = 0
    var radix_counts = Optional[DeviceBuffer[DType.uint32]]()
    var radix_offsets = Optional[DeviceBuffer[DType.uint32]]()
    var fault_groups = min(1024, max(1, (rows + 255) // 256))
    # One word flags a fault in any value this region writes, before a
    # filter, head or HAVING step can drop it (accelerator/faults.mojo).
    var fault_host = List[UInt32](length=1, fill=0)
    var fault_observed = False
    var fault_zeroed = False
    # Certificate regions observe their own faults in CPU order; plain
    # regions flag every written fault (accelerator/faults.mojo).
    var plain = plan.ordered_mode == 0
    var submit_started = Int(perf_counter_ns()) if profiling else 0
    if rows and cells and not projection_view:
        var ctx = scope.context()
        # All staging/program host owners are outside this protected block.
        try:
            var device_words = ctx.enqueue_create_buffer[DType.uint64](
                2 * cells
            )
            var device_valid = ctx.enqueue_create_buffer[DType.uint8](cells)
            var device_errors = ctx.enqueue_create_buffer[DType.uint32](cells)
            # Zeroed by the first observation only, so a region that
            # observes nothing moves no extra bytes.
            var fault_word = ctx.enqueue_create_buffer[DType.uint32](1)
            if _quantile_indices(plan):
                quantile_indices = ctx.enqueue_create_buffer[DType.uint32](rows)
            if resident_input:
                var mapping = List[Int64](capacity=plan.source.width())
                for column in plan.source._columns:
                    var found = -1
                    for i in range(resident_input.value().schema.width()):
                        ref field = resident_input.value().schema._columns[i]
                        if (
                            field.name() == column.name()
                            and field.dtype() == column.dtype()
                        ):
                            found = i
                            break
                    if found < 0:
                        raise Error(
                            "Mojo GPU resident source field does not match the"
                            " row plan"
                        )
                    mapping.append(Int64(found))
                _ = resident_input.value().copy_to(
                    scope,
                    device_words,
                    device_valid,
                    device_errors,
                    rows,
                    plan.slots,
                    mapping,
                )
            else:
                # Sub-buffer copies target their final column-major positions.
                # No temporary device matrix, repacking kernel or extra wait.
                ctx.enqueue_memset(device_words, UInt64(0))
                ctx.enqueue_memset(device_valid, UInt8(0))
                ctx.enqueue_memset(device_errors, UInt32(0))
                if host_cells:
                    ctx.enqueue_copy(
                        device_words.create_sub_buffer[DType.uint64](
                            0, host_cells
                        ),
                        words.unsafe_ptr(),
                    )
                    uploaded += host_cells * 8
                    if _source_high_words(plan):
                        ctx.enqueue_copy(
                            device_words.create_sub_buffer[DType.uint64](
                                cells, host_cells
                            ),
                            words.unsafe_ptr().unsafe_offset(host_cells),
                        )
                        uploaded += host_cells * 8
                    ctx.enqueue_copy(
                        device_valid.create_sub_buffer[DType.uint8](
                            0, host_cells
                        ),
                        validity.unsafe_ptr(),
                    )
                    uploaded += host_cells
            if remap_plan:
                remap_owner = submit_text(
                    scope,
                    remap_plan.take(),
                    device_words,
                    device_valid,
                    device_errors,
                    rows,
                )
            # Zero-step projections need no program buffers or kernel launches.
            if len(plan.steps):
                var device_code = ctx.enqueue_create_buffer[DType.int64](
                    len(code)
                )
                var device_literals = ctx.enqueue_create_buffer[DType.uint64](
                    len(literals)
                )
                var device_types = ctx.enqueue_create_buffer[DType.int64](
                    len(types)
                )
                ctx.enqueue_copy(device_code, code.unsafe_ptr())
                uploaded += len(code) * 8
                ctx.enqueue_copy(device_literals, literals.unsafe_ptr())
                uploaded += len(literals) * 8
                ctx.enqueue_copy(device_types, types.unsafe_ptr())
                uploaded += len(types) * 8
                if ordered:
                    device_ranks = ctx.enqueue_create_buffer[DType.uint32](
                        len(program.ordered_ranks[])
                    )
                    ctx.enqueue_copy(
                        device_ranks.value(),
                        program.ordered_ranks[].unsafe_ptr(),
                    )
                    uploaded += len(program.ordered_ranks[]) * 4
                    ordered_control = ctx.enqueue_create_buffer[DType.uint32](2)
                    if iana_certified:
                        iana_replay = ctx.enqueue_create_buffer[DType.uint64](3)
                        if iana_scalar_resources.cache_bytes:
                            iana_scalar_cache = ctx.enqueue_create_buffer[
                                DType.uint8
                            ](iana_scalar_resources.cache_bytes)
                            iana_active_scratch = ctx.enqueue_create_buffer[
                                DType.uint8
                            ](iana_scalar_resources.scratch_bytes)
                    if ordered_rows or ordered_precision:
                        ordered_partials = ctx.enqueue_create_buffer[
                            DType.uint32
                        ](2 * fault_groups)
                    if ordered_scalar:
                        ctx.enqueue_function[
                            ordered_faults.observe_scalar_inputs
                        ](
                            device_errors.unsafe_ptr(),
                            ordered_control.value().unsafe_ptr(),
                            UInt32(plan.source.width()),
                            grid_dim=1,
                            block_dim=1,
                        )
                        submissions += 1
                        var input_failure = scope.read_control(
                            ordered_control.value()
                        )
                        downloaded += 8
                        waits += 1
                        if input_failure[0]:
                            raise Error(
                                "Mojo GPU ordered scalar requires already"
                                " observed successful inputs"
                            )
                    if (
                        ordered_private
                        or ordered_decimal
                        or observed_producer
                        or ordered_precision
                        or observed_rank
                    ):
                        # Input cells are ordinary flags from an already
                        # observed producer. Prove every staged source cell is
                        # zero before any kernel can interpret ranked tokens.
                        ctx.enqueue_function[
                            ordered_faults.observe_input_partials
                        ](
                            device_errors.unsafe_ptr(),
                            ordered_partials.value().unsafe_ptr(),
                            UInt32(plan.source.width()),
                            UInt32(rows),
                            UInt32(rows),
                            UInt32(fault_groups),
                            grid_dim=(fault_groups + 127) // 128,
                            block_dim=128,
                        )
                        ctx.enqueue_function[
                            ordered_faults.observe_input_finish
                        ](
                            ordered_partials.value().unsafe_ptr(),
                            ordered_control.value().unsafe_ptr(),
                            UInt32(fault_groups),
                            grid_dim=1,
                            block_dim=1,
                        )
                        submissions += 2
                        var input_failure = scope.read_control(
                            ordered_control.value()
                        )
                        downloaded += 8
                        waits += 1
                        if input_failure[0]:
                            raise Error(
                                "Mojo GPU ordered row producer requires already"
                                " observed successful inputs"
                            )
                if len(plan.key_values):
                    var lookup = ctx.enqueue_create_buffer[DType.uint64](
                        len(plan.key_values)
                    )
                    ctx.enqueue_copy(lookup, plan.key_values.unsafe_ptr())
                    uploaded += len(plan.key_values) * 8
                    device_key_values = lookup^
                for step in plan.steps:
                    var observed_step = not observed_rank
                    if observed_rank:
                        for at in range(len(plan.steps)):
                            if plan.steps[at].start == step.start:
                                observed_step = plan.iana_mask_bases[at] >= 0
                                break
                    if current_rows == 0:
                        break
                    if step.grouped:
                        var blocks = (current_rows + 255) // 256
                        var keys = Int(code[4 * step.start + 1])
                        var index = ctx.enqueue_create_buffer[DType.uint32](
                            current_rows
                        )
                        var next_index = ctx.enqueue_create_buffer[
                            DType.uint32
                        ](current_rows)
                        var positions = ctx.enqueue_create_buffer[DType.uint32](
                            current_rows
                        )
                        var starts = ctx.enqueue_create_buffer[DType.uint32](
                            current_rows
                        )
                        var offsets = ctx.enqueue_create_buffer[DType.uint32](
                            blocks * 2
                        )
                        var metadata = ctx.enqueue_create_buffer[DType.uint32](
                            4
                        )
                        var device_slot_types = ctx.enqueue_create_buffer[
                            DType.int64
                        ](len(slot_types))
                        ctx.enqueue_copy(
                            device_slot_types, slot_types.unsafe_ptr()
                        )
                        uploaded += len(slot_types) * 8
                        var integer_partial_nodes = (
                            _grouped_integer_partial_nodes(
                                plan, step.start, step.nodes
                            )
                        )
                        var hash_route = False
                        var shape = List[UInt32]()
                        if current_rows >= 256 and hash_grouping.eligible(
                            plan, step
                        ):
                            var capacity = hash_grouping.capacity(current_rows)
                            var representatives = ctx.enqueue_create_buffer[
                                DType.uint32
                            ](capacity)
                            var counts = ctx.enqueue_create_buffer[
                                DType.uint32
                            ](capacity)
                            try:
                                ctx.enqueue_function[hash_grouping.initialize](
                                    representatives.unsafe_ptr(),
                                    counts.unsafe_ptr(),
                                    metadata.unsafe_ptr(),
                                    UInt32(capacity),
                                    grid_dim=(capacity + 255) // 256,
                                    block_dim=256,
                                )
                                ctx.enqueue_function[hash_grouping.build](
                                    device_words.unsafe_ptr(),
                                    device_valid.unsafe_ptr(),
                                    device_errors.unsafe_ptr(),
                                    device_code.unsafe_ptr(),
                                    device_types.unsafe_ptr(),
                                    representatives.unsafe_ptr(),
                                    counts.unsafe_ptr(),
                                    positions.unsafe_ptr(),
                                    metadata.unsafe_ptr(),
                                    UInt32(step.start),
                                    UInt32(keys),
                                    UInt32(current_rows),
                                    UInt32(rows),
                                    UInt32(plan.slots),
                                    UInt32(capacity),
                                    grid_dim=blocks,
                                    block_dim=256,
                                )
                                ctx.enqueue_function[
                                    hash_grouping.offsets_local
                                ](
                                    counts.unsafe_ptr(),
                                    representatives.unsafe_ptr(),
                                    next_index.unsafe_ptr(),
                                    metadata.unsafe_ptr(),
                                    UInt32(capacity),
                                    UInt32(
                                        hash_grouping.MINIMUM_PARALLEL_GROUP_ROWS
                                    ) if len(integer_partial_nodes) else UInt32(
                                        0
                                    ),
                                    grid_dim=(capacity + 255) // 256,
                                    block_dim=256,
                                )
                                ctx.enqueue_function[
                                    hash_grouping.offsets_blocks
                                ](
                                    next_index.unsafe_ptr(),
                                    metadata.unsafe_ptr(),
                                    UInt32((capacity + 255) // 256),
                                    grid_dim=1,
                                    block_dim=1,
                                )
                                ctx.enqueue_function[
                                    hash_grouping.offsets_finish
                                ](
                                    counts.unsafe_ptr(),
                                    representatives.unsafe_ptr(),
                                    next_index.unsafe_ptr(),
                                    starts.unsafe_ptr(),
                                    metadata.unsafe_ptr(),
                                    UInt32(capacity),
                                    grid_dim=(capacity + 255) // 256,
                                    block_dim=256,
                                )
                                ctx.enqueue_function[hash_grouping.scatter](
                                    positions.unsafe_ptr(),
                                    counts.unsafe_ptr(),
                                    index.unsafe_ptr(),
                                    metadata.unsafe_ptr(),
                                    UInt32(current_rows),
                                    grid_dim=blocks,
                                    block_dim=256,
                                )
                                submissions += 6
                                shape = scope.read_control(metadata)
                                downloaded += len(shape) * 4
                                waits += 1
                                hash_route = shape[2] == 0
                                if hash_route:
                                    grouped_int_max_qualifying_hash_group_rows = max(
                                        grouped_int_max_qualifying_hash_group_rows,
                                        Int(shape[3]),
                                    )
                                    trace_path("metal.hash_grouping")
                                else:
                                    trace_path(
                                        "metal.hash_grouping_probe_fallback"
                                    )
                            except error:
                                scope.synchronize()
                                raise error^
                        if not hash_route:
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
                                    UInt32(plan.slots),
                                    device_literals.unsafe_ptr(),
                                    grid_dim=merge_grid(
                                        current_rows, run_width
                                    ),
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
                                UInt32(plan.slots),
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
                            shape = scope.read_control(metadata)
                            downloaded += len(shape) * 4
                            waits += 1
                        if shape[1]:
                            raise data_fault(
                                "Mojo GPU group key failed: checked arithmetic"
                                " overflow or strict cast failed"
                            )
                        var gathered_words = ctx.enqueue_create_buffer[
                            DType.uint64
                        ](2 * cells)
                        var gathered_valid = ctx.enqueue_create_buffer[
                            DType.uint8
                        ](cells)
                        var gathered_errors = ctx.enqueue_create_buffer[
                            DType.uint32
                        ](cells)
                        if shape[0]:
                            if not hash_route:
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
                                    UInt32(plan.slots),
                                    grid_dim=blocks,
                                    block_dim=256,
                                )
                            if code[
                                4 * step.start
                            ] == ROW_GROUP_HEADER_WINDOW and literals[
                                step.start
                            ] & (
                                UInt64(1) << 32
                            ):
                                var lookup_pointer = (
                                    device_literals.unsafe_ptr()
                                    .unsafe_mut_cast[False]()
                                    .unsafe_origin_cast[ImmutAnyOrigin]()
                                )
                                if device_key_values:
                                    lookup_pointer = (
                                        device_key_values.value()
                                        .unsafe_ptr()
                                        .unsafe_mut_cast[False]()
                                        .unsafe_origin_cast[ImmutAnyOrigin]()
                                    )
                                ctx.enqueue_function[group_window_prepare](
                                    device_words.unsafe_ptr(),
                                    device_valid.unsafe_ptr(),
                                    index.unsafe_ptr(),
                                    starts.unsafe_ptr(),
                                    positions.unsafe_ptr(),
                                    device_code.unsafe_ptr(),
                                    device_literals.unsafe_ptr(),
                                    device_types.unsafe_ptr(),
                                    lookup_pointer,
                                    UInt32(step.start),
                                    shape[0],
                                    UInt32(current_rows),
                                    UInt32(rows),
                                    UInt32(plan.slots),
                                    grid_dim=1,
                                    block_dim=1,
                                )
                                var group_order = positions^
                                var next_group_order = next_index^
                                var group_run_width = 1
                                var group_start_slot = UInt32(
                                    code[4 * step.start + 2]
                                )
                                while group_run_width < Int(shape[0]):
                                    ctx.enqueue_function[group_window_merge](
                                        device_words.unsafe_ptr(),
                                        index.unsafe_ptr(),
                                        starts.unsafe_ptr(),
                                        group_order.unsafe_ptr(),
                                        next_group_order.unsafe_ptr(),
                                        group_start_slot,
                                        shape[0],
                                        UInt32(rows),
                                        UInt32(group_run_width),
                                        grid_dim=(Int(shape[0]) + 255) // 256,
                                        block_dim=256,
                                    )
                                    var previous_group_order = group_order^
                                    group_order = next_group_order^
                                    next_group_order = previous_group_order^
                                    group_run_width *= 2
                                    submissions += 1
                                ctx.enqueue_function[group_window_schedule](
                                    device_words.unsafe_ptr(),
                                    starts.unsafe_ptr(),
                                    group_order.unsafe_ptr(),
                                    metadata.unsafe_ptr(),
                                    group_start_slot,
                                    shape[0],
                                    UInt32(current_rows),
                                    UInt32(rows),
                                    UInt32(literals[step.start] & 0xFFFFFFFF),
                                    grid_dim=1,
                                    block_dim=1,
                                )
                                submissions += 2
                                positions = group_order^
                                next_index = next_group_order^
                            var partials_per_group = 1
                            var group_batch = 0
                            if (
                                len(integer_partial_nodes)
                                and shape[0] > 0
                                and (not hash_route or shape[3] != 0)
                            ):
                                var average_group_rows = (
                                    Int(current_rows) + Int(shape[0]) - 1
                                ) // Int(shape[0])
                                var desired_partials = (
                                    average_group_rows + 255
                                ) // 256
                                partials_per_group = min(
                                    4096,
                                    min(
                                        4096 // Int(shape[0]),
                                        desired_partials,
                                    ),
                                )
                                if partials_per_group >= 2:
                                    group_batch = min(
                                        Int(shape[0]),
                                        4096 // partials_per_group,
                                    )
                                    if not grouped_integer_scratch:
                                        grouped_integer_scratch = (
                                            ctx.enqueue_create_buffer[
                                                DType.uint64
                                            ](4096 * 8)
                                        )
                                        grouped_int_scratch_bytes = (
                                            4096 * STATE_WORDS * 8
                                        )
                                    grouped_int_parallel_outputs += len(
                                        integer_partial_nodes
                                    )
                                    grouped_int_max_partial_states = max(
                                        grouped_int_max_partial_states,
                                        group_batch * partials_per_group,
                                    )
                                    grouped_int_max_partials_per_group = max(
                                        grouped_int_max_partials_per_group,
                                        partials_per_group,
                                    )
                                    grouped_int_max_group_batch = max(
                                        grouped_int_max_group_batch,
                                        group_batch,
                                    )
                                    grouped_int_partial_tiles += (
                                        (Int(shape[0]) + group_batch - 1)
                                        // group_batch
                                        * len(integer_partial_nodes)
                                    )
                                else:
                                    partials_per_group = 1
                            ctx.enqueue_function[group_reduce](
                                device_words.unsafe_ptr(),
                                device_valid.unsafe_ptr(),
                                device_errors.unsafe_ptr(),
                                gathered_words.unsafe_ptr(),
                                gathered_valid.unsafe_ptr(),
                                gathered_errors.unsafe_ptr(),
                                index.unsafe_ptr(),
                                starts.unsafe_ptr(),
                                metadata.unsafe_ptr(),
                                device_code.unsafe_ptr(),
                                device_literals.unsafe_ptr(),
                                device_slot_types.unsafe_ptr(),
                                UInt32(step.start),
                                UInt32(step.nodes),
                                shape[0],
                                UInt32(current_rows),
                                UInt32(rows),
                                UInt32(plan.slots),
                                UInt32(hash_route),
                                UInt32(partials_per_group),
                                grid_dim=(Int(shape[0]) * step.nodes + 255)
                                // 256,
                                block_dim=256,
                            )
                            submissions += 1 + Int(not hash_route)
                            if partials_per_group > 1:
                                for node in integer_partial_nodes:
                                    for group_base in range(
                                        0, Int(shape[0]), group_batch
                                    ):
                                        var current_batch = min(
                                            group_batch,
                                            Int(shape[0]) - group_base,
                                        )
                                        ctx.enqueue_function[
                                            group_integer_partials
                                        ](
                                            device_words.unsafe_ptr(),
                                            device_valid.unsafe_ptr(),
                                            device_errors.unsafe_ptr(),
                                            index.unsafe_ptr(),
                                            starts.unsafe_ptr(),
                                            device_code.unsafe_ptr(),
                                            device_literals.unsafe_ptr(),
                                            device_slot_types.unsafe_ptr(),
                                            UInt32(step.start),
                                            node,
                                            UInt32(group_base),
                                            UInt32(current_batch),
                                            shape[0],
                                            UInt32(partials_per_group),
                                            UInt32(current_rows),
                                            UInt32(rows),
                                            grouped_integer_scratch.value().unsafe_ptr(),
                                            grid_dim=(
                                                current_batch
                                                * partials_per_group
                                            ),
                                            block_dim=256,
                                        )
                                        ctx.enqueue_function[
                                            group_integer_finish
                                        ](
                                            gathered_words.unsafe_ptr(),
                                            gathered_valid.unsafe_ptr(),
                                            gathered_errors.unsafe_ptr(),
                                            starts.unsafe_ptr(),
                                            device_code.unsafe_ptr(),
                                            device_literals.unsafe_ptr(),
                                            device_slot_types.unsafe_ptr(),
                                            UInt32(step.start),
                                            node,
                                            UInt32(group_base),
                                            UInt32(current_batch),
                                            shape[0],
                                            UInt32(partials_per_group),
                                            UInt32(current_rows),
                                            UInt32(rows),
                                            UInt32(plan.slots),
                                            grouped_integer_scratch.value().unsafe_ptr(),
                                            grid_dim=current_batch,
                                            block_dim=256,
                                        )
                                        submissions += 2
                            if _group_quantiles(plan, step.start, step.nodes):
                                ctx.enqueue_function[
                                    quantiles.correct_zero_groups
                                ](
                                    device_words.unsafe_ptr(),
                                    device_valid.unsafe_ptr(),
                                    gathered_words.unsafe_ptr(),
                                    gathered_valid.unsafe_ptr(),
                                    index.unsafe_ptr(),
                                    starts.unsafe_ptr(),
                                    device_code.unsafe_ptr(),
                                    device_literals.unsafe_ptr(),
                                    quantile_indices.value().unsafe_ptr(),
                                    UInt32(step.start),
                                    UInt32(step.nodes),
                                    shape[0],
                                    UInt32(current_rows),
                                    UInt32(rows),
                                    grid_dim=(Int(shape[0]) + 127) // 128,
                                    block_dim=128,
                                )
                                submissions += 1
                            # Starts, sort indices, and descriptor owners outlive all queued reads.
                            scope.synchronize()
                            waits += 1
                            _ = positions^
                            _ = next_index^
                        current_rows = Int(shape[0])
                        device_words = gathered_words^
                        device_valid = gathered_valid^
                        device_errors = gathered_errors^
                        # A later filter (HAVING) can drop a failed group.
                        for node in range(step.start, step.start + step.nodes):
                            if (
                                plain
                                and code[4 * node] == ROW_GROUP_OUTPUT
                                and _can_fault(
                                    plan.slot_dtypes[Int(code[4 * node + 2])]
                                )
                            ):
                                if not fault_zeroed:
                                    ctx.enqueue_copy(
                                        fault_word, fault_host.unsafe_ptr()
                                    )
                                    uploaded += 4
                                    fault_zeroed = True
                                ctx.enqueue_function[observe_slot](
                                    device_errors.unsafe_ptr(),
                                    fault_word.unsafe_ptr(),
                                    UInt32(code[4 * node + 2]),
                                    UInt32(current_rows),
                                    UInt32(rows),
                                    grid_dim=(current_rows + 255) // 256,
                                    block_dim=256,
                                )
                                submissions += 1
                                fault_observed = True
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
                        var next_index = ctx.enqueue_create_buffer[
                            DType.uint32
                        ](current_rows)
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
                                    UInt32(plan.slots),
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
                                    UInt32(plan.slots),
                                    device_literals.unsafe_ptr(),
                                    grid_dim=merge_grid(
                                        current_rows, run_width
                                    ),
                                    block_dim=256,
                                )
                            var previous = index^
                            index = next_index^
                            next_index = previous^
                            run_width *= 2
                            submissions += 1
                        scope.read_control_into(failures, host_failures)
                        downloaded += len(host_failures) * 4
                        waits += 1
                        for failure in host_failures:
                            if failure:
                                raise data_fault(
                                    "Mojo GPU window key failed: checked"
                                    " arithmetic overflow or strict cast failed"
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
                                    UInt32(plan.slots),
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
                                UInt32(plan.slots),
                                grid_dim=blocks,
                                block_dim=256,
                            )
                            submissions += 1
                            # Permutation, dense prefixes and block owners outlive
                            # their last queued reads before the next plan step.
                            scope.synchronize()
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
                                    UInt32(plan.slots),
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
                                    UInt32(plan.slots),
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
                            scope.read_control_into(failures, host_failures)
                            downloaded += len(host_failures) * 4
                            waits += 1
                            if is_duration:
                                for block in range(blocks):
                                    if host_failures[4 * block]:
                                        raise data_fault(
                                            "Mojo GPU time window index"
                                            " contains nulls"
                                        )
                                    if host_failures[4 * block + 1]:
                                        raise data_fault(
                                            "Mojo GPU time window index is not"
                                            " sorted within its group"
                                        )
                                    if host_failures[4 * block + 2]:
                                        raise data_fault(
                                            "Mojo GPU time window bounds"
                                            " overflow"
                                        )
                                    if host_failures[4 * block + 3]:
                                        raise data_fault(
                                            "Mojo GPU duration window decimal precision overflow" if plan.slot_dtypes[
                                                step.slot
                                            ].is_decimal() else "Mojo GPU duration window arithmetic overflow"
                                        )
                            else:
                                for failure in host_failures:
                                    if failure:
                                        raise data_fault(
                                            "Mojo GPU cumulative or rolling"
                                            " window overflow or input"
                                            " expression failure"
                                        )
                        continue
                    if step.order:
                        var blocks = (current_rows + 255) // 256
                        var index = ctx.enqueue_create_buffer[DType.uint32](
                            current_rows
                        )
                        var next_index = ctx.enqueue_create_buffer[
                            DType.uint32
                        ](current_rows)
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
                        var topk = _topk_limit(
                            plan, current_rows, terminal_boundary
                        )
                        if topk > 0:
                            # Sort bounded base runs with the shared comparator,
                            # then compact each run to its first K indices.
                            var run_width = 1
                            while (
                                run_width < current_rows
                                and run_width < TOPK_BASE_RUN_ROWS
                            ):
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
                                    UInt32(plan.slots),
                                    device_literals.unsafe_ptr(),
                                    grid_dim=merge_grid(
                                        current_rows, run_width
                                    ),
                                    block_dim=256,
                                )
                                var previous = index^
                                index = next_index^
                                next_index = previous^
                                run_width *= 2
                                submissions += 1
                            var keep_per_run = min(topk, TOPK_BASE_RUN_ROWS)
                            var active_rows = _topk_retained_rows(
                                current_rows, TOPK_BASE_RUN_ROWS, topk
                            )
                            ctx.enqueue_function[order_compact_runs](
                                index.unsafe_ptr(),
                                next_index.unsafe_ptr(),
                                UInt32(current_rows),
                                UInt32(TOPK_BASE_RUN_ROWS),
                                UInt32(keep_per_run),
                                UInt32(active_rows),
                                grid_dim=(active_rows + 255) // 256,
                                block_dim=256,
                            )
                            var previous = index^
                            index = next_index^
                            next_index = previous^
                            submissions += 1
                            var child_width = TOPK_BASE_RUN_ROWS
                            while child_width < current_rows:
                                var parent_width = 2 * child_width
                                var parent_keep = min(topk, parent_width)
                                active_rows = _topk_retained_rows(
                                    current_rows, parent_width, topk
                                )
                                var pairs = (
                                    current_rows + parent_width - 1
                                ) // parent_width
                                var tiles_per_pair = (
                                    parent_keep + 2047
                                ) // 2048
                                ctx.enqueue_function[order_merge_topk](
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
                                    UInt32(child_width),
                                    UInt32(min(topk, child_width)),
                                    UInt32(parent_keep),
                                    UInt32(plan.slots),
                                    device_literals.unsafe_ptr(),
                                    grid_dim=pairs * tiles_per_pair,
                                    block_dim=256,
                                )
                                var previous = index^
                                index = next_index^
                                next_index = previous^
                                child_width = parent_width
                                submissions += 1
                            current_rows = topk
                            result_capacity = topk
                        else:
                            var radix_pass_count = _radix_sort_passes(
                                plan, step
                            )
                            if radix_pass_count > 0 and current_rows > 0:
                                var tiles = blocks
                                var level_counts = List[Int]()
                                var level_offsets = List[Int]()
                                var level_count = tiles
                                var summary_count = 0
                                while level_count > 0:
                                    level_counts.append(level_count)
                                    level_offsets.append(summary_count)
                                    summary_count = _checked_add(
                                        summary_count, level_count
                                    )
                                    if level_count == 1:
                                        break
                                    level_count = (level_count + 255) // 256
                                radix_counts = ctx.enqueue_create_buffer[
                                    DType.uint32
                                ](16 * summary_count)
                                radix_offsets = ctx.enqueue_create_buffer[
                                    DType.uint32
                                ](16)
                                var node = UInt32(step.start)
                                var width = integer_width(
                                    _kind(plan.node_dtypes[step.start])
                                )
                                var digits = Int(width // 4) + 1
                                for pass_index in range(digits):
                                    var null_marker = UInt32(
                                        pass_index == Int(width // 4)
                                    )
                                    var shift = UInt32(pass_index * 4)
                                    ctx.enqueue_function[radix_histogram](
                                        device_words.unsafe_ptr(),
                                        device_valid.unsafe_ptr(),
                                        index.unsafe_ptr(),
                                        radix_counts.value().unsafe_ptr(),
                                        device_code.unsafe_ptr(),
                                        device_types.unsafe_ptr(),
                                        node,
                                        UInt32(current_rows),
                                        UInt32(rows),
                                        UInt32(summary_count),
                                        shift,
                                        null_marker,
                                        grid_dim=tiles,
                                        block_dim=256,
                                    )
                                    submissions += 1
                                    for level in range(len(level_counts) - 1):
                                        var count = level_counts[level]
                                        var parent_base = level_offsets[
                                            level + 1
                                        ]
                                        ctx.enqueue_function[radix_scan_level](
                                            radix_counts.value().unsafe_ptr(),
                                            UInt32(count),
                                            UInt32(level_offsets[level]),
                                            UInt32(parent_base),
                                            UInt32(summary_count),
                                            grid_dim=16
                                            * ((count + 255) // 256),
                                            block_dim=256,
                                        )
                                        submissions += 1
                                    for level in range(len(level_counts) - 1):
                                        var at = len(level_counts) - 2 - level
                                        var count = level_counts[at]
                                        if count <= 256:
                                            continue
                                        ctx.enqueue_function[
                                            radix_apply_carries
                                        ](
                                            radix_counts.value().unsafe_ptr(),
                                            UInt32(count),
                                            UInt32(level_offsets[at]),
                                            UInt32(level_offsets[at + 1]),
                                            UInt32(summary_count),
                                            grid_dim=16
                                            * ((count + 255) // 256),
                                            block_dim=256,
                                        )
                                        submissions += 1
                                    ctx.enqueue_function[radix_bucket_offsets](
                                        radix_counts.value().unsafe_ptr(),
                                        radix_offsets.value().unsafe_ptr(),
                                        UInt32(tiles),
                                        UInt32(summary_count),
                                        grid_dim=1,
                                        block_dim=256,
                                    )
                                    ctx.enqueue_function[radix_scatter](
                                        device_words.unsafe_ptr(),
                                        device_valid.unsafe_ptr(),
                                        index.unsafe_ptr(),
                                        next_index.unsafe_ptr(),
                                        radix_counts.value().unsafe_ptr(),
                                        radix_offsets.value().unsafe_ptr(),
                                        device_code.unsafe_ptr(),
                                        device_types.unsafe_ptr(),
                                        node,
                                        UInt32(current_rows),
                                        UInt32(rows),
                                        UInt32(summary_count),
                                        shift,
                                        null_marker,
                                        grid_dim=tiles,
                                        block_dim=256,
                                    )
                                    submissions += 2
                                    # Stable passes alternate only the index owners.
                                    var previous = index^
                                    index = next_index^
                                    next_index = previous^
                            else:
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
                                        UInt32(plan.slots),
                                        device_literals.unsafe_ptr(),
                                        grid_dim=merge_grid(
                                            current_rows, run_width
                                        ),
                                        block_dim=256,
                                    )
                                    # Swapping owners keeps both buffers alive across queued passes.
                                    var previous = index^
                                    index = next_index^
                                    next_index = previous^
                                    run_width *= 2
                                    submissions += 1
                        var selected_cells = (
                            _checked_mul(current_rows, plan.slots) if topk
                            > 0 else cells
                        )
                        var gather_blocks = (current_rows + 255) // 256
                        var gathered_words = ctx.enqueue_create_buffer[
                            DType.uint64
                        ](2 * selected_cells)
                        var gathered_valid = ctx.enqueue_create_buffer[
                            DType.uint8
                        ](selected_cells)
                        var gathered_errors = ctx.enqueue_create_buffer[
                            DType.uint32
                        ](selected_cells)
                        if step.gather_count:
                            var gathers = List[Int64](
                                capacity=step.gather_count
                            )
                            for column in range(step.gather_count):
                                gathers.append(
                                    plan.gathers[step.gather_start + column]
                                )
                            try:
                                var device_gathers = ctx.enqueue_create_buffer[
                                    DType.int64
                                ](len(gathers))
                                ctx.enqueue_copy(
                                    device_gathers, gathers.unsafe_ptr()
                                )
                                uploaded += len(gathers) * 8
                                if topk > 0:
                                    ctx.enqueue_function[order_gather_topk](
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
                                        UInt32(current_rows),
                                        UInt32(plan.slots),
                                        grid_dim=gather_blocks,
                                        block_dim=256,
                                    )
                                else:
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
                                        UInt32(plan.slots),
                                        grid_dim=blocks,
                                        block_dim=256,
                                    )
                                submissions += 1
                                # The gather metadata and source owners outlive all queued reads.
                                scope.synchronize()
                                waits += 1
                            except error:
                                scope.synchronize()
                                _ = gathers^
                                raise error^
                            _ = gathers^
                        scope.read_control_into(failures, host_failures)
                        downloaded += len(host_failures) * 4
                        waits += 1
                        for failure in host_failures:
                            if failure:
                                raise data_fault(
                                    "Mojo GPU sort key failed: checked"
                                    " arithmetic overflow or strict cast failed"
                                )
                        # The readback above drains every radix read. Drop and
                        # reinitialize the scratch owners before a later sort
                        # allocates its own histograms on this context.
                        _ = radix_counts^
                        _ = radix_offsets^
                        radix_counts = Optional[DeviceBuffer[DType.uint32]]()
                        radix_offsets = Optional[DeviceBuffer[DType.uint32]]()
                        device_words = gathered_words^
                        device_valid = gathered_valid^
                        device_errors = gathered_errors^
                        continue
                    if ordered and observed_step:
                        if iana_certified:
                            var mask_base = -1
                            for at in range(len(plan.steps)):
                                if plan.steps[at].start == step.start:
                                    mask_base = program.iana_mask_bases[at]
                                    break
                            if mask_base < 0:
                                raise Error(
                                    "IANA certificate scope plane is missing"
                                )
                            # Distinct existing control owners provide unused fallback
                            # pointers when there are no scalar slots. Explicit erased
                            # origins permit switching to the separately retained buffers.
                            var scalar_cache: Pointer[UInt8, MutAnyOrigin] = (
                                ordered_partials.value()
                                .unsafe_ptr()
                                .unsafe_bitcast[UInt8]()
                                .unsafe_origin_cast[MutAnyOrigin]()
                            )
                            var active_scratch: Pointer[UInt8, MutAnyOrigin] = (
                                ordered_control.value()
                                .unsafe_ptr()
                                .unsafe_bitcast[UInt8]()
                                .unsafe_origin_cast[MutAnyOrigin]()
                            )
                            if iana_scalar_resources.cache_bytes:
                                scalar_cache = (
                                    iana_scalar_cache.value()
                                    .unsafe_ptr()
                                    .unsafe_origin_cast[MutAnyOrigin]()
                                )
                                active_scratch = (
                                    iana_active_scratch.value()
                                    .unsafe_ptr()
                                    .unsafe_origin_cast[MutAnyOrigin]()
                                )
                            var policies = (
                                mask_base
                                + 2
                                + 3 * Int(program.literals[mask_base + 1])
                                + step.nodes
                            )
                            for node in range(step.nodes):
                                var scalar_slot = (
                                    program.literals[policies + node] >> 1
                                )
                                if not scalar_slot:
                                    continue
                                ctx.enqueue_function[row_program_iana](
                                    device_words.unsafe_ptr(),
                                    device_valid.unsafe_ptr(),
                                    device_errors.unsafe_ptr(),
                                    device_code.unsafe_ptr(),
                                    device_literals.unsafe_ptr(),
                                    device_types.unsafe_ptr(),
                                    device_ranks.value().unsafe_ptr(),
                                    iana_replay.value().unsafe_ptr(),
                                    scalar_cache,
                                    active_scratch,
                                    UInt32(step.start),
                                    UInt32(step.nodes),
                                    UInt32(step.slot),
                                    UInt32(current_rows),
                                    UInt32(rows),
                                    UInt32(plan.slots),
                                    UInt32(mask_base),
                                    UInt32.MAX,
                                    UInt32.MAX,
                                    UInt32(node),
                                    UInt32(ordered_batch_rows),
                                    grid_dim=(current_rows + 255) // 256,
                                    block_dim=256,
                                )
                                ctx.enqueue_function[
                                    iana_scalar_scopes.reduce_scalar_any
                                ](
                                    active_scratch,
                                    scalar_cache,
                                    UInt32(current_rows),
                                    UInt32(ordered_batch_rows),
                                    UInt32(
                                        (current_rows + ordered_batch_rows - 1)
                                        // ordered_batch_rows
                                    ),
                                    UInt32(scalar_slot - 1),
                                    grid_dim=(
                                        iana_scalar_resources.batches + 255
                                    )
                                    // 256,
                                    block_dim=256,
                                )
                                submissions += 2
                            ctx.enqueue_function[row_program_iana](
                                device_words.unsafe_ptr(),
                                device_valid.unsafe_ptr(),
                                device_errors.unsafe_ptr(),
                                device_code.unsafe_ptr(),
                                device_literals.unsafe_ptr(),
                                device_types.unsafe_ptr(),
                                device_ranks.value().unsafe_ptr(),
                                iana_replay.value().unsafe_ptr(),
                                scalar_cache,
                                active_scratch,
                                UInt32(step.start),
                                UInt32(step.nodes),
                                UInt32(step.slot),
                                UInt32(current_rows),
                                UInt32(rows),
                                UInt32(plan.slots),
                                UInt32(mask_base),
                                UInt32.MAX,
                                UInt32.MAX,
                                UInt32.MAX,
                                UInt32(ordered_batch_rows),
                                grid_dim=(current_rows + 255) // 256,
                                block_dim=256,
                            )
                        else:
                            ctx.enqueue_function[row_program_ordered](
                                device_words.unsafe_ptr(),
                                device_valid.unsafe_ptr(),
                                device_errors.unsafe_ptr(),
                                device_code.unsafe_ptr(),
                                device_literals.unsafe_ptr(),
                                device_types.unsafe_ptr(),
                                device_ranks.value().unsafe_ptr(),
                                UInt32(step.start),
                                UInt32(step.nodes),
                                UInt32(step.slot),
                                UInt32(current_rows),
                                UInt32(rows),
                                UInt32(plan.slots),
                                grid_dim=(current_rows + 255) // 256,
                                block_dim=256,
                            )
                        if ordered_precision:
                            ctx.enqueue_function[
                                ordered_faults.observe_precision_partials
                            ](
                                device_errors.unsafe_ptr(),
                                ordered_partials.value().unsafe_ptr(),
                                UInt32(step.slot),
                                UInt32(current_rows),
                                UInt32(rows),
                                UInt32(fault_groups),
                                grid_dim=(fault_groups + 127) // 128,
                                block_dim=128,
                            )
                            ctx.enqueue_function[
                                ordered_faults.observe_precision_finish
                            ](
                                ordered_partials.value().unsafe_ptr(),
                                ordered_control.value().unsafe_ptr(),
                                UInt32(fault_groups),
                                grid_dim=1,
                                block_dim=1,
                            )
                            submissions += 2
                        elif ordered_rows:
                            ctx.enqueue_function[
                                ordered_faults.observe_partials
                            ](
                                device_errors.unsafe_ptr(),
                                ordered_partials.value().unsafe_ptr(),
                                UInt32(step.slot),
                                UInt32(current_rows),
                                UInt32(rows),
                                UInt32(fault_groups),
                                UInt32(ordered_batch_rows),
                                grid_dim=(fault_groups + 127) // 128,
                                block_dim=128,
                            )
                            ctx.enqueue_function[ordered_faults.observe_finish](
                                ordered_partials.value().unsafe_ptr(),
                                ordered_control.value().unsafe_ptr(),
                                UInt32(fault_groups),
                                UInt32(ordered_batch_rows),
                                grid_dim=1,
                                block_dim=1,
                            )
                            submissions += 2
                        else:
                            ctx.enqueue_function[ordered_faults.observe](
                                device_errors.unsafe_ptr(),
                                ordered_control.value().unsafe_ptr(),
                                UInt32(step.slot),
                                UInt32(current_rows),
                                UInt32(rows),
                                UInt32(ordered_batch_rows),
                                grid_dim=1,
                                block_dim=1,
                            )
                            submissions += 1
                        var fault = scope.read_control(ordered_control.value())
                        downloaded += 8
                        waits += 1
                        if fault[0]:
                            var category = ordered_faults.category(fault[0])
                            if ordered_precision and category != 8:
                                raise data_fault(
                                    "Mojo GPU narrow Decimal precision proof"
                                    " was violated"
                                )
                            if iana_certified and category > 64:
                                var mask_base = -1
                                for at in range(len(plan.steps)):
                                    if plan.steps[at].start == step.start:
                                        mask_base = program.iana_mask_bases[at]
                                        break
                                # Distinct existing control owners provide unused fallback
                                # pointers when there are no scalar slots. Explicit erased
                                # origins permit switching to the separately retained buffers.
                                var scalar_cache: Pointer[
                                    UInt8, MutAnyOrigin
                                ] = (
                                    ordered_partials.value()
                                    .unsafe_ptr()
                                    .unsafe_bitcast[UInt8]()
                                    .unsafe_origin_cast[MutAnyOrigin]()
                                )
                                var active_scratch: Pointer[
                                    UInt8, MutAnyOrigin
                                ] = (
                                    ordered_control.value()
                                    .unsafe_ptr()
                                    .unsafe_bitcast[UInt8]()
                                    .unsafe_origin_cast[MutAnyOrigin]()
                                )
                                if iana_scalar_resources.cache_bytes:
                                    scalar_cache = (
                                        iana_scalar_cache.value()
                                        .unsafe_ptr()
                                        .unsafe_origin_cast[MutAnyOrigin]()
                                    )
                                    active_scratch = (
                                        iana_active_scratch.value()
                                        .unsafe_ptr()
                                        .unsafe_origin_cast[MutAnyOrigin]()
                                    )
                                ctx.enqueue_function[row_program_iana](
                                    device_words.unsafe_ptr(),
                                    device_valid.unsafe_ptr(),
                                    device_errors.unsafe_ptr(),
                                    device_code.unsafe_ptr(),
                                    device_literals.unsafe_ptr(),
                                    device_types.unsafe_ptr(),
                                    device_ranks.value().unsafe_ptr(),
                                    iana_replay.value().unsafe_ptr(),
                                    scalar_cache,
                                    active_scratch,
                                    UInt32(step.start),
                                    UInt32(step.nodes),
                                    UInt32(step.slot),
                                    UInt32(current_rows),
                                    UInt32(rows),
                                    UInt32(plan.slots),
                                    UInt32(mask_base),
                                    ordered_faults.rank(fault[0]),
                                    fault[1],
                                    UInt32.MAX,
                                    UInt32(ordered_batch_rows),
                                    grid_dim=1,
                                    block_dim=1,
                                )
                                var detail = List[UInt64](length=3, fill=0)
                                scope.read_control_into(
                                    iana_replay.value(), detail
                                )
                                downloaded += 24
                                waits += 1
                                submissions += 1
                                _raise_iana(
                                    plan,
                                    step.start,
                                    step.nodes,
                                    fault[0],
                                    detail,
                                )
                            raise_value_error(
                                ordered_faults.category(fault[0]),
                                (
                                    "Mojo GPU arithmetic overflow or strict"
                                    " cast failed"
                                ),
                            )
                    else:
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
                            UInt32(plan.slots),
                            grid_dim=(current_rows + 255) // 256,
                            block_dim=256,
                        )
                    submissions += 1
                    if plain and _step_can_fault(
                        code, types, step.start, step.nodes
                    ):
                        if not fault_zeroed:
                            ctx.enqueue_copy(
                                fault_word, fault_host.unsafe_ptr()
                            )
                            uploaded += 4
                            fault_zeroed = True
                        ctx.enqueue_function[observe_slot](
                            device_errors.unsafe_ptr(),
                            fault_word.unsafe_ptr(),
                            UInt32(step.slot),
                            UInt32(current_rows),
                            UInt32(rows),
                            grid_dim=(current_rows + 255) // 256,
                            block_dim=256,
                        )
                        submissions += 1
                        fault_observed = True
                    if step.filter:
                        var blocks = (current_rows + 255) // 256
                        var positions = ctx.enqueue_create_buffer[DType.uint32](
                            current_rows
                        )
                        var offsets = ctx.enqueue_create_buffer[DType.uint32](
                            blocks * 2
                        )
                        var metadata = ctx.enqueue_create_buffer[DType.uint32](
                            2
                        )
                        var gathered_words = ctx.enqueue_create_buffer[
                            DType.uint64
                        ](2 * cells)
                        var gathered_valid = ctx.enqueue_create_buffer[
                            DType.uint8
                        ](cells)
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
                        var shape = scope.read_control(metadata)
                        downloaded += len(shape) * 4
                        waits += 1
                        if shape[1]:
                            raise data_fault(
                                "Mojo GPU filter predicate failed: checked"
                                " arithmetic overflow or strict cast failed"
                            )
                        if shape[0] and step.gather_count:
                            var gathers = List[Int64](
                                capacity=step.gather_count
                            )
                            for column in range(step.gather_count):
                                gathers.append(
                                    plan.gathers[step.gather_start + column]
                                )
                            try:
                                var device_gathers = ctx.enqueue_create_buffer[
                                    DType.int64
                                ](len(gathers))
                                ctx.enqueue_copy(
                                    device_gathers, gathers.unsafe_ptr()
                                )
                                uploaded += len(gathers) * 8
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
                                    UInt32(plan.slots),
                                    grid_dim=blocks,
                                    block_dim=256,
                                )
                                submissions += 1
                                # Complete gathers before replacing their source owners.
                                scope.synchronize()
                                waits += 1
                            except error:
                                scope.synchronize()
                                _ = gathers^
                                raise error^
                            _ = gathers^
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
                var results = ctx.enqueue_create_buffer[DType.uint64](
                    2 * outputs
                )
                var result_validity = ctx.enqueue_create_buffer[DType.uint8](
                    outputs
                )
                var result_errors = ctx.enqueue_create_buffer[DType.uint32](
                    outputs
                )
                ctx.enqueue_copy(descriptors, reduction_code.unsafe_ptr())
                uploaded += len(reduction_code) * 8
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
                    UInt32(plan.slots),
                    UInt32(has_float_sums),
                    grid_dim=(outputs * groups + 127) // 128,
                    block_dim=128,
                )
                if has_float_sums:
                    ctx.enqueue_function[reduce_float_partials](
                        device_words.unsafe_ptr(),
                        device_valid.unsafe_ptr(),
                        device_errors.unsafe_ptr(),
                        descriptors.unsafe_ptr(),
                        partials.unsafe_ptr(),
                        UInt32(outputs),
                        UInt32(groups),
                        UInt32(current_rows),
                        UInt32(rows),
                        grid_dim=outputs * groups,
                        block_dim=256,
                    )
                    submissions += 1
                ctx.enqueue_function[reduce_finish](
                    partials.unsafe_ptr(),
                    descriptors.unsafe_ptr(),
                    results.unsafe_ptr(),
                    result_validity.unsafe_ptr(),
                    result_errors.unsafe_ptr(),
                    UInt32(outputs),
                    UInt32(groups),
                    UInt32(current_rows),
                    UInt32(1),
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
                    ctx.enqueue_function[quantiles.correct_zero_partials](
                        device_words.unsafe_ptr(),
                        device_valid.unsafe_ptr(),
                        descriptors.unsafe_ptr(),
                        partials.unsafe_ptr(),
                        quantile_indices.value().unsafe_ptr(),
                        UInt32(outputs),
                        UInt32(groups),
                        UInt32(current_rows),
                        UInt32(rows),
                        grid_dim=1,
                        block_dim=1,
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
                    submissions += 3
                # A head(0) or later consumer can skip exporting a failed state.
                for output in plan.outputs:
                    if plain and _can_fault(plan.slot_dtypes[output.slot]):
                        fault_observed = True
                if fault_observed:
                    if not fault_zeroed:
                        ctx.enqueue_copy(fault_word, fault_host.unsafe_ptr())
                        uploaded += 4
                    ctx.enqueue_function[observe_slot](
                        result_errors.unsafe_ptr(),
                        fault_word.unsafe_ptr(),
                        UInt32(0),
                        UInt32(outputs),
                        UInt32(outputs),
                        grid_dim=(outputs + 255) // 256,
                        block_dim=256,
                    )
                    submissions += 1
                scope.synchronize()
                waits += 1
                if fault_observed:
                    _observe_faults(scope, fault_word)
                    downloaded += 4
                    waits += 1
                resident_result = Table(
                    Matrix(
                        scope,
                        1,
                        outputs,
                        results^,
                        result_validity^,
                        result_errors^,
                    ),
                    schema.copy(),
                    output_slots.copy(),
                    count,
                )
            else:
                count = current_rows if plan.limit < 0 else min(
                    current_rows, plan.limit
                )
                scope.synchronize()
                waits += 1
                if fault_observed:
                    _observe_faults(scope, fault_word)
                    downloaded += 4
                    waits += 1
                resident_result = Table(
                    Matrix(
                        scope,
                        result_capacity,
                        plan.slots,
                        device_words^,
                        device_valid^,
                        device_errors^,
                    ),
                    schema.copy(),
                    output_slots.copy(),
                    count,
                )
        except error:
            scope.synchronize()
            _ = quantile_indices^
            _ = words^
            _ = validity^
            _ = errors^
            _ = program^
            _ = slot_types^
            _ = device_ranks^
            _ = iana_scalar_cache^
            _ = iana_active_scratch^
            _ = iana_replay^
            _ = ordered_control^
            _ = ordered_partials^
            _ = grouped_integer_scratch^
            _ = radix_counts^
            _ = radix_offsets^
            _ = remap_owner^
            raise error^
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
    var initialization_ns = scope.timing_ns()[0] - timing_before[0]
    var submit_wait_ns = (
        Int(perf_counter_ns())
        - submit_started
        - initialization_ns if profiling else 0
    )
    # Mojo destroys owners after their last use. Explicitly consume every
    # asynchronous upload owner only after the completed submission phase.
    _ = words^
    _ = validity^
    _ = errors^
    _ = program^
    _ = slot_types^
    _ = device_key_values^
    _ = quantile_indices^
    _ = device_ranks^
    _ = iana_scalar_cache^
    _ = iana_active_scratch^
    _ = iana_replay^
    _ = ordered_control^
    _ = ordered_partials^
    _ = grouped_integer_scratch^
    _ = radix_counts^
    _ = radix_offsets^
    _ = remap_owner^
    if not resident_result:
        if plan.reductions and count:
            resident_result = Table.identity(
                scope,
                HostMatrix(
                    schema.copy(),
                    count,
                    reduced_words^,
                    reduced_validity^,
                    reduced_errors^,
                ),
            )
        elif plan.reductions:
            resident_result = Table.empty(schema.copy())
        elif schema.width() == 0:
            resident_result = Table.rows_only(scope, count)
        else:
            resident_result = Table.empty(schema.copy())
    scope.record_upload(uploaded)
    scope.record_download(downloaded)
    scope.record_launches(submissions)
    var result = resident_result.value().copy()
    if text_domain:
        result.attach_text(plan.output_text, text_domain.value())
        if resident_input:
            result.category_leases = (
                resident_input.value().category_leases.copy()
            )
        for output in range(len(plan.outputs)):
            if not result.categories[output]:
                continue
            var slot = plan.outputs[output].slot
            if slot < 0 or slot >= plan.source.width():
                continue
            var name = plan.source._columns[slot].name()
            if resident_input:
                for column in range(resident_input.value().schema.width()):
                    if (
                        resident_input.value().schema._columns[column].name()
                        == name
                        and resident_input.value().categories[column]
                    ):
                        result.categories[output] = CategoryField(
                            resident_input.value()
                            .categories[column]
                            .value()
                            .owner.copy(),
                            result.categories[output].value().string_codes,
                        )
            else:
                for field in plan.source_text:
                    if field.name == name and field.category:
                        result.categories[output] = CategoryField(
                            field.category.value().owner.copy(),
                            result.categories[output].value().string_codes,
                        )
    if resident_input:
        for output in range(len(plan.outputs)):
            var slot = plan.outputs[output].slot
            if (
                plan.outputs[output].reduction < 0
                and slot >= 0
                and slot < plan.source.width()
            ):
                var name = plan.source._columns[slot].name()
                for column in range(resident_input.value().schema.width()):
                    if (
                        resident_input.value().schema._columns[column].name()
                        == name
                    ):
                        result.grouped_keys[output] = (
                            resident_input.value().grouped_keys[column].copy()
                        )
    if not terminal_boundary:
        result.observe_errors()
    var traffic = scope.transfers()
    submissions = traffic[2] - before[2]
    waits = traffic[3] - before[3]
    uploaded = traffic[0] - before[0]
    downloaded = traffic[1] - before[1]
    var report = ExecutionReport()
    report.record(
        plan.root,
        "ROW EXPRESSIONS",
        "metal",
        rows,
        count,
        algorithm="mojo_gpu_resident_projection_view" if projection_view else "mojo_gpu_ordered_implode_rows" if ordered_private else "mojo_gpu_grouped_rows" if grouping else "mojo_gpu_reductions" if plan.reductions else "mojo_gpu_windows" if ranking
        or windows else "mojo_gpu_ordered_rows" if ordering else "mojo_gpu_row_program",
        wall_ns=Int(perf_counter_ns()) - started,
    )
    var fields = report.frame()._columns.copy()
    if profiling:
        for field in List[Tuple[String, Int]](
            [
                ("host_prepare_ms", host_prepare_ns),
                ("staging_copy_ms", staging_ns),
                ("initialization_ms", initialization_ns),
                ("submit_wait_ms", submit_wait_ns),
                (
                    "synchronization_ms",
                    scope.timing_ns()[1] - timing_before[1],
                ),
            ]
        ):
            fields.append(
                Series(field[0], Column[Float64]([Float64(field[1]) / 1e6]))
            )
    fields.append(
        Series("kernel_launches", Column[Int64]([Int64(submissions)]))
    )
    fields.append(Series("synchronizations", Column[Int64]([Int64(waits)])))
    fields.append(Series("upload_bytes", Column[Int64]([Int64(uploaded)])))
    fields.append(Series("download_bytes", Column[Int64]([Int64(downloaded)])))
    fields.append(
        Series(
            "peak_requested_bytes",
            Column[Int64]([Int64(max(peak, scope.info()[1]))]),
        )
    )
    fields.append(Series("runtime_implementation", Column[String](["mojo"])))
    fields.append(
        Series(
            "grouped_int_algorithm",
            Column[String](
                [
                    "mojo_gpu_grouped_int_partials" if grouped_int_parallel_outputs else "legacy"
                ]
            ),
        )
    )
    fields.append(
        Series(
            "grouped_int_parallel_outputs",
            Column[Int64]([Int64(grouped_int_parallel_outputs)]),
        )
    )
    fields.append(
        Series(
            "grouped_int_max_partial_states",
            Column[Int64]([Int64(grouped_int_max_partial_states)]),
        )
    )
    fields.append(
        Series(
            "grouped_int_max_partials_per_group",
            Column[Int64]([Int64(grouped_int_max_partials_per_group)]),
        )
    )
    fields.append(
        Series(
            "grouped_int_max_group_batch",
            Column[Int64]([Int64(grouped_int_max_group_batch)]),
        )
    )
    fields.append(
        Series(
            "grouped_int_max_qualifying_hash_group_rows",
            Column[Int64]([Int64(grouped_int_max_qualifying_hash_group_rows)]),
        )
    )
    fields.append(
        Series(
            "grouped_int_partial_tiles",
            Column[Int64]([Int64(grouped_int_partial_tiles)]),
        )
    )
    fields.append(
        Series(
            "grouped_int_scratch_bytes",
            Column[Int64]([Int64(grouped_int_scratch_bytes)]),
        )
    )
    fields.append(Series("host_result_bytes", Column[Int64]([Int64(0)])))
    fields.append(
        Series(
            "resident_input_bytes",
            Column[Int64](
                [
                    Int64(
                        resident_input.value().retained_bytes() if resident_input else 0
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
            "query_wall_ms",
            Column[Float64]([Float64(Int(perf_counter_ns()) - started) / 1e6]),
        )
    )
    return (result^, DataFrame(fields^))


def describe(query: LazyFrame, *, ordered_source: Bool = False) -> String:
    from .fused_int32_plan import bind as bind_fused_int32
    from .wide_integer_plan import bind as bind_wide_integer

    try:
        var plan = _preflight(query, ordered_source=ordered_source)
        if bind_fused_int32(plan):
            return (
                "ENGINE accel: Mojo GPU fused Int32 predicate/arithmetic and "
                "Int64 SUM; native values, at most 4096 partial records, "
                "scalar-only export\n"
            )
        if bind_wide_integer(plan):
            return (
                "ENGINE accel: Mojo GPU native wide integer SUM; exact two-word"
                " state, at most 4096 shared partial records, scalar-only"
                " export\n"
            )
        return (
            "ENGINE accel: Mojo GPU row expressions; "
            + String(len(plan.steps))
            + (
                " steps; ordered projection/batch/AST faults" if plan.ordered_mode
                == 2 else " steps; certified IMPLODE batch/AST faults" if plan.ordered_mode
                == 3 else " steps"
            )
            + "; kernels and runtime execute directly in the dataframe"
            " library\n"
        )
    except error:
        return "ENGINE accel: " + String(error) + "\n"


def _can_fault(dtype: DataType) -> Bool:
    """Float and Boolean values never carry a new value fault."""
    return not (dtype.is_float() or dtype == DataType.BOOL)


def _step_can_fault(
    code: List[Int64], types: List[Int64], start: Int, nodes: Int
) -> Bool:
    """Whether a row-program step can write a new value fault.

    Float32/Float64/Bool arithmetic, data movement and Boolean logic never
    fault. Any other node touching an integer, Decimal or temporal value may;
    integer comparisons are counted too, which only costs one observation.
    """
    from dataframe.expr import WHEN, IS_NULL, IS_NOT_NULL, FILL_NULL, AND, OR
    from dataframe.expr import NOT, XOR

    for node in range(start, start + nodes):
        var op = Int(code[4 * node])
        if (
            op == COL
            or op == LIT_INT
            or op == LIT_FLOAT
            or op == LIT_BOOL
            or op == LIT_NULL
            or op == WHEN
            or op == IS_NULL
            or op == IS_NOT_NULL
            or op == FILL_NULL
            or op == AND
            or op == OR
            or op == NOT
            or op == XOR
        ):
            continue
        var kinds = List[Int64]([types[node]])
        for operand in [Int(code[4 * node + 1]), Int(code[4 * node + 2])]:
            if operand >= 0 and operand < nodes:
                kinds.append(types[start + operand])
        for kind in kinds:
            # 1 Float32, 4 Bool, 11 Float64 (program.kind).
            if kind != 1 and kind != 4 and kind != 11:
                return True
    return False


def _observe_faults(
    scope: Scope, fault_word: DeviceBuffer[DType.uint32]
) raises:
    """Raise a data fault when any value the region wrote failed.

    The CPU decides the query (accelerator/faults.mojo): it may never have
    evaluated a row that a head, filter or HAVING step dropped here.
    """
    # raise_value_error names the fault's category; all are data faults.
    raise_value_error(
        scope.read_control(fault_word)[0],
        "Mojo GPU arithmetic overflow or strict cast failed",
    )


def _raise_iana(
    plan: RowPlan, start: Int, nodes: Int, token: UInt32, detail: List[UInt64]
) raises:
    var rank = ordered_faults.rank(token)
    var status = ordered_faults.category(token) & 63
    if detail[1] != UInt64(rank) or detail[2] != UInt64(status):
        raise Error(
            "IANA diagnostic replay did not reproduce the observed occurrence"
        )
    var node = -1
    for at in range(start, start + nodes):
        if plan.ordered_ranks[][at] == rank:
            node = at
            break
    if node < 0 or plan.code[4 * node] != ROW_IANA_MUTATION:
        raise Error("IANA diagnostic occurrence is invalid")
    var base = Int(plan.code[4 * node + 2])
    var dtype = plan.node_dtypes[node]
    var name = dtype.time_zone()
    if status == 7:
        for at in range(len(plan.iana_descriptors)):
            if plan.iana_descriptors[at] == base:
                raise data_fault(plan.iana_diagnostics[at])
        raise Error("IANA interval diagnostic is missing")
    if status == 4:
        raise data_fault("temporal value overflows Int64")
    if status == 1:
        raise data_fault(
            "could not localize a shifted datetime to time zone '" + name + "'"
        )
    if status == 2 or status == 3:
        raise data_fault(
            "a shifted datetime is "
            + ("non-existent" if status == 2 else "ambiguous")
            + " in time zone '"
            + name
            + "'"
        )
    var wall = _naive_text(detail[0].cast[DType.int64](), dtype)
    if status == 5:
        raise data_fault(
            "datetime '"
            + wall
            + "' is non-existent in time zone '"
            + name
            + "'; use non_existent='null' to return null instead"
        )
    if status == 6:
        raise data_fault(
            "datetime '"
            + wall
            + "' is ambiguous in time zone '"
            + name
            + "'; use ambiguous='earliest', 'latest' or 'null'"
        )
    raise Error("IANA diagnostic status is invalid")
