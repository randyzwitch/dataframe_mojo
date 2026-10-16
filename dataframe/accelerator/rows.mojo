"""CPU-only lowering for bounded resident accelerator row-expression pipelines.

Expressions use the shared binder and retain their logical dtypes. Programs
are device descriptors, not a second public expression API. Rejections happen
before a provider context or device allocation is created.
"""
from .metadata import prune_projections
from . import (
    temporal_expressions,
    ordered_rows,
    implode_ordered,
    iana_mutations,
    iana_certificate,
    producer_phases,
    decimal_feed,
    decimal_group_feed,
    decimal_scalar_feed,
    decimal_row_feed,
)
from dataframe.producer_origin import RegionProof
from .code_domains import TextField
from .text_key_domains import append_partition_values
from .gpu.temporal_descriptors import (
    ROW_TEMPORAL_BINARY,
    ROW_TEMPORAL_CAST,
    binary_literal,
    cast_literal,
)
from .timezone_tables import serialize as serialize_zone
from .memory import checked_add, checked_mul
from dataframe.parse import parse_int64
from .integer_ranges import proven_integer_nodes, _decimal_scale_safe
from std.memory import ArcPointer, bitcast
from std.sys import size_of
from std.math import pow, abs
from dataframe.parallel import configured_workers
from .capabilities import RowCapabilities
from dataframe.identity_metadata import identity_column_source
from dataframe.binding import bind, BoundExpr, ROWS, op_name
from dataframe.dtype import DataType
from dataframe.expr import (
    Expr,
    RANK,
    CUM_SUM,
    CUM_MIN,
    CUM_MAX,
    CUM_COUNT,
    SHIFT,
    ROLLING_SUM,
    ROLLING_MEAN,
    ROLLING_MIN,
    ROLLING_MAX,
    ROLLING_VAR,
    ROLLING_STD,
    INTERPOLATE,
    INTERPOLATE_BY,
    ROLLING_SUM_BY,
    ROLLING_MEAN_BY,
    ROLLING_MIN_BY,
    ROLLING_MAX_BY,
    ROLLING_VAR_BY,
    ROLLING_STD_BY,
    is_rolling_by,
    FORWARD_FILL,
    BACKWARD_FILL,
    OVER,
    col,
    subtree,
    split_names,
    is_window,
    is_reduction,
    is_comparison,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    CLIP_LOW,
    CLIP_HIGH,
    FILL_NAN,
    KEEP_NULLS,
    ABS,
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
    WHEN,
    INT_IS_IN,
    COL,
    CAST,
    LIT_FLOAT,
    LIT_INT,
    LIT_BOOL,
    LIT_NULL,
    ADD,
    SUB,
    MUL,
    GT,
    LT,
    GE,
    LE,
    EQ,
    NE,
    AND,
    OR,
    XOR,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    FILL_NULL,
    NEG,
    SUM,
    MIN,
    MAX,
    COUNT,
    MEAN,
    VAR,
    SKEW,
    KURTOSIS,
    MEDIAN,
    QUANTILE,
    COV,
    CORR,
    STD,
    LEN,
    FIRST,
    LAST,
    ANY,
    ALL,
    NULL_COUNT,
    ARG_MIN,
    ARG_MAX,
    DT_YEAR,
    DT_TOTAL,
    DT_TIME,
    DT_TRUNCATE,
    DT_OFFSET_BY,
    DT_REPLACE_TZ,
    DT_CONVERT_TZ,
)
from dataframe.frame import DataFrame
from dataframe.temporal import parse_every, parse_interval, NANOS_PER_DAY
from dataframe.time_windows import positive_period, closed_sides
from dataframe.timezone import TimeZone, _fixed_offset
from dataframe.lazy import (
    LazyFrame,
    SCAN_FRAME,
    SELECT,
    WITH_COLUMNS,
    FILTER,
    SLICE,
    DROP,
    SORT,
    AGG,
)
from dataframe.series import Series

comptime ROW_GROUP_HEADER = 1002
comptime ROW_GROUP_KEY = 1003
comptime ROW_GROUP_OUTPUT = 1004
comptime ROW_GROUP_HEADER_LAST = 1005
comptime ROW_SLICE = 1006
comptime ROW_CUMULATIVE_KEY = 1009
comptime ROW_RANK_KEY = 1007
comptime ROW_GROUP_HEADER_WINDOW = 1008
# Backend-neutral quantile reductions preserve the complete binary64 q in
# min_count; the reduction identifies the interpolation method.
comptime ROW_QUANTILE_LINEAR = 1010
comptime ROW_QUANTILE_LOWER = 1011
comptime ROW_QUANTILE_HIGHER = 1012
comptime ROW_QUANTILE_NEAREST = 1013
comptime ROW_QUANTILE_MIDPOINT = 1014
comptime ROW_CALENDAR_TRUNCATE = 1015
comptime ROW_CALENDAR_OFFSET = 1016
comptime ROW_DURATION_KEY = 1017
comptime ROW_DURATION_BOUNDS = 1018
comptime ROW_DURATION_CALENDAR = 1019
comptime ROW_DURATION_UTC = 1020
# Temporal unit bits 0..26 retain storage/UTC/calendar-day metadata. Fixed
# local field operations carry seconds east of UTC in their right operand.
comptime ROW_FIXED_ZONE_FIELDS = Int64(1 << 27)
comptime ROW_IANA_ZONE_FIELDS = Int64(1 << 28)
comptime ROW_SORT_KEY = 1001
comptime ROW_MAX_NODES = 64
comptime ROW_MAX_SLOTS = 64
comptime ROW_MAX_STEPS = 64
comptime ROW_CODE_WORDS = 4


@fieldwise_init
struct RowStep(Copyable):
    var start: Int
    var nodes: Int
    var slot: Int
    var filter: Bool
    var gather_start: Int
    var gather_count: Int
    var order: Bool
    var grouped: Bool


struct RowOutput(Copyable):
    var name: String
    var dtype: DataType
    var slot: Int
    var reduction: Int
    var min_count: Int
    var field_metadata: Optional[ArcPointer[List[UInt8]]]

    def __init__(
        out self,
        name: String,
        dtype: DataType,
        slot: Int,
        reduction: Int,
        min_count: Int,
        field_metadata: Optional[ArcPointer[List[UInt8]]] = None,
    ):
        self.name = name
        self.dtype = dtype
        self.slot = slot
        self.reduction = reduction
        self.min_count = min_count
        self.field_metadata = field_metadata


@fieldwise_init
struct RowPlan(Movable):
    var source: DataFrame
    var dtype: DataType
    var code: List[Int64]
    var literals: List[Float64]
    var node_dtypes: List[DataType]
    var slot_dtypes: List[DataType]
    var gathers: List[Int64]
    var steps: List[RowStep]
    var outputs: List[RowOutput]
    var slots: Int
    var root: Int
    var limit: Int
    var reductions: Bool
    var reused_columns: Bool
    var key_values: List[UInt64]
    var source_failure: String
    var zone_words: List[UInt64]
    # Shared immutable ranks avoid a second host payload during Program prep.
    var ordered_ranks: ArcPointer[List[UInt32]]
    # 0 legacy flags, 1 private CUT scalar, 2 direct terminal ROWS projections,
    # 3 certified private IMPLODE ROWS producer, 4 observed Decimal producer.
    var ordered_mode: Int
    var source_text: List[TextField]
    var output_text: List[TextField]
    var code_domain: Optional[DataType]
    var source_domain: Optional[DataType]
    var category_sorts: List[Tuple[Int, Int, DataType]]
    var iana_mask_bases: List[Int]
    var iana_descriptors: List[Int]
    var iana_diagnostics: List[String]
    var iana_construction_peak_bytes: Int
    var producer_region: Optional[ArcPointer[RegionProof]]
    # Exact eager operand batches for checked scalar Decimal SUM replay.
    var decimal_feed_batch_rows: Int


def _contains_node(expression: Expr, root: Int, target: Int) -> Bool:
    var reachable = List[Bool](length=root + 1, fill=False)
    reachable[root] = True
    for reverse in range(root + 1):
        var index = root - reverse
        if not reachable[index]:
            continue
        if index == target:
            return True
        ref node = expression._nodes[index]
        if node.left >= 0:
            reachable[node.left] = True
        if node.right >= 0:
            reachable[node.right] = True
        if node.extra >= 0:
            reachable[node.extra] = True
    return False


def _empty(name: String, dtype: DataType) raises -> Series:
    return Series.full_null(name, dtype, 0)


def _projected_column(
    name: String, dtype: DataType, bound: BoundExpr, schema: List[Series]
) raises -> Series:
    # CPU cast_series preserves source owners for exact primitive identity
    # casts, including complete logical dtype metadata. Computations and
    # dtype-changing casts create a fresh field.
    var result = _empty(name, dtype)
    var source = identity_column_source(bound, len(bound.expr._nodes) - 1)
    if source >= 0 and schema[source].dtype() == dtype:
        result._field_metadata = schema[source]._field_metadata
    return result^


def _program(
    bound: BoundExpr,
    count: Int,
    slots: List[Int],
    dtype: DataType,
    mut code: List[Int64],
    mut literals: List[Float64],
    mut node_dtypes: List[DataType],
    capabilities: RowCapabilities,
    iana_certified: Bool = False,
) raises:
    if count < 1 or count > capabilities.max_nodes:
        capabilities.reject(
            "plan",
            "expression must contain 1 to "
            + String(capabilities.max_nodes)
            + " row-local nodes",
        )
    var program_start = len(literals)
    var memberships = List[Int]()
    var wide_literals = List[Int]()
    for i in range(count):
        ref node = bound.expr._nodes[i]
        var op = node.op
        capabilities.require_dtype(bound.dtypes[i])
        if (
            not capabilities.mixed_types
            and bound.dtypes[i] != dtype
            and bound.dtypes[i] != DataType.BOOL
        ):
            capabilities.reject(
                "plan", "mixed numeric dtypes or unsupported intermediate dtype"
            )
        if not (
            op == COL
            or (
                op == DIV
                and capabilities.decimal_multiplicative
                and bound.dtypes[i].is_decimal()
            )
            or (
                op == FLOORDIV
                and capabilities.float_division
                and bound.dtypes[i] == DataType.FLOAT32
            )
            or (
                (op == EXP or op == LOG)
                and capabilities.float_transcendentals
                and bound.dtypes[i].is_float()
            )
            or (
                op == POW
                and capabilities.float_power
                and bound.dtypes[i].is_float()
            )
            or (
                op == SQRT
                and capabilities.float_sqrt
                and bound.dtypes[i].is_float()
            )
            or (
                op == ROUND
                and capabilities.float_rounding
                and bound.dtypes[i].is_float()
            )
            or (
                (
                    (op == DIV or op == FLOORDIV)
                    and capabilities.float_division
                    or op == MOD
                    and capabilities.float_modulo
                )
                and bound.dtypes[i].is_float()
            )
            or (
                capabilities.temporal_fields
                and op >= DT_YEAR
                and op <= DT_TRUNCATE
            )
            or (op == DT_TRUNCATE and capabilities.calendar_truncation)
            or (op == DT_TOTAL and capabilities.duration_totals)
            or (
                capabilities.utc_temporal_conversions
                and (op == DT_REPLACE_TZ or op == DT_CONVERT_TZ)
            )
            or (op == DT_OFFSET_BY and capabilities.temporal_offsets)
            or (op == INT_IS_IN and capabilities.extended_integers)
            or (
                capabilities.row_extras
                and (
                    op == FLOORDIV
                    or op == MOD
                    or op == POW
                    or op == CLIP_LOW
                    or op == CLIP_HIGH
                    or op == FILL_NAN
                    or op == KEEP_NULLS
                    or op == ABS
                    or op == FLOOR
                    or op == CEIL
                    or op == ROUND
                    or op == IS_NAN
                    or op == IS_NOT_NAN
                    or op == IS_FINITE
                    or op == IS_INFINITE
                    or op == WHEN
                )
            )
            or (op == CAST and capabilities.casts)
            or op == LIT_INT
            or op == LIT_FLOAT
            or op == LIT_BOOL
            or op == LIT_NULL
            or op == ADD
            or op == SUB
            or op == MUL
            or op == NEG
            or op == GT
            or op == LT
            or op == GE
            or op == LE
            or op == EQ
            or op == NE
            or op == AND
            or op == OR
            or op == XOR
            or op == NOT
            or op == IS_NULL
            or op == IS_NOT_NULL
            or op == FILL_NULL
        ):
            capabilities.reject(
                "plan", "resident expression operation " + op_name(op)
            )
        var left_type = (
            bound.dtypes[node.left] if node.left >= 0 else bound.dtypes[i]
        )
        var right_type = (
            bound.dtypes[node.right] if node.right >= 0 else left_type
        )
        var iana_mutation = (
            capabilities.iana_zone_mutations
            and iana_certified
            and iana_mutations.mutation(bound, i)
        )
        var temporal_binary = (
            capabilities.temporal_binary_arithmetic
            and temporal_expressions.binary(bound, i)
        )
        var temporal_cast = (
            capabilities.temporal_physical_casts
            and temporal_expressions.cast(bound, i)
        )
        # Conversion changes validated zone metadata and preserves UTC ticks.
        # Local-time operations still require their own zone-aware support.
        if (
            not iana_mutation
            and op == DT_REPLACE_TZ
            and (
                left_type.time_zone() not in List[String](["", "UTC"])
                or bound.dtypes[i].time_zone() not in List[String](["", "UTC"])
            )
        ):
            capabilities.reject(
                "temporal",
                "Mojo time zone replacement requires naive or UTC values",
            )
        var temporal = (
            (op >= DT_YEAR and op <= DT_TRUNCATE)
            or op == DT_TOTAL
            or op == DT_OFFSET_BY
        )
        var iana_fields = (
            op >= DT_YEAR
            and op <= DT_TIME
            and left_type.is_datetime()
            and left_type.time_zone().byte_length() != 0
            and left_type.time_zone() != "UTC"
            and not _fixed_offset(left_type.time_zone())
            and capabilities.iana_zone_temporal_fields
        )
        var fixed_fields = False
        var field_offset = Int64(0)
        if (
            op >= DT_YEAR
            and op <= DT_TIME
            and left_type.is_datetime()
            and left_type.time_zone().byte_length()
            and left_type.time_zone() != "UTC"
            and capabilities.fixed_zone_temporal_fields
        ):
            var fixed = _fixed_offset(left_type.time_zone())
            if fixed:
                fixed_fields = True
                field_offset = fixed.value()
        var decimal_float_cast = (
            op == CAST
            and left_type.is_decimal()
            and bound.dtypes[i] == DataType.FLOAT64
            and capabilities.decimal_float_cast
            and (capabilities.float64 or capabilities.float64_emulated)
            and capabilities.casts
        )
        var decimal_cast = (
            op == CAST
            and capabilities.casts
            and capabilities.decimal_casts
            and (
                (
                    left_type.is_decimal()
                    and (
                        bound.dtypes[i].is_decimal()
                        or bound.dtypes[i].is_integer()
                        or bound.dtypes[i] == DataType.BOOL
                        or bound.dtypes[i] == DataType.FLOAT32
                    )
                )
                or (
                    bound.dtypes[i].is_decimal()
                    and (left_type.is_integer() or left_type == DataType.BOOL)
                )
            )
        )
        var decimal_arithmetic = left_type.is_decimal() and (
            capabilities.decimal_arithmetic
            and (op == ADD or op == SUB or op == NEG or op == ABS)
            or capabilities.decimal_multiplicative
            and (op == MUL or op == DIV)
        )
        # Decimal arithmetic descriptors preserve the exact bound result
        # precision, scale and physical output width (Float64 for DIV).
        if (
            temporal
            and left_type.is_datetime()
            and left_type.time_zone().byte_length()
            and not (
                capabilities.utc_temporal_conversions
                and left_type.time_zone() == "UTC"
            )
            and not fixed_fields
            and not iana_fields
            and not iana_mutation
        ):
            capabilities.reject(
                "temporal", "time zone field extraction requires CPU execution"
            )
        if (
            not capabilities.float64
            and (
                bound.dtypes[i] == DataType.FLOAT64
                or left_type == DataType.FLOAT64
                or right_type == DataType.FLOAT64
            )
            and not (
                op == COL
                or op == LIT_FLOAT
                or op == LIT_NULL
                or op == IS_NULL
                or op == IS_NOT_NULL
                or op == FILL_NULL
                or op == KEEP_NULLS
                or op == WHEN
                or (
                    capabilities.float64_emulated
                    and (
                        op == ADD
                        or op == SUB
                        or op == MUL
                        or op == NEG
                        or op == DIV
                        or op == FLOORDIV
                        or op == MOD
                        or op == SQRT
                        or (op == EXP or op == LOG)
                        and capabilities.float_transcendentals
                        or op == POW
                        and capabilities.float_power
                        or op == CAST
                        or op == FLOOR
                        or op == CEIL
                        or op == ROUND
                        or op == ABS
                        or op == GT
                        or op == LT
                        or op == GE
                        or op == LE
                        or op == EQ
                        or op == NE
                        or op == IS_NAN
                        or op == IS_NOT_NAN
                        or op == IS_FINITE
                        or op == IS_INFINITE
                        or op == FILL_NAN
                        or op == CLIP_LOW
                        or op == CLIP_HIGH
                    )
                )
            )
        ):
            capabilities.reject(
                "precision", "Float64 requires CPU execution for this operation"
            )
        if (left_type.is_decimal() and left_type.decimal_width() == 128) or (
            right_type.is_decimal() and right_type.decimal_width() == 128
        ):
            if (
                op != COL
                and op != LIT_NULL
                and op != IS_NULL
                and op != IS_NOT_NULL
                and op != WHEN
                and op != FILL_NULL
                and op != KEEP_NULLS
                and op != LIT_INT
                and op != GT
                and op != LT
                and op != GE
                and op != LE
                and op != EQ
                and op != NE
                and not decimal_float_cast
                and not decimal_cast
                and not decimal_arithmetic
            ):
                capabilities.reject(
                    "logical",
                    "Decimal128 row arithmetic requires CPU execution",
                )
        var logical = (
            bound.dtypes[i].is_temporal()
            or bound.dtypes[i].is_decimal()
            or left_type.is_temporal()
            or left_type.is_decimal()
            or right_type.is_temporal()
            or right_type.is_decimal()
        )
        if (
            logical
            and not decimal_float_cast
            and not decimal_cast
            and not decimal_arithmetic
            and not temporal_binary
            and not temporal_cast
            and (
                op == SQRT
                or op == EXP
                or op == LOG
                or op == ABS
                or op == FLOORDIV
                or op == MOD
                or op == POW
                or op == FLOOR
                or op == CEIL
                or op == ROUND
                or op == ADD
                or op == SUB
                or op == MUL
                or op == NEG
                or op == CAST
            )
        ):
            capabilities.reject(
                "logical", "logical arithmetic and casts require CPU execution"
            )
        if left_type.is_decimal() and (
            op == GT or op == LT or op == GE or op == LE or op == EQ or op == NE
        ):
            if not capabilities.decimal_comparisons and (
                left_type.scale() != right_type.scale()
                or (
                    left_type.physical() != right_type.physical()
                    and not (
                        capabilities.decimal128
                        and (
                            left_type.decimal_width() == 128
                            or right_type.decimal_width() == 128
                        )
                    )
                )
            ):
                capabilities.reject(
                    "logical",
                    (
                        "decimal comparisons require matching scales and"
                        " storage widths"
                    ),
                )
        if (
            op == ADD
            or op == SUB
            or op == MUL
            or op == NEG
            or op == ABS
            or op == FLOORDIV
            or op == POW
        ):
            capabilities.require_arithmetic(bound.dtypes[i])
        if (
            bound.dtypes[i].is_integer()
            and (
                op == ADD
                or op == SUB
                or op == MUL
                or op == NEG
                or op == ABS
                or op == FLOORDIV
                or op == POW
            )
            and bound.shapes[i] != ROWS
        ):
            capabilities.reject(
                "plan", "scalar integer arithmetic requires CPU execution"
            )
        if (
            op == CAST
            and node.integer
            and bound.shapes[i] != ROWS
            and not (
                capabilities.float64_emulated and bound.dtypes[i].is_float()
            )
            and not temporal_cast
        ):
            capabilities.reject(
                "plan", "scalar strict casts require CPU execution"
            )
        if (
            op == ROUND
            and bound.dtypes[i] == DataType.FLOAT32
            and node.integer != 0
            and not capabilities.float_rounding
        ):
            capabilities.reject(
                "precision",
                (
                    "Float32 decimal rounding requires Float64 intermediate"
                    " arithmetic"
                ),
            )
        if (
            (op == FLOORDIV or op == MOD or op == POW)
            and bound.dtypes[i].is_float()
            and not (
                op == FLOORDIV
                and capabilities.float_division
                or op == MOD
                and capabilities.float_modulo
                or op == POW
                and capabilities.float_power
            )
        ):
            capabilities.reject(
                "precision",
                "floating division, modulo, and power require CPU execution",
            )
        var source = slots[bound.sources[i]] if op == COL else (
            Int(node.integer) if op == CAST
            or op == ROUND else (node.extra if op == WHEN else -1)
        )
        if temporal:
            source = 0 if left_type.is_date() else (
                4 if left_type.is_time() else (
                    1 if left_type.per_second()
                    == 1000 else (2 if left_type.per_second() == 1000000 else 3)
                )
            )
        if op == DT_CONVERT_TZ or op == DT_REPLACE_TZ:
            source = (
                1 if left_type.per_second()
                == 1000 else 2 if left_type.per_second()
                == 1000000 else 3
            )
        var right = Int64(node.right)
        if fixed_fields:
            source |= Int(ROW_FIXED_ZONE_FIELDS)
            right = field_offset
        elif iana_fields:
            source |= Int(ROW_IANA_ZONE_FIELDS)
            # Filled with a zone-tail-relative header after all programs bind.
            right = -1
        if op == DT_CONVERT_TZ or op == DT_REPLACE_TZ:
            right = Int64(bound.dtypes[i].time_zone() == "UTC")
        if op == DT_TOTAL:
            var nanos_per_tick = Int64(1000000000) // left_type.per_second()
            var nanos_per_unit = Int64(1)
            if node.text == "days":
                nanos_per_unit = NANOS_PER_DAY
            elif node.text == "hours":
                nanos_per_unit = 3600000000000
            elif node.text == "minutes":
                nanos_per_unit = 60000000000
            elif node.text == "seconds":
                nanos_per_unit = 1000000000
            elif node.text == "milliseconds":
                nanos_per_unit = 1000000
            elif node.text == "microseconds":
                nanos_per_unit = 1000
            source = 1 if nanos_per_unit >= nanos_per_tick else 0
            right = (
                nanos_per_unit
                // nanos_per_tick if source else nanos_per_tick
                // nanos_per_unit
            )
        var lowered_op = op
        var calendar_ticks = Int64(0)
        if op == DT_OFFSET_BY and not iana_mutation:
            var interval = parse_every(node.text)
            if (
                capabilities.utc_temporal_conversions
                and left_type.is_datetime()
                and left_type.time_zone() == "UTC"
            ):
                var zoned = parse_interval(node.text)
                if zoned[1] < -131072 or zoned[1] > 131071:
                    capabilities.reject(
                        "temporal",
                        "UTC calendar-day offset exceeds descriptor range",
                    )
                # Bit 8 selects UTC's checked day/relocalization phase;
                # parse_every bounds days to signed 18 bits in bits 9..26.
                source |= 256 | ((Int(zoned[1]) & 0x3FFFF) << 9)
            var per_tick = (
                NANOS_PER_DAY if left_type.is_date() else Int64(1000000000)
                // left_type.per_second()
            )
            if (
                interval[0] != 0 and not capabilities.calendar_offsets
            ) or interval[1] % per_tick != 0:
                capabilities.reject(
                    "temporal",
                    "offset requires a fixed interval aligned to the storage unit",
                )
            right = interval[1] // per_tick
            if interval[0] != 0:
                if left_type.is_time():
                    capabilities.reject(
                        "temporal", "Time cannot be offset by calendar months"
                    )
                lowered_op = ROW_CALENDAR_OFFSET
                calendar_ticks = right
                right = interval[0]
        if op == DT_TRUNCATE and not iana_mutation:
            var interval = parse_every(node.text)
            if (
                capabilities.utc_temporal_conversions
                and left_type.is_datetime()
                and left_type.time_zone() == "UTC"
            ):
                source |= 256
            var per_tick = NANOS_PER_DAY if left_type.is_date() else (
                Int64(1000000000) // left_type.per_second()
            )
            if (
                interval[0] > 0
                and interval[1] == 0
                and capabilities.calendar_truncation
            ):
                if left_type.is_time():
                    capabilities.reject(
                        "temporal",
                        "Time cannot be truncated by calendar months",
                    )
                lowered_op = ROW_CALENDAR_TRUNCATE
                right = interval[0]
            else:
                if (
                    interval[0] != 0
                    or interval[1] <= 0
                    or interval[1] % per_tick != 0
                ):
                    capabilities.reject(
                        "temporal",
                        "truncate requires a positive fixed interval aligned to the storage unit or supported calendar months",
                    )
                right = interval[1] // per_tick
        node_dtypes.append(bound.dtypes[i])
        var temporal_payload = UInt64(0)
        if temporal_binary:
            capabilities.require_arithmetic(DataType.INT64)
            lowered_op = Int(ROW_TEMPORAL_BINARY)
            right = Int64(node.right)
            source = -1
            temporal_payload = binary_literal(Int64(op), left_type, right_type)
        elif temporal_cast:
            capabilities.require_arithmetic(DataType.INT64)
            lowered_op = Int(ROW_TEMPORAL_CAST)
            right = -1
            source = -1
            temporal_payload = cast_literal(
                left_type, bound.dtypes[i], node.integer != 0
            )
        if iana_mutation:
            lowered_op = Int(iana_mutations.ROW_IANA_MUTATION)
            right = -1
        code.append(Int64(lowered_op))
        code.append(Int64(node.left))
        code.append(right)
        code.append(Int64(source))
        # Integer payloads travel as raw 64-bit words, never through a float
        # conversion (Int64 literals beyond 2**53 must remain exact).
        literals.append(
            bitcast[DType.float64](temporal_payload) if temporal_binary
            or temporal_cast else bitcast[DType.float64](
                calendar_ticks
            ) if lowered_op
            == ROW_CALENDAR_OFFSET else bitcast[DType.float64](
                node.integer
            ) if op
            == LIT_INT else (
                Float64(node.integer) if op
                == LIT_BOOL else (
                    pow(Float64(10), Float64(abs(node.integer))) if op == ROUND
                    and capabilities.float_rounding
                    and bound.dtypes[i].is_float()
                    and node.integer != 0 else node.floating
                )
            )
        )
        if (
            op == LIT_INT
            and bound.dtypes[i].is_decimal()
            and bound.dtypes[i].decimal_width() == 128
        ):
            wide_literals.append(i)
        if op == INT_IS_IN:
            memberships.append(i)
    # Upper words are runtime payloads, like membership lists. Values do not
    # enter shader source or require a different pipeline cache entry.
    for i in wide_literals:
        ref node = bound.expr._nodes[i]
        var high = Int64(atol(node.text2)) if node.text2 else (
            Int64(-1) if node.integer < 0 else Int64(0)
        )
        code[4 * (program_start + i) + 3] = Int64(len(literals))
        code.extend([Int64(LIT_INT), Int64(-1), Int64(-1), Int64(-1)])
        literals.append(bitcast[DType.float64](high))
        node_dtypes.append(DataType.INT64)
    # Membership values are runtime literal payloads, outside the executable
    # expression's local node range. Changing values reuses the same shader.
    # Preserve exact signed/unsigned 64-bit words without Float64 conversion.
    for i in memberships:
        ref node = bound.expr._nodes[i]
        var entries = node.text.split(",")
        var size = len(entries) if node.text else 0
        code[4 * (program_start + i) + 2] = Int64(len(literals))
        code[4 * (program_start + i) + 3] = Int64(size)
        for j in range(size):
            code.extend([Int64(LIT_INT), Int64(-1), Int64(-1), Int64(-1)])
            literals.append(bitcast[DType.float64](Int64(atol(entries[j]))))
            node_dtypes.append(bound.dtypes[node.left])


def _append_gathers(slots: List[Int], mut gathers: List[Int64]) -> Int:
    # Several output names may refer to one resident value. Gather its data
    # and validity once while keeping visible column order in the schema.
    var unique = List[Int]()
    for slot in slots:
        if slot not in unique:
            unique.append(slot)
            gathers.append(Int64(slot))
    return len(unique)


def lower_rows(
    query: LazyFrame,
    capabilities: RowCapabilities = RowCapabilities(),
    *,
    terminal_boundary: Bool = True,
    source_rows: Int = -1,
    ordered_source: Bool = False,
) raises -> RowPlan:
    """Bind the entire resident region before GPU submission or source I/O.

    Regions feeding another operator are not observable terminal boundaries:
    that operator can prune rows and change whether checked work is evaluated.
    Resident callers may supply the actual source cardinality alongside a
    metadata-only frame; this never changes or reads the frame's storage.
    """
    if source_rows < -1:
        raise Error("Resident source row count must be >= -1")
    var root = len(query._nodes) - 1
    if root < 0:
        capabilities.reject("plan", "empty plan")
    var cursor = root
    var limit = -1
    if query._nodes[cursor].kind == SLICE and query._nodes[cursor].offset == 0:
        if query._nodes[cursor].length < -1:
            raise Error("Slice length must be nonnegative")
        limit = query._nodes[cursor].length
        cursor = query._nodes[cursor].left
    var chain = List[Int]()
    while cursor >= 0 and query._nodes[cursor].kind != SCAN_FRAME:
        var kind = query._nodes[cursor].kind
        if (
            kind != SELECT
            and kind != WITH_COLUMNS
            and kind != FILTER
            and kind != DROP
            and (kind != SLICE or not capabilities.slices)
            and (kind != SORT or not capabilities.order)
            and (kind != AGG or not capabilities.grouped)
        ):
            capabilities.reject(
                "plan",
                (
                    "resident region requires an in-memory scan and row-local"
                    " steps"
                ),
            )
        chain.append(cursor)
        cursor = query._nodes[cursor].left
        if len(chain) > capabilities.max_steps:
            capabilities.reject(
                "plan",
                "resident region supports at most "
                + String(capabilities.max_steps)
                + " plan steps",
            )
    if cursor < 0:
        capabilities.reject(
            "plan", "resident region requires an in-memory scan"
        )
    # Reuse core projection analysis after validating the resident chain.
    # This only narrows the source: no expression evaluation or row movement.
    var projected_query = query.copy()
    if not producer_phases.has_certified(query):
        prune_projections(projected_query)
    var frame = query._frames[query._nodes[cursor].offset].copy()
    var source_count = frame.height() if source_rows < 0 else source_rows
    if len(projected_query._nodes[cursor].names):
        frame = frame.select(projected_query._nodes[cursor].names)
    if (
        frame.width() < 1 and not capabilities.empty_inputs
    ) or frame.width() > capabilities.max_slots:
        capabilities.reject(
            "plan",
            "resident region requires "
            + String(0 if capabilities.empty_inputs else 1)
            + " to "
            + String(capabilities.max_slots)
            + " source columns",
        )
    var dtype = DataType.BOOL
    var schema = frame._columns.copy()
    var slots = List[Int]()
    for i in range(len(schema)):
        var type = schema[i].dtype()
        capabilities.require_dtype(type)
        if schema[i].is_chunked() and not capabilities.chunked_inputs:
            capabilities.reject(
                "plan", "chunked input; rechunk before accelerator execution"
            )
        if type != DataType.BOOL:
            if (
                not capabilities.mixed_types
                and dtype != DataType.BOOL
                and dtype != type
            ):
                capabilities.reject(
                    "plan",
                    "resident source numeric columns must share one dtype",
                )
            if dtype == DataType.BOOL:
                dtype = type
        slots.append(i)
    if dtype == DataType.BOOL:
        # Internal physical representation only; public columns remain Bool.
        dtype = DataType.FLOAT32
    var capsule = iana_mutations.Capsule()
    var iana_mask_bases = List[Int]()
    var code = List[Int64]()
    var literals = List[Float64]()
    var node_dtypes = List[DataType]()
    var slot_dtypes = List[DataType]()
    for column in schema:
        slot_dtypes.append(column.dtype())
    var gathers = List[Int64]()
    var steps = List[RowStep]()
    var outputs = List[RowOutput]()
    var key_values = List[UInt64]()
    var category_sorts = List[Tuple[Int, Int, DataType]]()
    var next_slot = len(schema)
    var reduced = False
    var reused_columns = False
    var ordered_ranks = List[UInt32]()
    var decimal_feed_batch_rows = 0
    var ordered_mode = 0
    var producer_region = Optional[ArcPointer[RegionProof]]()
    for k in range(len(chain) - 1, -1, -1):
        ref node = query._nodes[chain[k]]
        if reduced:
            capabilities.reject(
                "plan", "only final head may follow resident reductions"
            )
        if node.kind == SLICE:
            if node.length < -1:
                raise Error("Slice length must be nonnegative")
            if next_slot >= capabilities.max_slots:
                capabilities.reject(
                    "plan",
                    "slice exceeds "
                    + String(capabilities.max_slots)
                    + " value slots",
                )
            var start = len(literals)
            code.extend(
                [
                    Int64(ROW_SLICE),
                    Int64(start + 1),
                    Int64(start + 2),
                    Int64(-1),
                ]
            )
            literals.append(0)
            node_dtypes.append(DataType.BOOL)
            # Runtime scalar parameters keep shader caching independent of
            # offset and length. These two literal nodes are not evaluated.
            for parameter in List[Int]([node.offset, node.length]):
                code.extend([Int64(LIT_INT), Int64(-1), Int64(-1), Int64(-1)])
                literals.append(bitcast[DType.float64](Int64(parameter)))
                node_dtypes.append(DataType.INT64)
            var gather_start = len(gathers)
            var gather_count = _append_gathers(slots, gathers)
            slot_dtypes.append(DataType.BOOL)
            steps.append(
                RowStep(
                    start,
                    1,
                    next_slot,
                    True,
                    gather_start,
                    gather_count,
                    False,
                    False,
                )
            )
            next_slot += 1
            continue
        if node.kind == SORT:
            if len(node.names) == 0:
                continue
            if len(node.flags) != 2 * len(node.names):
                raise Error("sort needs direction and null placement per key")
            var start = len(literals)
            for i in range(len(node.names)):
                var found = -1
                for j in range(len(schema)):
                    if schema[j].name() == node.names[i]:
                        found = j
                if found < 0:
                    raise Error("Unknown column: " + node.names[i])
                capabilities.require_key(schema[found].dtype())
                for field in node.text_fields:
                    if (
                        field.name == node.names[i]
                        and field.category
                        and not field.category.value().string_codes
                    ):
                        category_sorts.append(
                            (
                                len(literals),
                                slots[found],
                                field.category.value().owner[].dtype,
                            )
                        )
                code.extend(
                    [
                        Int64(ROW_SORT_KEY),
                        Int64(slots[found]),
                        Int64(node.flags[i]),
                        Int64(node.flags[len(node.names) + i]),
                    ]
                )
                literals.append(0)
                node_dtypes.append(schema[found].dtype())
            var gather_start = len(gathers)
            var gather_count = _append_gathers(slots, gathers)
            steps.append(
                RowStep(
                    start,
                    len(node.names),
                    0,
                    False,
                    gather_start,
                    gather_count,
                    True,
                    False,
                )
            )
            continue
        if node.kind == DROP:
            var keep_schema = List[Series]()
            var keep_slots = List[Int]()
            for name in node.names:
                var found = False
                for column in schema:
                    found |= column.name() == name
                if not found:
                    raise Error("Unknown column: " + name)
            for i in range(len(schema)):
                if schema[i].name() not in node.names:
                    keep_schema.append(schema[i].copy())
                    keep_slots.append(slots[i])
            schema = keep_schema^
            slots = keep_slots^
            continue
        var group_keys = List[Int]()
        if node.kind == AGG:
            if len(node.names) == 0:
                raise Error("group_by requires at least one key")
            var key_names = List[String]()
            for name in node.names:
                if name in key_names:
                    raise Error("Duplicate group key: " + name)
                key_names.append(name)
                var found = -1
                for j in range(len(schema)):
                    if schema[j].name() == name:
                        found = j
                if found < 0:
                    raise Error("Unknown column: " + name)
                group_keys.append(found)
                capabilities.require_key(schema[found].dtype())
        var projected = List[Series]()
        var projected_slots = List[Int]()
        var aggregate_outputs = List[RowOutput]()
        # An explicit empty projection keeps its input row shape. A selector
        # that expands to nothing is normalized separately to a scalar row.
        var has_rows = node.kind == SELECT and len(node.exprs) == 0
        var has_aggregate = False
        var projection_names = List[String]()
        var ordered_terminal = (
            ordered_source
            and len(chain) == 1
            and len(query._nodes) == 2
            and cursor == 0
            and node.kind == SELECT
            and terminal_boundary
            and source_rows < 0
            and limit < 0
            and len(node.names2) == 0
            and len(query._nodes[cursor].names2) == 0
            and not query._nodes[cursor].source_failure
            and len(node.exprs) > 0
        )
        var needs_order = False
        var iana_terminal = False
        if ordered_terminal:
            for expression in node.exprs:
                var original_bound = bind(expression, schema)
                ordered_terminal &= ordered_rows.eligible(
                    original_bound, capabilities.iana_zone_mutations
                )
                needs_order |= ordered_rows.needed(original_bound)
                iana_terminal |= (
                    capabilities.iana_zone_mutations
                    and iana_mutations.contains(original_bound)
                )
        ordered_terminal &= needs_order
        if ordered_terminal:
            ordered_mode = 5 if iana_terminal else 2
        var ordered_decimal = (
            node.text == "metal_decimal_materialization"
            or node.text == "metal_integer_materialization"
        )
        if ordered_decimal:
            if (
                len(chain) != 1
                or not terminal_boundary
                or limit >= 0
                or (node.kind != SELECT and node.kind != WITH_COLUMNS)
                or len(node.names2) != 1 + len(node.exprs)
                or node.names2[0] != node.text
            ):
                capabilities.reject(
                    "ordered decimal", "invalid observed producer"
                )
            ordered_mode = 4
        var observed_sum = node.text == "metal_sum_materialization"
        if observed_sum:
            if (
                len(chain) != 1
                or not terminal_boundary
                or limit >= 0
                or node.kind != SELECT
                or len(node.names2) != 1 + len(node.exprs)
                or node.names2[0] != node.text
            ):
                capabilities.reject("observed sum", "invalid observed producer")
        var sum_expression = 0
        var precision_rows = node.text == decimal_row_feed.MATERIALIZATION
        if precision_rows:
            if (
                len(chain) != 1
                or not terminal_boundary
                or limit >= 0
                or (node.kind != SELECT and node.kind != WITH_COLUMNS)
                or len(node.names2) != 1 + len(node.exprs)
                or node.names2[0] != node.text
            ):
                capabilities.reject(
                    "decimal date rows", "invalid original ROW producer"
                )
            var original_bounds = List[BoundExpr]()
            for expression in node.exprs:
                original_bounds.append(bind(expression, schema))
            if not decimal_row_feed.eligible(original_bounds):
                capabilities.reject(
                    "decimal date rows", "original whole-ROW proof changed"
                )
            ordered_mode = decimal_row_feed.ORDERED_MODE
        var scalar_precision = node.text == decimal_scalar_feed.MATERIALIZATION
        var ordered_group = (
            node.text == decimal_group_feed.MATERIALIZATION or scalar_precision
        )
        if scalar_precision:
            if (
                len(chain) != 1
                or not terminal_boundary
                or limit >= 0
                or node.kind != SELECT
                or len(node.names2) != 1 + len(node.exprs)
                or node.names2[0] != node.text
            ):
                capabilities.reject(
                    "decimal scalar feed", "invalid original SUM projection"
                )
            var original_bounds = List[BoundExpr]()
            for expression in node.exprs:
                original_bounds.append(bind(expression, schema))
            if not decimal_scalar_feed.eligible(original_bounds, schema):
                capabilities.reject(
                    "decimal scalar feed",
                    "original operand/state proof changed",
                )
            ordered_mode = decimal_scalar_feed.ORDERED_MODE
        var observed_group = (
            node.text == "metal_group_materialization"
            or node.text == decimal_group_feed.MATERIALIZATION
        )
        if observed_group:
            if (
                len(chain) != 1
                or not terminal_boundary
                or limit >= 0
                or node.kind != AGG
                or len(node.names2) != 2 + len(node.names) + len(node.exprs)
            ):
                capabilities.reject(
                    "observed group", "invalid observed producer"
                )
            if node.names2[0] != node.text or node.names2[1] != String(
                node.maintain_order
            ):
                capabilities.reject("observed group", "group metadata changed")
            for at in range(len(node.names)):
                if node.names2[2 + at] != node.names[at]:
                    capabilities.reject("observed group", "group key changed")
        if ordered_group and not scalar_precision:
            for field in query._nodes[node.left].text_fields:
                if field.name in node.names and field.category:
                    # Only dictionary normalization can issue this owner.
                    # Bare coded String GROUP compares its certified HIGH
                    # original codes; visible output is plain String again.
                    if (
                        field.dtype != DataType.STRING
                        or not field.category.value().string_codes
                        or not field.category.value()
                        .owner[]
                        .dtype.is_categorical()
                    ):
                        capabilities.reject(
                            "decimal group feed",
                            "original code key lacks a coded String owner",
                        )
            var original_bounds = List[BoundExpr]()
            for expression in node.exprs:
                original_bounds.append(bind(expression, schema))
            if not decimal_group_feed.eligible(
                original_bounds, schema, node.names
            ):
                capabilities.reject(
                    "decimal group feed", "original whole-group proof changed"
                )
            ordered_mode = decimal_group_feed.ORDERED_MODE
        var grouped_expression = 0
        var decimal_expression = 0
        var ordered_producer = Bool(node.producer_origin)
        if ordered_producer:
            producer_phases.validate_region(query, node)
            if node.producer_origin.value()[].phase not in List[Int](
                [
                    producer_phases.OBSERVED_IANA,
                    producer_phases.OBSERVED_STRICT_CAST,
                ]
            ):
                capabilities.reject(
                    "observed temporal", "unknown producer phase"
                )
            if (
                len(chain) != 1
                or not terminal_boundary
                or limit >= 0
                or (node.kind != SELECT and node.kind != WITH_COLUMNS)
            ):
                capabilities.reject(
                    "observed temporal", "requires one certified row producer"
                )
            ordered_mode = producer_phases.OBSERVED_IANA
            producer_region = node.producer_region
        var producer_expression = 0
        var ordered_private = implode_ordered.tagged(node)
        if ordered_private:
            if (
                len(chain) != 1
                or not terminal_boundary
                or limit >= 0
                or len(node.exprs) != 1
            ):
                capabilities.reject(
                    "ordered implode",
                    "requires one observed private row producer",
                )
            ordered_mode = 3
        for expr in node.exprs:
            if node.kind != FILTER:
                if expr._name in projection_names:
                    capabilities.reject(
                        "plan", "duplicate output name: " + expr._name
                    )
                projection_names.append(expr._name)
            var bound = bind(expr, schema)
            var expression_ranks = List[UInt32]()
            if scalar_precision:
                if node.names2[
                    1 + decimal_expression
                ] != implode_ordered.signature(bound):
                    capabilities.reject(
                        "decimal scalar feed",
                        "producer AST or logical metadata changed",
                    )
                decimal_expression += 1
            if observed_sum:
                if node.names2[1 + sum_expression] != implode_ordered.signature(
                    bound
                ):
                    capabilities.reject(
                        "observed sum",
                        "producer AST or logical metadata changed",
                    )
                sum_expression += 1
            if precision_rows:
                if node.names2[
                    1 + decimal_expression
                ] != implode_ordered.signature(bound):
                    capabilities.reject(
                        "decimal date rows",
                        "producer AST or logical metadata changed",
                    )
                decimal_expression += 1
            if observed_group:
                if node.names2[
                    2 + len(node.names) + grouped_expression
                ] != implode_ordered.signature(bound):
                    capabilities.reject(
                        "observed group",
                        "producer AST or logical metadata changed",
                    )
                grouped_expression += 1
            if node.text == decimal_feed.MATERIALIZATION:
                if (
                    len(chain) != 1
                    or not terminal_boundary
                    or limit >= 0
                    or node.kind != SELECT
                    or len(node.exprs) != 1
                    or len(node.names2) != 2
                    or node.names2[0] != node.text
                    or node.names2[1] != implode_ordered.signature(bound)
                    or not decimal_feed.eligible(bound, schema)
                ):
                    capabilities.reject(
                        "decimal feed", "original scalar producer changed"
                    )
                decimal_feed_batch_rows = decimal_feed.BATCH_ROWS
            if ordered_decimal:
                if not ordered_rows.eligible(bound) or node.names2[
                    1 + decimal_expression
                ] != implode_ordered.signature(bound):
                    capabilities.reject(
                        "ordered decimal",
                        "producer AST or logical metadata changed",
                    )
                decimal_expression += 1
            if ordered_producer:
                producer_phases.validate_bound(node, bound, producer_expression)
                producer_expression += 1
            if ordered_private:
                implode_ordered.validate(node, bound)
            if (
                ordered_terminal
                or ordered_private
                or ordered_decimal
                or ordered_producer
                or ordered_group
                or precision_rows
            ):
                var cloned = ordered_rows.occurrences(
                    bound, capabilities.max_nodes
                )
                bound = cloned[0].copy()
                expression_ranks = cloned[1].copy()
            var ordered_scalar = (
                len(node.names2) >= 2 and node.names2[1] == "ordered_scalar_v1"
            )
            if ordered_scalar:
                ordered_mode = 1
                if (
                    node.text != "metal_window_materialization"
                    or node.names2[0] != "metal_explicit_cut_operand"
                    or node.kind != SELECT
                    or len(chain) != 1
                    or len(node.exprs) != 1
                    or not terminal_boundary
                    or limit >= 0
                    or source_count > 1
                    or len(node.names2) != 3 + len(expr._nodes)
                    or len(bound.expr._nodes) != len(expr._nodes)
                ):
                    capabilities.reject(
                        "ordered scalar",
                        "requires a private one-row CUT producer",
                    )
                var anchor = False
                for field in schema:
                    anchor |= (
                        field.name() == node.names2[2]
                        and field.dtype() == DataType.INT8
                    )
                if not anchor:
                    capabilities.reject(
                        "ordered scalar", "missing literal anchor provenance"
                    )
                var previous_rank = Int64(0)
                for at in range(len(expr._nodes)):
                    ref original = expr._nodes[at]
                    ref actual = bound.expr._nodes[at]
                    if (
                        original.op != actual.op
                        or original.left != actual.left
                        or original.right != actual.right
                        or original.extra != actual.extra
                        or is_window(actual.op)
                        or is_reduction(actual.op)
                    ):
                        capabilities.reject(
                            "ordered scalar", "binder changed the ranked AST"
                        )
                    var rank = parse_int64(node.names2[3 + at])
                    if rank < 0 or rank > 0x1FFFFFE:
                        capabilities.reject(
                            "ordered scalar", "fault rank exceeds token range"
                        )
                    if rank:
                        if rank <= previous_rank:
                            capabilities.reject(
                                "ordered scalar",
                                "requires canonical occurrence order",
                            )
                        previous_rank = rank
                    elif not (
                        actual.op == COL
                        or actual.op == LIT_INT
                        or actual.op == LIT_FLOAT
                        or actual.op == LIT_BOOL
                        or actual.op == LIT_NULL
                        or actual.op == IS_NOT_NULL
                        or actual.op == WHEN
                    ):
                        capabilities.reject(
                            "ordered scalar",
                            "generated checked node lacks an occurrence rank",
                        )
                    ordered_ranks.append(UInt32(rank))
            if (
                capabilities.temporal_binary_arithmetic
                or capabilities.temporal_physical_casts
            ) and temporal_expressions.contains(bound):
                if (
                    not ordered_private
                    and not ordered_producer
                    and (
                        len(chain) != 1
                        or node.kind != SELECT
                        or not terminal_boundary
                        or limit >= 0
                        or source_rows >= 0
                    )
                ):
                    capabilities.reject(
                        "temporal",
                        "prototype requires a direct scan and terminal projections without head (SELECT only)",
                    )
                if (
                    ordered_terminal
                    or ordered_private
                    or ordered_decimal
                    or ordered_producer
                ):
                    temporal_expressions.validate_pairs(bound)
                else:
                    temporal_expressions.validate(bound)
            if capabilities.calendar_truncation:
                for part in bound.expr._nodes:
                    if (
                        part.op == DT_TRUNCATE
                        and not (
                            ordered_terminal
                            and iana_terminal
                            or ordered_producer
                        )
                        and bound.dtypes[part.left].is_datetime()
                    ):
                        var interval = parse_every(part.text)
                        if interval[0] != 0 and (
                            not terminal_boundary
                            or k != 0
                            or node.kind == FILTER
                            or limit >= 0
                        ):
                            capabilities.reject(
                                "plan",
                                "checked calendar truncation requires terminal projections without head",
                            )
            var proven = List[Bool](length=len(bound.expr._nodes), fill=False)
            if (
                capabilities.proven_integer_operations
                or capabilities.decimal_comparisons
                or capabilities.decimal_arithmetic
                or capabilities.decimal_multiplicative
            ):
                proven = proven_integer_nodes(bound)
            for part_index in range(len(bound.expr._nodes)):
                if capabilities.utc_temporal_conversions:
                    ref utc_part = bound.expr._nodes[part_index]
                    var checked_utc = (
                        utc_part.op == DT_REPLACE_TZ and utc_part.text == "UTC"
                    ) or (
                        utc_part.op == DT_TRUNCATE
                        and bound.dtypes[utc_part.left].is_datetime()
                        and bound.dtypes[utc_part.left].time_zone() == "UTC"
                    )
                    if checked_utc and (
                        not terminal_boundary
                        or k != 0
                        or node.kind == FILTER
                        or limit >= 0
                    ):
                        capabilities.reject(
                            "plan",
                            "checked UTC operations require terminal projections without head",
                        )
                if (
                    bound.expr._nodes[part_index].op == DT_OFFSET_BY
                    and capabilities.temporal_offsets
                    and not (
                        ordered_terminal and iana_terminal or ordered_producer
                    )
                ):
                    var interval = parse_every(
                        bound.expr._nodes[part_index].text
                    )
                    if (
                        interval[1] != 0
                        or (
                            interval[0] != 0
                            and bound.dtypes[
                                bound.expr._nodes[part_index].left
                            ].is_datetime()
                        )
                    ) and (
                        not terminal_boundary
                        or k != 0
                        or node.kind == FILTER
                        or limit >= 0
                    ):
                        capabilities.reject(
                            "plan",
                            "checked temporal offsets require terminal projections without head",
                        )
                if bound.dtypes[part_index].is_integer():
                    ref part = bound.expr._nodes[part_index]
                    if (
                        part.op == ADD
                        or part.op == SUB
                        or part.op == MUL
                        or part.op == NEG
                        or part.op == ABS
                        or part.op == FLOORDIV
                        or part.op == POW
                    ):
                        # CPU pushdown can remove checked work. An operation
                        # without a static no-error proof still needs an
                        # observable terminal boundary, with no final head,
                        # unless the backend carries per-value errors through
                        # row selection and checks each observing consumer.
                        if not (
                            capabilities.proven_integer_operations
                            and proven[part_index]
                        ) and (
                            not terminal_boundary
                            or (
                                not capabilities.deferred_integer_errors
                                and (
                                    k != 0 or node.kind == FILTER or limit >= 0
                                )
                            )
                        ):
                            capabilities.reject(
                                "plan",
                                (
                                    "checked integer arithmetic requires"
                                    " terminal projections without head"
                                ),
                            )
                    if part.op == SUM and limit >= 0 and node.kind != AGG:
                        capabilities.reject(
                            "plan",
                            (
                                "integer sum with final head requires CPU"
                                " execution"
                            ),
                        )
            var conditional = False
            for candidate in bound.expr._nodes:
                conditional |= candidate.op == WHEN
            for part_index in range(len(bound.expr._nodes)):
                ref part = bound.expr._nodes[part_index]
                if (
                    bound.dtypes[part_index].is_decimal()
                    and (
                        capabilities.decimal_arithmetic
                        and (
                            part.op == ADD
                            or part.op == SUB
                            or part.op == NEG
                            or part.op == ABS
                        )
                        or capabilities.decimal_multiplicative
                        and (part.op == MUL or part.op == DIV)
                    )
                    and not proven[part_index]
                    and not ordered_group
                    and (
                        not terminal_boundary
                        or k != 0
                        or node.kind == FILTER
                        or limit >= 0
                    )
                ):
                    capabilities.reject(
                        "plan",
                        "checked decimal arithmetic requires terminal projections without head",
                    )
                if (
                    is_comparison(part.op)
                    and part.left >= 0
                    and bound.dtypes[part.left].is_decimal()
                    and capabilities.decimal_comparisons
                    and not proven[part_index]
                ):
                    var left = bound.dtypes[part.left]
                    var right = bound.dtypes[part.right]
                    # CPU's one-row Decimal128/Decimal64 comparison fast path
                    # rescales the wide operand without checking overflow.
                    # A resident filter can change the row count to one, so
                    # admit this combination only with a static proof.
                    if (
                        left.decimal_width() == 128
                        and right.decimal_width() == 64
                        and left.scale() < right.scale()
                    ) or (
                        right.decimal_width() == 128
                        and left.decimal_width() == 64
                        and right.scale() < left.scale()
                    ):
                        capabilities.reject(
                            "plan",
                            "checked decimal comparisons with a rescaled Decimal128 and Decimal64 require a static proof",
                        )
                    if (
                        not terminal_boundary
                        or k != 0
                        or node.kind == FILTER
                        or limit >= 0
                        or conditional
                    ):
                        capabilities.reject(
                            "plan",
                            "checked decimal comparisons require terminal projections without head or conditional branches",
                        )
                if (
                    part.op == CAST
                    and part.integer
                    and not ordered_producer
                    and not (
                        capabilities.proven_integer_operations
                        and proven[part_index]
                    )
                    and not (
                        capabilities.float64_emulated
                        and bound.dtypes[part_index].is_float()
                    )
                    and (
                        not terminal_boundary
                        or k != 0
                        or node.kind == FILTER
                        or limit >= 0
                    )
                ):
                    capabilities.reject(
                        "plan",
                        (
                            "strict casts require terminal projections without"
                            " head"
                        ),
                    )
            # Both window families evaluate their input first, then use a
            # stable partition permutation and scatter results to source rows.
            var rank_schema = schema.copy()
            var rank_slots = slots.copy()
            var ranked = bound.expr.copy()
            var original_rank_scopes = bound.expr.copy()
            var rank_scopes = List[Bool](length=len(ranked._nodes), fill=False)
            var has_rank = False
            for r in range(len(ranked._nodes)):
                var window_op = ranked._nodes[r].op
                var cumulative = (
                    window_op == CUM_SUM
                    or window_op == CUM_MIN
                    or window_op == CUM_MAX
                    or window_op == CUM_COUNT
                )
                var shift_fill = (
                    window_op == SHIFT
                    or window_op == FORWARD_FILL
                    or window_op == BACKWARD_FILL
                )
                var rolling_moments = (
                    window_op == ROLLING_VAR or window_op == ROLLING_STD
                )
                var rolling = (
                    rolling_moments
                    or window_op == ROLLING_SUM
                    or window_op == ROLLING_MEAN
                    or window_op == ROLLING_MIN
                    or window_op == ROLLING_MAX
                )
                var coordinate = window_op == INTERPOLATE_BY
                var interpolation = window_op == INTERPOLATE or coordinate
                var duration = is_rolling_by(window_op)
                if (
                    window_op != RANK
                    and not cumulative
                    and not shift_fill
                    and not rolling
                    and not interpolation
                    and not duration
                ):
                    continue
                if duration and not capabilities.duration_rolling:
                    capabilities.reject(
                        "plan",
                        "duration rolling windows require backend support",
                    )
                if coordinate and not capabilities.coordinate_interpolation:
                    capabilities.reject(
                        "plan",
                        "coordinate interpolation requires backend support",
                    )
                if (
                    interpolation
                    and not coordinate
                    and not capabilities.interpolation_windows
                ):
                    capabilities.reject(
                        "plan", "interpolation windows require backend support"
                    )
                if rolling_moments and not capabilities.rolling_moments:
                    capabilities.reject(
                        "plan",
                        "rolling variance and standard deviation require backend support",
                    )
                if rolling_moments and not (
                    capabilities.float64 or capabilities.float64_emulated
                ):
                    capabilities.reject(
                        "plan", "rolling moments require Float64 arithmetic"
                    )
                if rolling and not capabilities.rolling_windows:
                    capabilities.reject(
                        "plan", "rolling windows require backend support"
                    )
                if (
                    rolling
                    and window_op == ROLLING_SUM
                    and bound.dtypes[ranked._nodes[r].left].is_decimal()
                ):
                    capabilities.reject(
                        "logical",
                        "rolling_sum over a decimal column requires an explicit Float64 cast",
                    )
                if shift_fill and not capabilities.shift_fill_windows:
                    capabilities.reject(
                        "plan", "shift and fill windows require backend support"
                    )
                if cumulative and not capabilities.cumulative_windows:
                    capabilities.reject(
                        "plan", "cumulative windows require backend support"
                    )
                if (
                    not cumulative
                    and not shift_fill
                    and not rolling
                    and not interpolation
                    and not duration
                    and not capabilities.ranks
                ):
                    capabilities.reject("plan", "rank requires backend support")
                if node.kind == AGG:
                    capabilities.reject(
                        "plan",
                        (
                            "rank inside grouped aggregation requires CPU"
                            " execution"
                        ),
                    )
                var rank_node = ranked._nodes[r].copy()
                var replacement = r
                var partitions = List[String]()
                var scope = -1
                for parent in range(r + 1, len(ranked._nodes)):
                    ref candidate = original_rank_scopes._nodes[parent]
                    if candidate.op == OVER and _contains_node(
                        original_rank_scopes, candidate.left, r
                    ):
                        if scope >= 0:
                            capabilities.reject(
                                "plan",
                                "nested over scopes require CPU execution",
                            )
                        scope = parent
                if scope >= 0:
                    partitions = split_names(
                        original_rank_scopes._nodes[scope].text2
                    )
                    rank_scopes[scope] = True
                    if original_rank_scopes._nodes[scope].left == r:
                        replacement = scope
                var methods = List[String](
                    ["average", "min", "max", "dense", "ordinal"]
                )
                var method = -1
                for m in range(len(methods)):
                    if rank_node.text == methods[m]:
                        method = m
                if duration:
                    method = (
                        0 if window_op
                        == ROLLING_SUM_BY else 1 if window_op
                        == ROLLING_MEAN_BY else 2 if window_op
                        == ROLLING_MIN_BY else 3 if window_op
                        == ROLLING_MAX_BY else 4 if window_op
                        == ROLLING_VAR_BY else 5
                    )
                elif interpolation:
                    method = (
                        15 if coordinate else 13 if rank_node.text
                        == "linear" else 14
                    )
                elif rolling:
                    method = (
                        7 if window_op
                        == ROLLING_SUM else 8 if window_op
                        == ROLLING_MEAN else 9 if window_op
                        == ROLLING_MIN else 10 if window_op
                        == ROLLING_MAX else 11 if window_op
                        == ROLLING_VAR else 12
                    )
                elif shift_fill:
                    method = (
                        4 if window_op
                        == SHIFT else 5 if window_op
                        == FORWARD_FILL else 6
                    )
                elif cumulative:
                    method = 0 if window_op == CUM_SUM else (
                        1 if window_op
                        == CUM_MIN else (2 if window_op == CUM_MAX else 3)
                    )
                if method < 0:
                    raise Error("unknown rank method: " + rank_node.text)
                var child = bind(subtree(ranked, rank_node.left), rank_schema)
                if (
                    duration
                    and method == 0
                    and child.dtypes[len(child.dtypes) - 1].is_decimal()
                    and child.dtypes[len(child.dtypes) - 1].decimal_width()
                    != 128
                ):
                    var source_type = child.dtypes[len(child.dtypes) - 1]
                    # Every physical Int32/Int64 value fits precision 38 at
                    # the same scale, so this widening cannot discard a row.
                    child = bind(
                        child.expr.cast(
                            DataType.decimal(38, source_type.scale()),
                            strict=False,
                        ),
                        rank_schema,
                    )
                if (
                    rolling_moments
                    or (duration and (method == 1 or method >= 4))
                    or (interpolation and method != 14)
                    or (
                        rolling
                        and window_op == ROLLING_MEAN
                        and child.dtypes[len(child.dtypes) - 1].is_decimal()
                    )
                ):
                    var converted = child.expr.cast(DataType.FLOAT64)
                    child = bind(converted, rank_schema)
                var child_proven = List[Bool](
                    length=len(child.expr._nodes), fill=False
                )
                if capabilities.proven_integer_operations:
                    child_proven = proven_integer_nodes(child)
                for c in range(len(child.expr._nodes)):
                    var operation = child.expr._nodes[c].op
                    if not child_proven[c] and (
                        child.dtypes[c].is_integer()
                        and (
                            operation == ADD
                            or operation == SUB
                            or operation == MUL
                            or operation == NEG
                            or operation == ABS
                            or operation == FLOORDIV
                            or operation == POW
                        )
                        or operation == CAST
                        and child.expr._nodes[c].integer
                        and not child.dtypes[c].is_float()
                    ):
                        capabilities.reject(
                            "plan", "checked rank inputs require CPU execution"
                        )
                if (
                    1 + Int(duration) + len(partitions) > capabilities.max_nodes
                    or next_slot + 2 + Int(coordinate or duration)
                    > capabilities.max_slots
                    or len(steps) + 2 + Int(coordinate or duration)
                    > capabilities.max_steps
                ):
                    capabilities.reject(
                        "plan", "rank exceeds resident slot or step limits"
                    )
                var input_slot = next_slot
                var input_start = len(literals)
                _program(
                    child,
                    len(child.expr._nodes),
                    rank_slots,
                    dtype,
                    code,
                    literals,
                    node_dtypes,
                    capabilities,
                )
                slot_dtypes.append(child.dtypes[len(child.dtypes) - 1])
                steps.append(
                    RowStep(
                        input_start,
                        len(child.expr._nodes),
                        input_slot,
                        False,
                        0,
                        0,
                        False,
                        False,
                    )
                )
                next_slot += 1
                var axis_slot = -1
                var period_ticks = Int64(0)
                var period_months = Int64(0)
                var period_days = Int64(0)
                var zoned_duration = False
                var zone_offset = Int64(0)
                var period_unit = Int64(0)
                var closed = Int64(0)
                if coordinate or duration:
                    var axis_expr = subtree(ranked, rank_node.right)
                    if coordinate:
                        axis_expr = axis_expr.cast(DataType.FLOAT64)
                    var axis = bind(axis_expr, rank_schema)
                    var axis_type = axis.dtypes[len(axis.dtypes) - 1]
                    if duration:
                        zoned_duration = (
                            axis_type.is_datetime()
                            and axis_type.time_zone() != ""
                        )
                        if zoned_duration:
                            var zone_name = axis_type.time_zone()
                            if zone_name != "UTC":
                                var fixed = _fixed_offset(zone_name)
                                if not fixed:
                                    capabilities.reject(
                                        "temporal",
                                        "duration rolling requires a timezone-free or fixed-offset index",
                                    )
                                zone_offset = fixed.value()
                            if zone_offset:
                                if not capabilities.fixed_zone_duration_rolling:
                                    capabilities.reject(
                                        "temporal",
                                        "fixed-offset duration rolling requires backend support",
                                    )
                            elif not capabilities.utc_duration_rolling:
                                capabilities.reject(
                                    "temporal",
                                    "UTC duration rolling requires backend support",
                                )
                        positive_period(rank_node.text)
                        var sides = closed_sides(rank_node.text2)
                        closed = Int64(sides[0]) | (Int64(sides[1]) << 1)
                        var interval: Tuple[Int64, Int64]
                        if zoned_duration:
                            var zoned = parse_interval(rank_node.text)
                            interval = (zoned[0], zoned[2])
                            period_days = zoned[1]
                        else:
                            interval = parse_every(rank_node.text)
                        var per_tick = (
                            NANOS_PER_DAY if axis_type.is_date() else Int64(
                                1000000000
                            )
                            // axis_type.per_second()
                        )
                        if interval[1] % per_tick != 0:
                            capabilities.reject(
                                "temporal",
                                "duration rolling requires a fixed positive interval aligned to the storage unit",
                            )
                        period_ticks = interval[1] // per_tick
                        period_months = interval[0]
                        period_unit = Int64(
                            0 if axis_type.is_date() else 1 if axis_type.unit()
                            == "ms" else 2 if axis_type.unit()
                            == "us" else 3
                        )
                        if (
                            period_months
                            and not capabilities.calendar_duration_rolling
                        ):
                            capabilities.reject(
                                "temporal",
                                "calendar duration rolling requires backend support",
                            )
                    var axis_proven = List[Bool](
                        length=len(axis.expr._nodes), fill=False
                    )
                    if capabilities.proven_integer_operations:
                        axis_proven = proven_integer_nodes(axis)
                    for c in range(len(axis.expr._nodes)):
                        var operation = axis.expr._nodes[c].op
                        if not axis_proven[c] and (
                            axis.dtypes[c].is_integer()
                            and (
                                operation == ADD
                                or operation == SUB
                                or operation == MUL
                                or operation == NEG
                                or operation == ABS
                                or operation == FLOORDIV
                                or operation == POW
                            )
                            or operation == CAST
                            and axis.expr._nodes[c].integer
                            and not axis.dtypes[c].is_float()
                        ):
                            capabilities.reject(
                                "plan",
                                "checked duration indices require CPU execution" if duration else "checked interpolation axes require CPU execution",
                            )
                    axis_slot = next_slot
                    var axis_start = len(literals)
                    _program(
                        axis,
                        len(axis.expr._nodes),
                        rank_slots,
                        dtype,
                        code,
                        literals,
                        node_dtypes,
                        capabilities,
                    )
                    slot_dtypes.append(axis_type)
                    steps.append(
                        RowStep(
                            axis_start,
                            len(axis.expr._nodes),
                            axis_slot,
                            False,
                            0,
                            0,
                            False,
                            False,
                        )
                    )
                    next_slot += 1
                if (
                    1
                    + Int(duration)
                    + Int(period_months != 0)
                    + Int(zoned_duration)
                    + len(partitions)
                    > capabilities.max_nodes
                ):
                    capabilities.reject(
                        "plan", "window metadata exceeds backend node limits"
                    )
                var rank_start = len(literals)
                # Match the CPU Float64-to-Int conversion for min_samples.
                # A rounded positive value may wrap negative at Int64.MAX;
                # such a threshold accepts every nonempty window.
                code.extend(
                    [
                        Int64(
                            ROW_DURATION_KEY if duration else ROW_CUMULATIVE_KEY if cumulative
                            or shift_fill
                            or rolling
                            or interpolation else ROW_RANK_KEY
                        ),
                        Int64(input_slot),
                        Int64(axis_slot) if coordinate
                        or duration else (
                            rank_node.integer if rank_node.floating
                            < 0 else max(Int64(0), Int64(rank_node.floating))
                        ) if rolling else Int64(
                            method == 6
                        ) if shift_fill else Int64(
                            rank_node.min_count == 1
                        ),
                        Int64(method),
                    ]
                )
                var result_type = (
                    bound.dtypes[r] if cumulative
                    or shift_fill
                    or rolling
                    or interpolation
                    or duration else (
                        DataType.FLOAT64 if method == 0 else DataType.INT64
                    )
                )
                literals.append(
                    # Device row indices fit UInt32. Saturating
                    # larger bounds preserves every observable window/result.
                    bitcast[DType.float64](
                        period_ticks
                    ) if duration else bitcast[DType.float64](
                        min(rank_node.integer, Int64(UInt32.MAX))
                        | (
                            min(Int64(rank_node.min_count), Int64(UInt32.MAX))
                            << 32
                        )
                    ) if rolling_moments else bitcast[
                        DType.float64
                    ](
                        rank_node.integer
                    ) if shift_fill
                    or rolling else bitcast[DType.float64](
                        Int64(result_type.precision())
                    ) if cumulative
                    and result_type.is_decimal() else 0
                )
                node_dtypes.append(child.dtypes[len(child.dtypes) - 1])
                if duration:
                    code.extend(
                        [
                            Int64(ROW_DURATION_BOUNDS),
                            max(Int64(0), Int64(rank_node.floating)),
                            closed,
                            Int64(result_type.precision()) if method == 0
                            and result_type.is_decimal() else Int64(
                                rank_node.min_count
                            ),
                        ]
                    )
                    literals.append(0)
                    node_dtypes.append(slot_dtypes[axis_slot])
                    if period_months:
                        code.extend(
                            [
                                Int64(ROW_DURATION_CALENDAR),
                                period_months,
                                period_unit,
                                0,
                            ]
                        )
                        literals.append(0)
                        node_dtypes.append(DataType.INT64)
                    if zoned_duration:
                        code.extend(
                            [
                                Int64(ROW_DURATION_UTC),
                                period_days,
                                period_unit,
                                zone_offset,
                            ]
                        )
                        literals.append(0)
                        node_dtypes.append(DataType.INT64)
                for name in partitions:
                    var key = -1
                    for c in range(len(schema)):
                        if schema[c].name() == name:
                            key = c
                    if key < 0:
                        raise Error("Unknown partition column: " + name)
                    capabilities.require_key(schema[key].dtype())
                    code.extend(
                        [
                            Int64(ROW_SORT_KEY),
                            Int64(slots[key]),
                            Int64(0),
                            Int64(1),
                        ]
                    )
                    literals.append(0)
                    node_dtypes.append(schema[key].dtype())
                slot_dtypes.append(result_type)
                steps.append(
                    RowStep(
                        rank_start,
                        1
                        + Int(duration)
                        + Int(period_months != 0)
                        + Int(zoned_duration)
                        + len(partitions),
                        next_slot,
                        False,
                        0,
                        0,
                        True,
                        False,
                    )
                )
                var name = "__metal_rank_" + String(next_slot)
                var collision = True
                while collision:
                    collision = False
                    for column in rank_schema:
                        if column.name() == name:
                            collision = True
                    if collision:
                        name += "_"
                rank_schema.append(_empty(name, result_type))
                rank_slots.append(next_slot)
                next_slot += 1
                ranked._nodes[replacement] = col(name)._nodes[0].copy()
                if replacement != r:
                    ranked._nodes[r] = col(name)._nodes[0].copy()
                has_rank = True
            if has_rank:
                for scope in range(len(ranked._nodes)):
                    if (
                        not rank_scopes[scope]
                        or ranked._nodes[scope].op != OVER
                    ):
                        continue
                    var child = ranked._nodes[scope].left
                    var fragment = subtree(ranked, child)
                    for part in fragment._nodes:
                        if (
                            is_window(part.op)
                            or is_reduction(part.op)
                            or part.op == OVER
                        ):
                            capabilities.reject(
                                "plan",
                                (
                                    "rank scopes containing other windows or"
                                    " reductions require CPU execution"
                                ),
                            )
                    ranked._nodes[scope] = ranked._nodes[child].copy()
                bound = bind(
                    subtree(ranked, len(ranked._nodes) - 1), rank_schema
                )
            var count = len(bound.expr._nodes)
            var result_dtype = bound.dtypes[count - 1]
            var reduction = -1
            var min_count = 0
            var row_shape = bound.shape() == ROWS
            var last = bound.expr._nodes[count - 1].copy()
            if (
                last.op == MIN
                or last.op == FIRST
                or last.op == LAST
                or last.op == ANY
                or last.op == ALL
                or last.op == NULL_COUNT
                or last.op == ARG_MIN
                or last.op == ARG_MAX
                or last.op == MAX
                or last.op == SUM
                or last.op == COUNT
                or last.op == MEAN
                or last.op == COV
                or last.op == CORR
                or last.op == SKEW
                or last.op == KURTOSIS
                or last.op == MEDIAN
                or last.op == QUANTILE
                or last.op == VAR
                or last.op == STD
                or last.op == LEN
            ):
                if (node.kind != SELECT and node.kind != AGG) or bound.shapes[
                    last.left
                ] != ROWS:
                    capabilities.reject(
                        "plan",
                        "reductions require a row expression in final select",
                    )
                reduction = last.op
                capabilities.require_reduction(
                    bound.dtypes[last.left], reduction, source_count
                )
                capabilities.require_dtype(result_dtype)
                var checked_sum = (
                    result_dtype == DataType.INT32
                    or result_dtype == DataType.UINT32
                    or (
                        capabilities.bounded_integer_sum
                        and not bound.dtypes[last.left].is_decimal()
                        and (
                            bound.dtypes[last.left].physical() == DataType.INT64
                            or bound.dtypes[last.left] == DataType.UINT64
                        )
                    )
                )
                if reduction == SUM and checked_sum and not terminal_boundary:
                    capabilities.reject(
                        "plan", "checked sums require a terminal boundary"
                    )
                if (
                    node.kind == AGG
                    and reduction == SUM
                    and checked_sum
                    and (k != 0 or limit >= 0)
                ):
                    capabilities.reject(
                        "plan",
                        (
                            "checked grouped sums require terminal aggregation"
                            " without head"
                        ),
                    )
                min_count = (
                    Int(last.integer) if reduction == ANY
                    or reduction == ALL else last.min_count
                )
                count -= 1
                if (
                    last.left != count - 1
                    and reduction != COV
                    and reduction != CORR
                ):
                    capabilities.reject(
                        "plan",
                        "reduction must consume the complete row expression",
                    )
                if (
                    (reduction == SUM or reduction == MEAN)
                    and bound.dtypes[last.left] != dtype
                    and not capabilities.mixed_types
                ):
                    capabilities.reject(
                        "plan",
                        (
                            "resident numeric reduction requires the common"
                            " numeric input dtype"
                        ),
                    )
                if reduction == COV or reduction == CORR:
                    capabilities.require_reduction(
                        bound.dtypes[last.right], reduction, source_count
                    )
                    if bound.shapes[last.right] != ROWS:
                        capabilities.reject(
                            "plan",
                            "paired reductions require two row expressions",
                        )
                    if reduction == CORR and last.text != "pearson":
                        capabilities.reject(
                            "plan", "Spearman correlation requires rank staging"
                        )
                    if next_slot + 1 >= capabilities.max_slots:
                        capabilities.reject(
                            "plan", "paired reduction requires two value slots"
                        )
                    var left_slot = next_slot
                    # Keep the original left-to-right error observation order.
                    for side in List[Int]([last.left, last.right]):
                        var converted = subtree(bound.expr, side).cast(
                            DataType.FLOAT64
                        )
                        var child = bind(converted, rank_schema)
                        var child_count = len(child.expr._nodes)
                        var child_start = len(literals)
                        _program(
                            child,
                            child_count,
                            rank_slots,
                            dtype,
                            code,
                            literals,
                            node_dtypes,
                            capabilities,
                        )
                        slot_dtypes.append(DataType.FLOAT64)
                        steps.append(
                            RowStep(
                                child_start,
                                child_count,
                                next_slot,
                                False,
                                0,
                                0,
                                False,
                                False,
                            )
                        )
                        next_slot += 1
                    min_count = (
                        min(Int(last.integer), Int(Int32.MAX))
                        | (left_slot + 1) << 31
                        | min(configured_workers(), 65535) << 47
                    )
                    aggregate_outputs.append(
                        RowOutput(
                            expr._name,
                            result_dtype,
                            left_slot,
                            reduction,
                            min_count,
                        )
                    )
                    has_aggregate = True
                    continue
                if (
                    reduction == VAR
                    or reduction == STD
                    or reduction == SKEW
                    or reduction == KURTOSIS
                ):
                    # Counts fit Int32; every larger ddof yields null. The
                    # middle sixteen bits are reserved for paired moments.
                    # Higher moments use the low bits for bias/Fisher flags.
                    min_count = (
                        min(Int(last.integer), Int(Int32.MAX))
                        | min(configured_workers(), 65535) << 47
                    )
                    var converted = subtree(bound.expr, last.left).cast(
                        DataType.FLOAT64
                    )
                    bound = bind(converted, rank_schema)
                    count = len(bound.expr._nodes)
                if reduction == MEDIAN or reduction == QUANTILE:
                    min_count = Int(
                        bitcast[DType.int64](
                            last.floating if last.floating != 0 else Float64(0)
                        )
                    )
                    reduction = ROW_QUANTILE_LINEAR
                    if last.text == "lower":
                        reduction = ROW_QUANTILE_LOWER
                    elif last.text == "higher":
                        reduction = ROW_QUANTILE_HIGHER
                    elif last.text == "nearest":
                        reduction = ROW_QUANTILE_NEAREST
                    elif last.text == "midpoint":
                        reduction = ROW_QUANTILE_MIDPOINT
                    var converted = subtree(bound.expr, last.left).cast(
                        DataType.FLOAT64
                    )
                    bound = bind(converted, rank_schema)
                    count = len(bound.expr._nodes)
                has_aggregate = True
            elif bound.aggregated[len(bound.aggregated) - 1]:
                capabilities.reject(
                    "plan", "unsupported reduction or aggregate expression"
                )
            else:
                has_rows |= row_shape
            if node.kind == FILTER and result_dtype != DataType.BOOL:
                capabilities.reject("plan", "filter predicate must be Boolean")
            if (
                capabilities.reuse_column_slots
                and (node.kind == SELECT or node.kind == WITH_COLUMNS)
                and reduction < 0
                and row_shape
                and count == 1
                and bound.expr._nodes[0].op == COL
            ):
                var existing = rank_slots[bound.sources[0]]
                capabilities.require_dtype(result_dtype)
                if slot_dtypes[existing] == result_dtype and (
                    capabilities.mixed_types
                    or result_dtype == dtype
                    or result_dtype == DataType.BOOL
                ):
                    projected.append(
                        _projected_column(
                            expr._name, result_dtype, bound, rank_schema
                        )
                    )
                    projected_slots.append(existing)
                    reused_columns = True
                    continue
            if next_slot >= capabilities.max_slots:
                capabilities.reject(
                    "plan",
                    "resident region supports at most "
                    + String(capabilities.max_slots)
                    + " value slots",
                )
            var start = len(literals)
            _program(
                bound,
                count,
                rank_slots,
                dtype,
                code,
                literals,
                node_dtypes,
                capabilities,
                ordered_terminal and iana_terminal or ordered_producer,
            )
            if ordered_terminal and iana_terminal or ordered_producer:
                while len(iana_mask_bases) < len(steps):
                    iana_mask_bases.append(-1)
                var mask_words = iana_certificate.mask_words(
                    bound, capabilities.max_nodes
                )
                # occurrences(bound) is idempotent for the already-cloned tree.
                for at in range(count):
                    if (
                        code[4 * (start + at)]
                        == iana_mutations.ROW_IANA_MUTATION
                    ):
                        code[4 * (start + at) + 2] = Int64(
                            capsule.append(bound, at)
                        )
                iana_mask_bases.append(len(capsule.words))
                capsule.construction_peak_bytes = max(
                    capsule.construction_peak_bytes,
                    checked_add(
                        checked_mul(capsule.words.capacity(), 8),
                        checked_mul(mask_words.capacity(), 16),
                    ),
                )
                capsule.words.extend(mask_words^)
            if (
                ordered_terminal
                or ordered_private
                or ordered_decimal
                or ordered_producer
                or ordered_group
                or precision_rows
            ):
                while len(ordered_ranks) < start:
                    ordered_ranks.append(UInt32(0))
                if ordered_group:
                    # Scalar SUM and AGG strip the reduction root before
                    # emitting its intact original operand.
                    # Preserve precisely those original occurrence ranks.
                    for at in range(count):
                        ordered_ranks.append(expression_ranks[at])
                else:
                    ordered_ranks.extend(expression_ranks.copy())
            if (
                ordered_scalar
                or ordered_terminal
                or ordered_private
                or ordered_decimal
                or ordered_producer
                or ordered_group
                or precision_rows
            ):
                # Decimal high literal words and membership tails are not
                # evaluated nodes; their rank entries are never dereferenced.
                while len(ordered_ranks) < len(literals):
                    ordered_ranks.append(UInt32(0))
            slot_dtypes.append(bound.dtypes[count - 1])
            var gather_start = len(gathers)
            var gather_count = len(slots)
            if node.kind == FILTER:
                gather_count = _append_gathers(slots, gathers)
            steps.append(
                RowStep(
                    start,
                    count,
                    next_slot,
                    node.kind == FILTER,
                    gather_start,
                    gather_count,
                    False,
                    False,
                )
            )
            if reduction >= 0:
                aggregate_outputs.append(
                    RowOutput(
                        expr._name,
                        result_dtype,
                        next_slot,
                        reduction,
                        min_count,
                    )
                )
            else:
                projected.append(
                    _projected_column(
                        expr._name, result_dtype, bound, rank_schema
                    )
                )
                projected_slots.append(next_slot)
            next_slot += 1
        if node.kind == FILTER:
            if len(node.exprs) != 1:
                capabilities.reject("plan", "filter requires one predicate")
            continue
        if node.kind == AGG:
            if len(aggregate_outputs) != len(node.exprs):
                capabilities.reject(
                    "plan", "group aggregation requires reductions"
                )
            if next_slot + 3 + len(aggregate_outputs) > capabilities.max_slots:
                capabilities.reject(
                    "plan",
                    "group aggregation exceeds "
                    + String(capabilities.max_slots)
                    + " value slots",
                )
            var index_slot = next_slot
            var start_slot = next_slot + 1
            var first_slot = next_slot + 2
            next_slot += 3
            slot_dtypes.extend([DataType.INT64, DataType.INT64, DataType.INT64])
            var start = len(literals)
            code.extend(
                [
                    Int64(
                        ROW_GROUP_HEADER_WINDOW if node.text
                        == "metal_window_materialization" else ROW_GROUP_HEADER_LAST if len(
                            node.flags
                        )
                        and node.flags[0] else ROW_GROUP_HEADER
                    ),
                    Int64(len(group_keys)),
                    Int64(start_slot),
                    Int64(first_slot),
                ]
            )
            var group_metadata = UInt64(0)
            if node.text == "metal_window_materialization":
                group_metadata = UInt64(
                    min(configured_workers(), Int(Int32.MAX))
                )
                if (
                    capabilities.contiguous_window_reductions
                    and len(node.flags) > 1
                    and node.flags[1]
                ):
                    # Low32 retain configured workers; bit32 records that
                    # the original inner expression had no nested OVER.
                    group_metadata |= UInt64(1) << 32
            literals.append(bitcast[DType.float64](group_metadata))
            node_dtypes.append(DataType.INT64)
            var grouped_schema = List[Series]()
            var grouped_slots = List[Int]()
            var group_names = List[String]()
            for key in group_keys:
                var categorical_identity = False
                for field in query._nodes[node.left].text_fields:
                    if field.name == schema[key].name() and field.category:
                        categorical_identity = True
                var key_descriptor = UInt64(0)
                for original_key in node.window_keys:
                    if original_key.name != schema[key].name():
                        continue
                    # The original whole-expression binding may decode just
                    # this CAT key before preparing its OVER partition.
                    if original_key.string_hash:
                        categorical_identity = False
                    if (
                        original_key.dtype == DataType.STRING
                        or original_key.dtype._is_bytes()
                        or original_key.dtype.is_categorical()
                    ):
                        if UInt64(len(key_values)) >= UInt64(1) << 62:
                            raise Error(
                                "Metal window key lookup capacity exceeded"
                            )
                        key_descriptor = (UInt64(1) << 63) | UInt64(
                            len(key_values)
                        )
                        if not original_key.string_hash:
                            key_descriptor |= UInt64(1) << 62
                        if node.code_domain and (
                            original_key.dtype == DataType.STRING
                            or original_key.dtype.is_categorical()
                        ):
                            append_partition_values(
                                key_values,
                                node.code_domain.value(),
                                original_key,
                            )
                        else:
                            # Opaque byte dictionaries retain their own exact
                            # raw-byte family and are never text unions.
                            key_values.extend(original_key.values[].copy())
                    if (
                        categorical_identity
                        and original_key.dtype.is_categorical()
                        and not original_key.string_hash
                    ):
                        key_descriptor = (
                            (UInt64(1) << 63)
                            | (UInt64(1) << 62)
                            | (UInt64(1) << 61)
                        )
                    break
                code.extend(
                    [
                        Int64(ROW_GROUP_KEY),
                        Int64(slots[key]),
                        Int64(0),
                        Int64(2) if categorical_identity else Int64(0),
                    ]
                )
                literals.append(bitcast[DType.float64](key_descriptor))
                node_dtypes.append(schema[key].dtype())
                var key_column = _empty(schema[key].name(), schema[key].dtype())
                key_column._field_metadata = schema[key]._field_metadata
                grouped_schema.append(key_column^)
                grouped_slots.append(slots[key])
                group_names.append(schema[key].name())
            for output in aggregate_outputs:
                if output.name in group_names:
                    raise Error("Duplicate group output name: " + output.name)
                group_names.append(output.name)
                code.extend(
                    [
                        Int64(ROW_GROUP_OUTPUT),
                        Int64(output.slot),
                        Int64(next_slot),
                        Int64(output.reduction),
                    ]
                )
                literals.append(bitcast[DType.float64](Int64(output.min_count)))
                node_dtypes.append(output.dtype)
                slot_dtypes.append(output.dtype)
                grouped_schema.append(_empty(output.name, output.dtype))
                grouped_slots.append(next_slot)
                next_slot += 1
            steps.append(
                RowStep(
                    start,
                    1 + len(group_keys) + len(aggregate_outputs),
                    index_slot,
                    False,
                    0,
                    0,
                    False,
                    True,
                )
            )
            if node.maintain_order:
                var order_start = len(literals)
                code.extend(
                    [Int64(ROW_SORT_KEY), Int64(first_slot), Int64(0), Int64(0)]
                )
                literals.append(0)
                node_dtypes.append(DataType.INT64)
                var gather_start = len(gathers)
                var gather_count = _append_gathers(grouped_slots, gathers)
                steps.append(
                    RowStep(
                        order_start,
                        1,
                        0,
                        False,
                        gather_start,
                        gather_count,
                        True,
                        False,
                    )
                )
            schema = grouped_schema^
            slots = grouped_slots^
            continue
        if has_aggregate:
            if len(aggregate_outputs) != len(node.exprs):
                capabilities.reject(
                    "plan", "cannot mix row and reduction projections"
                )
            outputs = aggregate_outputs^
            reduced = True
        elif node.kind == SELECT:
            if not has_rows:
                capabilities.reject(
                    "plan",
                    "scalar-only select is not a resident row projection",
                )
            schema = projected^
            slots = projected_slots^
        else:
            for i in range(len(projected)):
                var found = -1
                for j in range(len(schema)):
                    if schema[j].name() == projected[i].name():
                        found = j
                if found < 0:
                    schema.append(projected[i].copy())
                    slots.append(projected_slots[i])
                else:
                    schema[found] = projected[i].copy()
                    slots[found] = projected_slots[i]
        var names = List[String]()
        if reduced:
            for output in outputs:
                if output.name in names:
                    capabilities.reject(
                        "plan", "duplicate output name: " + output.name
                    )
                names.append(output.name)
        else:
            for column in schema:
                if column.name() in names:
                    capabilities.reject(
                        "plan", "duplicate output name: " + column.name()
                    )
                names.append(column.name())
    if not reduced:
        for i in range(len(schema)):
            outputs.append(
                RowOutput(
                    schema[i].name(),
                    schema[i].dtype(),
                    slots[i],
                    -1,
                    0,
                    schema[i]._field_metadata,
                )
            )
    if len(steps) > capabilities.max_steps:
        capabilities.reject(
            "plan",
            "resident region supports at most "
            + String(capabilities.max_steps)
            + " device steps",
        )
    if len(outputs) > capabilities.max_slots:
        capabilities.reject(
            "plan",
            "resident region supports at most "
            + String(capabilities.max_slots)
            + " output columns",
        )
    if len(outputs) == 0 and not capabilities.empty_outputs:
        capabilities.reject(
            "plan", "resident region requires at least one output column"
        )
    if ordered_mode == 1 and (
        len(steps) != 1
        or steps[0].filter
        or steps[0].order
        or steps[0].grouped
        or reduced
        or len(outputs) != 1
        or len(ordered_ranks) != len(literals)
    ):
        capabilities.reject(
            "ordered scalar", "requires exactly one row expression"
        )
    if ordered_mode == 3 and (
        len(steps) != 1
        or len(outputs) != frame.width() + 1
        or limit >= 0
        or not terminal_boundary
    ):
        capabilities.reject(
            "ordered implode",
            "requires one fresh private output and preserved inputs",
        )
    if (
        ordered_mode == 2
        or ordered_mode == 3
        or ordered_mode == 4
        or ordered_mode == 5
        or ordered_mode == 6
        or ordered_mode == decimal_row_feed.ORDERED_MODE
    ):
        if reduced or len(ordered_ranks) != len(literals):
            capabilities.reject(
                "ordered rows", "invalid occurrence descriptor geometry"
            )
        for step in steps:
            if step.filter or step.order or step.grouped:
                capabilities.reject(
                    "ordered rows", "requires direct terminal row projections"
                )
    if ordered_mode == decimal_group_feed.ORDERED_MODE:
        # Header/key/reducer descriptors are not evaluated row occurrences.
        while len(ordered_ranks) < len(literals):
            ordered_ranks.append(UInt32(0))
        var grouped_steps = 0
        for step in steps:
            if step.filter or (step.order and not grouped_steps):
                capabilities.reject(
                    "decimal group feed", "row movement escaped original AGG"
                )
            grouped_steps += Int(step.grouped)
        if grouped_steps != (0 if reduced else 1) or len(ordered_ranks) != len(
            literals
        ):
            capabilities.reject(
                "decimal group feed", "invalid original AGG geometry"
            )
        if reduced:
            var floating_sum = False
            for output in outputs:
                var input_type = slot_dtypes[output.slot]
                if output.reduction != SUM:
                    capabilities.reject(
                        "decimal scalar feed", "requires original SUM states"
                    )
                if input_type == DataType.FLOAT64:
                    floating_sum = True
                elif not decimal_group_feed.infallible_sum(
                    input_type, Int(UInt32.MAX)
                ):
                    capabilities.reject(
                        "decimal scalar feed", "state exception bound changed"
                    )
            if not floating_sum:
                capabilities.reject(
                    "decimal scalar feed", "missing original Float64 SUM"
                )
    for step_index in range(len(steps)):
        ref step = steps[step_index]
        for occurrence in range(step.start, step.start + step.nodes):
            if code[4 * occurrence] == iana_mutations.ROW_IANA_MUTATION:
                var descriptor = Int(code[4 * occurrence + 2])
                if (
                    (ordered_mode != 5 and ordered_mode != 6)
                    or descriptor not in capsule.descriptors
                    or step_index >= len(iana_mask_bases)
                    or iana_mask_bases[step_index] < 0
                ):
                    capabilities.reject(
                        "temporal",
                        "IANA mutation requires an occurrence-certified phase",
                    )
                iana_mutations.validate_descriptor(capsule.words, descriptor)
        if ordered_mode == 5 or ordered_mode == 6:
            if step_index >= len(iana_mask_bases):
                capabilities.reject(
                    "temporal", "IANA mask certificate is missing"
                )
            iana_certificate.validate(
                capsule.words, iana_mask_bases[step_index], step.nodes
            )
    var capsule_peak = 0
    var zone_words = List[UInt64]()
    if len(capsule.words):
        for step in steps:
            for node in range(step.start, step.start + step.nodes):
                var op = code[4 * node]
                if (
                    op >= DT_YEAR
                    and op <= DT_TIME
                    and (code[4 * node + 3] & ROW_IANA_ZONE_FIELDS)
                ):
                    var source = step.start + Int(code[4 * node + 1])
                    var zone = capsule.zone(node_dtypes[source].time_zone())
                    code[4 * node + 2] = Int64(capsule.bases[zone])
        capsule.validate()
        capsule_peak = max(
            capsule.construction_peak_bytes,
            checked_add(
                checked_mul(capsule.words.capacity(), 16),
                checked_add(
                    checked_mul(capsule.descriptors.capacity(), 16),
                    checked_mul(
                        capsule.diagnostics.capacity(), 2 * size_of[String]()
                    ),
                ),
            ),
        )
        zone_words = capsule.words.copy()
    else:
        var zone_names = List[String]()
        var zone_bases = List[Int64]()
        for step in steps:
            for node in range(step.start, step.start + step.nodes):
                var op = code[4 * node]
                if (
                    op < DT_YEAR
                    or op > DT_TIME
                    or not (code[4 * node + 3] & ROW_IANA_ZONE_FIELDS)
                ):
                    continue
                var source = step.start + Int(code[4 * node + 1])
                var name = node_dtypes[source].time_zone()
                var found = -1
                for at in range(len(zone_names)):
                    if zone_names[at] == name:
                        found = at
                        break
                if found < 0:
                    var zone = TimeZone.load(name)
                    var packed = serialize_zone(zone)
                    _ = checked_add(len(zone_words), len(packed))
                    found = len(zone_names)
                    zone_names.append(name)
                    zone_bases.append(Int64(len(zone_words)))
                    zone_words.extend(packed^)
                code[4 * node + 2] = zone_bases[found]
    return RowPlan(
        frame^,
        dtype,
        code^,
        literals^,
        node_dtypes^,
        slot_dtypes^,
        gathers^,
        steps^,
        outputs^,
        next_slot,
        root,
        limit,
        reduced,
        reused_columns,
        key_values^,
        query._nodes[cursor].source_failure,
        zone_words^,
        ArcPointer(ordered_ranks^),
        ordered_mode,
        query._nodes[cursor].text_fields.copy(),
        query._nodes[root].text_fields.copy(),
        query._nodes[root].code_domain,
        query._nodes[cursor].code_domain,
        category_sorts^,
        iana_mask_bases^,
        capsule.descriptors.copy(),
        capsule.diagnostics.copy(),
        capsule_peak,
        producer_region,
        decimal_feed_batch_rows,
    )
