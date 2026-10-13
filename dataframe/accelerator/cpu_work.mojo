"""Common CPU work features for the shared bound numeric row subset."""
from std.memory import bitcast
from ..dtype import DataType
from ..expr import (
    DT_TOTAL,
    EXP,
    LOG,
    POW,
    DT_OFFSET_BY,
    CAST,
    LEN,
    MEAN,
    SUM,
    VAR,
    STD,
    COV,
    CORR,
    SKEW,
    KURTOSIS,
)
from ..lazy import LazyFrame
from .capabilities import RowCapabilities
from .rows import (
    RowPlan,
    lower_rows,
    ROW_GROUP_OUTPUT,
    ROW_RANK_KEY,
    ROW_CUMULATIVE_KEY,
    ROW_DURATION_KEY,
    ROW_CALENDAR_OFFSET,
    ROW_CALENDAR_TRUNCATE,
    ROW_QUANTILE_LINEAR,
    ROW_QUANTILE_MIDPOINT,
)
from .cost import COST_FEATURE_COUNT, cost_feature_names


def _width(dtype: DataType) -> Int:
    if dtype.is_decimal():
        return dtype.decimal_width() // 8
    if dtype == DataType.BOOL:
        return 1  # byte upper bound for packed storage
    if dtype == DataType.INT8 or dtype == DataType.UINT8:
        return 1
    if dtype == DataType.INT16 or dtype == DataType.UINT16:
        return 2
    if (
        dtype == DataType.INT32
        or dtype == DataType.UINT32
        or dtype == DataType.FLOAT32
    ):
        return 4
    return 8


def duration_family(plan: RowPlan, prefix: String) -> String:
    var family = String()
    for step in plan.steps:
        var start = step.start * 4
        if plan.code[start] != ROW_DURATION_KEY:
            continue
        family += (
            ";"
            + prefix
            + "duration-rolling-v1:m"
            + String(plan.code[start + 3])
            + ":p"
            + String(bitcast[DType.int64](plan.literals[step.start]))
            + ":c"
            + String(plan.code[start + 6])
            + ":n"
            + String(plan.code[start + 5])
            + ":d"
            + String(plan.code[start + 7])
        )
    return family^


def cpu_features(query: LazyFrame) raises -> Tuple[List[Float64], String]:
    """Bound worst-case work without inspecting selectivity or null values.

    Coefficients must be fitted on the declared dtype family across null and
    selectivity distributions. The CPU lower residual envelope is validated on
    separate generated query identities before use.
    """
    var plan = lower_rows(
        query,
        RowCapabilities(
            "analytical CPU",
            float64=True,
            float_division=True,
            float_rounding=True,
            float_modulo=True,
            float_transcendentals=True,
            float_power=True,
            float_sqrt=True,
            wide_integer=True,
            int64_arithmetic=True,
            extended_integers=True,
            mixed_types=True,
            casts=True,
            extrema=True,
            fixed_logical=True,
            duration_totals=True,
            calendar_offsets=True,
            calendar_truncation=True,
            temporal_offsets=True,
            decimal128=True,
            decimal128_reductions=True,
            decimal_float_cast=True,
            decimal_casts=True,
            decimal_comparisons=True,
            decimal_arithmetic=True,
            decimal_multiplicative=True,
            ranks=True,
            cumulative_windows=True,
            paired_moments=True,
            moments=True,
            quantiles=True,
            higher_moments=True,
            rolling_windows=True,
            rolling_moments=True,
            interpolation_windows=True,
            coordinate_interpolation=True,
            duration_rolling=True,
            empty_outputs=True,
            empty_inputs=True,
            chunked_inputs=True,
            shift_fill_windows=True,
        ),
    )
    var features = List[Float64](length=COST_FEATURE_COUNT, fill=0)
    features[0] = 1
    var rows = Float64(plan.source.height())
    features[1] = rows
    var family = String()
    for column in plan.source._columns:
        features[2] += rows * Float64(_width(column.dtype()))
        var current = column.dtype().name()
        if not family:
            family = current
        elif family != current:
            family = "mixed"
    var chunked = False
    var chunks = 0
    for column in plan.source._columns:
        chunked |= column.is_chunked()
        chunks += column.n_chunks()
    if chunked:
        family += ";chunked-inputs-v1:" + String(chunks)
    if not plan.source.width():
        family += ";empty-inputs-v1"
    if not len(plan.outputs):
        family += ";empty-outputs-v1"
    var paired = False
    for output in plan.outputs:
        paired |= output.reduction == COV or output.reduction == CORR
    for i in range(0, len(plan.code), 4):
        if plan.code[i] == ROW_GROUP_OUTPUT:
            paired |= plan.code[i + 3] == COV or plan.code[i + 3] == CORR
    if paired:
        family += ";paired-moments-v1"
    var exponentials = 0
    var logarithms = 0
    for i in range(0, len(plan.code), 4):
        exponentials += Int(plan.code[i] == EXP)
        logarithms += Int(plan.code[i] == LOG)
    if exponentials or logarithms:
        family += (
            ";transcendentals-v1:e"
            + String(exponentials)
            + ":l"
            + String(logarithms)
        )
    var powers = 0
    for i in range(0, len(plan.code), 4):
        powers += Int(
            plan.code[i] == POW and plan.node_dtypes[i // 4].is_float()
        )
    if powers:
        family += ";floating-power-v1:p" + String(powers)
    var calendar_offset = False
    for i in range(0, len(plan.code), 4):
        calendar_offset |= plan.code[i] == ROW_CALENDAR_OFFSET
    if calendar_offset:
        family += ";calendar-offsets-v1"
    var calendar = False
    for i in range(0, len(plan.code), 4):
        calendar |= plan.code[i] == ROW_CALENDAR_TRUNCATE
    if calendar:
        family += ";calendar-truncation-v1"
    var offsets = False
    for i in range(0, len(plan.code), 4):
        offsets |= plan.code[i] == DT_OFFSET_BY
    if offsets:
        family += ";fixed-offsets-v1"
    var totals = False
    for i in range(0, len(plan.code), 4):
        totals |= plan.code[i] == DT_TOTAL
    if totals:
        family += ";duration-totals-v1"
    var quantiles = False
    for output in plan.outputs:
        quantiles |= (
            ROW_QUANTILE_LINEAR <= output.reduction <= ROW_QUANTILE_MIDPOINT
        )
    for i in range(0, len(plan.code), 4):
        if plan.code[i] == ROW_GROUP_OUTPUT:
            quantiles |= (
                ROW_QUANTILE_LINEAR
                <= Int(plan.code[i + 3])
                <= ROW_QUANTILE_MIDPOINT
            )
    if quantiles:
        family += ";quantiles-v1"
    var higher = False
    for output in plan.outputs:
        higher |= output.reduction == SKEW or output.reduction == KURTOSIS
    for i in range(0, len(plan.code), 4):
        if plan.code[i] == ROW_GROUP_OUTPUT:
            higher |= plan.code[i + 3] == SKEW or plan.code[i + 3] == KURTOSIS
    if higher:
        family += ";higher-moments-v1"
    var moments = False
    for output in plan.outputs:
        moments |= output.reduction == VAR or output.reduction == STD
    for i in range(0, len(plan.code), 4):
        if plan.code[i] == ROW_GROUP_OUTPUT:
            moments |= plan.code[i + 3] == VAR or plan.code[i + 3] == STD
    if moments:
        family += ";moments-v1"
    var ranked = False
    var cumulative = False
    var rolling = False
    var rolling_moments = False
    var interpolation = False
    var coordinate_interpolation = False
    var shift_fill = False
    for step in plan.steps:
        ranked |= plan.code[step.start * 4] == ROW_RANK_KEY
        cumulative |= (
            plan.code[step.start * 4] == ROW_CUMULATIVE_KEY
            and plan.code[step.start * 4 + 3] < 4
        )
        shift_fill |= (
            plan.code[step.start * 4] == ROW_CUMULATIVE_KEY
            and plan.code[step.start * 4 + 3] >= 4
            and plan.code[step.start * 4 + 3] <= 6
        )
    for step in plan.steps:
        interpolation |= (
            plan.code[step.start * 4] == ROW_CUMULATIVE_KEY
            and plan.code[step.start * 4 + 3] >= 13
        )
        coordinate_interpolation |= (
            plan.code[step.start * 4] == ROW_CUMULATIVE_KEY
            and plan.code[step.start * 4 + 3] == 15
        )
        rolling_moments |= (
            plan.code[step.start * 4] == ROW_CUMULATIVE_KEY
            and plan.code[step.start * 4 + 3] >= 11
            and plan.code[step.start * 4 + 3] <= 12
        )
        rolling |= (
            plan.code[step.start * 4] == ROW_CUMULATIVE_KEY
            and plan.code[step.start * 4 + 3] >= 7
            and plan.code[step.start * 4 + 3] <= 12
        )
    if ranked:
        family += ";rank-v1"
    if cumulative:
        family += ";cumulative-v1"
    if shift_fill:
        family += ";shift-fill-v1"
    if rolling:
        family += ";rolling-v1"
    if rolling_moments:
        family += ";rolling-moments-v2"
    if interpolation:
        family += ";interpolation-v1"
    if coordinate_interpolation:
        family += ";coordinate-interpolation-v1"
    family += duration_family(plan, "")
    var wide_decimal = False
    for dtype in plan.node_dtypes:
        wide_decimal |= dtype.is_decimal() and dtype.decimal_width() == 128
    for output in plan.outputs:
        wide_decimal |= (
            output.dtype.is_decimal() and output.dtype.decimal_width() == 128
        )
    if wide_decimal:
        family += ";decimal128-v1"
    var wide_decimal_reduction = False
    for output in plan.outputs:
        wide_decimal_reduction |= (
            (output.reduction == SUM or output.reduction == MEAN)
            and plan.slot_dtypes[output.slot].is_decimal()
            and plan.slot_dtypes[output.slot].decimal_width() == 128
        )
    for i in range(0, len(plan.code), 4):
        if plan.code[i] == ROW_GROUP_OUTPUT:
            var source = plan.slot_dtypes[Int(plan.code[i + 1])]
            wide_decimal_reduction |= (
                (plan.code[i + 3] == SUM or plan.code[i + 3] == MEAN)
                and source.is_decimal()
                and source.decimal_width() == 128
            )
    if wide_decimal_reduction:
        family += ";decimal128-reductions-v1"
    for step in plan.steps:
        features[4 if step.filter else 3] += rows * Float64(step.nodes)
        for i in range(step.start, step.start + step.nodes):
            if plan.node_dtypes[i].is_integer():
                features[6] += rows
            if Int(plan.code[i * 4]) == CAST:
                features[7] += rows
    for output in plan.outputs:
        if output.reduction >= 0 and output.reduction != LEN:
            features[5] += rows
    features[14] = Float64(len(plan.steps))
    return (features^, family^)
