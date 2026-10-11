"""Common CPU work features for the shared bound numeric row subset."""
from ..dtype import DataType
from ..expr import CAST, LEN
from ..lazy import LazyFrame
from .capabilities import RowCapabilities
from .rows import lower_rows
from .cost import COST_FEATURE_COUNT, cost_feature_names


def _width(dtype: DataType) -> Int:
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
            wide_integer=True,
            int64_arithmetic=True,
            extended_integers=True,
            mixed_types=True,
            casts=True,
            extrema=True,
            fixed_logical=False,
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
