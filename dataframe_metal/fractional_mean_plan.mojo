"""Direct grouped means that support ordered binary64 accumulation on Metal."""
from dataframe import DataType, LazyFrame
from dataframe.expr import COL, MEAN
from dataframe.lazy import AGG, SCAN_FRAME


@fieldwise_init
struct FractionalMeans(Copyable, Movable):
    var key: String
    var key_dtype: DataType
    var values: List[String]
    var dtypes: List[DataType]
    var names: List[String]
    var fields: List[Int]
    var wide: Bool

    def identity(self) -> String:
        var result = String("grouped_ordered_means:v1:") + self.key_dtype.name()
        for i in range(len(self.values)):
            result += ":" + self.dtypes[i].name()
            if self.values[i] == self.key:
                result += ":key"
        for field in self.fields:
            result += ":out" + String(field)
        return result^


def plan_fractional_means(
    region: LazyFrame,
) raises -> Optional[FractionalMeans]:
    if (
        len(region._plan._nodes) != 2
        or region._plan._nodes[0].kind != SCAN_FRAME
    ):
        return None
    ref agg = region._plan._nodes[1]
    if agg.kind != AGG or agg.left != 0 or len(agg.group_keys()) != 1:
        return None
    ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
    ref key = agg.group_keys()[0]
    if key not in frame.columns() or frame.height() >= Int(Int32.MAX):
        return None
    var key_dtype = frame.column(key).dtype()
    if key_dtype != DataType.INT32 and key_dtype != DataType.INT64:
        return None
    var plan = FractionalMeans(
        key.copy(),
        key_dtype,
        List[String](),
        List[DataType](),
        List[String](),
        List[Int](),
        False,
    )
    var floating = False
    for expr in agg.exprs:
        if (
            len(expr._nodes) != 2
            or expr._nodes[0].op != COL
            or expr._nodes[1].op != MEAN
            or expr._nodes[1].left != 0
        ):
            return None
        ref name = expr._nodes[0].text
        if (
            name not in frame.columns()
            or expr._name == key
            or expr._name in plan.names
        ):
            return None
        var dtype = frame.column(name).dtype()
        if dtype not in [
            DataType.INT32,
            DataType.INT64,
            DataType.FLOAT32,
            DataType.FLOAT64,
        ]:
            return None
        floating = (
            floating or dtype == DataType.FLOAT32 or dtype == DataType.FLOAT64
        )
        plan.wide = plan.wide or dtype == DataType.FLOAT64
        var field = 0
        while field < len(plan.values) and plan.values[field] != name:
            field += 1
        if field == len(plan.values):
            if field == 4:
                return None
            plan.values.append(name.copy())
            plan.dtypes.append(dtype)
        plan.names.append(expr._name.copy())
        plan.fields.append(field)
    if not floating:
        return None
    return plan^
