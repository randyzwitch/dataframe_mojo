"""Schema-only matching of native MIN/MAX bundles over numeric ranges."""
from dataframe import col, LazyFrame
from dataframe.expr import COL, MIN, MAX
from dataframe.lazy import SCAN_FRAME, FILTER, SELECT
from .filter_plan import FilterSum, plan_region
from .multi_plan import ReductionField, ReductionOutput, MAX_FIELDS


@fieldwise_init
struct ExtremaBundle(Copyable, Movable):
    var predicate: FilterSum
    var fields: List[ReductionField]
    var outputs: List[ReductionOutput]


def plan_extrema(region: LazyFrame) raises -> Optional[ExtremaBundle]:
    var nodes = len(region._plan._nodes)
    if nodes < 2 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    ref root = region._plan._nodes[nodes - 1]
    if root.kind != SELECT or not root.exprs:
        return None
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    var fields = List[ReductionField]()
    var outputs = List[ReductionOutput]()
    var names = List[String]()
    var predicate = Optional[FilterSum]()
    for expression in root.exprs:
        if len(expression._nodes) != 2:
            return None
        ref leaf = expression._nodes[0]
        ref aggregate = expression._nodes[1]
        if leaf.op != COL or (aggregate.op != MIN and aggregate.op != MAX):
            return None
        if expression._name in names:
            return None
        names.append(expression._name.copy())
        var offered = frame.lazy()
        for i in range(1, nodes - 1):
            ref node = region._plan._nodes[i]
            if (
                node.kind != FILTER
                or len(node.exprs) != 1
                or node.left != i - 1
            ):
                return None
            offered = offered.filter(node.exprs[0])
        if root.left != nodes - 2:
            return None
        var matched = plan_region(offered.select(col(leaf.text).sum()))
        if not matched:
            return None
        if not predicate:
            predicate = matched.value().copy()
        var field = -1
        for i in range(len(fields)):
            if fields[i].column == leaf.text:
                field = i
        if field < 0:
            if len(fields) == MAX_FIELDS:
                return None
            field = len(fields)
            fields.append(
                ReductionField(leaf.text.copy(), matched.value().dtype, False)
            )
        outputs.append(
            ReductionOutput(field, expression._name.copy(), aggregate.op, 0)
        )
    return ExtremaBundle(predicate.take(), fields^, outputs^)


def extrema_key(bundle: ExtremaBundle) -> String:
    var key = String("native_extrema:v1:")
    key += bundle.predicate.filter_dtype.name()
    key += ":filtered" if bundle.predicate.filtered else ":plain"
    var shared = -1
    for i in range(len(bundle.fields)):
        if bundle.fields[i].column == bundle.predicate.filter_column:
            shared = i
        var minimum = False
        var maximum = False
        for output in bundle.outputs:
            if output.field == i:
                minimum = minimum or output.op == MIN
                maximum = maximum or output.op == MAX
        key += ":" + bundle.fields[i].dtype.name()
        key += ":min" if minimum else ""
        key += ":max" if maximum else ""
    return key + ":shared=" + String(shared)
