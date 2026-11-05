"""Schema-only matching and shared state for fused numeric aggregates."""
from dataframe import DataFrame, col
from dataframe.dtype import DataType
from dataframe.expr import COL, COUNT, MEAN, SUM
from dataframe.lazy import FILTER, LazyFrame, SCAN_FRAME, SELECT
from .filter_plan import FilterSum, plan_region

comptime MAX_FIELDS = 4


@fieldwise_init
struct ReductionField(Copyable, Movable):
    var column: String
    var dtype: DataType
    var needs_sum: Bool


@fieldwise_init
struct ReductionOutput(Copyable, Movable):
    var field: Int
    var name: String
    var op: Int
    var min_count: Int


@fieldwise_init
struct ReductionBundle(Copyable, Movable):
    var predicate: FilterSum
    var fields: List[ReductionField]
    var outputs: List[ReductionOutput]
    var precise_float: Bool


def plan_bundle(region: LazyFrame) raises -> Optional[ReductionBundle]:
    var nodes = len(region._plan._nodes)
    if nodes < 2 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    ref root = region._plan._nodes[nodes - 1]
    if root.kind != SELECT or len(root.exprs) == 0:
        return None
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    var fields = List[ReductionField]()
    var outputs = List[ReductionOutput]()
    var names = List[String]()
    var predicate = Optional[FilterSum]()
    var precise = False
    for expression in root.exprs:
        if len(expression._nodes) != 2:
            return None
        ref leaf = expression._nodes[0]
        ref aggregate = expression._nodes[1]
        if leaf.op != COL or (
            aggregate.op != SUM
            and aggregate.op != COUNT
            and aggregate.op != MEAN
        ):
            return None
        if expression._name in names:
            return None
        names.append(expression._name.copy())
        # Reuse the single-sum matcher for the predicate's exact range and
        # physical-type checks, without evaluating any input rows.
        var offered = frame.lazy()
        for i in range(1, nodes - 1):
            ref node = region._plan._nodes[i]
            if node.kind != FILTER or len(node.exprs) != 1:
                return None
            offered = offered.filter(node.exprs[0])
        offered = offered.select(col(leaf.text).sum())
        var matched = plan_region(offered)
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
        if aggregate.op != COUNT:
            fields[field].needs_sum = True
        if aggregate.op == MEAN and fields[field].dtype == DataType.FLOAT32:
            precise = True
        outputs.append(
            ReductionOutput(
                field,
                expression._name.copy(),
                aggregate.op,
                aggregate.min_count,
            )
        )
    return ReductionBundle(predicate.take(), fields^, outputs^, precise)


def bundle_key(bundle: ReductionBundle) -> String:
    var key = String("fused_reductions:") + bundle.predicate.filter_dtype.name()
    if bundle.predicate.filtered:
        var shared = -1
        for i in range(len(bundle.fields)):
            if bundle.fields[i].column == bundle.predicate.filter_column:
                shared = i
        key += ":filtered:" + (
            "external" if shared < 0 else "shared" + String(shared)
        )
    else:
        key += ":plain"
    for field in bundle.fields:
        key += (
            ":" + field.dtype.name() + (":sum" if field.needs_sum else ":count")
        )
    return key + (":exact_float" if bundle.precise_float else ":native")
