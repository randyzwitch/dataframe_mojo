"""Schema-only matching for bounded single integer-key grouped reductions."""
from dataframe import DataType, Expr, LazyFrame, col
from dataframe.expr import COL, LEN, COUNT
from dataframe.lazy import AGG, SCAN_FRAME, FILTER
from .predicate_plan import PredicateComposition, plan_predicates, predicate_key


@fieldwise_init
struct GroupedComposition(Copyable, Movable):
    var key: String
    var key_dtype: DataType
    var selected: PredicateComposition
    var row_counts: List[String]


def plan_grouped(region: LazyFrame) raises -> Optional[GroupedComposition]:
    var n = len(region._plan._nodes)
    if n < 2 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    ref root = region._plan._nodes[n - 1]
    if root.kind != AGG or root.left != n - 2 or len(root.group_keys()) != 1:
        return None
    var key = root.group_keys()[0].copy()
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    if key not in frame.columns():
        return None
    var dtype = frame.column(key).dtype()
    if dtype != DataType.INT32 and dtype != DataType.INT64:
        return None
    var offered = frame.lazy()
    for i in range(1, n - 1):
        ref node = region._plan._nodes[i]
        if node.kind != FILTER or node.left != i - 1 or len(node.exprs) != 1:
            return None
        offered = offered.filter(node.exprs[0])
    var expressions = List[Expr]()
    var counts = List[String]()
    for expression in root.exprs:
        if expression._name == key:
            return None
        var transformed = expression.copy()
        if len(transformed._nodes) == 2 and transformed._nodes[1].op == LEN:
            if (
                transformed._nodes[0].op != COL
                or transformed._nodes[1].left != 0
            ):
                return None
            # Rows, including null keys, come from the group-existence count.
            # A dummy native COUNT supplies the existing projection schema.
            transformed = col(key).count().alias(expression._name)
            counts.append(expression._name.copy())
        expressions.append(transformed^)
    if not expressions:
        return None
    var matched = plan_predicates(
        offered.select_exprs(expressions), allow_simple=True
    )
    if not matched:
        return None
    return GroupedComposition(key^, dtype, matched.take(), counts^)


def grouped_key(plan: GroupedComposition) -> String:
    var key = (
        String("grouped_reductions:v1:")
        + plan.key_dtype.name()
        + ":"
        + predicate_key(plan.selected)
    )
    var slot = -1
    for i in range(len(plan.selected.bundle.inputs)):
        if plan.selected.bundle.inputs[i] == plan.key:
            slot = i
    key += ":key" + String(slot) + ":limit256"
    for output in plan.selected.bundle.outputs:
        key += ":rows" if output.name in plan.row_counts else ":values"
    return key
