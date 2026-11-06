"""Schema-only matching for bounded single integer-key grouped reductions."""
from dataframe import DataType, Expr, LazyFrame, col
from dataframe.expr import COL, LEN, COUNT
from dataframe.lazy import AGG, SCAN_FRAME, FILTER
from .group_keys import GroupKeys, compile_group_keys, MAX_GROUP_KEYS
from .predicate_plan import PredicateComposition, plan_predicates, predicate_key


@fieldwise_init
struct GroupedComposition(Copyable, Movable):
    var key: String
    var key_dtype: DataType
    var selected: PredicateComposition
    var row_counts: List[String]
    var keys: GroupKeys


def plan_grouped(
    region: LazyFrame, max_keys: Int = 1
) raises -> Optional[GroupedComposition]:
    var n = len(region._plan._nodes)
    if n < 2 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    ref root = region._plan._nodes[n - 1]
    if (
        root.kind != AGG
        or root.left != n - 2
        or len(root.group_keys()) == 0
        or len(root.group_keys()) > min(MAX_GROUP_KEYS, max_keys)
    ):
        return None
    var key = root.group_keys()[0].copy()
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    var compiled_keys = compile_group_keys(root.group_keys(), frame)
    if not compiled_keys:
        return None
    var dtype = compiled_keys.value().dtypes[0]
    var offered = frame.lazy()
    for i in range(1, n - 1):
        ref node = region._plan._nodes[i]
        if node.kind != FILTER or node.left != i - 1 or len(node.exprs) != 1:
            return None
        offered = offered.filter(node.exprs[0])
    var expressions = List[Expr]()
    var counts = List[String]()
    for expression in root.exprs:
        if expression._name in compiled_keys.value().names:
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
    return GroupedComposition(
        key^, dtype, matched.take(), counts^, compiled_keys.take()
    )


def grouped_key(plan: GroupedComposition) -> String:
    if len(plan.keys.names) > 1:
        var identity = String("grouped_reductions:v2:tuple:")
        for i in range(len(plan.keys.names)):
            var slot = -1
            for j in range(len(plan.selected.bundle.inputs)):
                if plan.selected.bundle.inputs[j] == plan.keys.names[i]:
                    slot = j
            identity += (
                ":" + plan.keys.dtypes[i].name() + ":input" + String(slot)
            )
        identity += ":" + predicate_key(plan.selected)
        for output in plan.selected.bundle.outputs:
            identity += ":rows" if output.name in plan.row_counts else ":values"
        return identity
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
