"""Schema-only matching for bounded native integer-key grouped reductions."""
from dataframe import DataType, Expr, LazyFrame, col
from dataframe.expr import COL, LEN, COUNT
from dataframe.lazy import AGG, SCAN_FRAME, FILTER
from .multi_plan import ReductionOutput
from .group_keys import GroupKeys, compile_group_keys, MAX_GROUP_KEYS
from .predicate_plan import PredicateComposition, plan_predicates, predicate_key


@fieldwise_init
struct GroupedComposition(Copyable, Movable):
    var key: String
    var key_dtype: DataType
    var selected: PredicateComposition
    var row_counts: List[String]
    var keys: GroupKeys
    var outputs: List[ReductionOutput]


def plan_grouped(
    region: LazyFrame, max_keys: Int = MAX_GROUP_KEYS
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
    var names = List[String]()
    for expression in root.exprs:
        if (
            expression._name in compiled_keys.value().names
            or expression._name in names
        ):
            return None
        names.append(expression._name.copy())
        var transformed = expression.copy()
        if len(transformed._nodes) == 2 and transformed._nodes[1].op == LEN:
            if (
                transformed._nodes[0].op != COL
                or transformed._nodes[1].left != 0
                or transformed._nodes[0].text not in frame.columns()
            ):
                return None
            # Row counts already exist in the hash scan. They consume neither
            # an additional aggregate value nor a predicate/value input binding.
            counts.append(expression._name.copy())
            continue
        expressions.append(transformed^)
    if not names:
        return None
    if not expressions:
        # All-LEN queries still need one field to carry the existing schema
        # and dispatch machinery; its non-null count is never the row result.
        expressions.append(col(key).count().alias(counts[0]))
    var matched = plan_predicates(
        offered.select_exprs(expressions), allow_simple=True
    )
    if not matched:
        return None
    var outputs = List[ReductionOutput]()
    for name in names:
        if name in counts:
            outputs.append(ReductionOutput(-1, name.copy(), COUNT, 0))
        else:
            var found = False
            for output in matched.value().bundle.outputs:
                if output.name == name:
                    outputs.append(output.copy())
                    found = True
                    break
            if not found:
                return None
    return GroupedComposition(
        key^, dtype, matched.take(), counts^, compiled_keys.take(), outputs^
    )


def grouped_key(plan: GroupedComposition) -> String:
    var identity = String("grouped_reductions:v3:local_updates:")
    for i in range(len(plan.keys.names)):
        var slot = -1
        for j in range(len(plan.selected.bundle.inputs)):
            if plan.selected.bundle.inputs[j] == plan.keys.names[i]:
                slot = j
        identity += ":" + plan.keys.dtypes[i].name() + ":input" + String(slot)
    identity += ":limit65536:hybrid_v1:" + predicate_key(plan.selected)
    for output in plan.outputs:
        identity += ":rows" if output.name in plan.row_counts else ":values"
    return identity
