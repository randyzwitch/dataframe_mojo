"""Schema matching for fused moments; public execution is not enabled here.

The existing expression, predicate and grouping matchers own native type and
resource limits. MEAN is an internal matching surrogate only: moment outputs
retain their original operation and ddof, and need a separate square state.
"""
from dataframe import LazyFrame
from dataframe.expr import VAR, STD, MEAN
from dataframe.lazy import SELECT, AGG
from .region_plan import normalize_region
from .predicate_plan import PredicateComposition, plan_predicates, predicate_key
from .group_plan import GroupedComposition, plan_grouped, grouped_key


@fieldwise_init
struct MomentOutput(Copyable, Movable):
    var field: Int
    var name: String
    var op: Int
    var parameter: Int64


@fieldwise_init
struct MomentPlan(Copyable, Movable):
    var selected: PredicateComposition
    var grouped: Optional[GroupedComposition]
    var outputs: List[MomentOutput]
    var needs_moments: List[Bool]
    # MEAN-surrogate region for physical layout matching after inlining.
    # Original VAR/STD operations and parameters remain in outputs.
    var matching_region: LazyFrame


def moment_output_sources(outputs: List[MomentOutput]) -> List[Int]:
    """Share variance finishing across aliases and matching VAR/STD pairs."""
    var sources = List[Int]()
    for i in range(len(outputs)):
        var source = i
        if outputs[i].op == VAR or outputs[i].op == STD:
            for j in range(i):
                if (
                    (outputs[j].op == VAR or outputs[j].op == STD)
                    and outputs[j].field == outputs[i].field
                    and outputs[j].parameter == outputs[i].parameter
                ):
                    source = sources[j]
                    break
        sources.append(source)
    return sources^


def plan_moments(region: LazyFrame) raises -> Optional[MomentPlan]:
    var size = len(region._plan._nodes)
    if size < 2:
        return None
    ref root = region._plan._nodes[size - 1]
    if root.kind != SELECT and root.kind != AGG:
        return None
    var rewritten = region.copy()
    var operations = List[Int]()
    var parameters = List[Int64]()
    var any_moments = False
    for i in range(len(root.exprs)):
        ref expression = root.exprs[i]
        if not expression._nodes:
            return None
        var last = len(expression._nodes) - 1
        ref aggregate = expression._nodes[last]
        operations.append(aggregate.op)
        parameters.append(
            aggregate.integer if aggregate.op == VAR
            or aggregate.op == STD else Int64(aggregate.min_count)
        )
        if aggregate.op == VAR or aggregate.op == STD:
            if (
                aggregate.integer < 0
                or aggregate.right >= 0
                or aggregate.extra >= 0
            ):
                return None
            any_moments = True
            ref surrogate = (
                rewritten._plan._nodes[size - 1].exprs[i]._nodes[last]
            )
            surrogate.op = MEAN
            surrogate.integer = 0
            surrogate.min_count = 0
    if not any_moments:
        return None
    var normalized = normalize_region(rewritten)
    if not normalized:
        return None
    var offered = normalized.take()
    var grouped = Optional[GroupedComposition]()
    var selected: PredicateComposition
    if root.kind == AGG:
        var matched = plan_grouped(offered)
        if not matched:
            return None
        selected = matched.value().selected.copy()
        grouped = matched.take()
    else:
        var matched = plan_predicates(offered, allow_simple=True)
        if not matched:
            return None
        selected = matched.take()
    var bindings = selected.bundle.outputs.copy()
    if grouped:
        bindings = grouped.value().outputs.copy()
    if len(bindings) != len(operations):
        return None
    var outputs = List[MomentOutput]()
    var needs = List[Bool](length=len(selected.bundle.fields), fill=False)
    for i in range(len(bindings)):
        ref binding = bindings[i]
        if operations[i] == VAR or operations[i] == STD:
            if binding.field < 0 or binding.field >= len(needs):
                return None
            needs[binding.field] = True
        outputs.append(
            MomentOutput(
                binding.field, binding.name.copy(), operations[i], parameters[i]
            )
        )
    return MomentPlan(selected^, grouped^, outputs^, needs^, offered^)


def moment_key(plan: MomentPlan) -> String:
    var key = String("moments:v1:")
    if plan.grouped:
        key += grouped_key(plan.grouped.value())
    else:
        key += predicate_key(plan.selected)
    # Aliases do not change work. Each output's own ddof or min_count does.
    for output in plan.outputs:
        key += ":out" + String(output.field) + "," + String(output.op)
        key += "," + String(output.parameter)
    return key
