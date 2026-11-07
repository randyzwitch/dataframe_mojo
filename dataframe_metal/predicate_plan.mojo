"""Match aggregate regions using the shared native numeric predicate compiler."""
from std.memory import bitcast
from dataframe import LazyFrame
from dataframe.lazy import SCAN_FRAME, FILTER, SELECT
from .composition_plan import Composition, plan_composition, composition_key
from .predicate_program import (
    MAX_CLAUSES,
    MAX_INPUTS,
    RANGE_CLAUSE,
    NULL_CLAUSE,
    COLUMN_CLAUSE,
    MEMBERSHIP_CLAUSE,
    PredicateClause,
    PredicateProgram,
    _emit,
    _truth,
    bind_predicate_inputs,
)


@fieldwise_init
struct PredicateComposition(Copyable, Movable):
    var bundle: Composition
    var program: PredicateProgram


def plan_predicates(
    region: LazyFrame, allow_simple: Bool = False
) raises -> Optional[PredicateComposition]:
    var n = len(region._plan._nodes)
    if (
        n < (2 if allow_simple else 3)
        or region._plan._nodes[0].kind != SCAN_FRAME
    ):
        return None
    # Existing compositions already have their own specialized scan. Keep
    # their execution route aligned with their unchanged calibration key.
    if not allow_simple and plan_composition(region):
        return None
    ref root = region._plan._nodes[n - 1]
    if root.kind != SELECT or root.left != n - 2:
        return None
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    var clauses = List[PredicateClause]()
    var code = List[Int]()
    for i in range(1, n - 1):
        ref node = region._plan._nodes[i]
        if (
            node.kind != FILTER
            or node.left != i - 1
            or len(node.exprs) != 1
            or not node.exprs[0]._nodes
        ):
            return None
        if not _emit(
            node.exprs[0], len(node.exprs[0]._nodes) - 1, frame, clauses, code
        ):
            return None
        if i > 1:
            code.append(-1)
    var offered = plan_composition(
        frame.lazy().select_exprs(root.exprs), include_native=True
    )
    if not offered:
        return None
    var bundle = offered.take()
    var names = bundle.inputs.copy()
    var truth = List[UInt32](length=8, fill=0)
    if code:
        truth = _truth(code, len(clauses))
    else:
        truth[0] = 1
    var program = PredicateProgram(clauses^, truth^, List[Int]())
    if not bind_predicate_inputs(program, names):
        return None
    # Eight value bindings and a ninth independent predicate binding are the
    # existing physical ABI. Extra predicate columns use unused value slots.
    bundle.inputs = List[String]()
    for i in range(min(8, len(names))):
        bundle.inputs.append(names[i].copy())
    bundle.predicate.filter_column = names[len(names) - 1].copy()
    bundle.predicate.filter_dtype = frame.column(
        bundle.predicate.filter_column
    ).dtype()
    return PredicateComposition(bundle^, program^)


def predicate_key(plan: PredicateComposition) -> String:
    var advanced = False
    var members = False
    for clause in plan.program.clauses:
        advanced = advanced or clause.kind != RANGE_CLAUSE or clause.inverted
        members = members or clause.kind == MEMBERSHIP_CLAUSE
    var key = String(
        "predicate_reductions:v3:" if members else "predicate_reductions:v2:" if advanced else "predicate_reductions:v1:"
    ) + composition_key(plan.bundle)
    for i in range(len(plan.program.clauses)):
        ref leaf = plan.program.clauses[i]
        ref clause = leaf.bounds
        if advanced:
            key += (
                ":kind="
                + String(leaf.kind)
                + ":inverted="
                + String(Int(leaf.inverted))
                + ":compare="
                + String(leaf.comparison)
                + ":right="
                + String(leaf.other_input)
            )
        key += (
            ":p"
            + String(plan.program.slots[i])
            + ":"
            + clause.filter_dtype.name()
            + ":"
            + String(clause.low)
            + ":"
            + String(clause.high)
            + ":"
            + String(bitcast[DType.uint32](clause.low_float))
            + ":"
            + String(bitcast[DType.uint32](clause.high_float))
            + ":"
            + String(clause.inclusive)
        )
    for leaf in plan.program.clauses:
        if leaf.kind == MEMBERSHIP_CLAUSE:
            key += ":members"
            for member in leaf.members:
                key += "," + String(member)
    for bits in plan.program.truth:
        key += ":truth=" + String(bits)
    return key
