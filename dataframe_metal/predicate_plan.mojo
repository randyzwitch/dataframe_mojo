"""Schema-only compilation of positive numeric AND/OR predicates.

Each leaf is an existing checked native range. A small truth table combines
the leaves without a per-row expression interpreter or GPU stack. For filter
selection, SQL AND and OR compose the leaves' is-true bits even with nulls.
"""
from std.memory import bitcast
from dataframe import DataFrame, DataType, Expr, LazyFrame
from dataframe.expr import (
    AND,
    OR,
    COL,
    IS_NULL,
    IS_NOT_NULL,
    NOT,
    NE,
    EQ,
    GT,
    GE,
    LT,
    LE,
)
from dataframe.lazy import SCAN_FRAME, FILTER, SELECT
from .filter_plan import FilterSum, _predicate, supported_dtype
from .composition_plan import Composition, plan_composition, composition_key

comptime MAX_CLAUSES = 8
comptime MAX_INPUTS = 9


comptime RANGE_CLAUSE = 0
comptime NULL_CLAUSE = 1
comptime COLUMN_CLAUSE = 2


struct PredicateClause(Copyable, Movable):
    var bounds: FilterSum
    var kind: Int
    var inverted: Bool
    var other: String
    var comparison: Int
    var other_input: Int

    def __init__(
        out self,
        var bounds: FilterSum,
        kind: Int = RANGE_CLAUSE,
        inverted: Bool = False,
        other: String = "",
        comparison: Int = 0,
    ):
        self.bounds = bounds^
        self.kind = kind
        self.inverted = inverted
        self.other = other.copy()
        self.comparison = comparison
        self.other_input = -1


@fieldwise_init
struct PredicateProgram(Copyable, Movable):
    var clauses: List[PredicateClause]
    var truth: List[UInt32]
    var slots: List[Int]


@fieldwise_init
struct PredicateComposition(Copyable, Movable):
    var bundle: Composition
    var program: PredicateProgram


def _range(
    expression: Expr, at: Int, frame: DataFrame
) raises -> Optional[FilterSum]:
    var first = at
    while (
        expression._nodes[first].op == AND or expression._nodes[first].op == OR
    ):
        first = expression._nodes[first].left
    var left = expression._nodes[first].left
    if left < 0 or expression._nodes[left].op != COL:
        return None
    var name = expression._nodes[left].text.copy()
    if name not in frame.columns() or not supported_dtype(
        frame.column(name).dtype()
    ):
        return None
    var plan = FilterSum(name, name, frame.column(name).dtype())
    if not _predicate(plan, expression, at):
        return None
    return plan^


def _same(a: PredicateClause, b: PredicateClause) -> Bool:
    if (
        a.kind != b.kind
        or a.inverted != b.inverted
        or a.other != b.other
        or a.comparison != b.comparison
    ):
        return False
    return (
        a.bounds.filter_column == b.bounds.filter_column
        and a.bounds.low == b.bounds.low
        and a.bounds.high == b.bounds.high
        and a.bounds.inclusive == b.bounds.inclusive
        and bitcast[DType.uint32](a.bounds.low_float)
        == bitcast[DType.uint32](b.bounds.low_float)
        and bitcast[DType.uint32](a.bounds.high_float)
        == bitcast[DType.uint32](b.bounds.high_float)
    )


def _emit(
    expression: Expr,
    at: Int,
    frame: DataFrame,
    mut clauses: List[PredicateClause],
    mut code: List[Int],
    inverted: Bool = False,
) raises -> Bool:
    ref root = expression._nodes[at]
    if root.op == NOT:
        return _emit(expression, root.left, frame, clauses, code, not inverted)
    var unequal = root.op == NE
    var equal = expression.copy()
    if unequal:
        equal._nodes[at].op = EQ
    var bounds = _range(equal if unequal else expression, at, frame)
    var leaf = Optional[PredicateClause]()
    if bounds:
        leaf = PredicateClause(bounds.take(), inverted=inverted != unequal)
    else:
        ref node = expression._nodes[at]
        if node.op == IS_NULL or node.op == IS_NOT_NULL:
            ref input = expression._nodes[node.left]
            if input.op != COL or input.text not in frame.columns():
                return False
            var dtype = frame.column(input.text).dtype()
            if not supported_dtype(dtype):
                return False
            leaf = PredicateClause(
                FilterSum(input.text.copy(), input.text.copy(), dtype),
                NULL_CLAUSE,
                (node.op == IS_NOT_NULL) != inverted,
            )
    if not leaf and (
        root.op == GT
        or root.op == GE
        or root.op == LT
        or root.op == LE
        or root.op == EQ
        or root.op == NE
    ):
        ref left = expression._nodes[root.left]
        ref right = expression._nodes[root.right]
        if (
            left.op == COL
            and right.op == COL
            and left.text in frame.columns()
            and right.text in frame.columns()
        ):
            var dtype = frame.column(left.text).dtype()
            if (
                supported_dtype(dtype)
                and frame.column(right.text).dtype() == dtype
            ):
                leaf = PredicateClause(
                    FilterSum(left.text.copy(), left.text.copy(), dtype),
                    COLUMN_CLAUSE,
                    inverted != unequal,
                    right.text.copy(),
                    EQ if unequal else root.op,
                )
    if leaf:
        var slot = -1
        for i in range(len(clauses)):
            if _same(clauses[i], leaf.value()):
                slot = i
        if slot < 0:
            if len(clauses) == MAX_CLAUSES:
                return False
            slot = len(clauses)
            clauses.append(leaf.take())
        code.append(slot)
        return True
    ref node = expression._nodes[at]
    if node.op != AND and node.op != OR:
        return False
    if not _emit(
        expression, node.left, frame, clauses, code, inverted
    ) or not _emit(expression, node.right, frame, clauses, code, inverted):
        return False
    # Push NOT to leaves. Each inverted comparison still rejects nulls;
    # complementing an is-true mask would incorrectly admit UNKNOWN.
    code.append(-1 if (node.op == AND) != inverted else -2)
    return True


def _truth(code: List[Int], clauses: Int) -> List[UInt32]:
    var table = List[UInt32](length=8, fill=0)
    for mask in range(1 << clauses):
        var stack = List[Bool]()
        for token in code:
            if token >= 0:
                stack.append(Bool(mask & (1 << token)))
            else:
                var right = stack.pop()
                var left = stack.pop()
                stack.append(left and right if token == -1 else left or right)
        if stack[0]:
            table[mask >> 5] |= UInt32(1) << UInt32(mask & 31)
    return table^


def plan_predicates(region: LazyFrame) raises -> Optional[PredicateComposition]:
    var n = len(region._plan._nodes)
    if n < 3 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    # Existing compositions already have their own specialized scan. Keep
    # their execution route aligned with their unchanged calibration key.
    if plan_composition(region):
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
    for clause in clauses:
        if clause.bounds.filter_column not in names:
            names.append(clause.bounds.filter_column.copy())
        if clause.kind == COLUMN_CLAUSE and clause.other not in names:
            names.append(clause.other.copy())
    if len(names) > MAX_INPUTS:
        return None
    var slots = List[Int]()
    for index in range(len(clauses)):
        ref clause = clauses[index]
        for i in range(len(names)):
            if names[i] == clause.bounds.filter_column:
                slots.append(i)
            if clause.kind == COLUMN_CLAUSE and names[i] == clause.other:
                clause.other_input = i
    # Eight value bindings and a ninth independent predicate binding are the
    # existing physical ABI. Extra predicate columns use unused value slots.
    bundle.inputs = List[String]()
    for i in range(min(8, len(names))):
        bundle.inputs.append(names[i].copy())
    bundle.predicate.filter_column = names[len(names) - 1].copy()
    bundle.predicate.filter_dtype = frame.column(
        bundle.predicate.filter_column
    ).dtype()
    var truth = _truth(code, len(clauses))
    return PredicateComposition(
        bundle^, PredicateProgram(clauses^, truth^, slots^)
    )


def predicate_key(plan: PredicateComposition) -> String:
    var advanced = False
    for clause in plan.program.clauses:
        advanced = advanced or clause.kind != RANGE_CLAUSE or clause.inverted
    var key = String(
        "predicate_reductions:v2:" if advanced else "predicate_reductions:v1:"
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
    for bits in plan.program.truth:
        key += ":truth=" + String(bits)
    return key
