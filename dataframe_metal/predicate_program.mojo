"""Schema-only compilation of positive numeric AND/OR predicates.

Each leaf is an existing checked native range. A small truth table combines
the leaves without a per-row expression interpreter or GPU stack. For filter
selection, SQL AND and OR compose the leaves' is-true bits even with nulls.
"""
from std.memory import bitcast
from dataframe import DataFrame, DataType, Expr
from dataframe.expr import (
    AND,
    LIT_BOOL,
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
    EQ_MISSING,
    NE_MISSING,
    LIT_NULL,
    LIT_FLOAT,
    UNTYPED,
)
from .filter_plan import FilterSum, _predicate, supported_dtype
from .membership_plan import compile_membership

comptime MAX_CLAUSES = 8
comptime MAX_INPUTS = 9


comptime RANGE_CLAUSE = 0
comptime NULL_CLAUSE = 1
comptime COLUMN_CLAUSE = 2
comptime MEMBERSHIP_CLAUSE = 3


struct PredicateClause(Copyable, Movable):
    var bounds: FilterSum
    var kind: Int
    var inverted: Bool
    var other: String
    var comparison: Int
    var other_input: Int
    var members: List[UInt64]

    def __init__(
        out self,
        var bounds: FilterSum,
        kind: Int = RANGE_CLAUSE,
        inverted: Bool = False,
        other: String = "",
        comparison: Int = 0,
        var members: List[UInt64] = List[UInt64](),
    ):
        self.bounds = bounds^
        self.kind = kind
        self.inverted = inverted
        self.other = other.copy()
        self.comparison = comparison
        self.other_input = -1
        self.members = members^


@fieldwise_init
struct PredicateProgram(Copyable, Movable):
    var clauses: List[PredicateClause]
    var truth: List[UInt32]
    var slots: List[Int]


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
        or a.members != b.members
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


def _missing_literal(
    expression: Expr, at: Int, frame: DataFrame, inverted: Bool
) raises -> Optional[PredicateClause]:
    ref root = expression._nodes[at]
    var column = root.left
    var literal = root.right
    if column < 0 or literal < 0:
        return None
    if expression._nodes[column].op != COL:
        column = root.right
        literal = root.left
    ref input = expression._nodes[column]
    ref value = expression._nodes[literal]
    if input.op != COL or input.text not in frame.columns():
        return None
    var dtype = frame.column(input.text).dtype()
    if not supported_dtype(dtype):
        return None
    var plan = FilterSum(input.text.copy(), input.text.copy(), dtype)
    if value.op == LIT_NULL:
        if len(value.dtypes) != 1 or not value.dtypes[0]:
            return None
        if (
            value.dtypes[0].value() != DataType.NULL
            and value.dtypes[0].value() != dtype
        ):
            return None
        return PredicateClause(plan^, NULL_CLAUSE, inverted)
    if value.op == LIT_FLOAT and value.floating != value.floating:
        if dtype != DataType.FLOAT32:
            return None
        if value.text != UNTYPED and (
            len(value.dtypes) != 1
            or not value.dtypes[0]
            or value.dtypes[0].value() != DataType.FLOAT32
        ):
            return None
        # A total NaN-equality leaf: NaN matches NaN, and null is false.
        return PredicateClause(
            plan^, NULL_CLAUSE, inverted, comparison=EQ_MISSING
        )
    return None


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
    var missing = root.op == EQ_MISSING or root.op == NE_MISSING
    var unequal = root.op == NE or root.op == NE_MISSING
    var equal = expression.copy()
    if unequal or missing:
        equal._nodes[at].op = EQ
    if (
        missing
        and root.left >= 0
        and root.right >= 0
        and expression._nodes[root.left].op != COL
        and expression._nodes[root.right].op == COL
    ):
        equal._nodes[at].left = root.right
        equal._nodes[at].right = root.left
    var membership = compile_membership(expression, at, frame)
    var bounds = _range(equal if unequal or missing else expression, at, frame)
    var leaf = _missing_literal(
        expression, at, frame, inverted != unequal
    ) if missing else Optional[PredicateClause]()
    if leaf:
        pass
    elif membership:
        var matched = membership.take()
        leaf = PredicateClause(
            FilterSum(
                matched.column.copy(), matched.column.copy(), matched.dtype
            ),
            MEMBERSHIP_CLAUSE,
            inverted,
            members=matched.keys.copy(),
        )
    elif bounds:
        leaf = PredicateClause(
            bounds.take(),
            inverted=inverted != unequal,
            comparison=EQ_MISSING if missing else 0,
        )
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
        or missing
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
                    EQ_MISSING if missing else EQ if unequal else root.op,
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
    # Push NOT to leaves. Ordinary comparisons retain UNKNOWN; total missing
    # comparisons remain valid, including their null cases.
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


def compile_predicate(
    expression: Expr, frame: DataFrame, root: Int = -1
) raises -> Optional[PredicateProgram]:
    """Compile a row predicate without planning or executing an aggregate.

    Slots remain unbound until the caller combines predicate and value inputs.
    The same compiler serves WHERE and the conditions of lazy CASE branches.
    """
    var at = len(expression._nodes) - 1 if root < 0 else root
    if at < 0 or at >= len(expression._nodes):
        return None
    if expression._nodes[at].op == LIT_BOOL:
        var truth = List[UInt32](length=8, fill=0)
        truth[0] = UInt32(expression._nodes[at].integer != 0)
        return PredicateProgram(List[PredicateClause](), truth^, List[Int]())
    var clauses = List[PredicateClause]()
    var code = List[Int]()
    if not _emit(expression, at, frame, clauses, code):
        return None
    var truth = _truth(code, len(clauses))
    return PredicateProgram(clauses^, truth^, List[Int]())


def bind_predicate_inputs(
    mut program: PredicateProgram,
    mut names: List[String],
    limit: Int = MAX_INPUTS,
) -> Bool:
    """Assign shared native-input slots; leave both arguments intact on failure."""
    var bound = program.copy()
    var inputs = names.copy()
    bound.slots = List[Int]()
    for index in range(len(bound.clauses)):
        ref clause = bound.clauses[index]
        if clause.bounds.filter_column not in inputs:
            inputs.append(clause.bounds.filter_column.copy())
        for i in range(len(inputs)):
            if inputs[i] == clause.bounds.filter_column:
                bound.slots.append(i)
        if clause.kind == COLUMN_CLAUSE:
            if clause.other not in inputs:
                inputs.append(clause.other.copy())
            for i in range(len(inputs)):
                if inputs[i] == clause.other:
                    clause.other_input = i
    if len(inputs) > limit:
        return False
    program = bound^
    names = inputs^
    return True
