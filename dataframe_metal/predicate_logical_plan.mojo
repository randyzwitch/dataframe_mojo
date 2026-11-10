"""Schema-only mask reduction plans for existing global logical aggregates."""
from dataframe import DataType, Expr, LazyFrame, lit
from dataframe.binding import bind
from dataframe.expr import ANY, ALL, LEN, COUNT, NULL_COUNT, COL, EQ_MISSING
from dataframe.lazy import SCAN_FRAME, FILTER, SELECT
from .predicate_program import (
    PredicateProgram,
    compile_predicate,
    bind_predicate_inputs,
    NULL_CLAUSE,
    COLUMN_CLAUSE,
)
from .composition_plan import condition_key
from .expression_format import BOOL_COLUMN_COMPARISON


@fieldwise_init
struct PredicateLogicalOutput(Copyable, Movable):
    var name: String
    var op: Int
    var field: Int
    var ignore_nulls: Bool


@fieldwise_init
struct PredicateLogicalPlan(Copyable, Movable):
    var inputs: List[String]
    var values: List[Bool]
    var validity: List[Bool]
    var selection: PredicateProgram
    var positive: List[PredicateProgram]
    var negative: List[PredicateProgram]
    var outputs: List[PredicateLogicalOutput]
    var residual: List[Expr]
    var order: List[String]


def _roles(
    program: PredicateProgram, mut values: List[Bool], mut validity: List[Bool]
):
    for i in range(len(program.clauses)):
        ref clause = program.clauses[i]
        var slot = program.slots[i]
        validity[slot] = True
        if clause.kind != NULL_CLAUSE or clause.comparison in [
            BOOL_COLUMN_COMPARISON,
            EQ_MISSING,
        ]:
            values[slot] = True
        if clause.kind == COLUMN_CLAUSE:
            values[clause.other_input] = True
            validity[clause.other_input] = True


def plan_predicate_logical(
    region: LazyFrame,
) raises -> Optional[PredicateLogicalPlan]:
    var size = len(region._plan._nodes)
    if size < 2 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    ref root = region._plan._nodes[size - 1]
    if root.kind != SELECT or root.left != size - 2:
        return None
    ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
    var filter = Optional[Expr]()
    for i in range(1, size - 1):
        ref node = region._plan._nodes[i]
        if node.kind != FILTER or node.left != i - 1 or len(node.exprs) != 1:
            return None
        if filter:
            filter = filter.value() & node.exprs[0]
        else:
            filter = node.exprs[0].copy()
    var selected = compile_predicate(
        filter.value().copy() if filter else lit(True), frame
    )
    if not selected:
        return None
    var names = List[String]()
    if not bind_predicate_inputs(selected.value(), names):
        return None
    var positive = List[PredicateProgram]()
    var negative = List[PredicateProgram]()
    var keys = List[String]()
    var outputs = List[PredicateLogicalOutput]()
    var residual = List[Expr]()
    var order = List[String]()
    for expression in root.exprs:
        if not expression._nodes or expression._name in order:
            return None
        order.append(expression._name.copy())
        ref aggregate = expression._nodes[len(expression._nodes) - 1]
        if aggregate.op not in [ANY, ALL, LEN, COUNT, NULL_COUNT]:
            residual.append(expression.copy())
            continue
        var count = len(expression._nodes)
        if count < 2 or aggregate.left != count - 2:
            return None
        var bound = bind(expression, frame._columns, fuse=False)
        var dtype = bound.dtypes[count - 2]
        var value = expression.copy()
        _ = value._nodes.pop()
        var native_column = (
            count == 2
            and value._nodes[0].op == COL
            and dtype
            in [DataType.BOOL, DataType.INT32, DataType.INT64, DataType.FLOAT32]
        )
        if aggregate.op == LEN and native_column:
            outputs.append(
                PredicateLogicalOutput(expression._name.copy(), LEN, -1, True)
            )
            continue
        var t: Optional[PredicateProgram]
        var f: Optional[PredicateProgram]
        if dtype == DataType.BOOL:
            t = compile_predicate(value, frame)
            f = compile_predicate(~value, frame)
        elif native_column and aggregate.op in [COUNT, NULL_COUNT]:
            t = compile_predicate(value.is_not_null(), frame)
            f = compile_predicate(lit(False), frame)
        elif aggregate.op in [ANY, ALL]:
            return None
        else:
            residual.append(expression.copy())
            continue
        if not t or not f:
            return None
        var key = condition_key(t.value()) + "|" + condition_key(f.value())
        var field = keys.index(key) if key in keys else len(keys)
        if field == len(keys):
            if len(keys) == 8:
                return None
            if not bind_predicate_inputs(
                t.value(), names
            ) or not bind_predicate_inputs(f.value(), names):
                return None
            keys.append(key^)
            positive.append(t.take())
            negative.append(f.take())
        outputs.append(
            PredicateLogicalOutput(
                expression._name.copy(),
                aggregate.op,
                field,
                Bool(aggregate.integer),
            )
        )
    if not outputs:
        return None
    if not positive:
        var t = compile_predicate(lit(True), frame)
        var f = compile_predicate(lit(False), frame)
        positive.append(t.take())
        negative.append(f.take())
    var values = List[Bool](length=len(names), fill=False)
    var validity = List[Bool](length=len(names), fill=False)
    _roles(selected.value(), values, validity)
    for i in range(len(positive)):
        _roles(positive[i], values, validity)
        _roles(negative[i], values, validity)
    if not names:
        for name in frame.columns():
            if frame.column(name).dtype() in [
                DataType.BOOL,
                DataType.INT32,
                DataType.INT64,
                DataType.FLOAT32,
            ]:
                names.append(name.copy())
                values.append(False)
                validity.append(False)
                break
    if not names:
        return None
    return PredicateLogicalPlan(
        names^,
        values^,
        validity^,
        selected.take(),
        positive^,
        negative^,
        outputs^,
        residual^,
        order^,
    )
