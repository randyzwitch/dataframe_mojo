"""Schema-only native FIRST/LAST/ARG_MIN/ARG_MAX regions and tuple keys."""
from std.memory import bitcast
from dataframe import DataFrame, DataType, LazyFrame
from dataframe.binding import bind
from dataframe.expr import COL, FIRST, LAST, ARG_MIN, ARG_MAX, DROP_NULLS
from dataframe.lazy import SCAN_FRAME, FILTER, SELECT, AGG
from .group_keys import GroupKeys, compile_group_keys
from .predicate_program import (
    PredicateProgram,
    _emit,
    _truth,
    bind_predicate_inputs,
)


@fieldwise_init
struct OrderedField(Copyable, Movable):
    var column: String
    var dtype: DataType
    var drop_nulls: Bool
    var operations: UInt32


@fieldwise_init
struct OrderedOutput(Copyable, Movable):
    var name: String
    var field: Int
    var op: Int


@fieldwise_init
struct OrderedPlan(Copyable, Movable):
    var fields: List[OrderedField]
    var outputs: List[OrderedOutput]
    var inputs: List[String]
    var predicate: PredicateProgram
    var keys: Optional[GroupKeys]


def ordered_tag(dtype: DataType) -> Int:
    return (
        0 if dtype
        == DataType.INT32 else 1 if dtype
        == DataType.INT64 else 2 if dtype
        == DataType.FLOAT32 else 3 if dtype
        == DataType.BOOL else -1
    )


def ordered_side(op: Int) -> Int:
    return 0 if op == FIRST else 1 if op == LAST else 2 if op == ARG_MIN else 3


def plan_ordered(region: LazyFrame) raises -> Optional[OrderedPlan]:
    var count = len(region._plan._nodes)
    if count < 2 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    ref root = region._plan._nodes[count - 1]
    if root.kind != SELECT and root.kind != AGG:
        return None
    ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
    if frame.height() >= Int(Int32.MAX):
        return None
    var keys = Optional[GroupKeys]()
    if root.kind == AGG:
        keys = compile_group_keys(root.group_keys(), frame)
        if not keys:
            return None
    var inputs = List[String]()
    var fields = List[OrderedField]()
    var outputs = List[OrderedOutput]()
    var names = List[String]()
    for expression in root.exprs:
        var size = len(expression._nodes)
        if size != 2 and size != 3:
            return None
        ref value = expression._nodes[0]
        ref aggregate = expression._nodes[size - 1]
        if (
            value.op != COL
            or value.text not in frame.columns()
            or aggregate.op not in [FIRST, LAST, ARG_MIN, ARG_MAX]
            or aggregate.left != size - 2
            or aggregate.right >= 0
            or aggregate.extra >= 0
        ):
            return None
        if expression._name in names:
            return None
        if keys and expression._name in keys.value().names:
            return None
        names.append(expression._name.copy())
        var drop = size == 3
        if drop:
            ref transform = expression._nodes[1]
            if (
                transform.op != DROP_NULLS
                or transform.left != 0
                or transform.right >= 0
                or transform.extra >= 0
            ):
                return None
        var dtype = frame.column(value.text).dtype()
        if ordered_tag(dtype) < 0:
            return None
        try:
            _ = bind(expression, frame._columns, fuse=False)
        except:
            return None
        var field = -1
        for i in range(len(fields)):
            if fields[i].column == value.text and fields[i].drop_nulls == drop:
                field = i
        if field < 0:
            if len(fields) == 4:
                return None
            field = len(fields)
            fields.append(OrderedField(value.text.copy(), dtype, drop, 0))
        fields[field].operations |= UInt32(1) << UInt32(
            ordered_side(aggregate.op)
        )
        if value.text not in inputs:
            inputs.append(value.text.copy())
        outputs.append(
            OrderedOutput(expression._name.copy(), field, aggregate.op)
        )
    if not outputs:
        return None
    from .predicate_program import PredicateClause

    var clauses = List[PredicateClause]()
    var code = List[Int]()
    for i in range(1, count):
        ref node = region._plan._nodes[i]
        if node.left != i - 1:
            return None
        if i == count - 1:
            continue
        if node.kind != FILTER or len(node.exprs) != 1:
            return None
        ref expression = node.exprs[0]
        if not expression._nodes:
            return None
        if not _emit(
            expression, len(expression._nodes) - 1, frame, clauses, code
        ):
            return None
        if i > 1:
            code.append(-1)
    var truth = List[UInt32](length=8, fill=0)
    if code:
        truth = _truth(code, len(clauses))
    else:
        truth[0] = 1
    var predicate = PredicateProgram(clauses^, truth^, List[Int]())
    if not bind_predicate_inputs(predicate, inputs):
        return None
    return OrderedPlan(fields^, outputs^, inputs^, predicate^, keys^)


def ordered_key(plan: OrderedPlan) -> String:
    var result = String("ordered_reductions:v1:")
    for field in plan.fields:
        var input = -1
        for i in range(len(plan.inputs)):
            if plan.inputs[i] == field.column:
                input = i
        result += (
            ":input"
            + String(input)
            + ":"
            + field.dtype.name()
            + ":drop"
            + String(Int(field.drop_nulls))
            + ":ops"
            + String(field.operations)
        )
    if plan.keys:
        for dtype in plan.keys.value().dtypes:
            result += ":key:" + dtype.name()
    for output in plan.outputs:
        result += ":out" + String(output.field) + ":" + String(output.op)
    ref program = plan.predicate
    for i in range(len(program.clauses)):
        ref leaf = program.clauses[i]
        ref bounds = leaf.bounds
        result += (
            ":p"
            + String(program.slots[i])
            + ":"
            + String(leaf.kind)
            + ":"
            + String(leaf.comparison)
            + ":"
            + String(Int(leaf.inverted))
            + ":"
            + String(leaf.other_input)
        )
        result += (
            ":"
            + bounds.filter_dtype.name()
            + ":"
            + String(bounds.low)
            + ":"
            + String(bounds.high)
            + ":"
            + String(bitcast[DType.uint32](bounds.low_float))
            + ":"
            + String(bitcast[DType.uint32](bounds.high_float))
            + ":"
            + String(bounds.inclusive)
        )
        for member in leaf.members:
            result += ":m" + String(member)
    for word in program.truth:
        result += ":t" + String(word)
    return result^
