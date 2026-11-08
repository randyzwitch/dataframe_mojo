"""Schema-only exact N_UNIQUE matching over native values and tuple keys."""
from dataframe import DataType, LazyFrame
from dataframe.binding import bind
from dataframe.expr import COL, N_UNIQUE, DROP_NULLS
from dataframe.lazy import SCAN_FRAME, FILTER, SELECT, AGG
from .ordered_plan import ordered_tag
from .group_keys import GroupKeys, compile_group_keys
from .predicate_program import (
    PredicateProgram,
    PredicateClause,
    _emit,
    _truth,
    bind_predicate_inputs,
)


@fieldwise_init
struct DistinctField(Copyable, Movable):
    var column: String
    var dtype: DataType


@fieldwise_init
struct DistinctOutput(Copyable, Movable):
    var name: String
    var field: Int
    var ignore_nulls: Bool


@fieldwise_init
struct DistinctPlan(Copyable, Movable):
    var fields: List[DistinctField]
    var outputs: List[DistinctOutput]
    var inputs: List[String]
    var predicate: PredicateProgram
    var keys: Optional[GroupKeys]


def plan_distinct(region: LazyFrame) raises -> Optional[DistinctPlan]:
    var size = len(region._plan._nodes)
    if size < 2 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    ref root = region._plan._nodes[size - 1]
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
    var fields = List[DistinctField]()
    var outputs = List[DistinctOutput]()
    var inputs = List[String]()
    var names = List[String]()
    for expression in root.exprs:
        var n = len(expression._nodes)
        if n != 2 and n != 3:
            return None
        ref value = expression._nodes[0]
        ref aggregate = expression._nodes[n - 1]
        if (
            value.op != COL
            or value.text not in frame.columns()
            or aggregate.op != N_UNIQUE
            or aggregate.left != n - 2
            or aggregate.right >= 0
            or aggregate.extra >= 0
        ):
            return None
        if expression._name in names or (
            keys and expression._name in keys.value().names
        ):
            return None
        names.append(expression._name.copy())
        var drop = n == 3
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
            if fields[i].column == value.text:
                field = i
        if field < 0:
            if len(fields) == 4:
                return None
            field = len(fields)
            fields.append(DistinctField(value.text.copy(), dtype))
            inputs.append(value.text.copy())
        outputs.append(
            DistinctOutput(
                expression._name.copy(), field, drop or Bool(aggregate.integer)
            )
        )
    if not outputs:
        return None
    var clauses = List[PredicateClause]()
    var code = List[Int]()
    for i in range(1, size):
        ref node = region._plan._nodes[i]
        if node.left != i - 1:
            return None
        if i == size - 1:
            continue
        if node.kind != FILTER or len(node.exprs) != 1:
            return None
        ref expression = node.exprs[0]
        if not expression._nodes or not _emit(
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
    return DistinctPlan(fields^, outputs^, inputs^, predicate^, keys^)
