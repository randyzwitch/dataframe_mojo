"""Schema-only matching for arithmetic and column aggregates in one region."""
from std.memory import bitcast
from dataframe import DataFrame, DataType, Expr, LazyFrame, col
from dataframe.expr import COL, SUM, COUNT, MEAN
from dataframe.lazy import FILTER, SCAN_FRAME, SELECT
from .filter_plan import FilterSum, plan_region
from .arithmetic_plan import plan_arithmetic
from .multi_plan import MAX_FIELDS, ReductionOutput

comptime MAX_VALUE_INPUTS = 8


@fieldwise_init
struct ValueField(Copyable, Movable):
    var left: String
    var right: String
    var dtype: DataType
    var op: Int
    var literal_side: Int
    var integer: Int64
    var floating: Float32
    var needs_sum: Bool
    var needs_mean: Bool
    var left_input: Int
    var right_input: Int

    def same_value(self, other: Self) -> Bool:
        return (
            self.left == other.left
            and self.right == other.right
            and self.dtype == other.dtype
            and self.op == other.op
            and self.literal_side == other.literal_side
            and self.integer == other.integer
            and bitcast[DType.uint32](self.floating)
            == bitcast[DType.uint32](other.floating)
        )


@fieldwise_init
struct Composition(Copyable, Movable):
    var predicate: FilterSum
    var fields: List[ValueField]
    var outputs: List[ReductionOutput]
    var inputs: List[String]
    var precise_float: Bool


def _input_slot(mut names: List[String], name: String) -> Int:
    for i in range(len(names)):
        if names[i] == name:
            return i
    var slot = len(names)
    names.append(name.copy())
    return slot


def plan_composition(region: LazyFrame) raises -> Optional[Composition]:
    var n = len(region._plan._nodes)
    if n < 2 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    ref root = region._plan._nodes[n - 1]
    if root.kind != SELECT or len(root.exprs) == 0:
        return None
    # Preserve the existing standalone binary-SUM fast path and its costs.
    if (
        len(root.exprs) == 1
        and len(root.exprs[0]._nodes) == 4
        and root.exprs[0]._nodes[3].op == SUM
    ):
        return None
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    var offered = frame.lazy()
    for i in range(1, n - 1):
        ref node = region._plan._nodes[i]
        if node.kind != FILTER or len(node.exprs) != 1:
            return None
        offered = offered.filter(node.exprs[0])
    var fields = List[ValueField]()
    var outputs = List[ReductionOutput]()
    var output_names = List[String]()
    var predicate = Optional[FilterSum]()
    var arithmetic = False
    var precise = False
    for expression in root.exprs:
        var size = len(expression._nodes)
        if size != 2 and size != 4:
            return None
        ref aggregate = expression._nodes[size - 1]
        if (
            aggregate.op != SUM
            and aggregate.op != COUNT
            and aggregate.op != MEAN
        ):
            return None
        if expression._name in output_names:
            return None
        output_names.append(expression._name.copy())
        var value: ValueField
        var total = expression.copy()
        total._nodes[size - 1].op = SUM
        if size == 2:
            if expression._nodes[0].op != COL or aggregate.left != 0:
                return None
            var matched = plan_region(offered.select(total))
            if not matched:
                return None
            var p = matched.take()
            value = ValueField(
                p.column.copy(),
                p.column.copy(),
                p.dtype,
                COL,
                0,
                0,
                0,
                False,
                False,
                -1,
                -1,
            )
            if not predicate:
                predicate = p^
        else:
            var matched = plan_arithmetic(offered.select(total))
            if not matched:
                return None
            var p = matched.take()
            value = ValueField(
                p.left.copy(),
                p.right.copy(),
                p.predicate.dtype,
                p.op,
                p.literal_side,
                p.integer,
                p.floating,
                False,
                False,
                -1,
                -1,
            )
            if not predicate:
                predicate = p.predicate.copy()
            arithmetic = True
        var slot = -1
        for i in range(len(fields)):
            if fields[i].same_value(value):
                slot = i
        if slot < 0:
            if len(fields) == MAX_FIELDS:
                return None
            slot = len(fields)
            fields.append(value^)
        if aggregate.op != COUNT:
            fields[slot].needs_sum = True
        if aggregate.op == MEAN:
            fields[slot].needs_mean = True
            if fields[slot].dtype == DataType.FLOAT32:
                precise = True
        outputs.append(
            ReductionOutput(
                slot, expression._name.copy(), aggregate.op, aggregate.min_count
            )
        )
    if not arithmetic:
        return None
    var single_float = True
    var binary_slot = -1
    var binary_count = 0
    for i in range(len(fields)):
        single_float = single_float and fields[i].dtype == DataType.FLOAT32
        if fields[i].op != COL:
            binary_slot = i
            binary_count += 1
    single_float = single_float and binary_count == 1
    if precise or single_float:
        var order = List[Int]()
        if single_float:
            order.append(binary_slot)
            for mean in [True, False]:
                for i in range(len(fields)):
                    if i != binary_slot and fields[i].needs_mean == mean:
                        order.append(i)
        else:
            for floating in [True, False]:
                for i in range(len(fields)):
                    if (fields[i].dtype == DataType.FLOAT32) == floating:
                        order.append(i)
        var packed = List[ValueField]()
        var remap = List[Int](length=len(fields), fill=0)
        for i in range(len(order)):
            packed.append(fields[order[i]].copy())
            remap[order[i]] = i
        for i in range(len(outputs)):
            outputs[i].field = remap[outputs[i].field]
        _ = fields^
        fields = packed^
    var inputs = List[String]()
    for i in range(len(fields)):
        if fields[i].literal_side != 1:
            fields[i].left_input = _input_slot(inputs, fields[i].left)
        if fields[i].op != COL and fields[i].literal_side != 2:
            fields[i].right_input = _input_slot(inputs, fields[i].right)
    if len(inputs) > MAX_VALUE_INPUTS:
        return None
    return Composition(predicate.take(), fields^, outputs^, inputs^, precise)


def composition_key(plan: Composition) -> String:
    # Names do not affect costs; canonical source slots describe reuse.
    var key = (
        String("composed_reductions:v1:") + plan.predicate.filter_dtype.name()
    )
    key += ":filtered" if plan.predicate.filtered else ":plain"
    var shared = -1
    for i in range(len(plan.inputs)):
        if plan.inputs[i] == plan.predicate.filter_column:
            shared = i
    key += ":predicate" + String(shared)
    for field in plan.fields:
        key += (
            ":"
            + field.dtype.name()
            + ":"
            + String(field.op)
            + ":"
            + String(field.left_input)
            + ":"
            + String(field.right_input)
            + ":literal"
            + String(field.literal_side)
            + (":sum" if field.needs_sum else ":count")
            + (":mean" if field.needs_mean else "")
        )
    key += ":exact_float" if plan.precise_float else ":native"
    return key + (":single_float_v1" if single_float_composition(plan) else "")


def single_float_composition(plan: Composition) -> Bool:
    var arithmetic = 0
    for field in plan.fields:
        if field.dtype != DataType.FLOAT32:
            return False
        arithmetic += Int(field.op != COL)
    return arithmetic == 1
