"""Schema-only matching for arithmetic and column aggregates in one region."""
from std.memory import bitcast
from dataframe import DataFrame, DataType, Expr, LazyFrame, col
from dataframe.expr import (
    COL,
    SUM,
    COUNT,
    MEAN,
    MIN,
    MAX,
    ADD,
    SUB,
    MUL,
    LIT_INT,
)
from dataframe.lazy import FILTER, SCAN_FRAME, SELECT
from .filter_plan import FilterSum, plan_region
from .arithmetic_plan import plan_arithmetic
from .multi_plan import MAX_FIELDS, ReductionOutput
from .expression_plan import ValueNode, compile_expression
from .expression_format import (
    TREE_OP,
    TYPED_HEADER,
    CASE_HEADER,
    CASE_TEST,
    CASE_JUMP,
    CASE_NULL,
)
from .conditional_plan import compile_conditional
from .typed_expression_plan import compile_typed_expression
from .predicate_program import (
    PredicateProgram,
    bind_predicate_inputs,
    MAX_INPUTS,
)

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
    var tree: List[ValueNode]
    var conditions: List[PredicateProgram]

    def same_value(self, other: Self) -> Bool:
        if len(self.conditions) != len(other.conditions):
            return False
        for i in range(len(self.conditions)):
            if condition_key(self.conditions[i]) != condition_key(
                other.conditions[i]
            ):
                return False
        if len(self.tree) != len(other.tree):
            return False
        for i in range(len(self.tree)):
            if not self.tree[i].same_value(other.tree[i]):
                return False
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


def plan_composition(
    region: LazyFrame, include_native: Bool = False
) raises -> Optional[Composition]:
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
        and root.exprs[0]._nodes[2].op in [ADD, SUB, MUL]
        and not include_native
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
    var extrema = False
    var ordinary = False
    var precise = False
    for expression in root.exprs:
        var size = len(expression._nodes)
        if size < 2:
            return None
        ref aggregate = expression._nodes[size - 1]
        if (
            aggregate.op != SUM
            and aggregate.op != COUNT
            and aggregate.op != MEAN
            and aggregate.op != MIN
            and aggregate.op != MAX
        ):
            return None
        if aggregate.op == MIN or aggregate.op == MAX:
            extrema = True
        else:
            ordinary = True
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
                List[ValueNode](),
                List[PredicateProgram](),
            )
            if not predicate:
                predicate = p^
        elif size == 4 and expression._nodes[2].op in [ADD, SUB, MUL]:
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
                List[ValueNode](),
                List[PredicateProgram](),
            )
            if not predicate:
                predicate = p.predicate.copy()
            arithmetic = True
        else:
            if aggregate.left != size - 2:
                return None
            var conditional = compile_conditional(expression, frame, size - 2)
            var conditions = List[PredicateProgram]()
            var tree_nodes = List[ValueNode]()
            var seed = String()
            var dtype = DataType.INT32
            if conditional:
                var conditional_value = conditional.take()
                seed = conditional_value.seed.copy()
                dtype = conditional_value.dtype
                tree_nodes = conditional_value.nodes.copy()
                conditions = conditional_value.conditions.copy()
            else:
                var compiled = compile_expression(expression, frame, size - 2)
                if not compiled:
                    compiled = compile_typed_expression(
                        expression, frame, size - 2
                    )
                if not compiled:
                    return None
                var tree = compiled.take()
                seed = tree.seed.copy()
                dtype = tree.dtype
                tree_nodes = tree.nodes.copy()
            var matched = plan_region(offered.select(col(seed).sum()))
            if (
                not matched
                and n == 2
                and frame.column(seed).dtype() == DataType.BOOL
            ):
                # Bool columns only supply CASE conditions here. Numeric
                # constants form the value; no Bool payload is read as Int64.
                var condition_seed = FilterSum(seed.copy(), seed.copy(), dtype)
                condition_seed.filter_dtype = DataType.BOOL
                matched = condition_seed^
            if not matched:
                return None
            var p = matched.take()
            value = ValueField(
                seed.copy(),
                seed.copy(),
                dtype,
                TREE_OP,
                0,
                0,
                0,
                False,
                False,
                -1,
                -1,
                tree_nodes^,
                conditions^,
            )
            if not predicate:
                predicate = p^
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
        if aggregate.op == SUM or aggregate.op == MEAN:
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
    if not arithmetic and not (extrema and ordinary) and not include_native:
        return None
    var single_float = True
    var binary_slot = -1
    var binary_count = 0
    for i in range(len(fields)):
        single_float = (
            single_float
            and fields[i].dtype == DataType.FLOAT32
            and not fields[i].tree
        )
        if fields[i].op != COL:
            binary_slot = i
            binary_count += 1
    single_float = single_float and binary_count == 1 and not extrema
    var order = List[Int]()
    if single_float:
        order.append(binary_slot)
        for mean in [True, False]:
            for i in range(len(fields)):
                if i != binary_slot and fields[i].needs_mean == mean:
                    order.append(i)
    else:
        # Group physical types so the kernel can resolve every value
        # read and accumulator type at compile time.
        for dtype in [DataType.FLOAT32, DataType.INT32, DataType.INT64]:
            for i in range(len(fields)):
                if fields[i].dtype == dtype:
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
        if fields[i].tree:
            for j in range(len(fields[i].tree)):
                if fields[i].tree[j].op == COL:
                    fields[i].tree[j].input = _input_slot(
                        inputs, fields[i].tree[j].column
                    )
            continue
        if fields[i].literal_side != 1:
            fields[i].left_input = _input_slot(inputs, fields[i].left)
        if fields[i].op != COL and fields[i].literal_side != 2:
            fields[i].right_input = _input_slot(inputs, fields[i].right)
    if len(inputs) > MAX_VALUE_INPUTS:
        return None
    for i in range(len(fields)):
        for j in range(len(fields[i].conditions)):
            if not bind_predicate_inputs(
                fields[i].conditions[j], inputs, limit=MAX_INPUTS
            ):
                return None
    var physical_predicate = predicate.take()
    if len(inputs) == MAX_INPUTS:
        # The ninth physical binding is shared by all predicates. A distinct
        # WHERE column is handled by plan_predicates' compound route, whose
        # truth program can refer to any of the nine canonical inputs.
        if (
            physical_predicate.filtered
            and physical_predicate.filter_column != inputs[8]
        ):
            return None
        physical_predicate.filter_column = inputs[8].copy()
        physical_predicate.filter_dtype = frame.column(inputs[8]).dtype()
    return Composition(physical_predicate^, fields^, outputs^, inputs^, precise)


def composition_key(plan: Composition) -> String:
    # Names do not affect costs; canonical source slots describe reuse.
    var key = (
        String(
            "typed_reductions:v2:scalar_chain:" if has_typed_values(
                plan
            ) else "conditional_reductions:v1:" if has_conditions(
                plan
            ) else "expression_reductions:v1:" if has_trees(
                plan
            ) else "mixed_reductions:v1:" if has_extrema(
                plan
            ) else (
                "composed_reductions:v1:" if single_float_composition(
                    plan
                ) else "composed_reductions:v2:typed_fields:"
            )
        )
        + plan.predicate.filter_dtype.name()
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
    for field in plan.fields:
        if field.tree:
            key += ":tree"
            for node in field.tree:
                key += (
                    ":"
                    + String(node.op)
                    + ","
                    + String(node.input)
                    + ","
                    + String(node.literal)
                    + (",type" + String(node.tag) if node.tag >= 0 else "")
                )
    if has_extrema(plan):
        for i in range(len(plan.fields)):
            key += ":ext" + String(i)
            for op in [MIN, MAX]:
                for output in plan.outputs:
                    if output.field == i and output.op == op:
                        key += ":min" if op == MIN else ":max"
                        break
    key += ":exact_float" if plan.precise_float else ":native"
    for field in plan.fields:
        for condition in field.conditions:
            key += ":" + condition_key(condition, names=False)
    return key + (":single_float_v1" if single_float_composition(plan) else "")


def has_extrema(plan: Composition) -> Bool:
    for output in plan.outputs:
        if output.op == MIN or output.op == MAX:
            return True
    return False


def has_trees(plan: Composition) -> Bool:
    for field in plan.fields:
        if field.tree:
            return True
    return False


def single_float_composition(plan: Composition) -> Bool:
    if has_extrema(plan) or has_trees(plan):
        return False
    var arithmetic = 0
    for field in plan.fields:
        if field.dtype != DataType.FLOAT32:
            return False
        arithmetic += Int(field.op != COL)
    return arithmetic == 1


def condition_key(program: PredicateProgram, names: Bool = True) -> String:
    var key = String("case_predicate:v1")
    for i in range(len(program.clauses)):
        ref leaf = program.clauses[i]
        ref p = leaf.bounds
        key += ":" + (p.filter_column if names else String(program.slots[i]))
        key += ":" + (leaf.other if names else String(leaf.other_input))
        key += (
            ":"
            + p.filter_dtype.name()
            + ":"
            + String(leaf.kind)
            + ":"
            + String(Int(leaf.inverted))
            + ":"
            + String(leaf.comparison)
        )
        key += (
            ":"
            + String(p.low)
            + ":"
            + String(p.high)
            + ":"
            + String(p.inclusive)
        )
        key += (
            ":"
            + String(bitcast[DType.uint32](p.low_float))
            + ":"
            + String(bitcast[DType.uint32](p.high_float))
        )
    for leaf in program.clauses:
        if leaf.kind == 3:
            key += ":members"
            for member in leaf.members:
                key += "," + String(member)
    for word in program.truth:
        key += ":" + String(word)
    return key


def has_conditions(plan: Composition) -> Bool:
    for field in plan.fields:
        if field.conditions:
            return True
    return False


def only_conditional_trees(plan: Composition) -> Bool:
    """Native and binary fields bypass the tree evaluator entirely."""
    for field in plan.fields:
        if field.op == TREE_OP and not field.conditions:
            return False
    return has_conditions(plan)


def simple_case_layout(nodes: List[ValueNode]) -> Int:
    """Classify only CASE layouts implemented by the register evaluator."""
    if not nodes or nodes[0].op != CASE_HEADER:
        return -1
    for shape in range(4):
        var then_binary = Bool(shape & 1)
        var else_binary = Bool(shape & 2)
        var otherwise = 6 if then_binary else 4
        var jump = otherwise - 1
        var size = otherwise + (3 if else_binary else 1)
        if len(nodes) != size:
            continue
        if (
            nodes[1].op != CASE_TEST
            or nodes[1].literal != UInt64(otherwise)
            or nodes[jump].op != CASE_JUMP
            or nodes[jump].literal != UInt64(size)
        ):
            continue
        var eligible = True
        for at in [2, otherwise]:
            eligible = eligible and nodes[at].op in [COL, LIT_INT, CASE_NULL]
        if then_binary:
            eligible = eligible and nodes[3].op in [COL, LIT_INT, CASE_NULL]
            eligible = eligible and nodes[4].op in [ADD, SUB, MUL]
        if else_binary:
            eligible = eligible and nodes[otherwise + 1].op in [
                COL,
                LIT_INT,
                CASE_NULL,
            ]
            eligible = eligible and nodes[otherwise + 2].op in [ADD, SUB, MUL]
        if eligible:
            return shape
    return -1


def simple_conditional_trees(plan: Composition) -> Bool:
    """Prove the CASE stack evaluator is unnecessary for this projection."""
    for field in plan.fields:
        if field.op == TREE_OP and (
            not field.conditions or simple_case_layout(field.tree) < 0
        ):
            return False
    return has_conditions(plan)


def has_typed_values(plan: Composition) -> Bool:
    for field in plan.fields:
        if field.tree and field.tree[0].op == TYPED_HEADER:
            return True
    return False


def needs_general_values(plan: Composition) -> Bool:
    """Use the general value path for branches or per-node native types."""
    return has_conditions(plan) or has_typed_values(plan)
