"""Bind native casts without changing CPU promotion or checked intermediates."""
from std.memory import bitcast
from dataframe import DataFrame, DataType, Expr
from dataframe.binding import bind, BoundExpr
from dataframe.expr import (
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_NULL,
    CAST,
    ADD,
    SUB,
    MUL,
    DIV,
    FLOORDIV,
    MOD,
    NEG,
    ABS,
    SIGN,
    FLOOR,
    CEIL,
    TRUNCATE,
    ROUND,
    CLIP_LOW,
    CLIP_HIGH,
    FILL_NULL,
    FILL_NAN,
)
from .expression_plan import ValueNode, ExpressionTree
from .expression_format import (
    TYPED_HEADER,
    TYPED_CHAIN,
    TYPED_REGISTERS,
    MAX_EXPRESSION_NODES,
    MAX_EXPRESSION_STACK,
    CASE_NULL,
)


def numeric_tag(dtype: DataType) -> Int:
    return (
        0 if dtype
        == DataType.INT32 else 1 if dtype
        == DataType.INT64 else 2 if dtype
        == DataType.FLOAT32 else -1
    )


def compile_typed_expression(
    expression: Expr, frame: DataFrame, root: Int = -1
) raises -> Optional[ExpressionTree]:
    var last = len(expression._nodes) - 1 if root < 0 else root
    if (
        last < 0
        or last >= len(expression._nodes)
        or last >= MAX_EXPRESSION_NODES
    ):
        return None
    var has_native_program = False
    for i in range(last + 1):
        has_native_program = has_native_program or expression._nodes[i].op in [
            CAST,
            DIV,
            FLOORDIV,
            MOD,
            NEG,
            ABS,
            SIGN,
            FLOOR,
            CEIL,
            TRUNCATE,
            ROUND,
            CLIP_LOW,
            CLIP_HIGH,
            FILL_NULL,
            FILL_NAN,
        ]
    if not has_native_program:
        return None
    var bound: BoundExpr
    try:
        bound = bind(expression, frame._columns, fuse=False)
    except:
        return None
    var nodes = List[ValueNode]([ValueNode(TYPED_HEADER, "", -1, 0)])
    var stack = List[Int]()
    var seed = String()
    var register_slots = UInt64(0)
    var register_count = 0
    for i in range(last + 1):
        ref original = expression._nodes[i]
        ref node = bound.expr._nodes[i]
        var tag = numeric_tag(bound.dtypes[i])
        if node.op == LIT_NULL:
            # Invalid payloads have no observed width; the consuming typed
            # operation owns the value dtype after binding.
            tag = 0
        if tag < 0 or node.extra >= 0:
            return None
        if node.op == COL:
            if (
                node.left >= 0
                or node.right >= 0
                or node.text not in frame.columns()
            ):
                return None
            if not seed:
                seed = node.text.copy()
            nodes.append(ValueNode(COL, node.text.copy(), -1, 0, tag))
        elif node.op == LIT_NULL:
            if node.left >= 0 or node.right >= 0:
                return None
            nodes.append(ValueNode(CASE_NULL, "", -1, 0, tag))
        elif node.op == LIT_INT or node.op == LIT_FLOAT:
            if node.left >= 0 or node.right >= 0:
                return None
            var bits = UInt64(0)
            if tag == 2:
                var value = Float32(
                    node.floating
                ) if node.op == LIT_FLOAT else Float32(node.integer)
                bits = UInt64(bitcast[DType.uint32](value))
            elif node.op == LIT_INT:
                bits = bitcast[DType.uint64](node.integer)
            else:
                return None
            nodes.append(ValueNode(LIT_INT, "", -1, bits, tag))
        elif node.op == CAST:
            if not stack or node.right >= 0 or original.left != stack.pop():
                return None
            var source = numeric_tag(bound.dtypes[node.left])
            if source < 0 or (node.integer != 0 and node.integer != 1):
                return None
            if source == 2 and tag != 2:
                nodes[0].literal = 1
            nodes.append(
                ValueNode(
                    CAST,
                    "",
                    -1,
                    (UInt64(source) << 8) | UInt64(node.integer),
                    tag,
                )
            )
        elif node.op in [NEG, ABS, SIGN, FLOOR, CEIL, TRUNCATE, ROUND]:
            if not stack or node.right >= 0 or original.left != stack.pop():
                return None
            # Decimal scaling currently requires CPU Float64 arithmetic.
            # Integer round retains the existing identity semantics.
            if tag == 2 and node.op == ROUND and node.integer != 0:
                return None
            nodes.append(
                ValueNode(
                    node.op, "", -1, bitcast[DType.uint64](node.integer), tag
                )
            )
        elif node.op in [
            ADD,
            SUB,
            MUL,
            DIV,
            FLOORDIV,
            MOD,
            CLIP_LOW,
            CLIP_HIGH,
            FILL_NULL,
            FILL_NAN,
        ]:
            if len(stack) < 2:
                return None
            var right = stack.pop()
            var left = stack.pop()
            if original.left != left or original.right != right:
                return None
            if node.op == FILL_NAN:
                if tag != 2:
                    return None
                # COUNT observes replacement validity, so NaN detection must
                # read Float32 payloads even without numeric output work.
                nodes[0].literal = 1
            nodes.append(ValueNode(node.op, "", -1, 0, tag))
        else:
            return None
        register_slots |= UInt64(len(stack)) << UInt64(i * 3)
        register_count = max(register_count, len(stack) + 1)
        stack.append(i)
        if len(stack) > MAX_EXPRESSION_STACK:
            return None
    if len(stack) != 1 or stack[0] != last or not seed:
        return None
    var chain = scalar_chain_schedule(nodes)
    if chain:
        nodes[0].literal |= chain
    else:
        nodes[0].literal |= UInt64(TYPED_REGISTERS) | (
            UInt64(register_count) << 8
        )
        nodes[0].input = Int(register_slots)
    return ExpressionTree(bound.dtypes[last], nodes^, seed^)


def scalar_chain_schedule(nodes: List[ValueNode]) -> UInt64:
    """A left-to-right accumulator, preserving every eager checked operation."""
    if len(nodes) < 2 or nodes[0].op != TYPED_HEADER:
        return UInt64(0)
    if nodes[1].op not in [COL, LIT_INT, CASE_NULL]:
        return UInt64(0)
    var at = 2
    var steps = 0
    var unary = UInt64(0)
    while at < len(nodes):
        var op = nodes[at].op
        if op in [CAST, NEG, ABS, SIGN, FLOOR, CEIL, TRUNCATE, ROUND]:
            unary |= UInt64(1) << UInt64(steps)
            at += 1
        elif op in [COL, LIT_INT, CASE_NULL]:
            if at + 1 >= len(nodes) or nodes[at + 1].op not in [
                ADD,
                SUB,
                MUL,
                DIV,
                FLOORDIV,
                MOD,
                CLIP_LOW,
                CLIP_HIGH,
                FILL_NULL,
                FILL_NAN,
            ]:
                return UInt64(0)
            at += 2
        else:
            return UInt64(0)
        steps += 1
    return UInt64(TYPED_CHAIN) | (UInt64(steps) << 8) | (unary << 16)
