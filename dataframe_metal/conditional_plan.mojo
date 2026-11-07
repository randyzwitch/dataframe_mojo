"""Lower native CASE values to bounded forward-only branch programs.

Conditions use the same numeric predicate compiler as WHERE. Only the taken
branch executes: a checked error in an untaken branch stays unobservable.
This module does not enable a region until its device lowering is installed.
"""
from std.memory import bitcast
from dataframe import DataFrame, DataType, Expr
from dataframe.binding import bind, BoundExpr
from dataframe.expr import (
    COL,
    ADD,
    SUB,
    MUL,
    WHEN,
    LIT_INT,
    LIT_FLOAT,
    LIT_NULL,
)
from .expression_plan import ValueNode, _integer_fits_float32
from .predicate_program import PredicateProgram, compile_predicate
from .filter_plan import supported_dtype

from .expression_format import (
    CASE_HEADER,
    CASE_TEST,
    CASE_JUMP,
    CASE_NULL,
    MAX_CASE_INSTRUCTIONS,
    MAX_CASE_SOURCE_NODES,
    MAX_CASE_CONDITIONS,
    MAX_CASE_STACK,
)


@fieldwise_init
struct ConditionalValue(Copyable, Movable):
    var dtype: DataType
    var seed: String
    var nodes: List[ValueNode]
    var conditions: List[PredicateProgram]


def _mark_condition(expression: Expr, at: Int, mut visited: List[Bool]) -> Bool:
    var pending = List[Int]([at])
    while pending:
        var index = pending.pop()
        if index < 0 or index >= len(visited) or visited[index]:
            return False
        visited[index] = True
        ref node = expression._nodes[index]
        for child in [node.left, node.right, node.extra]:
            if child >= 0:
                if child >= index:
                    return False
                pending.append(child)
    return True


def _emit_value(
    expression: Expr,
    typed: Expr,
    types: List[DataType],
    frame: DataFrame,
    at: Int,
    dtype: DataType,
    depth: Int,
    mut visited: List[Bool],
    mut nodes: List[ValueNode],
    mut conditions: List[PredicateProgram],
) raises -> Bool:
    if at < 0 or at >= len(visited) or visited[at] or depth >= MAX_CASE_STACK:
        return False
    visited[at] = True
    ref node = expression._nodes[at]
    if node.op != LIT_NULL and types[at] != dtype:
        return False
    if node.op == WHEN:
        if (
            node.left < 0
            or node.left >= at
            or node.right < 0
            or node.right >= at
            or node.extra >= at
            or len(conditions) == MAX_CASE_CONDITIONS
            or not _mark_condition(expression, node.left, visited)
        ):
            return False
        var predicate = compile_predicate(expression, frame, root=node.left)
        if not predicate:
            return False
        var slot = len(conditions)
        conditions.append(predicate.take())
        var test_at = len(nodes)
        nodes.append(ValueNode(CASE_TEST, "", slot, 0))
        if not _emit_value(
            expression,
            typed,
            types,
            frame,
            node.right,
            dtype,
            depth,
            visited,
            nodes,
            conditions,
        ):
            return False
        var jump_at = len(nodes)
        nodes.append(ValueNode(CASE_JUMP, "", -1, 0))
        nodes[test_at].literal = UInt64(len(nodes))
        if node.extra < 0:
            nodes.append(ValueNode(CASE_NULL, "", -1, 0))
        elif not _emit_value(
            expression,
            typed,
            types,
            frame,
            node.extra,
            dtype,
            depth,
            visited,
            nodes,
            conditions,
        ):
            return False
        nodes[jump_at].literal = UInt64(len(nodes))
    elif node.op == ADD or node.op == SUB or node.op == MUL:
        if (
            node.left < 0
            or node.left >= at
            or node.right < 0
            or node.right >= at
            or not _emit_value(
                expression,
                typed,
                types,
                frame,
                node.left,
                dtype,
                depth,
                visited,
                nodes,
                conditions,
            )
            or not _emit_value(
                expression,
                typed,
                types,
                frame,
                node.right,
                dtype,
                depth + 1,
                visited,
                nodes,
                conditions,
            )
        ):
            return False
        nodes.append(ValueNode(node.op, "", -1, 0))
    elif node.op == COL:
        if (
            node.text not in frame.columns()
            or frame.column(node.text).dtype() != dtype
        ):
            return False
        nodes.append(ValueNode(COL, node.text.copy(), -1, 0))
    elif node.op == LIT_NULL:
        nodes.append(ValueNode(CASE_NULL, "", -1, 0))
    elif node.op == LIT_INT or node.op == LIT_FLOAT:
        ref literal = typed._nodes[at]
        var bits = UInt64(0)
        if dtype == DataType.FLOAT32:
            if node.op == LIT_INT and not _integer_fits_float32(
                literal.integer
            ):
                return False
            var value = (
                Float64(literal.integer) if node.op
                == LIT_INT else literal.floating
            )
            var narrow = Float32(value)
            if Float64(narrow) != value:
                return False
            bits = UInt64(bitcast[DType.uint32](narrow))
        else:
            if node.op != LIT_INT:
                return False
            if dtype == DataType.INT32 and (
                literal.integer < Int64(Int32.MIN)
                or literal.integer > Int64(Int32.MAX)
            ):
                return False
            bits = bitcast[DType.uint64](literal.integer)
        nodes.append(ValueNode(LIT_INT, "", -1, bits))
    else:
        return False
    return len(nodes) <= MAX_CASE_INSTRUCTIONS


def compile_conditional(
    expression: Expr, frame: DataFrame, root: Int = -1
) raises -> Optional[ConditionalValue]:
    """Preserve native branch types and CPU binding without reading row values."""
    var at = len(expression._nodes) - 1 if root < 0 else root
    if at < 0 or at >= len(expression._nodes) or at >= MAX_CASE_SOURCE_NODES:
        return None
    var present = False
    for i in range(at + 1):
        present = present or expression._nodes[i].op == WHEN
    if not present:
        return None
    var bound: BoundExpr
    try:
        bound = bind(expression, frame._columns, fuse=False)
    except:
        return None
    var dtype = bound.dtypes[at]
    if not supported_dtype(dtype):
        return None
    var nodes = List[ValueNode]([ValueNode(CASE_HEADER, "", -1, 0)])
    var conditions = List[PredicateProgram]()
    var visited = List[Bool](length=at + 1, fill=False)
    if (
        not _emit_value(
            expression,
            bound.expr,
            bound.dtypes,
            frame,
            at,
            dtype,
            0,
            visited,
            nodes,
            conditions,
        )
        or not conditions
    ):
        return None
    var seed = String()
    for i in range(at + 1):
        if not visited[i]:
            return None
        if expression._nodes[i].op == COL and not seed:
            seed = expression._nodes[i].text.copy()
    if not seed:
        return None
    return ConditionalValue(dtype, seed^, nodes^, conditions^)
