"""Compile checked native arithmetic without evaluating input values."""
from std.memory import bitcast
from dataframe import DataFrame, DataType, Expr
from dataframe.expr import COL, LIT_INT, LIT_FLOAT, ADD, SUB, MUL, UNTYPED
from .filter_plan import supported_dtype
from .expression_format import MAX_EXPRESSION_NODES, MAX_EXPRESSION_STACK


struct ValueNode(Copyable, Movable):
    var op: Int
    var column: String
    var input: Int
    var literal: UInt64
    var tag: Int

    def __init__(
        out self,
        op: Int,
        column: String,
        input: Int,
        literal: UInt64,
        tag: Int = -1,
    ):
        self.op = op
        self.column = column.copy()
        self.input = input
        self.literal = literal
        self.tag = tag

    def same_value(self, other: Self) -> Bool:
        return (
            self.op == other.op
            and self.column == other.column
            and self.literal == other.literal
            and self.tag == other.tag
        )


@fieldwise_init
struct ExpressionTree(Copyable, Movable):
    var dtype: DataType
    var nodes: List[ValueNode]
    var seed: String


def _integer_fits_float32(value: Int64) -> Bool:
    # Testing through Float64 would falsely accept Int64.MAX: that first
    # conversion already rounds it to 2**63. Count significant bits instead.
    var magnitude = UInt64(value)
    if value < 0:
        magnitude = UInt64(0) - magnitude
    if magnitude == 0:
        return True
    while (magnitude & 1) == 0:
        magnitude >>= 1
    return magnitude < (UInt64(1) << 24)


def compile_expression(
    expression: Expr, frame: DataFrame, root: Int = -1
) raises -> Optional[ExpressionTree]:
    var last = len(expression._nodes) - 1 if root < 0 else root
    if (
        last < 0
        or last >= len(expression._nodes)
        or last >= MAX_EXPRESSION_NODES
    ):
        return None
    var seed = String()
    var dtype = DataType.INT32
    for i in range(last + 1):
        ref node = expression._nodes[i]
        if node.op == COL:
            if node.text not in frame.columns():
                return None
            var column_dtype = frame.column(node.text).dtype()
            if not supported_dtype(column_dtype):
                return None
            if not seed:
                seed = node.text.copy()
                dtype = column_dtype
            elif dtype != column_dtype:
                return None
    if not seed:
        return None
    var nodes = List[ValueNode]()
    var stack = List[Int]()
    var columns = List[Bool]()
    for i in range(last + 1):
        ref node = expression._nodes[i]
        var has_column = node.op == COL
        if node.op == COL:
            if node.left >= 0 or node.right >= 0:
                return None
            nodes.append(ValueNode(COL, node.text.copy(), -1, 0))
        elif node.op == LIT_INT or node.op == LIT_FLOAT:
            if node.text != UNTYPED:
                var literal_dtype = (
                    DataType.INT64 if node.op == LIT_INT else DataType.FLOAT64
                )
                if len(node.dtypes) == 1 and node.dtypes[0]:
                    literal_dtype = node.dtypes[0].value()
                if literal_dtype != dtype:
                    return None
            var bits = UInt64(0)
            if dtype == DataType.FLOAT32:
                if node.op == LIT_INT and not _integer_fits_float32(
                    node.integer
                ):
                    return None
                var value = (
                    Float64(node.integer) if node.op
                    == LIT_INT else node.floating
                )
                var narrow = Float32(value)
                if Float64(narrow) != value:
                    return None
                bits = UInt64(bitcast[DType.uint32](narrow))
            else:
                if node.op != LIT_INT or (
                    dtype == DataType.INT32
                    and (
                        node.integer < Int64(Int32.MIN)
                        or node.integer > Int64(Int32.MAX)
                    )
                ):
                    return None
                bits = bitcast[DType.uint64](node.integer)
            nodes.append(ValueNode(LIT_INT, "", -1, bits))
        elif node.op == ADD or node.op == SUB or node.op == MUL:
            if len(stack) < 2:
                return None
            var right = stack.pop()
            var left = stack.pop()
            var right_column = columns.pop()
            var left_column = columns.pop()
            has_column = left_column or right_column
            # Literal-only arithmetic may bind its intermediate dtype before
            # the surrounding column supplies a native dtype. Decline it
            # rather than silently changing those intermediate semantics.
            if not has_column:
                return None
            # Require a real postfix tree, not arbitrary DAG edges or dead
            # work that might hide checked errors when lowering the program.
            if node.left != left or node.right != right:
                return None
            nodes.append(ValueNode(node.op, "", -1, 0))
        else:
            return None
        stack.append(i)
        columns.append(has_column)
        if len(stack) > MAX_EXPRESSION_STACK:
            return None
    if len(stack) != 1 or stack[0] != last:
        return None
    return ExpressionTree(dtype, nodes^, seed^)
