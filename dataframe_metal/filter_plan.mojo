"""Schema-only matching of Metal filtered reduction regions."""
from std.utils.numerics import inf, isnan
from dataframe.dtype import DataType
from dataframe.expr import (
    AND,
    COL,
    EQ,
    Expr,
    GE,
    GT,
    LE,
    LIT_FLOAT,
    LIT_INT,
    LT,
    SUM,
    UNTYPED,
)
from dataframe.lazy import FILTER, LazyFrame, SCAN_FRAME, SELECT


def supported_dtype(dtype: DataType) -> Bool:
    return (
        dtype == DataType.INT32
        or dtype == DataType.INT64
        or dtype == DataType.FLOAT32
    )


struct FilterSum(Copyable, Movable):
    """An accepted `select(col(c).sum())` over range filters on `c`."""

    var column: String
    var output: String
    var filter_column: String
    var filter_dtype: DataType
    var dtype: DataType
    var min_count: Int
    var filtered: Bool
    # Integer columns: the inclusive range kept.
    var low: Int64
    var high: Int64
    # Float32 columns: the range kept, with bit 0 making `low_float`
    # inclusive and bit 1 `high_float`.
    var low_float: Float32
    var high_float: Float32
    var inclusive: UInt32

    def __init__(out self, column: String, output: String, dtype: DataType):
        self.column = column
        self.output = output
        self.filter_column = column
        self.filter_dtype = dtype
        self.dtype = dtype
        self.min_count = 0
        self.filtered = False
        self.low = Int64.MIN
        self.high = Int64.MAX
        self.low_float = -inf[DType.float32]()
        self.high_float = inf[DType.float32]()
        self.inclusive = 3

    def _narrow(mut self, op: Int, integer: Int64, floating: Float64) -> Bool:
        """Keep only values satisfying `c <op> literal`; False when the
        literal cannot be represented exactly in the column's type."""
        self.filtered = True
        if self.filter_dtype == DataType.FLOAT32:
            var bound = floating.cast[DType.float32]()
            if isnan(bound) or bound.cast[DType.float64]() != floating:
                return False
            if op == GT or op == GE or op == EQ:
                var exclusive = op == GT
                if bound > self.low_float or (
                    bound == self.low_float and exclusive
                ):
                    self.low_float = bound
                    self.inclusive = (self.inclusive & 2) | (
                        UInt32(0) if exclusive else UInt32(1)
                    )
            if op == LT or op == LE or op == EQ:
                var exclusive = op == LT
                if bound < self.high_float or (
                    bound == self.high_float and exclusive
                ):
                    self.high_float = bound
                    self.inclusive = (self.inclusive & 1) | (
                        UInt32(0) if exclusive else UInt32(2)
                    )
            return True
        var lowest = Int64(
            Int32.MIN
        ) if self.filter_dtype == DataType.INT32 else (Int64.MIN)
        var highest = Int64(
            Int32.MAX
        ) if self.filter_dtype == DataType.INT32 else (Int64.MAX)
        if integer < lowest or integer > highest:
            return False
        if op == GT:
            if integer == highest:
                self.low = highest
                self.high = lowest
                return True
            self.low = max(self.low, integer + 1)
        elif op == GE:
            self.low = max(self.low, integer)
        elif op == LT:
            if integer == lowest:
                self.low = highest
                self.high = lowest
                return True
            self.high = min(self.high, integer - 1)
        elif op == LE:
            self.high = min(self.high, integer)
        else:
            self.low = max(self.low, integer)
            self.high = min(self.high, integer)
        return True


def _predicate(mut plan: FilterSum, predicate: Expr, at: Int) raises -> Bool:
    """Narrow `plan` by the predicate subtree rooted at node `at`."""
    ref node = predicate._nodes[at]
    if node.op == AND:
        return _predicate(plan, predicate, node.left) and _predicate(
            plan, predicate, node.right
        )
    if not (
        node.op == GT
        or node.op == GE
        or node.op == LT
        or node.op == LE
        or node.op == EQ
    ):
        return False
    ref left = predicate._nodes[node.left]
    ref right = predicate._nodes[node.right]
    if left.op != COL or left.text != plan.filter_column:
        return False
    if right.text != UNTYPED:
        return False
    if right.op == LIT_INT:
        if plan.filter_dtype == DataType.FLOAT32:
            return plan._narrow(node.op, 0, Float64(right.integer))
        return plan._narrow(node.op, right.integer, 0)
    if right.op == LIT_FLOAT and plan.filter_dtype == DataType.FLOAT32:
        return plan._narrow(node.op, 0, right.floating)
    return False


def plan_region(region: LazyFrame) raises -> Optional[FilterSum]:
    """The accepted shape of `region`, or None to leave it on the CPU."""
    var nodes = len(region._plan._nodes)
    if nodes < 2 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    ref root = region._plan._nodes[nodes - 1]
    if root.kind != SELECT or len(root.exprs) != 1:
        return None
    ref total = root.exprs[0]
    if len(total._nodes) != 2:
        return None
    if total._nodes[0].op != COL or total._nodes[1].op != SUM:
        return None
    var name = total._nodes[0].text
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    if name not in frame.columns():
        return None
    var dtype = frame.column(name).dtype()
    if not (
        dtype == DataType.INT32
        or dtype == DataType.INT64
        or dtype == DataType.FLOAT32
    ):
        return None
    var plan = FilterSum(name, total._name, dtype)
    plan.min_count = total._nodes[1].min_count
    var filter_name = String()
    for i in range(1, nodes - 1):
        for expression in region._plan._nodes[i].exprs:
            for expr_node in expression._nodes:
                if expr_node.op == COL:
                    if filter_name and filter_name != expr_node.text:
                        return None
                    filter_name = expr_node.text.copy()
    if filter_name:
        if filter_name not in frame.columns():
            return None
        var filter_dtype = frame.column(filter_name).dtype()
        if not supported_dtype(filter_dtype):
            return None
        plan.filter_column = filter_name
        plan.filter_dtype = filter_dtype
    for i in range(1, nodes - 1):
        ref node = region._plan._nodes[i]
        if node.kind != FILTER or len(node.exprs) != 1:
            return None
        if not _predicate(plan, node.exprs[0], len(node.exprs[0]._nodes) - 1):
            return None
    return plan^
