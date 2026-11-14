"""Expression folding and common-subexpression helpers."""
from dataframe.binding import (
    BoundExpr,
    bind,
)
from dataframe.storage.column import (
    Column,
)
from dataframe.dtype import (
    DataType,
    NUMERIC_DTYPES,
)
from dataframe.execution import _total_cast, evaluate
from dataframe.expr import (
    ADD,
    AND,
    CAST,
    COL,
    DIV,
    MEAN,
    MUL,
    NEG,
    SUB,
    SUM,
    EQ,
    GE,
    GT,
    LE,
    LT,
    IS_NULL,
    IS_NOT_NULL,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    COUNT,
    LEN,
    NULL_COUNT,
    MIN,
    MAX,
    FIRST,
    LAST,
    Expr,
    HORIZONTAL_ARGS,
    LIT_BOOL,
    LIT_FLOAT,
    LIT_INT,
    LIT_NULL,
    LIT_STRING,
    NE,
    NOT,
    Node,
    OR,
    OVER,
    SELECTOR,
    UNTYPED,
    WHEN,
    _set_node_children,
    col,
    is_reduction,
    is_window,
    node_children,
    subtree,
)
from dataframe.planning.properties import (
    _expr_signature,
)
from dataframe.series import (
    Series,
)
from std.collections import (
    Dict,
    Optional,
)


def _cannot_raise(expr: Expr) -> Bool:
    """Whether evaluation is total after schema binding.

    Row locality alone does not permit removing input rows: checked arithmetic,
    casts, parsing and other partial operations can fail on a discarded row.
    Keep unknown operations behind the same evaluation boundary. Add an opcode
    here only when all of its bound implementations are total.
    """
    for node in expr._nodes:
        if not _total_op(node.op):
            return False
    return True


def _total_op(op: Int) -> Bool:
    """Whether every bound implementation of `op` is total."""
    return op in [
        COL,
        SELECTOR,
        LIT_BOOL,
        LIT_INT,
        LIT_FLOAT,
        LIT_STRING,
        LIT_NULL,
        EQ,
        NE,
        LT,
        LE,
        GT,
        GE,
        AND,
        OR,
        NOT,
        IS_NULL,
        IS_NOT_NULL,
        IS_NAN,
        IS_NOT_NAN,
        IS_FINITE,
        IS_INFINITE,
        COUNT,
        LEN,
        NULL_COUNT,
        MIN,
        MAX,
        FIRST,
        LAST,
    ]


def _cannot_raise(exprs: List[Expr]) -> Bool:
    for expr in exprs:
        if not _cannot_raise(expr):
            return False
    return True


def _cannot_raise(exprs: List[Expr], input: List[Series]) -> Bool:
    """Whether evaluating `exprs` over columns typed as `input` is total.

    Some operations raise for one input type and never for another: integer
    and decimal arithmetic and sums are checked for overflow, while float
    arithmetic follows IEEE 754 and yields an infinity or NaN instead; a
    cast raises only when some source value has no target value. So an
    expression the untyped check rejects can still be total once its
    operand types are known. An expression that does not bind keeps the
    untyped answer."""
    if _cannot_raise(exprs):
        return True
    for expr in exprs:
        try:
            if not _cannot_raise(bind(expr, input, fuse=False)):
                return False
        except:
            return False
    return True


def _cannot_raise(bound: BoundExpr) -> Bool:
    """Whether every operation of a bound expression is total for the
    types it was bound to: the untyped total operations; float addition,
    subtraction, multiplication, division and negation; sums and means of
    floats; and casts that are defined for every value (to the same type,
    between numeric types when not strict, or strict into a type that
    holds every source value). Decimal targets are excluded: a decimal
    cast can overflow its precision."""
    ref nodes = bound.expr._nodes
    for i in range(len(nodes)):
        ref node = nodes[i]
        if _total_op(node.op):
            continue
        var op = node.op
        if op == ADD or op == SUB or op == MUL or op == DIV or op == NEG:
            if not bound.dtypes[i].is_float():
                return False
            for child in [node.left, node.right]:
                if child >= 0 and not bound.dtypes[child].is_float():
                    return False
            continue
        if op == SUM or op == MEAN:
            if node.left < 0 or not bound.dtypes[node.left].is_float():
                return False
            continue
        if op == CAST and node.left >= 0:
            var source = bound.dtypes[node.left]
            var target = bound.dtypes[i]
            if source == target:
                continue
            if (
                target.is_numeric()
                and not target.is_decimal()
                and source.is_numeric()
                and (not source.is_decimal() or target.is_float())
                and (node.integer != 1 or _total_cast(source, target))
            ):
                continue
        return False
    return True


def _same_literal(a: Node, b: Node) -> Int:
    """1 when two literal nodes hold equal values, 0 when unequal, -1
    when that is unknown (not both literals of one type, or a null)."""
    if a.op != b.op:
        return -1
    if a.op == LIT_STRING:
        return 1 if a.text == b.text else 0
    if a.op == LIT_BOOL:
        return 1 if a.integer == b.integer else 0
    if (
        a.op == LIT_INT
        and a.text != UNTYPED
        and a.text == b.text
        and a.text2 == b.text2
    ):
        return 1 if a.integer == b.integer else 0
    return -1


def _fold_constants(expr: Expr) -> Expr:
    """`expr` with its constant boolean parts evaluated: an equality of
    two literals (a projection's constant substituted into a filter,
    `_through_projection`), and AND, OR and NOT over known values, under
    SQL's three-valued logic (false AND null is false, true OR null is
    true). This allows filters to be simplified before they move."""
    ref nodes = expr._nodes
    var n = len(nodes)
    # -1 unknown, 0 false, 1 true.
    var value = List[Int](length=n, fill=-1)
    var folded = False
    for i in range(n):
        ref node = nodes[i]
        var a = value[node.left] if node.left >= 0 else -1
        var b = value[node.right] if node.right >= 0 else -1
        if node.op == LIT_BOOL:
            value[i] = 1 if node.integer != 0 else 0
            continue
        if (node.op == EQ or node.op == NE) and node.left >= 0:
            var same = _same_literal(nodes[node.left], nodes[node.right])
            if same >= 0:
                value[i] = same if node.op == EQ else 1 - same
        elif node.op == AND:
            if a == 0 or b == 0:
                value[i] = 0
            elif a == 1 and b == 1:
                value[i] = 1
            elif a == 1 or b == 1:
                folded = True
        elif node.op == OR:
            if a == 1 or b == 1:
                value[i] = 1
            elif a == 0 and b == 0:
                value[i] = 0
            elif a == 0 or b == 0:
                folded = True
        elif node.op == NOT and a >= 0:
            value[i] = 1 - a
        folded = folded or value[i] >= 0
    if not folded:
        return expr.copy()
    var out = List[Node]()
    _ = _emit_folded(nodes, value, n - 1, out)
    return Expr(out^, expr._name)


def _fold_casts(expr: Expr) raises -> Optional[Expr]:
    """`expr` with each cast of a column-free subexpression replaced by the
    literal it evaluates to, or None when there is none to fold. A cast
    that raises is left in place, so its error is raised where it was."""
    ref nodes = expr._nodes
    var replaced = Dict[Int, Node]()
    for i in range(len(nodes)):
        ref node = nodes[i]
        if node.op != CAST or node.left < 0:
            continue
        if not _column_free(expr, i):
            continue
        try:
            var no_columns = List[Series]()
            var value = evaluate(
                bind(subtree(expr, i), no_columns), no_columns, 1
            )
            var literal = _literal_node(value)
            if literal:
                replaced[i] = literal.take()
        except:
            continue
    if len(replaced) == 0:
        return None
    var out = List[Node]()
    _ = _emit_replaced(nodes, replaced, len(nodes) - 1, out)
    return Optional(Expr(out^, expr._name))


def _column_free(expr: Expr, root: Int) -> Bool:
    """Whether the subexpression at `root` reads no column, selector,
    aggregate or window: one value whatever the rows."""
    var stack: List[Int] = [root]
    while len(stack) > 0:
        var i = stack.pop()
        ref node = expr._nodes[i]
        if (
            node.op == COL
            or node.op == SELECTOR
            or is_reduction(node.op)
            or is_window(node.op)
        ):
            return False
        for child in node_children(node):
            stack.append(child)
    return True


def _literal_node(value: Series) raises -> Optional[Node]:
    """A one-row result as a typed literal node: integers, temporal values
    and decimals as `LIT_INT` (a 128-bit decimal's high word in `text2`),
    floats as `LIT_FLOAT`, Booleans, strings and nulls as theirs. None for
    anything else (a list, a struct, a categorical)."""
    if len(value) != 1:
        return None
    var dtype = value.dtype()
    var dtypes: List[Optional[DataType]] = [Optional(dtype)]
    if value.null_count() == 1:
        return Optional(
            Node(LIT_NULL, -1, -1, dtype.name(), 0, 0, 0, -1, "", dtypes^)
        )
    if dtype == DataType.BOOL:
        return Optional(
            Node(
                LIT_BOOL,
                -1,
                -1,
                "",
                Int64(1) if value.get(0).bool() else Int64(0),
                0,
                0,
                -1,
                "",
                List[Optional[DataType]](),
            )
        )
    if dtype == DataType.STRING:
        return Optional(
            Node(
                LIT_STRING,
                -1,
                -1,
                value.get(0).string(),
                0,
                0,
                0,
                -1,
                "",
                List[Optional[DataType]](),
            )
        )
    if dtype.is_categorical() or dtype.is_nested():
        return None
    if value._data.isa[Column[Int128]]():
        if not dtype.is_decimal():
            return None
        var raw = value._data[Column[Int128]]._get(0)
        var low = Int64(UInt64(UInt128(raw) & UInt128(0xFFFFFFFFFFFFFFFF)))
        var high = Int64(raw >> 64)
        return Optional(
            Node(LIT_INT, -1, -1, "", low, 0, 0, -1, String(high), dtypes^)
        )
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if value._data.isa[Column[Scalar[D]]]():
            var x = value._data[Column[Scalar[D]]]._get(0)
            comptime if D.is_floating_point():
                return Optional(
                    Node(
                        LIT_FLOAT,
                        -1,
                        -1,
                        "",
                        0,
                        x.cast[DType.float64](),
                        0,
                        -1,
                        "",
                        dtypes^,
                    )
                )
            else:
                return Optional(
                    Node(
                        LIT_INT,
                        -1,
                        -1,
                        "",
                        x.cast[DType.int64](),
                        0,
                        0,
                        -1,
                        "",
                        dtypes^,
                    )
                )
    return None


def _shared_text(expr: Expr, root: Int) -> Optional[String]:
    """The text of the subexpression of `expr` at node `root` when common
    subexpression elimination may compute it once: row-local, reading a
    column, and more than a column or a literal; None otherwise."""
    var op = expr._nodes[root].op
    if (
        op == COL
        or op == LIT_INT
        or op == LIT_FLOAT
        or op == LIT_BOOL
        or op == LIT_STRING
        or op == LIT_NULL
        or op == SELECTOR
        # A horizontal reduction's input list is evaluated by the reduction
        # above it, never as a column of its own.
        or op == HORIZONTAL_ARGS
    ):
        return None
    var part = subtree(expr, root)
    var reads = False
    for node in part._nodes:
        # Row-local only (`_row_local`): no reduction or window.
        if (
            node.op == SELECTOR
            or node.op == OVER
            or is_reduction(node.op)
            or is_window(node.op)
        ):
            return None
        reads = reads or node.op == COL
    if not reads:
        return None
    part._name = String()
    return _expr_signature(part)


def count_shared(
    expr: Expr,
    i: Int,
    short: Bool,
    mut every_row: Dict[String, Int],
    mut seen: Dict[String, Int],
):
    """Count each subexpression of `expr` under node `i` that common
    subexpression elimination may compute once (`_shared_text`): in
    `seen` every occurrence, in `every_row` those evaluated on every row.

    A conditional evaluates its branches only on the rows that take them,
    and a conjunction may skip its later operands, so an occurrence there
    (`short`) may never be evaluated on a row where it would raise. One
    computed once for every row is safe only when some occurrence is
    evaluated on every row anyway. A reduction or window evaluates its
    input on every row whatever conditional holds its result."""
    var text = _shared_text(expr, i)
    if text:
        seen[text.value()] = seen.get(text.value(), 0) + 1
        if not short:
            every_row[text.value()] = every_row.get(text.value(), 0) + 1
    ref node = expr._nodes[i]
    var below = short
    if is_reduction(node.op) or is_window(node.op) or node.op == OVER:
        below = False
    var first = True
    for child in node_children(node):
        var conditional = (
            node.op == WHEN or node.op == AND or node.op == OR
        ) and not first
        count_shared(expr, child, below or conditional, every_row, seen)
        first = False


def _emit_shared(
    expr: Expr,
    i: Int,
    seen: Dict[String, Int],
    mut names: Dict[String, String],
    mut shared: List[Expr],
    taken: List[String],
    mut out: List[Node],
) raises -> Int:
    """Append node `i` of `expr` to `out` after its inputs, a subexpression
    `seen` more than once as a read of the hidden column computing it
    (added to `shared` the first time); its index in `out`."""
    var text = _shared_text(expr, i)
    if text and seen.get(text.value(), 0) > 1:
        if text.value() not in names:
            var name = "__cse_" + String(len(shared))
            while name in taken:
                name = "_" + name
            names[text.value()] = name
            shared.append(subtree(expr, i).alias(name))
        out.append(col(names[text.value()])._nodes[0].copy())
        return len(out) - 1
    var copy = expr._nodes[i].copy()
    var children = node_children(copy)
    for k in range(len(children)):
        children[k] = _emit_shared(
            expr, children[k], seen, names, shared, taken, out
        )
    _set_node_children(copy, children)
    out.append(copy^)
    return len(out) - 1


def _emit_replaced(
    nodes: List[Node], replaced: Dict[Int, Node], i: Int, mut out: List[Node]
) raises -> Int:
    """Append node `i` to `out` after its inputs, a folded subexpression as
    its literal; its index in `out`."""
    if i in replaced:
        out.append(replaced[i].copy())
        return len(out) - 1
    var copy = nodes[i].copy()
    var children = node_children(copy)
    for k in range(len(children)):
        children[k] = _emit_replaced(nodes, replaced, children[k], out)
    _set_node_children(copy, children)
    out.append(copy^)
    return len(out) - 1


def _emit_folded(
    nodes: List[Node], value: List[Int], i: Int, mut out: List[Node]
) -> Int:
    """Append node `i` of a folded expression to `out` after its inputs,
    a known value as a literal and an AND (OR) with a true (false) input
    as its other input; its index in `out`."""
    if value[i] >= 0:
        out.append(
            Node(
                LIT_BOOL,
                -1,
                -1,
                "",
                Int64(value[i]),
                0,
                0,
                -1,
                "",
                List[Optional[DataType]](),
            )
        )
        return len(out) - 1
    ref node = nodes[i]
    if node.op == AND or node.op == OR:
        var identity = 1 if node.op == AND else 0
        if value[node.left] == identity:
            return _emit_folded(nodes, value, node.right, out)
        if value[node.right] == identity:
            return _emit_folded(nodes, value, node.left, out)
    var copy = node.copy()
    var children = node_children(copy)
    for k in range(len(children)):
        children[k] = _emit_folded(nodes, value, children[k], out)
    _set_node_children(copy, children)
    out.append(copy^)
    return len(out) - 1
