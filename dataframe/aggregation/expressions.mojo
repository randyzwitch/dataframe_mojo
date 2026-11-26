"""Binding and expression eligibility shared by grouping and frame evaluation."""
from dataframe.binding import (
    BoundExpr,
    bind,
)
from dataframe.expr import (
    ALL,
    ANY,
    EXPR_MAX_BY,
    EXPR_MIN_BY,
    ARG_MAX,
    ARG_MIN,
    COL,
    CORR,
    COUNT,
    COV,
    DROP_NULLS,
    EXPR_FILTER,
    Expr,
    FIRST,
    IMPLODE,
    KURTOSIS,
    LAST,
    LEN,
    MAX,
    MEAN,
    MEDIAN,
    MIN,
    MODE,
    NULL_COUNT,
    N_UNIQUE,
    OVER,
    PRODUCT,
    QUANTILE,
    ROW_LEN,
    SELECTOR,
    SKEW,
    STD,
    SUM,
    VAR,
    is_reduction,
    is_sequence,
    is_window,
    Node,
    _set_node_children,
    col,
    node_children,
    subtree,
)
from dataframe.selectors import (
    expand_all,
)
from dataframe.series import (
    Series,
)
from std.memory import bitcast
from std.collections import (
    Dict,
    Optional,
)


def _bind_all(
    expressions: List[Expr], columns: List[Series]
) raises -> List[BoundExpr]:
    """Expand selectors, reject duplicate output names, then bind all."""
    var expanded = expand_all(expressions, columns)
    var bound = List[BoundExpr]()
    var names = Dict[String, Bool]()
    for expression in expanded:
        if expression._name in names:
            raise Error("Duplicate expression output name: " + expression._name)
        names[expression._name] = True
        bound.append(bind(expression, columns))
    return bound^


def _column_names(expr: Expr) -> List[String]:
    var names = List[String]()
    for node in expr._nodes:
        if node.op == COL and node.text not in names:
            names.append(node.text)
    return names^


def count_rows_by(expression: Expr, key: String) -> Expr:
    """`expression` with each `len()` as `col(key).len()`, a reduction with a state in the
    aggregate table: every row has a value of the key, null or not, so the
    counts are the same. Unchanged when it has none, or when the result is
    no streaming reduction (`len()` inside a window or a sequence)."""
    var found = False
    for node in expression._nodes:
        if node.op == ROW_LEN:
            found = True
    if not found:
        return expression.copy()
    var counted = col(key).len()
    var nodes = List[Node](capacity=len(expression._nodes) + 1)
    var place = List[Int](capacity=len(expression._nodes))
    for j in range(len(expression._nodes)):
        var node = expression._nodes[j].copy()
        if node.op == ROW_LEN:
            nodes.append(counted._nodes[0].copy())
            var length = counted._nodes[1].copy()
            length.left = len(nodes) - 1
            nodes.append(length^)
        else:
            var children = node_children(node)
            for c in range(len(children)):
                children[c] = place[children[c]]
            _set_node_children(node, children)
            nodes.append(node^)
        place.append(len(nodes) - 1)
    var rewritten = expression.copy()
    rewritten._nodes = nodes^
    if not _stream_reductions([rewritten.copy()]):
        return expression.copy()
    return rewritten^


def _row_local(exprs: List[Expr]) -> Bool:
    """True when every output row depends only on the same input row."""
    for e in exprs:
        for node in e._nodes:
            if (
                is_reduction(node.op)
                or is_sequence(node.op)
                or is_window(node.op)
                or node.op == OVER
            ):
                return False
    return True


def filter_value(expression: Expr, at: Int) -> Int:
    """The node a reduction's input node `at` takes its values from: the
    filtered values of an aggregate filter (`aggregate_filter`), else `at`
    itself."""
    if at >= 0 and aggregate_filter(expression, at):
        return expression._nodes[at].left
    return at


def aggregate_filter(expression: Expr, at: Int) -> Optional[Expr]:
    """The predicate of an aggregate filter: a reduction's input node `at`
    that keeps some rows of row-local values, `x.filter(p)` (the rows where
    `p` is true) or `x.drop_nulls()` (where `x` is not null). Reducing it
    is reducing `x` over only those rows, so every row still finds its
    group and the reduction's state skips the rows the predicate drops.
    None for any other input."""
    if at < 0:
        return None
    ref node = expression._nodes[at]
    if node.op == EXPR_FILTER:
        var value = subtree(expression, node.left)
        var predicate = subtree(expression, node.right)
        if not _row_local([value.copy()]) or not _row_local([predicate.copy()]):
            return None
        return predicate^
    if node.op == DROP_NULLS:
        var value = subtree(expression, node.left)
        if not _row_local([value.copy()]):
            return None
        return value.is_not_null()
    return None


def same_nodes(a: Expr, b: Expr) -> Bool:
    """Whether two expressions are the same nodes, field by field (names
    aside): they keep the same rows."""
    if len(a._nodes) != len(b._nodes):
        return False
    for k in range(len(a._nodes)):
        ref x = a._nodes[k]
        ref y = b._nodes[k]
        if (
            x.op != y.op
            or x.left != y.left
            or x.right != y.right
            or x.text != y.text
            or x.integer != y.integer
            or bitcast[DType.uint64](x.floating)
            != bitcast[DType.uint64](y.floating)
            or x.min_count != y.min_count
            or x.extra != y.extra
            or x.text2 != y.text2
            or len(x.dtypes) != len(y.dtypes)
        ):
            return False
        for d in range(len(x.dtypes)):
            if Bool(x.dtypes[d]) != Bool(y.dtypes[d]):
                return False
            if x.dtypes[d] and (
                x.dtypes[d].value().name() != y.dtypes[d].value().name()
            ):
                return False
    return True


def reduction_filter(expression: Expr, at: Int) -> Optional[Expr]:
    """The predicate of reduction node `at`'s aggregate filter: its input's
    (`aggregate_filter`), or for a correlation or covariance, the one
    predicate both its inputs filter by, since it pairs the rows the two
    keep. None when the reduction reads no aggregate filter, or a pair's
    inputs keep rows by different predicates (or one keeps every row)."""
    ref node = expression._nodes[at]
    var left = aggregate_filter(expression, node.left)
    if node.op != CORR and node.op != COV:
        return left^
    if not left:
        return None
    var right = aggregate_filter(expression, node.right)
    if not right or not same_nodes(left.value(), right.value()):
        return None
    return left^


def without_filters(expressions: List[Expr]) -> List[Expr]:
    """`expressions` with each aggregate filter replaced by the values it
    filters: the same reductions, of the same types, over every row."""
    var result = List[Expr](capacity=len(expressions))
    for expression in expressions:
        var nodes = expression._nodes.copy()
        var changed = False
        for i in range(len(nodes)):
            if not is_reduction(nodes[i].op) or nodes[i].left < 0:
                continue
            if reduction_filter(expression, i):
                nodes[i].left = filter_value(expression, nodes[i].left)
                if nodes[i].op == CORR or nodes[i].op == COV:
                    nodes[i].right = filter_value(expression, nodes[i].right)
                changed = True
        if changed:
            var compact = Expr(nodes^, expression._name)
            result.append(subtree(compact, len(compact._nodes) - 1))
        else:
            result.append(expression.copy())
    return result^


def has_aggregate_filter(expressions: List[Expr]) -> Bool:
    """Whether some reduction reads an aggregate filter."""
    for expression in expressions:
        for node in expression._nodes:
            if is_reduction(node.op) and aggregate_filter(
                expression, node.left
            ):
                return True
    return False


def _stream_reductions(
    expressions: List[Expr],
    holistic: Bool = False,
    listed: Bool = False,
    filtered: Bool = False,
) -> Bool:
    """Whether every expression is reductions of row-local inputs, combined
    row by row. `holistic` admits medians and quantiles; `listed` also
    mode, implode, max_by and min_by, whose states the aggregate table holds by address;
    `filtered` also a reduction of an aggregate filter
    (`reduction_filter`), which the aggregate table applies per
    reduction."""
    if len(expressions) == 0:
        return False
    for expression in expressions:
        ref nodes = expression._nodes
        var reachable = List[Bool](length=len(nodes), fill=False)
        reachable[len(nodes) - 1] = True
        var saw_reduction = False
        # The aggregate filters admitted, each below its reduction (seen
        # first: a node's children precede it).
        var filters = List[Int]()
        for reverse in range(len(nodes)):
            var i = len(nodes) - 1 - reverse
            ref node = nodes[i]
            if i in filters:
                continue
            # A max_by or min_by, a sequence step over a group's rows, is
            # a reduction the table holds by address (`listed`).
            var picks = node.op == EXPR_MAX_BY or node.op == EXPR_MIN_BY
            if (
                (
                    (is_window(node.op) or is_sequence(node.op))
                    and not (listed and picks)
                )
                or node.op == OVER
                or node.op == SELECTOR
            ):
                return False
            if not reachable[i]:
                continue
            if is_reduction(node.op):
                if holistic and (node.op == MEDIAN or node.op == QUANTILE):
                    pass
                elif listed and node.op in [
                    MODE,
                    IMPLODE,
                    EXPR_MAX_BY,
                    EXPR_MIN_BY,
                ]:
                    pass
                elif node.op not in [
                    SUM,
                    COUNT,
                    MIN,
                    MAX,
                    MEAN,
                    FIRST,
                    LAST,
                    STD,
                    VAR,
                    LEN,
                    ANY,
                    ALL,
                    NULL_COUNT,
                    N_UNIQUE,
                    ARG_MIN,
                    ARG_MAX,
                    SKEW,
                    KURTOSIS,
                    CORR,
                    COV,
                    PRODUCT,
                ]:
                    return False
                if filtered and reduction_filter(expression, i):
                    filters.append(node.left)
                    if node.op == CORR or node.op == COV:
                        filters.append(node.right)
                    saw_reduction = True
                    continue
                # Reduction inputs must be row-local; reduction-of-reduction
                # and windows retain the materializing evaluator.
                if not _row_local([subtree(expression, node.left)]):
                    return False
                if node.right >= 0 and not _row_local(
                    [subtree(expression, node.right)]
                ):
                    return False
                saw_reduction = True
                continue
            if node.op == COL:
                return False
            for child in node_children(node):
                reachable[child] = True
        if not saw_reduction:
            return False
    return True
