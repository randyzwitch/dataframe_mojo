"""Binding and expression eligibility shared by grouping and frame evaluation."""
from dataframe.binding import (
    BoundExpr,
    bind,
)
from dataframe.expr import (
    ALL,
    ANY,
    ARG_MAX,
    ARG_MIN,
    COL,
    CORR,
    COUNT,
    COV,
    Expr,
    FIRST,
    KURTOSIS,
    LAST,
    LEN,
    MAX,
    MEAN,
    MEDIAN,
    MIN,
    NULL_COUNT,
    N_UNIQUE,
    OVER,
    PRODUCT,
    QUANTILE,
    SELECTOR,
    SKEW,
    STD,
    SUM,
    VAR,
    is_reduction,
    is_window,
    node_children,
    subtree,
)
from dataframe.selectors import (
    expand_all,
)
from dataframe.series import (
    Series,
)
from std.collections import (
    Dict,
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


def _row_local(exprs: List[Expr]) -> Bool:
    """True when every output row depends only on the same input row."""
    for e in exprs:
        for node in e._nodes:
            if is_reduction(node.op) or is_window(node.op) or node.op == OVER:
                return False
    return True


def _stream_reductions(expressions: List[Expr], holistic: Bool = False) -> Bool:
    if len(expressions) == 0:
        return False
    for expression in expressions:
        ref nodes = expression._nodes
        var reachable = List[Bool](length=len(nodes), fill=False)
        reachable[len(nodes) - 1] = True
        var saw_reduction = False
        for reverse in range(len(nodes)):
            var i = len(nodes) - 1 - reverse
            ref node = nodes[i]
            if is_window(node.op) or node.op == OVER or node.op == SELECTOR:
                return False
            if not reachable[i]:
                continue
            if is_reduction(node.op):
                if holistic and (node.op == MEDIAN or node.op == QUANTILE):
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
