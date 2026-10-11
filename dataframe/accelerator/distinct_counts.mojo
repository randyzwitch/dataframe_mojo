"""Distinct counts through resident deduplication and row counting."""
from std.collections import Dict
from dataframe.binding import bind
from dataframe.expr import Expr, COL, N_UNIQUE, col, subtree
from dataframe.lazy import LazyFrame, PlanNode, SELECT, AGG, UNIQUE, _plan_node


def _eligible(operation: PlanNode) -> Bool:
    if operation.kind != SELECT and operation.kind != AGG:
        return False
    if len(operation.exprs) != 1:
        return False
    ref expression = operation.exprs[0]
    return (
        len(expression._nodes) > 0
        and expression._nodes[len(expression._nodes) - 1].op == N_UNIQUE
    )


def has_distinct_counts(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        if _eligible(operation):
            return True
    return False


def lower_distinct_counts(query: LazyFrame) raises -> LazyFrame:
    """Count a single distinct expression, including its one null category.

    Input expressions keep the ordinary nonterminal checked-error restrictions.
    Other aggregate siblings require independent branches and remain unsupported.
    No source rows are evaluated while constructing these native GPU operators.
    """
    var expected = query._execute(len(query._nodes) - 1, True)
    var names = Dict[String, Bool]()
    for frame in query._frames:
        for name in frame.columns():
            names[name] = True
    for operation in query._nodes:
        for name in operation.names:
            names[name] = True
        for expression in operation.exprs:
            names[expression._name] = True
            for node in expression._nodes:
                if node.op == COL:
                    names[node.text] = True
    var result = query.copy()
    result._nodes = List[PlanNode]()
    var roots = List[Int]()
    for index in range(len(query._nodes)):
        var operation = query._nodes[index].copy()
        if operation.left >= 0:
            operation.left = roots[operation.left]
        if operation.right >= 0:
            operation.right = roots[operation.right]
        if not _eligible(operation):
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        var schema = query._execute(query._nodes[index].left, True)
        var bound = bind(operation.exprs[0], schema._columns)
        var name = "__dataframe_metal_distinct_value"
        while name in names:
            name += "_"
        names[name] = True
        var projected = List[Expr]()
        var keys = operation.names.copy()
        for key in keys:
            projected.append(col(key))
        ref top = bound.expr._nodes[len(bound.expr._nodes) - 1]
        projected.append(subtree(bound.expr, top.left).alias(name))
        result._nodes.append(
            _plan_node(SELECT, operation.left, exprs=projected)
        )
        keys.append(name)
        result._nodes.append(
            _plan_node(
                UNIQUE,
                len(result._nodes) - 1,
                names=keys,
                text="first",
                maintain_order=True,
            )
        )
        operation.left = len(result._nodes) - 1
        operation.exprs = [col(name).len().alias(operation.exprs[0]._name)]
        result._nodes.append(operation^)
        roots.append(len(result._nodes) - 1)
    var actual = result._execute(len(result._nodes) - 1, True)
    if actual.columns() != expected.columns():
        raise Error("Metal unsupported [distinct count]: output names differ")
    for i in range(actual.width()):
        if actual._columns[i].dtype() != expected._columns[i].dtype():
            raise Error(
                "Metal unsupported [distinct count]: output dtypes differ"
            )
    return result^
