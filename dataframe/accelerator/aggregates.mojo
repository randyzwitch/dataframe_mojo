"""Metadata-only decomposition of aggregate expressions for resident Metal."""
from std.collections import Dict
from dataframe.binding import bind
from dataframe.expr import (
    Expr,
    Node,
    COL,
    SELECTOR,
    OVER,
    col,
    is_reduction,
    is_window,
    subtree,
    _node,
)
from dataframe.lazy import LazyFrame, PlanNode, SELECT, AGG, _plan_node


def _composed(expression: Expr) -> Bool:
    if len(expression._nodes) == 0 or is_reduction(
        expression._nodes[len(expression._nodes) - 1].op
    ):
        return False
    var aggregate = False
    for node in expression._nodes:
        if node.op == OVER or is_window(node.op):
            return False
        aggregate |= is_reduction(node.op)
    return aggregate


def has_aggregate_compositions(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        if operation.kind == SELECT or operation.kind == AGG:
            for expression in operation.exprs:
                if _composed(expression):
                    return True
    return False


def lower_aggregate_compositions(query: LazyFrame) raises -> LazyFrame:
    """Compute aggregate leaves on GPU, then project their resident results.

    Binding and schema execution inspect no source rows. Checked aggregate
    leaves retain the normal child-region restrictions; this transformation
    does not broaden which errors may be observed through a later projection.
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
    var serial = 0
    for index in range(len(query._nodes)):
        var operation = query._nodes[index].copy()
        if operation.left >= 0:
            operation.left = roots[operation.left]
        if operation.right >= 0:
            operation.right = roots[operation.right]
        var changed = False
        if operation.kind == SELECT or operation.kind == AGG:
            for expression in operation.exprs:
                changed |= _composed(expression)
        if not changed:
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        var schema = query._execute(query._nodes[index].left, True)
        var leaves = List[Expr]()
        var projected = List[Expr]()
        if operation.kind == AGG:
            for key in operation.names:
                projected.append(col(key))
        for expression in operation.exprs:
            for node in expression._nodes:
                if node.op == SELECTOR or node.op == OVER or is_window(node.op):
                    raise Error(
                        "Metal unsupported [aggregate composition]: selectors and windows"
                    )
            var bound = bind(expression, schema._columns)
            var positions = List[Int](length=len(bound.expr._nodes), fill=-1)
            var rewritten = List[Node]()
            for n in range(len(bound.expr._nodes)):
                var node = bound.expr._nodes[n].copy()
                positions[n] = len(rewritten)
                if is_reduction(node.op):
                    var name = "__dataframe_metal_aggregate_" + String(serial)
                    serial += 1
                    while name in names:
                        name += "_"
                    names[name] = True
                    leaves.append(subtree(bound.expr, n).alias(name))
                    rewritten.append(_node(COL, text=name))
                else:
                    if node.left >= 0:
                        node.left = positions[node.left]
                    if node.right >= 0:
                        node.right = positions[node.right]
                    if node.extra >= 0:
                        node.extra = positions[node.extra]
                    rewritten.append(node^)
            var rewritten_expression = Expr(rewritten^, expression._name)
            # Remove input references that belonged exclusively to aggregate
            # subtrees; the following projection sees only aggregate outputs.
            projected.append(
                subtree(
                    rewritten_expression, len(rewritten_expression._nodes) - 1
                )
            )
        operation.exprs = leaves^
        result._nodes.append(operation^)
        result._nodes.append(
            _plan_node(SELECT, len(result._nodes) - 1, exprs=projected)
        )
        roots.append(len(result._nodes) - 1)
    try:
        var actual = result._execute(len(result._nodes) - 1, True)
        if actual.columns() != expected.columns():
            raise Error("output names differ")
        for i in range(actual.width()):
            if actual._columns[i].dtype() != expected._columns[i].dtype():
                raise Error("output dtypes differ")
    except error:
        raise Error(
            "Metal unsupported [aggregate composition]: " + String(error)
        )
    return result^
