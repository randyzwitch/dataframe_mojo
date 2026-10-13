"""Metadata-only decomposition of aggregate expressions for resident Metal."""
from .metadata import schema_frame
from std.collections import Dict
from dataframe.binding import bind, ROWS
from dataframe.dtype import DataType
from dataframe.frame import DataFrame
from dataframe.expr import (
    Expr,
    Node,
    COL,
    SELECTOR,
    OVER,
    WHEN,
    MEAN,
    ADD,
    SUB,
    MUL,
    col,
    is_reduction,
    is_window,
    subtree,
    _node,
)
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    SELECT,
    AGG,
    FILTER,
    WITH_COLUMNS,
    JOIN,
    DROP,
    SCAN_FRAME,
    _is_scan,
    _plan_node,
)


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


def _needs_composition(operation: PlanNode) -> Bool:
    if (
        operation.kind != SELECT
        and operation.kind != AGG
        and operation.kind != FILTER
        and operation.kind != WITH_COLUMNS
    ):
        return False
    var reduction = False
    var other = False
    for expression in operation.exprs:
        if _composed(expression):
            return True
        other |= len(expression._nodes) == 0 or not is_reduction(
            expression._nodes[len(expression._nodes) - 1].op
        )
        for node in expression._nodes:
            reduction |= is_reduction(node.op)
    return reduction and (
        other or operation.kind == FILTER or operation.kind == WITH_COLUMNS
    )


def has_aggregate_compositions(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        if _needs_composition(operation):
            return True
    return False


def _in_memory_sources(query: LazyFrame) raises:
    var pending = List[Int]([len(query._nodes) - 1])
    var seen = List[Bool](length=len(query._nodes), fill=False)
    while len(pending):
        var index = pending.pop()
        if index < 0 or index >= len(query._nodes):
            raise Error("Metal unsupported [plan]: invalid child index")
        if seen[index]:
            continue
        seen[index] = True
        ref node = query._nodes[index]
        if _is_scan(node.kind) and node.kind != SCAN_FRAME:
            raise Error(
                "Metal unsupported [plan]: resident regions require in-memory"
                " scans"
            )
        if node.left >= 0:
            pending.append(node.left)
        if node.right >= 0:
            pending.append(node.right)


def _conditional_leaf(
    expression: Expr, schema: DataFrame, exact_float32: Bool
) raises:
    # A lifted scalar is evaluated before its row consumer. Keep data-dependent
    # precision failures out of conditionally unobserved scalar branches.
    var bound = bind(expression, schema._columns)
    for i in range(len(bound.expr._nodes)):
        ref node = bound.expr._nodes[i]
        if (
            not exact_float32
            and bound.dtypes[i] == DataType.FLOAT32
            and (node.op == ADD or node.op == SUB or node.op == MUL)
        ):
            raise Error(
                "Metal unsupported [precision]: conditional scalar Float32"
                " arithmetic requires CPU execution"
            )


def lower_aggregate_compositions(
    query: LazyFrame, *, exact_float32: Bool = False
) raises -> LazyFrame:
    """Compute aggregate leaves on GPU, then project their resident results.

    Binding and schema execution inspect no source rows. Checked aggregate
    leaves retain the normal child-region restrictions; this transformation
    does not broaden which errors may be observed through a later projection.
    """
    _in_memory_sources(query)
    var expected = schema_frame(query, len(query._nodes) - 1)
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
        var changed = _needs_composition(operation)
        if not changed:
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        var schema = schema_frame(query, query._nodes[index].left)
        var row_mode = (
            operation.kind == FILTER or operation.kind == WITH_COLUMNS
        )
        if operation.kind == SELECT:
            for expression in operation.exprs:
                var bound = bind(expression, schema._columns)
                row_mode |= bound.shapes[len(bound.shapes) - 1] == ROWS
        var leaves = List[Expr]()
        var leaf_names = List[String]()
        var projected = List[Expr]()
        if operation.kind == AGG:
            for key in operation.names:
                projected.append(col(key))
        for expression in operation.exprs:
            for node in expression._nodes:
                if node.op == SELECTOR or node.op == OVER or is_window(node.op):
                    raise Error(
                        "Metal unsupported [aggregate composition]: selectors"
                        " and windows"
                    )
            var bound = bind(expression, schema._columns)
            var conditional = False
            for node in bound.expr._nodes:
                conditional |= node.op == WHEN
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
                    var leaf = subtree(bound.expr, n).alias(name)
                    if conditional:
                        _conditional_leaf(leaf, schema, exact_float32)
                    leaves.append(leaf^)
                    leaf_names.append(name)
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
        if row_mode:
            # One resident prefix feeds both the row side and scalar leaves.
            # Its identity projection forces a GPU owner before the fan-out;
            # the DAG planner reuses that exact prefix task.
            var columns = List[Expr]()
            for name in schema.columns():
                columns.append(col(name))
            result._nodes.append(
                _plan_node(SELECT, operation.left, exprs=columns)
            )
            var prefix = len(result._nodes) - 1
            result._nodes.append(_plan_node(SELECT, prefix, exprs=leaves))
            var scalar = len(result._nodes) - 1
            result._nodes.append(
                _plan_node(
                    JOIN,
                    prefix,
                    scalar,
                    names2=["_right"],
                    text="cross",
                    how=6,
                )
            )
            operation.left = len(result._nodes) - 1
            operation.exprs = projected^
            var kind = operation.kind
            result._nodes.append(operation^)
            if kind == FILTER or kind == WITH_COLUMNS:
                result._nodes.append(
                    _plan_node(DROP, len(result._nodes) - 1, names=leaf_names)
                )
            roots.append(len(result._nodes) - 1)
            continue
        operation.exprs = leaves^
        result._nodes.append(operation^)
        result._nodes.append(
            _plan_node(
                SELECT,
                len(result._nodes) - 1,
                exprs=projected,
                text=query._nodes[index].text,
            )
        )
        roots.append(len(result._nodes) - 1)
    try:
        var actual = schema_frame(result, len(result._nodes) - 1)
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
