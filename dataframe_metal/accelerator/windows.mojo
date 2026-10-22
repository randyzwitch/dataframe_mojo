"""Partition aggregates and nested window scopes materialized on the GPU."""
from .metadata import schema_frame
from dataframe.plan_types import WindowKey
from std.collections import Dict
from dataframe.binding import bind, ROWS
from dataframe.dtype import DataType
from dataframe.expr import (
    Expr,
    Node,
    COL,
    OVER,
    CUM_SUM,
    CUM_MIN,
    CUM_MAX,
    CUM_COUNT,
    SHIFT,
    ROLLING_SUM,
    ROLLING_MEAN,
    ROLLING_MIN,
    ROLLING_MAX,
    ROLLING_VAR,
    ROLLING_STD,
    FORWARD_FILL,
    BACKWARD_FILL,
    SELECTOR,
    col,
    null,
    is_reduction,
    is_window,
    subtree,
    split_names,
    _node,
)
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    SELECT,
    FILTER,
    WITH_COLUMNS,
    AGG,
    JOIN,
    DROP,
    _plan_node,
)
from .aggregates import _in_memory_sources


def _staged_over(expression: Expr, index: Int) -> Bool:
    if expression._nodes[index].op != OVER:
        return False
    var inner = subtree(expression, expression._nodes[index].left)
    for node in inner._nodes:
        if (
            is_reduction(node.op)
            or node.op == OVER
            or node.op == CUM_SUM
            or node.op == CUM_MIN
            or node.op == CUM_MAX
            or node.op == CUM_COUNT
            or node.op == ROLLING_SUM
            or node.op == ROLLING_MEAN
            or node.op == ROLLING_MIN
            or node.op == ROLLING_MAX
            or node.op == ROLLING_VAR
            or node.op == ROLLING_STD
            or node.op == SHIFT
            or node.op == FORWARD_FILL
            or node.op == BACKWARD_FILL
        ):
            return True
    return False


def has_partition_aggregates(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        if operation.text == "metal_window_materialization":
            continue
        for expression in operation.exprs:
            for i in range(len(expression._nodes)):
                if _staged_over(expression, i):
                    return True
    return False


def lower_partition_aggregates(query: LazyFrame) raises -> LazyFrame:
    """Stage scopes bottom-up, aligning aggregates and preserving row windows.

    CPU evaluate() prepares every OVER state before evaluating its consumer.
    Only these internal producers are marked as observable materializations;
    checked expressions elsewhere retain the normal child-region restrictions.
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
        var changed = False
        for expression in operation.exprs:
            for i in range(len(expression._nodes)):
                changed |= _staged_over(expression, i)
        if not changed or operation.text == "metal_window_materialization":
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        if (
            operation.kind != SELECT
            and operation.kind != FILTER
            and operation.kind != WITH_COLUMNS
        ):
            raise Error(
                "Metal unsupported [window]: partition aggregates require a row"
                " consumer"
            )
        var schema = schema_frame(query, query._nodes[index].left)
        var columns = List[Expr]()
        for name in schema.columns():
            columns.append(col(name))
        result._nodes.append(
            _plan_node(
                SELECT,
                operation.left,
                exprs=columns,
                text="metal_shared_prefix",
            )
        )
        var prefix = len(result._nodes) - 1
        var aligned = prefix
        var temporary = List[String]()
        var projected = List[Expr]()
        for expression in operation.exprs:
            var bound = bind(expression, schema._columns)
            var positions = List[Int](length=len(bound.expr._nodes), fill=-1)
            var rewritten = List[Node]()
            for n in range(len(bound.expr._nodes)):
                var node = bound.expr._nodes[n].copy()
                positions[n] = len(rewritten)
                if node.op == OVER:
                    # Earlier OVER nodes already name their aligned resident
                    # results. Extract from that rewritten graph so a nested
                    # window consumes the inner window's rows.
                    var prepared = Expr(rewritten.copy(), expression._name)
                    var inner = subtree(prepared, positions[node.left])
                    var dependent = False
                    var windowed = False
                    for child in inner._nodes:
                        if child.op == OVER or child.op == SELECTOR:
                            raise Error(
                                "Metal unsupported [window]: nested windows and"
                                " selectors"
                            )
                        dependent |= child.op == COL and child.text in temporary
                        windowed |= is_window(child.op)
                    var producer = prefix
                    if dependent:
                        # A row owner above the preceding join is shared by
                        # the outer aggregate and its broadcast side.
                        var current = schema_frame(result, aligned)
                        var shared = List[Expr]()
                        for column in current.columns():
                            shared.append(col(column))
                        result._nodes.append(
                            _plan_node(
                                SELECT,
                                aligned,
                                exprs=shared,
                                text="metal_shared_prefix",
                            )
                        )
                        producer = len(result._nodes) - 1
                        aligned = producer
                    var producer_schema = schema_frame(result, producer)
                    var inner_bound = bind(inner, producer_schema._columns)
                    var name = "__dataframe_metal_window_" + String(serial)
                    serial += 1
                    while name in names:
                        name += "_"
                    names[name] = True
                    temporary.append(name)
                    var keys = split_names(node.text2)
                    var window_keys = List[WindowKey]()
                    for key in keys:
                        for column in schema._columns:
                            if column.name() != key:
                                continue
                            var dtype = column.dtype()
                            var string_hash = (
                                dtype == DataType.STRING or dtype._is_bytes()
                            )
                            if dtype.is_categorical():
                                # CPU evaluate() prepares the whole bound
                                # expression before any OVER state. A COL
                                # read anywhere decodes this categorical;
                                # key names in OVER.text2 alone do not.
                                for original in bound.expr._nodes:
                                    string_hash |= (
                                        original.op == COL
                                        and original.text == key
                                    )
                            window_keys.append(
                                WindowKey(key, dtype, string_hash)
                            )
                    # CPU prepares the entire bound expression before OVER.
                    # A read CAT key therefore groups and broadcasts by decoded
                    # String bytes. Use a private key on BOTH edges; visible CAT
                    # columns still retain their original code/dictionary owner.
                    var comparison_keys = keys.copy()
                    var decoded = List[Expr]()
                    for key in range(len(window_keys)):
                        if (
                            not window_keys[key].dtype.is_categorical()
                            or not window_keys[key].string_hash
                        ):
                            continue
                        var alias = "__dataframe_metal_window_key_" + String(
                            serial
                        )
                        serial += 1
                        while alias in names:
                            alias += "_"
                        names[alias] = True
                        temporary.append(alias)
                        decoded.append(
                            col(keys[key])
                            .fill_null(null(DataType.STRING))
                            .alias(alias)
                        )
                        comparison_keys[key] = alias
                        window_keys[key] = WindowKey(
                            alias, DataType.STRING, True
                        )
                    if len(decoded):
                        var original_producer = producer
                        result._nodes.append(
                            _plan_node(
                                WITH_COLUMNS,
                                producer,
                                exprs=decoded.copy(),
                                text="metal_shared_prefix",
                            )
                        )
                        producer = len(result._nodes) - 1
                        if aligned == original_producer:
                            aligned = producer
                        else:
                            result._nodes.append(
                                _plan_node(
                                    WITH_COLUMNS,
                                    aligned,
                                    exprs=decoded.copy(),
                                    text="metal_shared_prefix",
                                )
                            )
                            aligned = len(result._nodes) - 1
                    if inner_bound.shape() == ROWS:
                        result._nodes.append(
                            _plan_node(
                                WITH_COLUMNS,
                                aligned,
                                exprs=[
                                    (
                                        inner.over(
                                            comparison_keys
                                        ) if windowed else inner.copy()
                                    ).alias(name)
                                ],
                                text="metal_window_materialization",
                            )
                        )
                    else:
                        result._nodes.append(
                            _plan_node(
                                AGG,
                                producer,
                                exprs=[inner.alias(name)],
                                names=comparison_keys,
                                # The rewritten graph has no OVER nodes;
                                # retain the original nested eligibility.
                                # Flag0 remains first/last representation.
                                flags=[False, not dependent],
                                window_keys=window_keys,
                                maintain_order=True,
                                text="metal_window_materialization",
                            )
                        )
                        var grouped = len(result._nodes) - 1
                        result._nodes.append(
                            _plan_node(
                                JOIN,
                                aligned,
                                grouped,
                                names=comparison_keys,
                                right_keys=comparison_keys,
                                names2=["_right"],
                                how=1,
                                text="left",
                                flags=[True],
                            )
                        )
                    aligned = len(result._nodes) - 1
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
            projected.append(
                subtree(
                    rewritten_expression, len(rewritten_expression._nodes) - 1
                )
            )
        operation.left = aligned
        operation.exprs = projected^
        var kind = operation.kind
        result._nodes.append(operation^)
        if kind == FILTER or kind == WITH_COLUMNS:
            result._nodes.append(
                _plan_node(DROP, len(result._nodes) - 1, names=temporary)
            )
        roots.append(len(result._nodes) - 1)
    var actual = schema_frame(result, len(result._nodes) - 1)
    if actual.columns() != expected.columns():
        raise Error("Metal unsupported [window]: output names differ")
    for i in range(actual.width()):
        if actual._columns[i].dtype() != expected._columns[i].dtype():
            raise Error("Metal unsupported [window]: output dtypes differ")
    return result^
