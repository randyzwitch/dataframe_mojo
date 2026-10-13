"""Resolve CPU selectors against metadata before accelerator normalization."""
from dataframe.expr import SELECTOR
from dataframe.lazy import LazyFrame, SELECT, WITH_COLUMNS, AGG, FILTER
from dataframe.selectors import expand_all
from .aggregates import _in_memory_sources
from .metadata import schema_frame


def _live_nodes(query: LazyFrame) -> List[Bool]:
    var live = List[Bool](length=len(query._nodes), fill=False)
    var pending = List[Int]([len(query._nodes) - 1])
    while len(pending):
        var index = pending.pop()
        if index < 0 or index >= len(live) or live[index]:
            continue
        live[index] = True
        ref operation = query._nodes[index]
        if operation.left >= 0:
            pending.append(operation.left)
        if operation.right >= 0:
            pending.append(operation.right)
    return live^


def has_selectors(query: LazyFrame) -> Bool:
    var live = _live_nodes(query)
    for index in range(len(live)):
        if not live[index]:
            continue
        ref operation = query._nodes[index]
        for expression in operation.exprs:
            for node in expression._nodes:
                if node.op == SELECTOR:
                    return True
    return False


def lower_selectors(query: LazyFrame) raises -> LazyFrame:
    """Expand names, dtypes, exclusions and positions using the CPU API.

    Expansion reads only schemas. Later lowering binds every expanded producer
    and retains its normal observation, precision and capability checks.
    """
    _in_memory_sources(query)
    var result = query.copy()
    var live = _live_nodes(query)
    for index in range(len(result._nodes)):
        if not live[index]:
            continue
        var operation = result._nodes[index].copy()
        var selected = False
        for expression in operation.exprs:
            for node in expression._nodes:
                selected |= node.op == SELECTOR
        if not selected:
            continue
        if (
            operation.kind != SELECT
            and operation.kind != WITH_COLUMNS
            and operation.kind != AGG
            and operation.kind != FILTER
        ):
            raise Error("Metal unsupported [selector]: expression context")
        var schema = schema_frame(result, operation.left)
        var expanded = expand_all(operation.exprs, schema._columns)
        if operation.kind == FILTER and len(expanded) != 1:
            raise Error("A filter selector must match exactly one column")
        if operation.kind == SELECT and not len(expanded):
            raise Error(
                "Metal unsupported [selector]: empty selections require zero-column resident results"
            )
        operation.exprs = expanded^
        result._nodes[index] = operation^
    return result^
