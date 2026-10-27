"""Eager adapters for grouping sets, time windows, pivot, and gather."""
from dataframe.binding import (
    bind,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.expr import (
    Expr,
    col,
    exclude,
    grouping_value,
    is_grouping,
    lit,
    null,
)
from dataframe.frame import (
    DataFrame,
    _pivot_aggregate,
    concat,
)
from dataframe.join_type import (
    JOIN_CROSS,
    join_type,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.graph import (
    plan_live_parents,
    plan_order_free_above,
)
from dataframe.planning.model import (
    PlanNode,
    ROLLING,
)
from dataframe.planning.properties import (
    _join_schema,
    _set_members,
)
from dataframe.series import (
    Series,
)
from std.collections import (
    Dict,
    Optional,
)


def _set_rows(
    node: PlanNode,
    var part: DataFrame,
    members: List[String],
    dtypes: List[DataType],
) raises -> DataFrame:
    """One set's grouped rows (its keys, then the aggregations) in the
    node's output shape: every key in order, null where the set does not
    group by it, and the `grouping` columns as that set's constants."""
    var constants = List[Expr]()
    var marked = False
    for e in node.exprs:
        if is_grouping(e):
            marked = True
            constants.append(lit(grouping_value(e, members)).alias(e._name))
    for k in range(len(node.group_keys())):
        if node.group_keys()[k] not in members:
            constants.append(null(dtypes[k]).alias(node.group_keys()[k]))
    if len(constants) > 0:
        part = part.with_columns(constants)
    var order = List[Expr]()
    for name in node.group_keys():
        order.append(col(name))
    if marked:
        # Output names are known only for single-output expressions.
        for e in node.exprs:
            order.append(col(e._name))
    else:
        order.append(exclude(node.group_keys()))
    return part.select_exprs(order)


def _run_grouping_sets(node: PlanNode, input: DataFrame) raises -> DataFrame:
    """A GROUPING_SETS node over its collected input: one group-by per set,
    stacked. The stream's single sink replaces this when the aggregate
    table keeps every state."""
    var dtypes = List[DataType]()
    for name in node.group_keys():
        dtypes.append(input.column(name).dtype())
    var aggregates = List[Expr]()
    for e in node.exprs:
        if not is_grouping(e):
            aggregates.append(e.copy())
    var members = _set_members(node)
    var parts = List[DataFrame](capacity=len(members))
    for one in members:
        var part: DataFrame
        if len(one) == 0:
            part = input.select_exprs(aggregates)
        else:
            part = input.group_by(one).agg(aggregates)
        parts.append(_set_rows(node, part^, one, dtypes))
    return concat(parts)


def _run_time_window(node: PlanNode, input: DataFrame) raises -> DataFrame:
    """The eager `group_by_dynamic` or `rolling` aggregation a time window
    node records, over its executed input."""
    var by = List[String](node.window_columns()[1:])
    if node.kind == ROLLING:
        return input.rolling(
            node.window_columns()[0],
            node.window_period(),
            Optional(node.window_offset()),
            node.window_closed(),
            by,
        ).agg(node.exprs, batch_size=node.window_batch_size())
    return input.group_by_dynamic(
        node.window_columns()[0],
        node.window_every(),
        Optional(node.window_period()),
        Optional(node.window_offset()),
        node.window_closed(),
        node.window_label(),
        by,
    ).agg(node.exprs, batch_size=node.window_batch_size())


def _pivoted(input: DataFrame, node: PlanNode) raises -> DataFrame:
    """A PIVOT node's result: DataFrame.pivot's, its columns chosen and
    ordered by the node's names, a name the input has no cell for filled
    with nulls of the aggregate's type."""
    var index = List[String](node.pivot_columns()[: node.pivot_index_count()])
    var values = node.pivot_values()
    var wide = input.pivot(
        node.pivot_on(),
        index=index,
        values=values,
        aggregate_function=node.text,
    )
    var bound = bind(_pivot_aggregate(values, node.text), input._columns)
    var dtype = bound.dtypes[len(bound.dtypes) - 1]
    var present = Dict[String, Bool]()
    for name in wide.columns():
        present[name] = True
    var columns = List[Series](
        capacity=len(index) + len(node.pivot_output_columns())
    )
    for name in index:
        columns.append(wide.column(name))
    for name in node.pivot_output_columns():
        if name in present:
            columns.append(wide.column(name))
        else:
            columns.append(Series.full_null(name, dtype, wide.height()))
    return DataFrame(columns^, height=wide.height())


def _gathered(
    frame: DataFrame, indices: DataFrame, null_on_oob: Bool
) raises -> DataFrame:
    """A GATHER node's result: `frame`'s rows at the positions in the one
    integer column of `indices`. Null and out-of-range handling is
    determined by `null_on_oob`."""
    if indices.width() != 1:
        raise Error("gather indices must be a single column")
    var column = indices._columns[0].copy()
    if not column.dtype().is_integer():
        raise Error(
            "gather indices must have an integer dtype, not "
            + column.dtype().name()
        )
    column = column.cast(DataType.INT64)
    var rows = List[Int](capacity=len(column))
    var valid = List[Bool](capacity=len(column))
    for k in range(len(column)):
        var value = column.get(k)
        valid.append(not value.is_null())
        rows.append(0 if value.is_null() else Int(value.int64()))
    return frame._gather(rows, valid, null_on_oob)


def plan_join_frames(
    self: QueryContext,
    index: Int,
    var input: DataFrame,
    var right: DataFrame,
    empty: Bool,
) raises -> DataFrame:
    """Join node `index`'s executed inputs."""
    ref node = self._nodes[index]
    if node.how == JOIN_CROSS:
        return input.join(right, how=node.text, suffix=node.join_suffix())
    if empty and input.height() == 0 and right.height() == 0:
        var schema = _join_schema(node, input, right)
        if schema:
            return schema.take()
    # Consumers that ignore row order let the join skip restoring
    # left-major order after probing.
    # A schema has no row order to keep.
    var free = empty or plan_order_free_above(
        self, index, plan_live_parents(self, len(self._nodes) - 1), True
    )
    return input._join_impl(
        right,
        left_on=node.join_left_keys(),
        right_on=node.join_right_keys(),
        # Parsed here, as `DataFrame.join` would: an unknown type
        # raises.
        how=join_type(node.text),
        suffix=node.join_suffix(),
        coalesce=node.coalesce,
        keep_order=not free,
    )
