"""Metadata-only lowering of set operations to resident GPU operators."""
from dataframe.expr import Expr, col, lit
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    UNIQUE,
    AGG,
    FILTER,
    SELECT,
    WITH_COLUMNS,
    JOIN,
    CONCAT,
    _plan_node,
)
from dataframe.join_type import JOIN_FULL


def has_set_operations(query: LazyFrame) -> Bool:
    for node in query._nodes:
        if node.kind == UNIQUE or node.kind == CONCAT:
            return True
    return False


def lower_set_operations(query: LazyFrame) raises -> LazyFrame:
    """Keep first/last rows, or exclusively singleton rows, using GPU groups.

    Schema execution reads no source rows. Group first retains null payloads;
    maintain_order sorts groups by their first source position. No source
    values or duplicate decisions are evaluated by this transformation.
    """
    _ = query._execute(len(query._nodes) - 1, True)
    var result = query.copy()
    result._nodes = List[PlanNode]()
    var roots = List[Int]()
    for index in range(len(query._nodes)):
        var node = query._nodes[index].copy()
        if node.left >= 0:
            node.left = roots[node.left]
        if node.right >= 0:
            node.right = roots[node.right]
        if node.kind == CONCAT:
            var schema = query._execute(query._nodes[index].left, True)
            var names = schema.columns()
            var tag = "__dataframe_concat_side"
            while tag in names:
                tag += "_"
            var suffix = "__dataframe_concat_right"
            var collision = True
            while collision:
                collision = False
                for name in names:
                    collision |= name + suffix in names or name + suffix == tag
                if collision:
                    suffix += "_"
            # Opposite valid Boolean keys can never match. Full-join output
            # is therefore all left rows followed by all right rows, with
            # native cardinality checks, residency, and memory accounting.
            result._nodes.append(
                _plan_node(
                    WITH_COLUMNS, node.left, exprs=[lit(False).alias(tag)]
                )
            )
            var left = len(result._nodes) - 1
            result._nodes.append(
                _plan_node(
                    WITH_COLUMNS, node.right, exprs=[lit(True).alias(tag)]
                )
            )
            result._nodes.append(
                _plan_node(
                    JOIN,
                    left,
                    len(result._nodes) - 1,
                    names=[tag],
                    names2=[suffix],
                    text="full",
                    how=JOIN_FULL,
                    right_keys=[tag],
                )
            )
            var projected = List[Expr]()
            for name in names:
                # Each output row has exactly one present branch. Fill-null
                # retains a null source cell because the other branch is also
                # null; it never merges cells from two actual source rows.
                projected.append(
                    col(name).fill_null(col(name + suffix)).alias(name)
                )
            result._nodes.append(
                _plan_node(SELECT, len(result._nodes) - 1, exprs=projected)
            )
            roots.append(len(result._nodes) - 1)
            continue
        if node.kind != UNIQUE:
            result._nodes.append(node^)
            roots.append(len(result._nodes) - 1)
            continue
        var schema = query._execute(query._nodes[index].left, True)
        var names = schema.columns()
        var keys = node.names.copy()
        if len(keys) == 0:
            keys = names.copy()
        if len(keys) == 0:
            raise Error("Metal unsupported [plan]: unique requires columns")
        var aggregates = List[Expr]()
        for name in names:
            if name not in keys:
                aggregates.append(
                    col(name).last().alias(name) if node.text
                    == "last" else col(name).first().alias(name)
                )
        var count_name = "__dataframe_unique_count"
        while count_name in names:
            count_name += "_"
        if node.text == "none":
            aggregates.append(col(keys[0]).len().alias(count_name))
        result._nodes.append(
            _plan_node(
                AGG,
                node.left,
                exprs=aggregates,
                names=keys,
                # Internal grouping header chooses the representative and
                # position used to restore surviving source-row order.
                flags=[node.text == "last"],
                maintain_order=True,
            )
        )
        if node.text == "none":
            result._nodes.append(
                _plan_node(
                    FILTER,
                    len(result._nodes) - 1,
                    exprs=[col(count_name) == lit(Int64(1))],
                )
            )
        var projected = List[Expr]()
        for name in names:
            projected.append(col(name))
        result._nodes.append(
            _plan_node(SELECT, len(result._nodes) - 1, exprs=projected)
        )
        roots.append(len(result._nodes) - 1)
    return result^
