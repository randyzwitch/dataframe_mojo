"""Stage Spearman's pairwise Float64 ranks before resident correlation."""
from std.collections import Dict
from .metadata import schema_frame
from .aggregates import _in_memory_sources
from dataframe.binding import bind
from dataframe.dtype import DataType
from dataframe.expr import Expr, Node, COL, CORR, corr, col, lit, when, subtree
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    SELECT,
    AGG,
    FILTER,
    WITH_COLUMNS,
    DROP,
    _plan_node,
)


def has_spearman(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        for expression in operation.exprs:
            for node in expression._nodes:
                if node.op == CORR and node.text == "spearman":
                    return True
    return False


def lower_spearman(query: LazyFrame) raises -> LazyFrame:
    """Rank only paired valid rows, then correlate ranks in original row order.

    Float64 conversion precedes ranking, as in the CPU aggregate. A paired
    NaN is retained in one ranked operand so Pearson's state returns NaN;
    unpaired NaNs are excluded. Metadata binding never reads source values.
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
            for node in expression._nodes:
                changed |= node.op == CORR and node.text == "spearman"
        if not changed:
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        if (
            operation.kind != SELECT
            and operation.kind != AGG
            and operation.kind != FILTER
            and operation.kind != WITH_COLUMNS
        ):
            raise Error(
                "Metal unsupported [Spearman]: correlation requires a row consumer"
            )
        var schema = schema_frame(query, query._nodes[index].left)
        var prefix = operation.left
        var temporary = List[String]()
        var projected = List[Expr]()
        for expression in operation.exprs:
            var bound = bind(expression, schema._columns)
            var positions = List[Int](length=len(bound.expr._nodes), fill=-1)
            var rewritten = List[Node]()
            for n in range(len(bound.expr._nodes)):
                var node = bound.expr._nodes[n].copy()
                if node.left >= 0:
                    node.left = positions[node.left]
                if node.right >= 0:
                    node.right = positions[node.right]
                if node.extra >= 0:
                    node.extra = positions[node.extra]
                if node.op != CORR or node.text != "spearman":
                    positions[n] = len(rewritten)
                    rewritten.append(node^)
                    continue
                var prepared = Expr(rewritten.copy(), expression._name)
                var left = subtree(prepared, node.left)
                var right = subtree(prepared, node.right)
                for side in List[Int](
                    [bound.expr._nodes[n].left, bound.expr._nodes[n].right]
                ):
                    if bound.dtypes[side].is_decimal():
                        raise Error(
                            "Metal unsupported [logical]: decimal moments require an explicit Float64 cast"
                        )
                var stem = "__dataframe_metal_spearman_" + String(serial)
                serial += 1
                while (
                    stem + "_x" in names
                    or stem + "_y" in names
                    or stem + "_rx" in names
                    or stem + "_ry" in names
                ):
                    stem += "_"
                var x = stem + "_x"
                var y = stem + "_y"
                var rx = stem + "_rx"
                var ry = stem + "_ry"
                for name in List[String]([x, y, rx, ry]):
                    names[name] = True
                    temporary.append(name)
                result._nodes.append(
                    _plan_node(
                        WITH_COLUMNS,
                        prefix,
                        exprs=[
                            left.cast(DataType.FLOAT64).alias(x),
                            right.cast(DataType.FLOAT64).alias(y),
                        ],
                    )
                )
                prefix = len(result._nodes) - 1
                var paired = col(x).is_not_null() & col(y).is_not_null()
                var rank_x = when(paired).then(col(x)).end().rank("average")
                var rank_y = when(paired).then(col(y)).end().rank("average")
                if operation.kind == AGG:
                    rank_x = rank_x.over(operation.names)
                    rank_y = rank_y.over(operation.names)
                result._nodes.append(
                    _plan_node(
                        WITH_COLUMNS,
                        prefix,
                        exprs=[rank_x.alias(rx), rank_y.alias(ry)],
                    )
                )
                prefix = len(result._nodes) - 1
                var invalid = col(x).is_nan() | col(y).is_nan()
                var ranked_x = (
                    when(invalid)
                    .then(lit(Float64(0) / Float64(0)))
                    .otherwise(col(rx))
                )
                var replacement = corr(ranked_x, col(ry))
                var offset = len(rewritten)
                for child in replacement._nodes:
                    var copy = child.copy()
                    if copy.left >= 0:
                        copy.left += offset
                    if copy.right >= 0:
                        copy.right += offset
                    if copy.extra >= 0:
                        copy.extra += offset
                    rewritten.append(copy^)
                positions[n] = len(rewritten) - 1
            var final = Expr(rewritten^, expression._name)
            projected.append(subtree(final, len(final._nodes) - 1))
        operation.left = prefix
        operation.exprs = projected^
        result._nodes.append(operation^)
        if (
            query._nodes[index].kind == FILTER
            or query._nodes[index].kind == WITH_COLUMNS
        ):
            result._nodes.append(
                _plan_node(DROP, len(result._nodes) - 1, names=temporary)
            )
        roots.append(len(result._nodes) - 1)
    var actual = schema_frame(result, len(result._nodes) - 1)
    if actual.columns() != expected.columns():
        raise Error("Metal unsupported [Spearman]: output names differ")
    for i in range(actual.width()):
        if actual._columns[i].dtype() != expected._columns[i].dtype():
            raise Error("Metal unsupported [Spearman]: output dtypes differ")
    return result^
