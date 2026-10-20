"""Private observed-pair boundary for bounded resident text concatenation.

The marker carries metadata only. Its placeholder establishes a fresh String
field; the GPU executor replaces it after discovering the actual operand pairs.
"""
from dataframe import DataType, Expr
from dataframe.expr import col, lit, null, when
from dataframe.lazy import LazyFrame, WITH_COLUMNS, _plan_node

comptime TAG = "metal_observed_text_concat"
comptime VERSION = "observed_pairs_v1"


@fieldwise_init
struct Controls(Copyable, Movable):
    var left: String
    var right: String
    var output: String
    var separator: String


def is_marker(query: LazyFrame) -> Bool:
    if not len(query._nodes):
        return False
    ref node = query._nodes[len(query._nodes) - 1]
    return (
        node.kind == WITH_COLUMNS
        and node.text == "metal_window_materialization"
        and len(node.names2) >= 1
        and node.names2[0] == TAG
    )


def has_markers(query: LazyFrame) -> Bool:
    for node in query._nodes:
        if len(node.names2) and node.names2[0] == TAG:
            return True
    return False


def controls(query: LazyFrame) raises -> Controls:
    if not is_marker(query):
        raise Error("Observed concat marker is missing")
    ref node = query._nodes[len(query._nodes) - 1]
    if len(node.names2) != 6 or node.names2[1] != VERSION:
        raise Error("Observed concat marker version or shape differs")
    if not node.names2[2] or not node.names2[3] or not node.names2[4]:
        raise Error("Observed concat operand or output name is empty")
    if node.names2[4] == node.names2[2] or node.names2[4] == node.names2[3]:
        raise Error("Observed concat output must be a fresh field")
    if len(node.exprs) != 1 or node.exprs[0]._name != node.names2[4]:
        raise Error("Observed concat placeholder output differs")
    return Controls(
        node.names2[2], node.names2[3], node.names2[4], node.names2[5]
    )


def stage(
    mut query: LazyFrame,
    left: Int,
    left_name: String,
    right_name: String,
    output: String,
    separator: String,
) -> Expr:
    var placeholder = (
        when(col(left_name).is_not_null() & col(right_name).is_not_null())
        .then(lit(""))
        .otherwise(null(DataType.STRING))
        .alias(output)
    )
    query._nodes.append(
        _plan_node(
            WITH_COLUMNS,
            left,
            exprs=[placeholder^],
            text="metal_window_materialization",
            names2=[TAG, VERSION, left_name, right_name, output, separator],
        )
    )
    return col(output)
