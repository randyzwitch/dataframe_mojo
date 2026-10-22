"""Isolated QCUT custom-label resident marker; numeric admission is pending.

No input values are inspected. Original binding belongs to the shared CUT
normalizer. A private marker stages control bits and literal label positions;
ordinary dictionary recoding keeps those node positions unchanged.
"""
from std.memory import bitcast
from dataframe import DataType, Expr
from dataframe.expr import (
    Node,
    LIT_STRING,
    SEP,
    col,
    lit,
    null,
    when,
    as_struct,
    split_names,
)
from dataframe.lazy import LazyFrame, WITH_COLUMNS, _plan_node
from dataframe.parse import parse_float64, parse_int64

comptime QCUT_TAG = "metal_custom_qcut_operand"
comptime QCUT_VERSION = "custom_labels_v1"
comptime QCUT_DEFAULT = "default_labels_v1"


def is_qcut_marker(query: LazyFrame) -> Bool:
    if not len(query._nodes):
        return False
    ref node = query._nodes[len(query._nodes) - 1]
    return (
        node.kind == WITH_COLUMNS
        and node.text == "metal_window_materialization"
        and len(node.names2) >= 7
        and node.names2[0] == QCUT_TAG
        and (node.names2[1] == QCUT_VERSION or node.names2[1] == QCUT_DEFAULT)
    )


def stage_custom(
    mut query: LazyFrame,
    left: Int,
    operand: String,
    category_name: String,
    breakpoint_name: String,
    node: Node,
) raises -> Expr:
    """Append an observed control/classifier boundary with safe empty schema."""
    var probabilities = List[Float64]()
    if node.text.byte_length():
        for part in node.text.split(SEP):
            probabilities.append(parse_float64(String(part)))
    var labels = split_names(node.text2) if node.text2.byte_length() else List[
        String
    ]()
    var defaults = not len(labels)
    if not defaults and len(labels) != len(probabilities) + 1:
        raise Error("qcut labels must have one more element than quantiles")
    var flags = node.integer & 7
    # Empty literal is a String schema/control placeholder only; a dynamic
    # dictionary is published after GPU edges have been observed.
    var category = lit("") if defaults else lit(labels[len(labels) - 1])
    for reverse in range(len(labels) - 1):
        var index = len(labels) - 2 - reverse
        category = (
            when(col(operand).is_not_null())
            .then(category)
            .otherwise(lit(labels[index]))
        )
    category = (
        when(col(operand).is_not_null())
        .then(category)
        .otherwise(null(DataType.STRING))
        .alias(category_name)
    )
    var metadata: List[String] = [
        QCUT_TAG,
        QCUT_DEFAULT if defaults else QCUT_VERSION,
        operand,
        category_name,
        breakpoint_name if flags & 2 else String(),
        String(flags),
        String(len(probabilities)),
    ]
    for probability in probabilities:
        metadata.append(String(bitcast[DType.int64](probability)))
    for label in labels:
        var found = -1
        for index in range(len(category._nodes)):
            ref literal = category._nodes[index]
            if literal.op == LIT_STRING and literal.text == label:
                found = index
                break
        if found < 0:
            raise Error("QCUT custom label literal marker differs")
        metadata.append(String(found))
    var expressions: List[Expr] = [category^]
    if flags & 2:
        expressions.append(
            when(col(operand).is_not_null())
            .then(lit(Float64(0)))
            .otherwise(null(DataType.FLOAT64))
            .alias(breakpoint_name)
        )
    query._nodes.append(
        _plan_node(
            WITH_COLUMNS,
            left,
            exprs=expressions,
            text="metal_window_materialization",
            names2=metadata,
        )
    )
    if flags & 2:
        return as_struct(
            [
                col(breakpoint_name).alias("breakpoint"),
                col(category_name).alias("category"),
            ],
            "",
        )
    return col(category_name)


@fieldwise_init
struct Controls(Movable):
    var operand: String
    var category: String
    var breakpoint: String
    var flags: UInt32
    var probabilities: List[UInt64]
    var label_nodes: List[Int]
    var defaults: Bool


def controls(query: LazyFrame) raises -> Controls:
    """Parse only bounded literal metadata; never execute the dummy row AST."""
    if not is_qcut_marker(query):
        raise Error("QCUT private resident marker is missing")
    ref operation = query._nodes[len(query._nodes) - 1]
    ref metadata = operation.names2
    var flags = parse_int64(metadata[5])
    var count = parse_int64(metadata[6])
    if flags < 0 or flags > 7 or count < 0 or count > Int64(Int32.MAX):
        raise Error("QCUT private control shape is invalid")
    var defaults = metadata[1] == QCUT_DEFAULT
    var expected = 7 + Int(count) if defaults else 8 + Int(count) * 2
    if len(metadata) != expected or len(operation.exprs) != (
        2 if flags & 2 else 1
    ):
        raise Error("QCUT private control payload differs")
    if operation.exprs[0]._name != metadata[3] or (
        flags & 2 and operation.exprs[1]._name != metadata[4]
    ):
        raise Error("QCUT private output names differ")
    var probabilities = List[UInt64](capacity=Int(count))
    for index in range(Int(count)):
        probabilities.append(
            parse_int64(metadata[7 + index]).cast[DType.uint64]()
        )
    var nodes = List[Int](capacity=Int(count) + 1)
    for index in range(0 if defaults else Int(count) + 1):
        var at = parse_int64(metadata[7 + Int(count) + index])
        if at < 0 or at >= Int64(len(operation.exprs[0]._nodes)):
            raise Error("QCUT private label node is outside expression")
        nodes.append(Int(at))
    return Controls(
        metadata[2],
        metadata[3],
        metadata[4],
        UInt32(flags),
        probabilities^,
        nodes^,
        defaults,
    )


def has_default_qcuts(query: LazyFrame) -> Bool:
    from .partition_cuts import TAG as PARTITION_CUT_TAG

    for node in query._nodes:
        if len(node.names2) > 1 and (
            (node.names2[0] == QCUT_TAG and node.names2[1] == QCUT_DEFAULT)
            or node.names2[0] == PARTITION_CUT_TAG
        ):
            return True
    return False
