"""Original public partition CUT/QCUT metadata, before private normalization.

This certificate captures binding and controls without reading source values.
It is deliberately separate from admission: the execution scope must also
prove eager OVER state order, shared WindowLayouts, and resident key codes.
"""
from std.memory import bitcast
from dataframe import DataFrame, DataType, Expr
from dataframe.binding import BoundExpr, bind
from dataframe.expr import CUT, QCUT, OVER, COL, SEP, subtree, split_names
from dataframe.parse import parse_float64
from .implode_ordered import signature
from .partition_operands import certificate as operand_certificate
from .window_keys import WindowKey


@fieldwise_init
struct Capture(Copyable):
    var original: String
    var over_node: Int
    var operand: Expr
    var operand_signature: String
    var keys: List[WindowKey]
    var key_set: String
    var specifications: List[UInt64]
    var labels: List[String]
    var qcut: Bool
    var left_closed: Bool
    var include_breaks: Bool
    var allow_duplicates: Bool


def capture(
    original: BoundExpr, source: DataFrame, over_node: Int
) raises -> Capture:
    """Keep immutable controls and original AST/logical key interpretation.

    Bound siblings must be captured before lowering any occurrence. A key
    that is also a COL anywhere in the original expression uses decoded text
    hashing, matching CPU evaluate's whole-expression column preparation.
    """
    if over_node < 0 or over_node >= len(original.expr._nodes):
        raise Error("Partition CUT original OVER occurrence is missing")
    ref over = original.expr._nodes[over_node]
    if over.op != OVER or over.left < 0:
        raise Error("Partition CUT requires an original OVER occurrence")
    ref operation = original.expr._nodes[over.left]
    if operation.op != CUT and operation.op != QCUT:
        raise Error("Partition CUT requires a direct original bin state")
    if operation.left < 0:
        raise Error("Partition CUT original operand is missing")
    var operand = subtree(original.expr, operation.left)
    var proof = operand_certificate(operand, source)
    var specifications = List[UInt64]()
    if operation.text.byte_length():
        for part in operation.text.split(SEP):
            specifications.append(
                bitcast[DType.uint64](parse_float64(String(part)))
            )
    var labels = List[String]()
    if operation.text2.byte_length():
        labels = split_names(operation.text2)
    if len(labels) and len(labels) != len(specifications) + 1:
        raise Error("Partition CUT literal label count changed after binding")
    var keys = List[WindowKey]()
    for name in split_names(over.text2):
        var field = source._index(name)
        var dtype = source._columns[field].dtype()
        var string_hash = dtype == DataType.STRING or dtype._is_bytes()
        if dtype.is_categorical():
            for node in original.expr._nodes:
                string_hash |= node.op == COL and node.text == name
        keys.append(WindowKey(name, dtype, string_hash))
    if not len(keys):
        raise Error("Partition CUT original key set is empty")
    return Capture(
        signature(original),
        over_node,
        operand^,
        proof^,
        keys^,
        over.text2,
        specifications^,
        labels^,
        operation.op == QCUT,
        (operation.integer & 1) != 0,
        (operation.integer & 2) != 0,
        (operation.integer & 4) != 0,
    )


def validate(saved: Capture, expression: Expr, source: DataFrame) raises:
    """Rebinding must preserve the entire expression before any row lifting."""
    var rebound = bind(expression, source._columns)
    if signature(rebound) != saved.original:
        raise Error("Partition CUT original expression metadata changed")
    var current = capture(rebound, source, saved.over_node)
    if (
        current.operand_signature != saved.operand_signature
        or current.key_set != saved.key_set
        or current.specifications != saved.specifications
        or current.labels != saved.labels
        or current.qcut != saved.qcut
        or current.left_closed != saved.left_closed
        or current.include_breaks != saved.include_breaks
        or current.allow_duplicates != saved.allow_duplicates
        or len(current.keys) != len(saved.keys)
    ):
        raise Error("Partition CUT immutable control certificate changed")
    for at in range(len(current.keys)):
        if (
            current.keys[at].name != saved.keys[at].name
            or current.keys[at].dtype != saved.keys[at].dtype
            or current.keys[at].string_hash != saved.keys[at].string_hash
        ):
            raise Error("Partition CUT original logical key metadata changed")
