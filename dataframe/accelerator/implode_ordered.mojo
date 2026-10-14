"""Positive metadata proof for private primitive IMPLODE row producers.

The certificate binds a normalizer-owned producer to its exact CPU-bound AST.
It does not grant original-entry provenance to generic normalized row regions.
"""
from std.memory import bitcast
from dataframe.binding import BoundExpr
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    SCAN_FRAME,
    SELECT,
    WITH_COLUMNS,
    SLICE,
    _expr_signature,
)
from . import ordered_rows

comptime TAG = "metal_implode_ordered_rows_v1"
comptime MATERIALIZATION = "metal_implode_materialization"


def original_source(query: LazyFrame, root: Int) -> Bool:
    """Prove the original producer prefix, not a synthetic region scan."""
    ref select = query._nodes[root]
    if select.kind != SELECT or select.left != 0:
        return False
    ref scan = query._nodes[0]
    if (
        scan.kind != SCAN_FRAME
        or len(scan.names2)
        or scan.source_failure
        or scan.left >= 0
        or scan.right >= 0
    ):
        return False
    for at in range(1, len(query._nodes)):
        if at != root and query._nodes[at].kind != SLICE:
            return False
    return True


def signature(bound: BoundExpr) -> String:
    # The generic expression signature encodes operation payloads, all child
    # edges and names with lengths. Add raw float bits and bound logical dtype
    # and shape, so rebinds cannot silently change the proof's interpretation.
    var result = _expr_signature(bound.expr)
    for at in range(len(bound.expr._nodes)):
        result += (
            ";bits="
            + String(bitcast[DType.uint64](bound.expr._nodes[at].floating))
            + ";dtype="
            + bound.dtypes[at].name()
            + ";shape="
            + String(bound.shapes[at])
        )
    return result^


def certificate(bound: BoundExpr, source_proved: Bool) raises -> List[String]:
    if not ordered_rows.eligible(bound) or not ordered_rows.needed(bound):
        return List[String]()
    if not source_proved:
        raise Error(
            "Metal unsupported [ordered implode]: original raw in-memory source provenance is not proved"
        )
    return [TAG, signature(bound)]


def tagged(node: PlanNode) -> Bool:
    return len(node.names2) > 0 and node.names2[0] == TAG


def validate(node: PlanNode, bound: BoundExpr) raises:
    if (
        node.kind != WITH_COLUMNS
        or node.text != MATERIALIZATION
        or len(node.exprs) != 1
        or len(node.names2) != 2
        or not ordered_rows.eligible(bound)
        or node.names2[1] != signature(bound)
    ):
        raise Error(
            "Metal unsupported [ordered implode]: private producer AST or logical metadata changed"
        )
