"""Immutable original-expression provenance for private accelerator phases.

Tokens are issued before structural lowering. Bound ASTs and prefix anchors
are retained together; a synthetic scan or copied marker string is not proof.
This metadata module deliberately does not import LazyFrame or GPU owners.
"""
from std.memory import ArcPointer, bitcast
from dataframe.binding import BoundExpr
from dataframe.series import Series


def bound_signature(bound: BoundExpr) -> String:
    var result = String(bound.expr._name.byte_length()) + ":" + bound.expr._name
    for at in range(len(bound.expr._nodes)):
        ref node = bound.expr._nodes[at]
        result += (
            ";"
            + String(node.op)
            + ","
            + String(node.left)
            + ","
            + String(node.right)
            + ","
            + String(node.extra)
            + ","
            + String(node.integer)
            + ","
            + String(bitcast[DType.uint64](node.floating))
            + ","
            + String(node.min_count)
        )
        for text in [node.text, node.text2]:
            result += "," + String(text.byte_length()) + ":" + text
        for dtype in node.dtypes:
            var name = dtype.value().name() if dtype else String("_")
            result += "," + String(name.byte_length()) + ":" + name
        result += (
            ";source="
            + String(bound.sources[at])
            + ";aggregate="
            + String(bound.aggregated[at])
            + ";fusible="
            + String(bound.fusible[at])
            + ";dtype="
            + bound.dtypes[at].name()
            + ";shape="
            + String(bound.shapes[at])
        )
    return result^


@fieldwise_init
struct Prefix(Copyable):
    var anchor: ArcPointer[Int]
    var signature: String


@fieldwise_init
struct ProducerOrigin(Movable):
    var phase: Int
    var phase_geometry: String
    var phase_anchor: ArcPointer[Int]
    var original: List[BoundExpr]
    var signatures: List[String]
    var prefix: List[Prefix]
    var sources: List[Series]
    var input_names: List[String]
    var input_types: List[String]
    var source_failure: String
    var retained_bytes: Int


@fieldwise_init
struct RegionProof(Movable):
    """A DAG-validated original prefix, rebound to one exact region input."""

    var origin: ArcPointer[ProducerOrigin]
    var source_task: Int
    var input_names: List[String]
    var input_types: List[String]
    var raw_sources: List[Series]
    var raw_signature: String
    var source_failure: String
    var retained_bytes: Int
