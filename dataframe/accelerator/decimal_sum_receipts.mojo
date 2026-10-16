"""Immutable original GROUP receipts for infallible Decimal consumers.

Issuance is metadata-only. Authorization is conditional on the DAG executor
successfully observing the exact producer task before its consumer runs.
"""
from std.memory import ArcPointer
from std.sys import size_of
from dataframe import Series, DataFrame
from dataframe.binding import bind, BoundExpr
from dataframe.dtype import DataType
from dataframe.expr import SUM
from dataframe.lazy import LazyFrame, SCAN_FRAME, AGG
from . import decimal_sum_bounds as raw
from . import decimal_group_feed
from .implode_ordered import signature
from .metadata import schema_frame
from .decimal_prefix import prefix_infallible
from .producer_phases import _ancestors, _prefix_signature, _bound_bytes
from .memory import checked_add, checked_mul


@fieldwise_init
struct Origin(Movable):
    var original_root: Int
    var original_input: Int
    var input_rows: Int
    var maintain_order: Bool
    var keys: List[String]
    var original: List[BoundExpr]
    var signatures: List[String]
    var source_owners: List[Series]
    var prefix_signatures: List[String]
    var outputs: List[Series]
    var maxima: List[raw.Magnitude]
    var retained_bytes: Int


@fieldwise_init
struct Receipt(Movable):
    var origin: ArcPointer[Origin]
    var producer_task: Int
    var retained_bytes: Int


def capture(query: LazyFrame, group: Int) raises -> ArcPointer[Origin]:
    """Called only for an original eligible mode7 GROUP, BEFORE pruning."""
    if group < 0 or group >= len(query._nodes):
        raise Error("Decimal SUM bound original GROUP index invalid")
    ref node = query._nodes[group]
    if (
        node.kind != AGG
        or not len(node.names)
        or not prefix_infallible(query, node.left)
    ):
        raise Error(
            "Decimal SUM bound requires an original infallible GROUP input"
        )
    var input = schema_frame(query, node.left)
    var output_schema = schema_frame(query, group)
    var outputs = output_schema._columns.copy()
    var original = List[BoundExpr]()
    var signatures = List[String]()
    var maxima = List[raw.Magnitude](length=len(outputs), fill=raw.unknown())
    var empty = List[raw.Magnitude]()
    for i in range(len(node.exprs)):
        var bound = bind(node.exprs[i], input._columns)
        signatures.append(signature(bound))
        var root = len(bound.expr._nodes) - 1
        if root >= 0 and bound.expr._nodes[root].op == SUM:
            var child = bound.expr._nodes[root].left
            var values = raw.analyze(bound, empty, observed_success=True)
            if child >= 0:
                var maximum = raw.sum_bound(values[0][child], Int(UInt32.MAX))
                var slot = len(node.names) + i
                if slot < len(outputs) and raw.precision_safe(
                    maximum, outputs[slot].dtype()
                ):
                    maxima[slot] = maximum.copy()
        original.append(bound^)
    if not decimal_group_feed.eligible(original, input._columns, node.names):
        raise Error("Decimal SUM bound original GROUP operand proof missing")
    var owners = List[Series]()
    var prefixes = List[String]()
    for at in _ancestors(query, group):
        prefixes.append(_prefix_signature(query, at))
        ref ancestor = query._nodes[at]
        if ancestor.kind == SCAN_FRAME:
            for column in query._frames[ancestor.offset]._columns:
                owners.append(column.copy())
    var keys = node.names.copy()
    var retained = size_of[Origin]() + 8
    retained = checked_add(
        retained, checked_mul(maxima.capacity(), size_of[raw.Magnitude]())
    )
    retained = checked_add(
        retained, checked_mul(original.capacity(), size_of[BoundExpr]())
    )
    retained = checked_add(
        retained,
        checked_mul(owners.capacity() + outputs.capacity(), size_of[Series]()),
    )
    retained = checked_add(
        retained,
        checked_mul(
            keys.capacity() + signatures.capacity() + prefixes.capacity(),
            size_of[String](),
        ),
    )
    for bound in original:
        retained = checked_add(retained, _bound_bytes(bound))
    for text in keys:
        retained = checked_add(
            retained, checked_mul(text.byte_length() + 32, 2)
        )
    for text in signatures:
        retained = checked_add(
            retained, checked_mul(text.byte_length() + 32, 2)
        )
    for text in prefixes:
        retained = checked_add(
            retained, checked_mul(text.byte_length() + 32, 2)
        )
    for column in owners:
        retained = checked_add(
            retained, checked_mul(column.name().byte_length() + 32, 2)
        )
    for column in outputs:
        retained = checked_add(
            retained, checked_mul(column.name().byte_length() + 32, 2)
        )
    return ArcPointer(
        Origin(
            group,
            node.left,
            Int(UInt32.MAX),
            node.maintain_order,
            keys^,
            original^,
            signatures^,
            owners^,
            prefixes^,
            outputs^,
            maxima^,
            retained,
        )
    )


def matches(origin: Origin, query: LazyFrame, group: Int) raises -> Bool:
    if group != origin.original_root:
        return False
    ref node = query._nodes[group]
    if (
        node.left != origin.original_input
        or node.maintain_order != origin.maintain_order
        or node.names != origin.keys
        or len(node.exprs) != len(origin.original)
    ):
        return False
    if node.text != decimal_group_feed.MATERIALIZATION:
        return False
    var input = schema_frame(query, node.left)
    for i in range(len(node.exprs)):
        if (
            signature(bind(node.exprs[i], input._columns))
            != origin.signatures[i]
        ):
            return False
    var outputs = schema_frame(query, group)
    return schema_matches(origin, outputs)


def schema_matches(origin: Origin, frame: DataFrame) -> Bool:
    if frame.width() != len(origin.outputs):
        return False
    for i in range(frame.width()):
        if (
            frame._columns[i].name() != origin.outputs[i].name()
            or frame._columns[i].dtype() != origin.outputs[i].dtype()
        ):
            return False
    return True


def bind_receipt(
    origin: ArcPointer[Origin], producer_task: Int
) -> ArcPointer[Receipt]:
    return ArcPointer(Receipt(origin, producer_task, size_of[Receipt]() + 8))


def validate(
    receipt: Receipt,
    source_task: Int,
    producer_query: LazyFrame,
    consumer_query: LazyFrame,
) raises:
    if source_task != receipt.producer_task or source_task < 0:
        raise Error("Decimal SUM bound producer task lineage differs")
    ref origin = receipt.origin[]
    ref producer = producer_query._nodes[len(producer_query._nodes) - 1]
    if (
        producer.text != decimal_group_feed.MATERIALIZATION
        or producer.names != origin.keys
        or producer.maintain_order != origin.maintain_order
        or len(producer.exprs) != len(origin.original)
    ):
        raise Error("Decimal SUM bound original GROUP differs")
    var input = schema_frame(producer_query, producer.left)
    for i in range(len(producer.exprs)):
        if (
            signature(bind(producer.exprs[i], input._columns))
            != origin.signatures[i]
        ):
            raise Error("Decimal SUM bound original operand differs")
    if not schema_matches(origin, consumer_query._frames[0]):
        raise Error("Decimal SUM bound consumer slot or dtype differs")
