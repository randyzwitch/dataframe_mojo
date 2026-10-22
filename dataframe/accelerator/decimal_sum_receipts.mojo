"""Immutable original GROUP receipts for infallible Decimal consumers.

Issuance is metadata-only. Authorization is conditional on the DAG executor
successfully observing the exact producer task before its consumer runs.
"""
from std.memory import ArcPointer, Pointer
from std.sys import size_of
from dataframe import Series, DataFrame
from dataframe.binding import bind, BoundExpr
from dataframe.dtype import DataType
from dataframe.expr import SUM, COL, Expr, ADD, SUB, MUL, is_comparison
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    SCAN_FRAME,
    AGG,
    SELECT,
    FILTER,
    WITH_COLUMNS,
    _plan_node,
)
from . import decimal_sum_bounds as raw
from . import decimal_group_feed
from .implode_ordered import signature
from .metadata import schema_frame
from .decimal_prefix import prefix_infallible
from .producer_phases import (
    _ancestors,
    _prefix_signature,
    _bound_bytes,
    _source_layout,
)
from .window_keys import WindowKey
from .code_domains import TextField
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
    var prefix_categories: List[String]
    var prefix_indices: List[Int]
    var prefix_nodes: List[PlanNode]
    var source_counts: List[Int]
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
    var categories = List[String]()
    var indices = List[Int]()
    var snapshots = List[PlanNode]()
    var source_counts = List[Int]()
    for at in _ancestors(query, group):
        prefixes.append(_prefix_signature(query, at))
        categories.append(_category_signature(query._nodes[at].text_fields))
        indices.append(at)
        snapshots.append(query._nodes[at].copy())
        source_counts.append(0)
        ref ancestor = query._nodes[at]
        if not ancestor.lineage_anchor:
            raise Error("Decimal SUM bound original occurrence anchor missing")
        if ancestor.kind == SCAN_FRAME:
            source_counts[len(source_counts) - 1] = query._frames[
                ancestor.offset
            ].width()
            for column in query._frames[ancestor.offset]._columns:
                owners.append(column.copy())
    var keys = node.names.copy()
    var retained = size_of[Origin]() + 8
    retained = checked_add(
        retained,
        checked_mul(
            indices.capacity() + source_counts.capacity(), size_of[Int]()
        ),
    )
    retained = checked_add(
        retained, checked_mul(snapshots.capacity(), size_of[PlanNode]())
    )
    for at in indices:
        retained = checked_add(retained, _snapshot_bytes(query, at))
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
            keys.capacity()
            + signatures.capacity()
            + prefixes.capacity()
            + categories.capacity(),
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
    for text in categories:
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
            categories^,
            indices^,
            snapshots^,
            source_counts^,
            outputs^,
            maxima^,
            retained,
        )
    )


def matches(origin: Origin, query: LazyFrame, group: Int) raises -> Bool:
    if not prefix_matches(origin, query, group):
        return False
    ref node = query._nodes[group]
    if (
        node.maintain_order != origin.maintain_order
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
        or not _same_anchor(producer, origin.prefix_nodes[0])
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


def _snapshot_bytes(query: LazyFrame, at: Int) raises -> Int:
    ref node = query._nodes[at]
    var retained = 8  # shared original occurrence anchor
    retained = checked_add(
        retained,
        checked_mul(
            node.exprs.capacity() + node.partition_exprs.capacity(),
            size_of[Expr](),
        ),
    )
    retained = checked_add(
        retained,
        checked_mul(
            node.names.capacity()
            + node.names2.capacity()
            + node.right_keys.capacity(),
            size_of[String](),
        ),
    )
    retained = checked_add(
        retained, checked_mul(node.flags.capacity(), size_of[Bool]())
    )
    retained = checked_add(
        retained, checked_mul(node.window_keys.capacity(), size_of[WindowKey]())
    )
    retained = checked_add(
        retained, checked_mul(node.text_fields.capacity(), size_of[TextField]())
    )
    retained = checked_add(
        retained,
        checked_mul(node.partition_columns.capacity(), size_of[Series]()),
    )
    retained = checked_add(
        retained,
        checked_mul(
            node.text.byte_length() + node.source_failure.byte_length() + 64, 2
        ),
    )
    for name in node.names:
        retained = checked_add(
            retained, checked_mul(name.byte_length() + 32, 2)
        )
    for name in node.names2:
        retained = checked_add(
            retained, checked_mul(name.byte_length() + 32, 2)
        )
    for name in node.right_keys:
        retained = checked_add(
            retained, checked_mul(name.byte_length() + 32, 2)
        )
    for key in node.window_keys:
        retained = checked_add(
            retained, checked_mul(key.name.byte_length() + 32, 2)
        )
    for field in node.text_fields:
        retained = checked_add(
            retained, checked_mul(field.name.byte_length() + 32, 2)
        )
    for column in node.partition_columns:
        retained = checked_add(
            retained, checked_mul(column.name().byte_length() + 32, 2)
        )
    if len(node.exprs) or len(node.partition_exprs):
        var input = schema_frame(query, node.left)
        for expression in node.exprs:
            retained = checked_add(
                retained, _bound_bytes(bind(expression, input._columns))
            )
        for expression in node.partition_exprs:
            retained = checked_add(
                retained, _bound_bytes(bind(expression, node.partition_columns))
            )
    return retained


def _same_anchor(left: PlanNode, right: PlanNode) -> Bool:
    return (
        left.lineage_anchor
        and right.lineage_anchor
        and Int(Pointer(to=left.lineage_anchor.value()[]))
        == Int(Pointer(to=right.lineage_anchor.value()[]))
    )


def _plain_transport(node: PlanNode) -> Bool:
    # Exactly the additional SELECT emitted by prune_projections: no checked
    # expression, alias, source fault, counter, state or alternate input.
    if (
        node.kind != SELECT
        or node.lineage_anchor
        or node.left < 0
        or node.right != -1
        or node.text
        or len(node.names)
        or len(node.names2)
        or len(node.flags)
        or node.source_failure
        or node._executions
        or node.producer_origin
        or node.producer_region
        or node.asof_tolerance
        or len(node.window_keys)
        or len(node.partition_exprs)
        or len(node.partition_columns)
        or not len(node.exprs)
    ):
        return False
    var names = List[String]()
    for expression in node.exprs:
        if len(expression._nodes) != 1:
            return False
        ref part = expression._nodes[0]
        if (
            part.op != COL
            or expression._name != part.text
            or part.text in names
        ):
            return False
        names.append(part.text)
    return True


def _original_edge(origin: Origin, query: LazyFrame, child: Int) raises -> Int:
    var cursor = child
    if cursor == -1:
        return -1
    # Projection pruning inserts unanchored plain column transports and can
    # reorder indices. Strip only those transports, retaining every original
    # checked occurrence and resolving edges by the original anchor identity.
    var remaining = len(query._nodes)
    while cursor >= 0 and cursor < len(query._nodes) and remaining:
        remaining -= 1
        ref node = query._nodes[cursor]
        for at in range(len(origin.prefix_nodes)):
            if _same_anchor(node, origin.prefix_nodes[at]):
                return origin.prefix_indices[at]
        if not _transport_matches(query, cursor):
            return -2
        cursor = node.left
    return -2


def prefix_matches(origin: Origin, query: LazyFrame, group: Int) raises -> Bool:
    if (
        group < 0
        or group >= len(query._nodes)
        or not len(origin.prefix_nodes)
        or not _same_anchor(query._nodes[group], origin.prefix_nodes[0])
    ):
        return False
    var live = _ancestors(query, group)
    # No extra computation can be smuggled in as projection transport.
    for at in live:
        var anchored = False
        for original in origin.prefix_nodes:
            anchored |= _same_anchor(query._nodes[at], original)
        if not anchored and not _transport_matches(query, at):
            return False
    var staged = query.copy()
    var source_slot = 0
    for i in range(len(origin.prefix_nodes)):
        ref original = origin.prefix_nodes[i]
        var current = -1
        for at in live:
            if _same_anchor(query._nodes[at], original):
                if current >= 0:
                    return False
                current = at
        if current < 0:
            return False
        var normalized = query._nodes[current].copy()
        normalized.left = _original_edge(origin, query, normalized.left)
        normalized.right = _original_edge(origin, query, normalized.right)
        if normalized.left == -2 or normalized.right == -2:
            return False
        if original.kind == SCAN_FRAME:
            if normalized.offset < 0 or normalized.offset >= len(query._frames):
                return False
            ref frame = query._frames[normalized.offset]
            if frame.width() != origin.source_counts[i]:
                return False
            # Compare the retained ORIGINAL source owners, offsets, chunks,
            # validity owners and metadata, including unused physical columns.
            # Equal values from a replacement allocation are not provenance.
            for column in frame._columns:
                if source_slot >= len(origin.source_owners) or _source_layout(
                    column
                ) != _source_layout(origin.source_owners[source_slot]):
                    return False
                source_slot += 1
            var allowed = original.names.copy()
            if not len(allowed):
                allowed = frame.columns()
            var previous = -1
            for name in normalized.names:
                var position = -1
                for at in range(len(allowed)):
                    if allowed[at] == name:
                        position = at
                if position <= previous:
                    return False
                previous = position
            if len(original.names) and not len(normalized.names):
                return False
            normalized.names = original.names.copy()
        if i == 0:
            # The observed GROUP marker is checked separately by matches().
            normalized.text = original.text
            normalized.names2 = original.names2.copy()
        staged._nodes[current] = normalized^
        if (
            _prefix_signature(staged, current) != origin.prefix_signatures[i]
            or _category_signature(query._nodes[current].text_fields)
            != origin.prefix_categories[i]
        ):
            return False
    return source_slot == len(origin.source_owners)


def needs_bounds(query: LazyFrame, group: Int) raises -> Bool:
    """Capture only a direct first Decimal arithmetic consumer.

    Pure transports, terminal groups and unrelated integer/float expressions
    do not create anchors or retained provider receipts. A later replacing
    projection is intentionally outside this first-consumer proof.
    """
    var output = schema_frame(query, group)
    for node in query._nodes:
        if node.left != group or node.kind not in [
            SELECT,
            FILTER,
            WITH_COLUMNS,
        ]:
            continue
        for expression in node.exprs:
            var bound = bind(expression, output._columns)
            for at in range(len(bound.expr._nodes)):
                ref part = bound.expr._nodes[at]
                if part.op in [ADD, SUB, MUL] and bound.dtypes[at].is_decimal():
                    return True
                if (
                    is_comparison(part.op)
                    and part.left >= 0
                    and bound.dtypes[part.left].is_decimal()
                ):
                    return True
    return False


def _transport_matches(query: LazyFrame, at: Int) raises -> Bool:
    ref node = query._nodes[at]
    if not _plain_transport(node) or node.left >= len(query._nodes):
        return False
    var fields = List[TextField]()
    var names = List[String]()
    for expression in node.exprs:
        names.append(expression._nodes[0].text)
    for field in query._nodes[node.left].text_fields:
        if field.name in names:
            fields.append(field.copy())
    # Replay only the documented pruning adapter, from the guarded child:
    # exact column ASTs plus unchanged selected text owners/code domain and
    # default controls. Metadata cannot introduce a new interpretation.
    var expected = _plan_node(
        SELECT,
        node.left,
        exprs=node.exprs,
        code_domain=query._nodes[node.left].code_domain,
        text_fields=fields,
    )
    var staged = query.copy()
    staged._nodes[at] = expected^
    return _prefix_signature(query, at) == _prefix_signature(
        staged, at
    ) and _category_signature(node.text_fields) == _category_signature(
        staged._nodes[at].text_fields
    )


def _category_signature(fields: List[TextField]) -> String:
    """Snapshot private code interpretation omitted by the shared signature.

    Includes immutable original dictionary/code-buffer identities, offsets and
    validity owners. No dictionary labels or source code payloads are read.
    """
    var result = String(len(fields)) + ";"
    for field in fields:
        result += String(field.name.byte_length()) + ":" + field.name
        result += ";category=" + String(Bool(field.category))
        if not field.category:
            continue
        ref category = field.category.value()
        var owner = category.owner.copy()
        result += ";string_codes=" + String(category.string_codes)
        result += ";owner=" + String(Int(Pointer(to=owner[])))
        var dtype_name = owner[].dtype.name()
        result += (
            ";dtype=" + String(dtype_name.byte_length()) + ":" + dtype_name
        )
        if owner[].dtype.is_categorical():
            result += ";dictionary=" + String(
                Int(Pointer(to=owner[].dtype.dictionary()[]))
            )
        result += ";codes=" + String(len(owner[].codes))
        for code in owner[].codes:
            result += ";values=" + String(Int(Pointer(to=code._data[])))
            result += ";valid=" + String(Int(Pointer(to=code._bits[])))
            result += (
                ";offset="
                + String(code._offset)
                + ";length="
                + String(code._length)
            )
        result += ";"
    return result^
