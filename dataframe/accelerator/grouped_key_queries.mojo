"""Stage only original key domains after complete logical metadata validation.

Numeric payload ASTs, source origins, row counts and original projection order
are unchanged. No grouping, query operand or source value is evaluated on CPU.
"""
from dataframe import DataFrame, DataType
from dataframe.binding import bind
from dataframe.expr import COL, COUNT, NULL_COUNT, LIST_LEN, LIST_GET
from dataframe.lazy import LazyFrame, SCAN_FRAME, AGG, SELECT, SLICE
from .metadata import schema_frame
from .memory import checked_add
from .window_keys import WindowKey
from .grouped_lists import prepare as prepare_grouped
from .grouped_key_domains import KeyDomain, reservation, FIXED


@fieldwise_init
struct KeyQuery(Movable):
    var query: LazyFrame
    var schema: DataFrame
    var domains: List[KeyDomain]
    var names: List[String]
    var rows: Int
    var staging_bytes: Int
    var key_outputs: List[Int]


def _shape(query: LazyFrame) raises -> Tuple[LazyFrame, Int, Int]:
    var original = query.copy()
    original._reorder(len(original._nodes) - 1)
    original._compact_scan_slots()
    var root = len(original._nodes) - 1
    while root >= 0 and original._nodes[root].kind == SLICE:
        root = original._nodes[root].left
    var output = root
    if root >= 0 and original._nodes[root].kind == SELECT:
        root = original._nodes[root].left
    if (
        root != 1
        or original._nodes[root].kind != AGG
        or original._nodes[root].left != 0
        or original._nodes[0].kind != SCAN_FRAME
    ):
        raise Error(
            "Metal unsupported [grouped implode]: direct raw source and terminal AGG required"
        )
    if original._nodes[0].source_failure:
        raise Error(original._nodes[0].source_failure)
    if len(original._nodes[0].names2):
        raise Error(
            "Metal unsupported [grouped implode]: original raw source provenance required"
        )
    if not len(original._nodes[root].names):
        raise Error(
            "Metal unsupported [grouped implode]: at least one typed key required"
        )
    return (original^, root, output)


def _validate_original_operands(
    query: LazyFrame, group: Int, frame: DataFrame
) raises:
    # A transported key must not make text arithmetic/code-order comparisons
    # appear numeric, or grant occurrence ranks to a rewritten payload AST.
    ref operation = query._nodes[group]
    for expression in operation.exprs:
        var bound = bind(expression, frame._columns)
        var top = len(bound.expr._nodes) - 1
        for at in range(top):
            ref node = bound.expr._nodes[at]
            var dtype = bound.dtypes[at]
            if not (
                dtype == DataType.STRING
                or dtype.is_categorical()
                or dtype._is_bytes()
            ):
                continue
            var mask_count = (
                bound.expr._nodes[top].op == COUNT
                or bound.expr._nodes[top].op == NULL_COUNT
            ) and bound.expr._nodes[top].left == at
            var key = False
            for name in operation.names:
                key |= node.op == COL and node.text == name
            if not (mask_count and key):
                raise Error(
                    "Metal unsupported [grouped implode]: transported text keys may only feed direct count/null_count; primitive List operands required"
                )


def _validate_projection(query: LazyFrame, group: Int, output: Int) raises:
    if group == output:
        return
    var schema = schema_frame(query, group)
    for expression in query._nodes[output].exprs:
        var bound = bind(expression, schema._columns)
        ref top = bound.expr._nodes[len(bound.expr._nodes) - 1]
        if top.op == COL:
            continue
        if (
            (top.op == LIST_LEN or top.op == LIST_GET)
            and top.left >= 0
            and bound.expr._nodes[top.left].op == COL
        ):
            continue
        raise Error(
            "Metal unsupported [grouped implode]: direct grouped outputs or List len/get projection required"
        )


def _replace_provenance(mut values: List[WindowKey], key: WindowKey):
    for index in range(len(values)):
        if values[index].name == key.name:
            values[index] = key.copy()
            return
    values.append(key.copy())


def prepare_keys(
    query: LazyFrame,
    *,
    memory_limit_bytes: Int = -1,
    metadata_only: Bool = False,
    ordered_source: Bool = False,
    eager_feed: Bool = False,
) raises -> KeyQuery:
    var shape = _shape(query)
    var original = shape[0].copy()
    var group = shape[1]
    var schema = schema_frame(original, shape[2]).clear()
    var frame = original._frames[original._nodes[0].offset].copy()
    if len(original._nodes[0].names):
        frame = frame.select(original._nodes[0].names)
    _validate_original_operands(original, group, frame)
    _validate_projection(original, group, shape[2])
    var names = original._nodes[group].names.copy()
    var rows = frame.height()
    # Bind the COMPLETE encoded plan using empty carriers before staging any
    # selected dictionary codes, raw bytes or source-level lookup owners.
    var metadata = original.copy()
    var empty = frame.clear()
    for name in names:
        var domain = KeyDomain(frame.column(name), metadata_only=True)
        if domain.kind != FIXED:
            empty = empty.with_column(domain.encoded.copy())
    metadata._frames[metadata._nodes[0].offset] = empty^
    var bound_plan = prepare_grouped(
        metadata,
        source_rows_override=rows,
        ordered_source=ordered_source,
        eager_feed=eager_feed,
    )
    var key_outputs = List[Int]()
    for consumer in bound_plan.consumers:
        key_outputs.append(
            consumer.source if consumer.source < len(names) else -1
        )
    var empty_only = metadata_only or not rows
    var staging = 0
    if not empty_only:
        for name in names:
            staging = checked_add(staging, reservation(frame.column(name)))
        if memory_limit_bytes >= 0 and staging > memory_limit_bytes:
            raise Error(
                "Metal grouped key staging exceeds memory budget before submission"
            )
    var domains = List[KeyDomain]()
    var staged = original.copy()
    var encoded = frame.clear() if metadata_only else frame.copy()
    var provenance = staged._nodes[group].window_keys.copy()
    for name in names:
        var domain = KeyDomain(frame.column(name), metadata_only=empty_only)
        if domain.kind != FIXED:
            encoded = encoded.with_column(domain.encoded.copy())
            _replace_provenance(provenance, domain.provenance)
        domains.append(domain^)
    staged._frames[staged._nodes[0].offset] = encoded^
    staged._nodes[group].window_keys = provenance^
    return KeyQuery(
        staged^, schema^, domains^, names^, rows, staging, key_outputs^
    )
