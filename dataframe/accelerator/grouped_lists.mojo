"""Bound terminal grouped IMPLODE producers; no CPU query evaluation.

The first integration admits fixed-width primitive keys/children, independent
primitive sibling reductions and serial-group checked producer ordering. Large
checked/stateful bucket scheduling remains an explicit subsequent proof.
"""
from std.collections import Dict
from std.memory import ArcPointer
from dataframe import DataFrame, DataType, Expr, Series, col
from dataframe.frame import _stream_reductions
from dataframe.binding import bind, ROWS
from dataframe.expr import (
    IMPLODE,
    LIST_LEN,
    LIST_GET,
    COL,
    CAST,
    ADD,
    SUB,
    MUL,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    NEG,
    ABS,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    DT_OFFSET_BY,
    DT_TRUNCATE,
    DT_REPLACE_TZ,
    SUM,
    MEAN,
    MIN,
    MAX,
    COUNT,
    LEN,
    NULL_COUNT,
    ARG_MIN,
    ARG_MAX,
    FIRST,
    LAST,
    ANY,
    ALL,
    VAR,
    STD,
    SKEW,
    KURTOSIS,
    is_reduction,
    subtree,
)
from dataframe.lazy import LazyFrame, PlanNode, SCAN_FRAME, AGG, SELECT, SLICE
from .metadata import schema_frame
from .selectors import _live_nodes
from .integer_ranges import proven_integer_nodes
from .rows import RowPlan
from .window_keys import WindowKey
from .grouped_feeds import FeedPlan, prepare_feed
from .gpu.runtime import _preflight
from .grouped_key_queries import KeyQuery
from .opaque_dictionary import ByteDomain


@fieldwise_init
struct GroupedConsumer(Copyable):
    var source: Int
    var kind: Int
    var name: String
    var index: Int64


@fieldwise_init
struct GroupedListsPlan(Movable):
    var source: RowPlan
    var keys: RowPlan
    var producers: List[ArcPointer[RowPlan]]
    var list_outputs: List[Bool]
    var feed_plans: List[Optional[FeedPlan]]
    var schema: DataFrame
    var grouped_schema: DataFrame
    var consumers: List[GroupedConsumer]
    var names: List[String]
    var slices: List[PlanNode]
    var rows: Int
    var root: Int
    var scan: PlanNode
    var scheduled: Bool
    var key_provenance: List[WindowKey]
    var original_keys: Optional[ArcPointer[KeyQuery]]
    var publication_bytes: Optional[ArcPointer[ByteDomain]]


def _retain_key_proofs(mut query: LazyFrame, scan: PlanNode, group: PlanNode):
    for index in range(len(query._nodes)):
        ref node = query._nodes[index]
        node.code_domain = scan.code_domain
        node.text_fields = scan.text_fields.copy()
        if node.kind == AGG:
            node.window_keys = group.window_keys.copy()


def has_grouped_implode(query: LazyFrame) -> Bool:
    var live = _live_nodes(query)
    for index in range(len(query._nodes)):
        if not live[index] or query._nodes[index].kind != AGG:
            continue
        for expression in query._nodes[index].exprs:
            for node in expression._nodes:
                if node.op == IMPLODE:
                    return True
    return False


def _fixed(dtype: DataType) -> Bool:
    return (
        not dtype.is_nested()
        and dtype != DataType.STRING
        and not dtype._is_bytes()
        and not dtype.is_categorical()
        and (
            dtype.physical() != DataType.of(DType.int128) or dtype.is_decimal()
        )
    )


def prepare(
    query: LazyFrame,
    *,
    ordered_source: Bool = False,
    eager_feed: Bool = False,
    source_rows_override: Int = -1,
) raises -> GroupedListsPlan:
    var original = query.copy()
    original._reorder(len(original._nodes) - 1)
    original._compact_scan_slots()
    var root = len(original._nodes) - 1
    var reverse = List[PlanNode]()
    while original._nodes[root].kind == SLICE:
        reverse.append(original._nodes[root].copy())
        root = original._nodes[root].left
    var output_root = root
    var projection = -1
    if original._nodes[root].kind == SELECT:
        projection = root
        root = original._nodes[root].left
    if (
        original._nodes[root].kind != AGG
        or original._nodes[root].left != 0
        or original._nodes[0].kind != SCAN_FRAME
        or root != 1
    ):
        raise Error(
            "Metal unsupported [grouped implode]: direct raw source and terminal AGG required"
        )
    ref scan = original._nodes[0]
    if scan.source_failure:
        raise Error(scan.source_failure)
    if len(scan.names2):
        raise Error(
            "Metal unsupported [grouped implode]: original raw source provenance required"
        )
    ref operation = original._nodes[root]
    var frame = original._frames[scan.offset].copy()
    if len(scan.names):
        frame = frame.select(scan.names)
    if source_rows_override < -1 or source_rows_override > Int(Int32.MAX):
        raise Error("Grouped source row override is invalid")
    if source_rows_override >= 0 and frame.height():
        raise Error("Grouped source row override requires metadata-only source")
    var rows = (
        source_rows_override if source_rows_override >= 0 else frame.height()
    )
    # This validates original CPU names, logical types and aggregate shapes.
    var grouped_schema = schema_frame(original, root).clear()
    var expected = schema_frame(original, output_root).clear()
    var consumers = List[GroupedConsumer]()
    if projection < 0:
        for column in range(grouped_schema.width()):
            ref field = grouped_schema._columns[column]
            consumers.append(
                GroupedConsumer(
                    column, Int(field.dtype().is_list()), field.name(), 0
                )
            )
    else:
        for expression in original._nodes[projection].exprs:
            var bound = bind(expression, grouped_schema._columns)
            ref node = bound.expr._nodes[len(bound.expr._nodes) - 1]
            var name = node.text
            var kind = 0
            if node.op == LIST_LEN or node.op == LIST_GET:
                if node.left < 0 or bound.expr._nodes[node.left].op != COL:
                    raise Error(
                        "Metal unsupported [grouped implode]: direct resident List len/get required"
                    )
                name = bound.expr._nodes[node.left].text
                kind = 2 if node.op == LIST_LEN else 3
            elif node.op != COL:
                raise Error(
                    "Metal unsupported [grouped implode]: direct grouped outputs or List len/get projection required"
                )
            var source = -1
            for column in range(grouped_schema.width()):
                if grouped_schema._columns[column].name() == name:
                    source = column
                    break
            if source < 0:
                raise Error("Grouped IMPLODE projection source is missing")
            if grouped_schema._columns[source].dtype().is_list():
                if not kind:
                    kind = 1
            elif kind:
                raise Error("Grouped IMPLODE len/get source is not a List")
            consumers.append(
                GroupedConsumer(source, kind, expression._name, node.integer)
            )
    var needed = Dict[String, Bool]()
    for name in operation.names:
        needed[name] = True
        if not _fixed(frame.column(name).dtype()):
            raise Error(
                "Metal unsupported [grouped implode]: primitive fixed-width keys required"
            )
    var producers = List[ArcPointer[RowPlan]]()
    var lists = List[Bool]()
    var feeds = List[Optional[FeedPlan]]()
    var scheduled = False
    for expression in operation.exprs:
        var bound = bind(expression, frame._columns)
        var top = len(bound.expr._nodes) - 1
        ref node = bound.expr._nodes[top]
        var list = node.op == IMPLODE
        if not list and node.op not in List[Int](
            [
                SUM,
                MEAN,
                MIN,
                MAX,
                COUNT,
                LEN,
                NULL_COUNT,
                ARG_MIN,
                ARG_MAX,
                FIRST,
                LAST,
                ANY,
                ALL,
                VAR,
                STD,
                SKEW,
                KURTOSIS,
            ]
        ):
            raise Error(
                "Metal unsupported [grouped implode]: primitive aggregate output roots required"
            )
        var checked = 0
        var proven = proven_integer_nodes(bound)
        for at in range(top):
            ref child = bound.expr._nodes[at]
            if is_reduction(child.op) or child.op == IMPLODE:
                raise Error(
                    "Metal unsupported [grouped implode]: state-containing operands need grouped phase proof"
                )
            if not _fixed(bound.dtypes[at]):
                raise Error(
                    "Metal unsupported [grouped implode]: fixed-width primitive operands required"
                )
            if child.op == COL:
                needed[child.text] = True
            if child.op == CAST and child.integer and not proven[at]:
                checked += 1
            elif (
                not proven[at]
                and (
                    bound.dtypes[at].is_integer()
                    or bound.dtypes[at].is_decimal()
                    or bound.dtypes[at].is_temporal()
                )
                and child.op
                in List[Int]([ADD, SUB, MUL, DIV, FLOORDIV, MOD, POW, NEG, ABS])
            ):
                checked += 1
            elif child.op in List[Int](
                [DT_OFFSET_BY, DT_TRUNCATE, DT_REPLACE_TZ]
            ):
                # Some temporal forms wrap while others check intermediate
                # ticks. Conservatively rank the whole family until exact
                # per-op static infallibility is certified.
                checked += 1
            elif child.op in List[Int]([EQ, NE, LT, LE, GT, GE]) and (
                (child.left >= 0 and bound.dtypes[child.left].is_decimal())
                or (child.right >= 0 and bound.dtypes[child.right].is_decimal())
            ):
                checked += 1
        if list:
            if node.left < 0 or bound.shapes[node.left] != ROWS:
                raise Error(
                    "Metal unsupported [grouped implode]: row-shaped List operands required"
                )
            if checked > 1:
                raise Error(
                    "Metal unsupported [grouped implode]: competing checked operand faults need CPU 1024-row proof"
                )
        var feed_plan = Optional[FeedPlan]()
        var whole_state = (
            not list
            and ordered_source
            and rows < 32768
            and (eager_feed or not _stream_reductions(operation.exprs))
            and (node.op == SUM or node.op == MEAN)
            and bound.dtypes[node.left].is_decimal()
        )
        if not list and (checked or whole_state):
            if not eager_feed and _stream_reductions(operation.exprs):
                raise Error(
                    "Metal unsupported [grouped feed]: streaming scalar-only feed sequence is not yet certified"
                )
            # _agg_whole is the exact serial grouped route below 32768 rows.
            # Ranked operands remain private until all batch/feed/finish phases
            # have succeeded. Larger bucket/job routes need their own maps.
            if rows >= 32768:
                raise Error(
                    "Metal unsupported [grouped feed]: outer bucket/job feed sequence is not yet certified"
                )
            feed_plan = prepare_feed(
                expression,
                frame,
                grouped_schema._columns[len(operation.names) + len(producers)],
                rows,
                ordered_source=ordered_source,
            )
        var reduction_fault = (
            not list
            and (node.op == SUM or node.op == MEAN)
            and (
                bound.dtypes[top].is_decimal()
                or bound.dtypes[node.left].physical() == DataType.INT64
                or bound.dtypes[node.left].physical() == DataType.UINT64
            )
        )
        var parallel_association = not list and (
            node.op == MEAN
            or node.op == VAR
            or node.op == STD
            or node.op == SKEW
            or node.op == KURTOSIS
            or (node.op == SUM and bound.dtypes[node.left].is_float())
        )
        if rows >= 32768 and (reduction_fault or parallel_association):
            raise Error(
                "Metal unsupported [grouped implode]: parallel bucket/job fault and state association proof required"
            )
        scheduled = scheduled or (list and checked != 0 and rows >= 32768)
        var metadata = frame.clear().lazy()
        var producer = metadata.select(
            subtree(bound.expr, node.left).alias(expression._name)
        ) if list else metadata.group_by(
            operation.names, maintain_order=True
        ).agg(
            expression
        )
        _retain_key_proofs(producer, scan, operation)
        if feed_plan:
            producers.append(feed_plan.value().operand.copy())
        else:
            producers.append(ArcPointer(_preflight(producer, source_rows=rows)))
        feeds.append(feed_plan.copy())
        lists.append(list)
    var names = List[String]()
    var columns = List[Expr]()
    for column in frame._columns:
        if column.name() in needed:
            names.append(column.name())
            columns.append(col(column.name()))
    var input_query = frame.lazy().select_exprs(columns)
    _retain_key_proofs(input_query, scan, operation)
    var input = _preflight(input_query, source_rows=source_rows_override)
    var key_query = (
        frame.clear()
        .lazy()
        .group_by(operation.names, maintain_order=True)
        .agg(List[Expr]())
    )
    _retain_key_proofs(key_query, scan, operation)
    var keys = _preflight(key_query, source_rows=rows)
    var slices = List[PlanNode]()
    for index in range(len(reverse)):
        slices.append(reverse[len(reverse) - 1 - index].copy())
    return GroupedListsPlan(
        input^,
        keys^,
        producers^,
        lists^,
        feeds^,
        expected^,
        grouped_schema^,
        consumers^,
        operation.names.copy(),
        slices^,
        rows,
        root,
        scan.copy(),
        scheduled,
        operation.window_keys.copy(),
        operation.grouped_keys.copy(),
        operation.grouped_byte_domain.copy(),
    )
