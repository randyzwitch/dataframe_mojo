"""Metadata lowering for direct fixed-width source List len/get consumers."""
from std.collections import Dict
from dataframe import DataFrame, DataType, Expr, Series
from dataframe.binding import bind, ROWS
from dataframe.expr import COL, CAST, LIST_LEN, LIST_GET, Node, col
from dataframe.lazy import LazyFrame, SCAN_FRAME, SELECT
from .metadata import schema_frame
from .selectors import _live_nodes


@fieldwise_init
struct ListConsumer(Copyable):
    var source: String
    var op: Int
    var index: Int64
    var name: String
    var prototype: Series


@fieldwise_init
struct SourceLists(Movable):
    var query: LazyFrame
    var source: DataFrame
    var consumers: List[ListConsumer]
    var ordinary: List[String]
    var schema: DataFrame
    var prototypes: List[Series]
    var source_failure: String


def has_source_lists(query: LazyFrame) -> Bool:
    var live = _live_nodes(query)
    for index in range(len(query._nodes)):
        if live[index] and query._nodes[index].kind == SCAN_FRAME:
            for column in query._frames[query._nodes[index].offset]._columns:
                if column.dtype().is_list():
                    return True
    return False


def fixed(dtype: DataType) -> Bool:
    if (
        dtype.is_nested()
        or dtype == DataType.STRING
        or dtype._is_bytes()
        or dtype.is_categorical()
    ):
        return False
    if dtype.is_decimal() and dtype.decimal_width() == 128:
        return True
    var physical = dtype.physical()
    return (
        physical == DataType.BOOL
        or physical == DataType.INT8
        or physical == DataType.UINT8
        or physical == DataType.INT16
        or physical == DataType.UINT16
        or physical == DataType.INT32
        or physical == DataType.UINT32
        or physical == DataType.INT64
        or physical == DataType.UINT64
        or physical == DataType.FLOAT32
        or physical == DataType.FLOAT64
    )


def _name(mut used: Dict[String, Bool], serial: Int) -> String:
    var name = "__dataframe_list_" + String(serial)
    while name in used:
        name += "_"
    used[name] = True
    return name^


def child_prototype(column: Series) raises -> Series:
    # Inspect only typed immutable owners, never rechunk a List's payload.
    var result = Series.full_null("item", column.dtype().inner(), 0)
    var found = False
    for part in column.chunks():
        var child = part.list_column().child()
        if not found:
            result = child.slice(0, 0)
            found = True
        else:
            result._check_metadata(child)
    return result^


def lower_source_lists(query: LazyFrame) raises -> SourceLists:
    var original = query.copy()
    original._reorder(len(original._nodes) - 1)
    original._compact_scan_slots()
    if (
        len(original._nodes) != 2
        or original._nodes[0].kind != SCAN_FRAME
        or original._nodes[1].kind != SELECT
        or original._nodes[1].left != 0
    ):
        raise Error(
            "Metal unsupported [source list]: direct scan and terminal SELECT required"
        )
    ref scan = original._nodes[0]
    var source = original._frames[scan.offset].copy()
    if len(scan.names):
        source = source.select(scan.names)
    # Original type/name validation occurs before transport schema erasure.
    var expected = schema_frame(original, 1).clear()
    var consumers = List[ListConsumer]()
    var ordinary = List[String]()
    var ordinary_seen = Dict[String, Bool]()
    var used = Dict[String, Bool]()
    for name in source.columns():
        used[name] = True
    for expression in original._nodes[1].exprs:
        used[expression._name] = True
    var rewritten = List[Expr]()
    var prototypes = List[Series]()
    for output in range(len(original._nodes[1].exprs)):
        ref expression = original._nodes[1].exprs[output]
        var bound = bind(expression, source._columns)
        var top = len(bound.expr._nodes) - 1
        if bound.shape() != ROWS or not fixed(bound.dtypes[top]):
            raise Error(
                "Metal unsupported [source list]: primitive row outputs required"
            )
        var nodes = expression._nodes.copy()
        var roots = List[Int](length=len(nodes), fill=-1)
        var emitted = List[Node]()
        for index in range(len(nodes)):
            var node = nodes[index].copy()
            if node.op == COL:
                var column = source.column(node.text)
                if column.dtype().is_list():
                    continue
                if not fixed(column.dtype()):
                    raise Error(
                        "Metal unsupported [source list]: ordinary fields must be fixed-width"
                    )
                if node.text not in ordinary_seen:
                    ordinary_seen[node.text] = True
                    ordinary.append(node.text)
            if node.op == LIST_LEN or node.op == LIST_GET:
                if node.left < 0 or nodes[node.left].op != COL:
                    raise Error(
                        "Metal unsupported [source list]: direct source List required"
                    )
                var source_name = nodes[node.left].text
                var column = source.column(source_name)
                if not column.dtype().is_list() or not fixed(
                    column.dtype().inner()
                ):
                    raise Error(
                        "Metal unsupported [source list]: fixed-width primitive child required"
                    )
                var found = -1
                for candidate in range(len(consumers)):
                    if (
                        consumers[candidate].source == source_name
                        and consumers[candidate].op == node.op
                        and consumers[candidate].index == node.integer
                    ):
                        found = candidate
                        break
                if found < 0:
                    var name = _name(used, len(consumers))
                    var prototype = Series.full_null(name, DataType.INT64, 0)
                    if node.op == LIST_GET:
                        prototype = (
                            column.list_column()
                            .child()
                            .slice(0, 0)
                            .renamed(name)
                        )
                    found = len(consumers)
                    consumers.append(
                        ListConsumer(
                            source_name^,
                            node.op,
                            node.integer,
                            name^,
                            prototype^,
                        )
                    )
                node = col(consumers[found].name)._nodes[0].copy()
            else:
                for child in [node.left, node.right, node.extra]:
                    if child >= 0 and roots[child] < 0:
                        raise Error(
                            "Metal unsupported [source list]: bare or intermediate List consumer"
                        )
                if node.left >= 0:
                    node.left = roots[node.left]
                if node.right >= 0:
                    node.right = roots[node.right]
                if node.extra >= 0:
                    node.extra = roots[node.extra]
            roots[index] = len(emitted)
            emitted.append(node^)
        if roots[top] < 0:
            raise Error("Metal unsupported [source list]: bare List output")
        rewritten.append(Expr(emitted^, expression._name))
        var prototype = expected._columns[output].copy()
        # Match the direct GET/identity metadata owner without evaluating data.
        var owner = top
        while (
            bound.expr._nodes[owner].op == CAST
            and bound.dtypes[owner]
            == bound.dtypes[bound.expr._nodes[owner].left]
        ):
            owner = bound.expr._nodes[owner].left
        ref selected = bound.expr._nodes[owner]
        if (
            selected.op == LIST_GET
            and selected.left >= 0
            and bound.expr._nodes[selected.left].op == COL
        ):
            prototype._field_metadata = child_prototype(
                source.column(bound.expr._nodes[selected.left].text)
            )._field_metadata
        prototypes.append(prototype^)
    var columns = List[Series]()
    for name in ordinary:
        columns.append(source.column(name).slice(0, 0))
    for consumer in consumers:
        columns.append(consumer.prototype.copy())
    var flattened = original.copy()
    flattened._frames[scan.offset] = DataFrame(columns^)
    flattened._nodes[0].names = List[String]()
    flattened._nodes[1].exprs = rewritten^
    return SourceLists(
        flattened^,
        source^,
        consumers^,
        ordinary^,
        expected^,
        prototypes^,
        scan.source_failure,
    )


def attach_metadata(frame: DataFrame, plan: SourceLists) raises -> DataFrame:
    if frame.width() != len(plan.prototypes):
        raise Error("Metal source List output width differs")
    var result = frame.copy()
    for output in range(result.width()):
        result._columns[output]._field_metadata = plan.prototypes[
            output
        ]._field_metadata
    return result^
