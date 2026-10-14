"""Nullable source Struct field transport without CPU expression evaluation.

Only fixed-width primitive children are enabled here. A Struct's children
and parent bitmap remain immutable shared source owners. Field extraction
becomes the existing GPU WHEN operation, so parent-null rows discard hidden
child errors before consumers perform checked arithmetic.
"""
from std.collections import Dict
from std.memory import ArcPointer
from dataframe import (
    BoolColumn,
    DataFrame,
    DataType,
    Expr,
    Series,
    StructColumn,
)
from dataframe.expr import (
    COL,
    CAST,
    STRUCT_FIELD,
    STRUCT_PACK,
    Node,
    col,
    when,
    null,
)
from dataframe.binding import bind
from dataframe.identity_metadata import identity_column_source
from dataframe.lazy import SCAN_FRAME, SELECT
from .memory import checked_add, checked_mul
from .selectors import _live_nodes
from dataframe.lazy import LazyFrame


@fieldwise_init
struct FieldMapping(Copyable):
    var struct_name: String
    var field_name: String
    var raw_name: String
    var parent_name: String
    var prototype: Series


@fieldwise_init
struct SourceLayout(Movable):
    var source: DataFrame
    var flat: DataFrame
    var fields: List[FieldMapping]
    var parents: Dict[String, String]


@fieldwise_init
struct Projection(Movable):
    var query: LazyFrame
    var prototypes: List[Series]
    var staging_bytes: Int


def _name(mut used: Dict[String, Bool], serial: Int) -> String:
    var value = "__struct_source_" + String(serial)
    while value in used:
        value += "_"
    used[value] = True
    return value^


def parent_values(column: StructColumn) -> BoolColumn:
    """Expose parent validity as always-valid Bool values, sharing its bits.

    This is transport interpretation, not a CPU expression or child mask.
    Non-byte-aligned slices retain the same source bit offset. An absent
    bitmap needs only the ordinary packed constant-true transport buffer.
    """
    var result = BoolColumn(values=List[UInt8](), bits=List[UInt8](), length=0)
    result._length = len(column)
    if len(column._bits[]):
        result._data = column._bits.copy()
        result._offset = column._offset
    else:
        result._data = ArcPointer(
            List[UInt8](length=(len(column) + 7) // 8, fill=255)
        )
    return result^


def flatten_source(frame: DataFrame) raises -> SourceLayout:
    var used = Dict[String, Bool]()
    for name in frame.columns():
        used[name] = True
    var columns = List[Series]()
    var fields = List[FieldMapping]()
    var parents = Dict[String, String]()
    var serial = 0
    for original in frame._columns:
        if original.dtype().is_list():
            raise Error(
                "Metal unsupported [source struct]: does not transport List columns"
            )
        if not original.dtype().is_struct():
            columns.append(original.copy())
            continue
        var source = (
            original.rechunk() if original.is_chunked() else original.copy()
        )
        var structure = source.struct_column()
        var parent = _name(used, serial)
        serial += 1
        parents[original.name()] = parent
        columns.append(Series(parent, parent_values(structure)))
        for index in range(structure.field_count()):
            var child = structure.field(index)
            if child.dtype().is_nested():
                raise Error(
                    "Metal unsupported [source struct]: requires primitive child fields"
                )
            if (
                child.dtype() == DataType.STRING
                or child.dtype().is_categorical()
                or child.dtype()._is_bytes()
            ):
                raise Error(
                    "Metal unsupported [source struct]: variable-byte/dictionary child transport is pending"
                )
            var raw = _name(used, serial)
            serial += 1
            columns.append(child.renamed(raw))
            fields.append(
                FieldMapping(
                    original.name(),
                    child.name(),
                    raw,
                    parent,
                    child.slice(0, 0),
                )
            )
    return SourceLayout(
        frame.copy(),
        DataFrame(columns^, height=frame.height()),
        fields^,
        parents^,
    )


def field_mapping(
    layout: SourceLayout, name: String, field: String
) raises -> FieldMapping:
    for item in layout.fields:
        if item.struct_name == name and item.field_name == field:
            return item.copy()
    raise Error(
        "Metal unsupported [source struct]: field is unavailable: "
        + name
        + "."
        + field
    )


def field_expression(item: FieldMapping) -> Expr:
    return (
        when(col(item.parent_name))
        .then(col(item.raw_name))
        .otherwise(null(item.prototype.dtype()))
    )


def _append(mut nodes: List[Node], fragment: Expr) -> Int:
    var offset = len(nodes)
    for item in fragment._nodes:
        var node = item.copy()
        if node.left >= 0:
            node.left += offset
        if node.right >= 0:
            node.right += offset
        if node.extra >= 0:
            node.extra += offset
        nodes.append(node^)
    return len(nodes) - 1


def rewrite(layout: SourceLayout, expression: Expr) raises -> Expr:
    """Resolve direct source field access inside ordinary primitive Exprs."""
    var nodes = List[Node]()
    var roots = List[Int](length=len(expression._nodes), fill=-1)
    for index in range(len(expression._nodes)):
        ref original = expression._nodes[index]
        if original.op == COL and original.text in layout.parents:
            continue
        if original.op == STRUCT_PACK:
            raise Error(
                "Metal unsupported [source struct]: intermediate packing is pending"
            )
        if original.op == STRUCT_FIELD:
            ref source = expression._nodes[original.left]
            if source.op != COL or source.text not in layout.parents:
                raise Error(
                    "Metal unsupported [source struct]: field() requires a direct source Struct"
                )
            var item = field_mapping(layout, source.text, original.text)
            roots[index] = _append(nodes, field_expression(item))
            continue
        var node = original.copy()
        for child in [original.left, original.right, original.extra]:
            if child >= 0 and roots[child] < 0:
                raise Error(
                    "Metal unsupported [source struct]: bare Struct consumer is pending"
                )
        if node.left >= 0:
            node.left = roots[node.left]
        if node.right >= 0:
            node.right = roots[node.right]
        if node.extra >= 0:
            node.extra = roots[node.extra]
        roots[index] = len(nodes)
        nodes.append(node^)
    if not len(roots) or roots[len(roots) - 1] < 0:
        raise Error(
            "Metal unsupported [source struct]: bare Struct output needs assembly"
        )
    return Expr(nodes^, expression._name)


def projection(
    layout: SourceLayout, expressions: List[Expr]
) raises -> Projection:
    var rewritten = List[Expr]()
    var prototypes = List[Series]()
    for expression in expressions:
        rewritten.append(rewrite(layout, expression))
        ref root = expression._nodes[len(expression._nodes) - 1]
        var bound = bind(expression, layout.source._columns)
        var field = len(bound.dtypes) - 1
        while bound.expr._nodes[field].op == CAST:
            var child = bound.expr._nodes[field].left
            if (
                child < 0
                or bound.dtypes[field].is_nested()
                or bound.dtypes[field] != bound.dtypes[child]
            ):
                break
            field = child
        ref field_root = bound.expr._nodes[field]
        if field_root.op == STRUCT_FIELD:
            var source = bound.expr._nodes[field_root.left].text
            var item = field_mapping(layout, source, field_root.text)
            prototypes.append(item.prototype.renamed(expression._name))
        elif identity_column_source(bound, len(bound.expr._nodes) - 1) >= 0:
            var source = identity_column_source(
                bound, len(bound.expr._nodes) - 1
            )
            prototypes.append(
                layout.source._columns[source]
                .slice(0, 0)
                .renamed(expression._name)
            )
        else:
            # Only the metadata owner matters for a computed result. Its
            # authoritative dtype remains the normal GPU binder's result.
            prototypes.append(
                Series.full_null(expression._name, DataType.BOOL, 0)
            )
    return Projection(
        layout.flat.lazy().select_exprs(rewritten), prototypes^, 0
    )


def has_source_structs(query: LazyFrame) -> Bool:
    var live = _live_nodes(query)
    for index in range(len(query._nodes)):
        if not live[index] or query._nodes[index].kind != SCAN_FRAME:
            continue
        var offset = query._nodes[index].offset
        if offset < 0 or offset >= len(query._frames):
            continue
        for column in query._frames[offset]._columns:
            if column.dtype().is_struct():
                return True
    return False


def _staging_bytes(frame: DataFrame) raises -> Int:
    """Conservative extra host transport reservation; original buffers share.

    Chunked Struct rechunking can retain new children and transient growth
    buffers. Reserve twice a complete compact primitive payload plus a parent
    bitmap. This is deliberately conservative, not measured device allocation.
    """
    var total = 0
    for column in frame._columns:
        if not column.dtype().is_struct():
            continue
        var rows = len(column)
        var bitmap = checked_add(rows, 7) // 8
        if column.is_chunked():
            var payload = bitmap
            for index in range(column.dtype().field_count()):
                var dtype = column.dtype().field_dtype(index)
                if (
                    dtype.is_nested()
                    or dtype == DataType.STRING
                    or dtype.is_categorical()
                    or dtype._is_bytes()
                ):
                    raise Error(
                        "Metal unsupported [source struct]: fixed-width primitive fields required"
                    )
                var width = (
                    dtype.decimal_width() if dtype.is_decimal() else dtype.physical().bit_width()
                )
                var bytes = bitmap if dtype == DataType.BOOL else checked_mul(
                    rows, max(1, width // 8)
                )
                payload = checked_add(payload, checked_add(bytes, bitmap))
            # Struct concatenation also builds an unpacked parent Bool list.
            total = checked_add(
                total, checked_add(checked_mul(payload, 2), rows)
            )
        elif not len(column.struct_column()._bits[]):
            total = checked_add(total, bitmap)
    return total


def lower_source_structs(
    query: LazyFrame,
    *,
    memory_limit_bytes: Int = -1,
    metadata_only: Bool = False,
) raises -> Projection:
    """Normalize only a direct source Struct row projection; no value evaluation.

    Copy the original plan/scan fields, including source_failure and measured
    source cardinality metadata. File staging remains the outer provider hook.
    """
    if (
        len(query._nodes) != 2
        or query._nodes[0].kind != SCAN_FRAME
        or query._nodes[1].kind != SELECT
        or query._nodes[1].left != 0
    ):
        raise Error(
            "Metal unsupported [source struct]: direct scan and terminal SELECT required"
        )
    ref scan = query._nodes[0]
    if scan.offset < 0 or scan.offset >= len(query._frames):
        raise Error("Metal unsupported [source struct]: invalid source slot")
    var frame = query._frames[scan.offset].copy()
    if len(scan.names):
        frame = frame.select(scan.names)
    var staging = _staging_bytes(frame)
    if memory_limit_bytes >= 0 and staging > memory_limit_bytes:
        raise Error("Metal source struct staging exceeds memory budget")
    # Bind original logical schema before replacing Struct columns. In
    # explanation mode only zero-row views are flattened; no payload copies.
    for expression in query._nodes[1].exprs:
        var bound = bind(expression, frame._columns)
        var root = len(bound.dtypes) - 1
        if bound.dtypes[root].is_nested():
            raise Error(
                "Metal unsupported [source struct]: primitive outputs required"
            )
    var layout = flatten_source(
        frame.clear() if metadata_only else frame.copy()
    )
    var prepared = projection(layout, query._nodes[1].exprs)
    var rewritten = query.copy()
    rewritten._frames[scan.offset] = layout.flat.copy()
    rewritten._nodes[0].names = List[String]()
    rewritten._nodes[1].exprs = prepared.query._nodes[1].exprs.copy()
    return Projection(rewritten^, prepared.prototypes.copy(), staging)


def attach_metadata(frame: DataFrame, plan: Projection) raises -> DataFrame:
    """Attach original child owners to bare fields; payload owners stay shared."""
    if frame.width() != len(plan.prototypes):
        raise Error("Metal source struct output width differs")
    var result = frame.copy()
    for index in range(frame.width()):
        result._columns[index]._field_metadata = plan.prototypes[
            index
        ]._field_metadata
    return result^
