"""Original AST certificates for observed resident temporal producer phases.

Only the original-entry normalizer issues typed certificates. CPU-equivalent
head pushdown/projection pruning precede certification; later structural
rewrites must preserve the exact phase or use a separately verified bridge.
"""
from std.memory import ArcPointer, Pointer, bitcast
from std.sys import size_of
from std.collections import Optional
from dataframe.binding import bind, BoundExpr, ROWS
from dataframe.series import Series
from dataframe.frame import DataFrame
from dataframe.dtype import NUMERIC_DTYPES, DataType
from dataframe.column import Column
from dataframe.bool_column import BoolColumn
from dataframe.string_column import StringColumn
from dataframe.nested_column import ListColumn, StructColumn
from dataframe.expr import (
    Node,
    CAST,
    DT_REPLACE_TZ,
    DT_TRUNCATE,
    DT_OFFSET_BY,
    is_reduction,
    is_window,
)
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    SCAN_FRAME,
    SELECT,
    WITH_COLUMNS,
    _expr_signature,
)
from dataframe.producer_origin import (
    ProducerOrigin,
    Prefix,
    RegionProof,
    bound_signature,
)
from .metadata import schema_frame, prune_projections
from . import iana_mutations, ordered_rows
from .memory import checked_add, checked_mul
from .integer_ranges import proven_integer_nodes

comptime OBSERVED_IANA = 6
comptime OBSERVED_STRICT_CAST = 7
comptime MATERIALIZATION = "metal_original_temporal_producer"
comptime STRICT_CAST_MATERIALIZATION = "metal_original_strict_cast_producer"


def _marker(phase: Int) -> String:
    return (
        STRICT_CAST_MATERIALIZATION if phase
        == OBSERVED_STRICT_CAST else MATERIALIZATION
    )


def _strict_cast(bound: BoundExpr) -> Bool:
    var proven = proven_integer_nodes(bound)
    for at in range(len(bound.expr._nodes)):
        ref node = bound.expr._nodes[at]
        if (
            node.op == CAST
            and node.integer
            and not proven[at]
            and not bound.dtypes[at].is_float()
        ):
            return True
    return False


def _bound_bytes(bound: BoundExpr) raises -> Int:
    var retained = size_of[BoundExpr]()
    retained = checked_add(
        retained, checked_mul(bound.expr._nodes.capacity(), size_of[Node]())
    )
    retained = checked_add(
        retained, checked_mul(bound.dtypes.capacity(), size_of[DataType]())
    )
    retained = checked_add(
        retained,
        checked_mul(
            bound.shapes.capacity() + bound.sources.capacity(), size_of[Int]()
        ),
    )
    retained = checked_add(
        retained,
        checked_mul(
            bound.aggregated.capacity() + bound.fusible.capacity(),
            size_of[Bool](),
        ),
    )
    retained = checked_add(
        retained, checked_mul(bound.expr._name.byte_length() + 32, 2)
    )
    for node in bound.expr._nodes:
        retained = checked_add(
            retained,
            checked_mul(node.dtypes.capacity(), size_of[Optional[DataType]]()),
        )
        retained = checked_add(
            retained,
            checked_mul(
                node.text.byte_length() + node.text2.byte_length() + 64, 2
            ),
        )
    return retained


def _ancestors(query: LazyFrame, root: Int) raises -> List[Int]:
    var result = List[Int]()
    var seen = List[Bool](length=len(query._nodes), fill=False)
    var pending = List[Int]([root])
    while len(pending):
        var index = pending.pop()
        if index < 0 or index >= len(query._nodes):
            raise Error("Original producer prefix has an invalid edge")
        if seen[index]:
            continue
        seen[index] = True
        result.append(index)
        ref node = query._nodes[index]
        for child in [node.right, node.left]:
            if child >= index:
                raise Error("Original producer prefix is not topological")
            if child >= 0:
                pending.append(child)
    return result^


def _field(text: String) -> String:
    return String(text.byte_length()) + ":" + text


def _source_layout(column: Series) raises -> String:
    var result = _field(column.name()) + _field(column.dtype().name())
    result += (
        ";rows=" + String(len(column)) + ";chunks=" + String(column.n_chunks())
    )
    if column.dtype().is_categorical():
        result += ";dictionary=" + String(
            Int(Pointer(to=column.dtype().dictionary()[]))
        )
    if column._field_metadata:
        result += ";field=" + String(
            Int(Pointer(to=column._field_metadata.value()[]))
        )
    if column.is_chunked():
        result += ";chunk-owner=" + String(
            Int(Pointer(to=column._chunked.value()[]))
        )
        for part in column.chunks():
            result += _field(_source_layout(part))
        return result^
    comptime for at in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[at]
        if column._data.isa[Column[Scalar[D]]]():
            ref data = column._data[Column[Scalar[D]]]
            return (
                result
                + ";values="
                + String(Int(Pointer(to=data._data[])))
                + ";valid="
                + String(Int(Pointer(to=data._bits[])))
                + ";offset="
                + String(data._offset)
                + ";length="
                + String(data._length)
            )
    if column._data.isa[Column[Int128]]():
        ref data = column._data[Column[Int128]]
        return (
            result
            + ";values="
            + String(Int(Pointer(to=data._data[])))
            + ";valid="
            + String(Int(Pointer(to=data._bits[])))
            + ";offset="
            + String(data._offset)
            + ";length="
            + String(data._length)
        )
    if column._data.isa[ListColumn]():
        ref data = column._data[ListColumn]
        return (
            result
            + ";child="
            + String(Int(Pointer(to=data._child[])))
            + ";offsets="
            + String(Int(Pointer(to=data._offsets[])))
            + ";valid="
            + String(Int(Pointer(to=data._bits[])))
            + ";offset="
            + String(data._offset)
            + ";length="
            + String(data._length)
        )
    if column._data.isa[StructColumn]():
        ref data = column._data[StructColumn]
        return (
            result
            + ";fields="
            + String(Int(Pointer(to=data._fields[])))
            + ";valid="
            + String(Int(Pointer(to=data._bits[])))
            + ";offset="
            + String(data._offset)
            + ";length="
            + String(data._length)
        )
    if column._data.isa[BoolColumn]():
        ref data = column._data[BoolColumn]
        return (
            result
            + ";values="
            + String(Int(Pointer(to=data._data[])))
            + ";valid="
            + String(Int(Pointer(to=data._bits[])))
            + ";offset="
            + String(data._offset)
            + ";length="
            + String(data._length)
        )
    if column._data.isa[StringColumn]():
        ref data = column._data[StringColumn]
        result += (
            ";bytes="
            + String(Int(Pointer(to=data._bytes[])))
            + ";offsets="
            + String(Int(Pointer(to=data._offsets[])))
            + ";valid="
            + String(Int(Pointer(to=data._bits[])))
            + ";offset="
            + String(data._offset)
            + ";length="
            + String(data._length)
        )
        if data._view_storage:
            ref views = data._view_storage.value()
            result += (
                ";views="
                + String(Int(Pointer(to=views._views[])))
                + ";buffers="
                + String(Int(Pointer(to=views._buffers[])))
                + ";view-valid="
                + String(Int(Pointer(to=views._bits[])))
            )
        if data._codes:
            result += ";codes=" + String(Int(Pointer(to=data._codes.value()[])))
        return result^
    raise Error(
        "Original producer source has unsupported immutable owner metadata"
    )


def _append_names(mut result: String, names: List[String]):
    # Borrow each plane directly; preserve its field order and trailing marker.
    for name in names:
        result += _field(name)
    result += ";"


def _controls(node: PlanNode) -> String:
    var result = String("")
    if node._executions:
        result += ";counter=" + String(
            Int(Pointer(to=node._executions.value()[]))
        )
    if node.asof_tolerance:
        ref tolerance = node.asof_tolerance.value()
        result += (
            ";tolerance="
            + _field(tolerance.dtype().name())
            + String(tolerance._valid)
            + ","
            + String(tolerance._int)
            + ","
            + String(bitcast[DType.uint64](tolerance._float))
            + ","
            + String(tolerance._bool)
            + _field(tolerance._string)
        )
    for key in node.window_keys:
        result += (
            ";window="
            + _field(key.name)
            + _field(key.dtype.name())
            + String(key.string_hash)
            + ","
            + String(Int(Pointer(to=key.values[])))
        )
        if key.dtype.is_categorical():
            result += "," + String(Int(Pointer(to=key.dtype.dictionary()[])))
    return result^


def _phase_geometry(node: PlanNode) -> String:
    var result = (
        String(node.kind)
        + ","
        + String(node.right)
        + ","
        + String(node.offset)
        + ","
        + String(node.length)
        + ","
        + String(node.maintain_order)
        + ","
        + String(node.how)
        + ","
        + String(node.coalesce)
        + _field(node.source_failure)
    )
    _append_names(result, node.names)
    _append_names(result, node.names2)
    _append_names(result, node.right_keys)
    for flag in node.flags:
        result += String(flag) + ","
    result += _controls(node)
    if node.code_domain:
        var dtype = node.code_domain.value()
        result += _field(dtype.name())
        if dtype.is_categorical():
            result += String(Int(Pointer(to=dtype.dictionary()[])))
    for field in node.text_fields:
        result += _field(field.name) + _field(field.dtype.name())
        if field.dtype.is_categorical():
            result += String(Int(Pointer(to=field.dtype.dictionary()[])))
    return result^


def _prefix_signature(query: LazyFrame, index: Int) raises -> String:
    ref node = query._nodes[index]
    var result = (
        String(node.kind)
        + ","
        + String(node.left)
        + ","
        + String(node.right)
        + ","
        + String(node.offset)
        + ","
        + String(node.length)
        + ","
        + String(node.maintain_order)
        + ","
        + String(node.how)
        + ","
        + String(node.coalesce)
        + ";"
        + _field(node.source_failure)
        + _field(node.text)
    )
    _append_names(result, node.names)
    _append_names(result, node.names2)
    _append_names(result, node.right_keys)
    for flag in node.flags:
        result += String(flag) + ","
    for expression in node.exprs:
        result += _field(_expr_signature(expression))
        for occurrence in expression._nodes:
            result += ";bits=" + String(
                bitcast[DType.uint64](occurrence.floating)
            )
            for dtype in occurrence.dtypes:
                if dtype and dtype.value().is_categorical():
                    result += ";dictionary=" + String(
                        Int(Pointer(to=dtype.value().dictionary()[]))
                    )
    result += _controls(node)
    if node.code_domain:
        var dtype = node.code_domain.value()
        result += _field(dtype.name())
        if dtype.is_categorical():
            result += String(Int(Pointer(to=dtype.dictionary()[])))
    for field in node.text_fields:
        result += _field(field.name) + _field(field.dtype.name())
        if field.dtype.is_categorical():
            result += String(Int(Pointer(to=field.dtype.dictionary()[])))
    if node.kind == SCAN_FRAME:
        if node.offset < 0 or node.offset >= len(query._frames):
            raise Error("Original producer source frame is invalid")
        ref frame = query._frames[node.offset]
        result += ";rows=" + String(frame.height())
        for column in frame._columns:
            result += _field(_source_layout(column))
    return result^


def has_certified(query: LazyFrame) -> Bool:
    for node in query._nodes:
        if node.producer_origin:
            return True
    return False


def validate_prefix(query: LazyFrame, index: Int) raises:
    ref node = query._nodes[index]
    if not node.producer_origin:
        raise Error("Original producer typed provenance is missing")
    var origin = node.producer_origin.value()
    if (
        node.text != _marker(origin[].phase)
        or _phase_geometry(node) != origin[].phase_geometry
        or not node.lineage_anchor
        or Int(Pointer(to=node.lineage_anchor.value()[]))
        != Int(Pointer(to=origin[].phase_anchor[]))
    ):
        raise Error("Original producer phase geometry changed")
    var ancestors = _ancestors(query, node.left)
    if len(ancestors) != len(origin[].prefix):
        raise Error("Original producer prefix cardinality changed")
    for at in range(len(ancestors)):
        ref prefix = query._nodes[ancestors[at]]
        if (
            not prefix.lineage_anchor
            or Int(Pointer(to=prefix.lineage_anchor.value()[]))
            != Int(Pointer(to=origin[].prefix[at].anchor[]))
            or _prefix_signature(query, ancestors[at])
            != origin[].prefix[at].signature
        ):
            raise Error("Original producer source/prefix lineage changed")
    var input = schema_frame(query, node.left)
    if input.width() != len(origin[].input_names):
        raise Error("Original producer input schema cardinality changed")
    for column in range(input.width()):
        if (
            input._columns[column].name() != origin[].input_names[column]
            or input._columns[column].dtype().name()
            != origin[].input_types[column]
        ):
            raise Error("Original producer input logical schema changed")
    if len(node.exprs) != len(origin[].signatures):
        raise Error("Original producer phase expression count changed")
    for at in range(len(node.exprs)):
        validate_bound(node, bind(node.exprs[at], input._columns), at)


def validate_bound(node: PlanNode, bound: BoundExpr, index: Int) raises:
    if not node.producer_origin:
        raise Error("Original producer typed provenance is missing")
    var origin = node.producer_origin.value()
    if (
        node.text != _marker(origin[].phase)
        or _phase_geometry(node) != origin[].phase_geometry
    ):
        raise Error("Original producer phase geometry changed")
    if (
        index < 0
        or index >= len(origin[].signatures)
        or bound_signature(bound) != origin[].signatures[index]
        or bound.shape() != ROWS
    ):
        raise Error("Original producer AST or logical metadata changed")


def prepare_original(
    query: LazyFrame, optimize: Bool, original_entry: Bool
) raises -> LazyFrame:
    if not original_entry:
        raise Error(
            "Only original entry may issue temporal producer provenance"
        )
    var result = query.copy()
    if has_certified(result):
        raise Error("Original entry already carries a private producer token")
    # These are explicit CPU-equivalent bridges, performed before recording
    # the phase; no later generic subset-of-signatures admission is allowed.
    if optimize:
        result._push_slices()
        prune_projections(result)
    for node in result._nodes:
        if node.kind == SCAN_FRAME and (len(node.names2) or node.code_domain):
            raise Error(
                "Private encoded scans cannot claim original temporal provenance"
            )
    for node in result._nodes:
        if node.lineage_anchor or node.producer_region:
            raise Error(
                "Original entry contains private producer lineage metadata"
            )
    for at in range(len(result._nodes)):
        ref operation = result._nodes[at]
        if (
            (operation.kind != SELECT and operation.kind != WITH_COLUMNS)
            or (
                operation.text.byte_length()
                and operation.text != "metal_integer_materialization"
                and operation.text != "metal_decimal_materialization"
            )
            or (len(operation.names2) and not operation.text)
        ):
            continue
        var input = schema_frame(result, operation.left)
        var bounds = List[BoundExpr]()
        var signatures = List[String]()
        var temporal = False
        var strict_cast = False
        var eligible = True
        for expression in operation.exprs:
            var bound = bind(expression, input._columns)
            eligible &= bound.shape() == ROWS
            temporal |= iana_mutations.contains(bound)
            strict_cast |= _strict_cast(bound)
            for node in bound.expr._nodes:
                eligible &= not is_reduction(node.op) and not is_window(node.op)
            eligible &= ordered_rows.eligible(bound, allow_iana=True)
            signatures.append(bound_signature(bound))
            bounds.append(bound^)
        # Preserve the established direct-original mode5 route.
        var direct = (
            len(result._nodes) == 2
            and at == 1
            and operation.kind == SELECT
            and operation.left == 0
            and result._nodes[0].kind == SCAN_FRAME
        )
        if not eligible or not (temporal or strict_cast) or direct:
            continue
        # The original-entry bridge may already have marked this exact row
        # projection as checked integer/decimal work. Its typed certificate
        # supersedes that marker before retaining any prefix signatures.
        if operation.text and not strict_cast:
            continue
        if operation.text:
            operation.text = ""
            operation.names2.clear()
        operation.lineage_anchor = ArcPointer(Int(0))
        var prefix = List[Prefix]()
        var sources = List[Series]()
        var retained = size_of[ProducerOrigin]()
        for ancestor in _ancestors(result, operation.left):
            if not result._nodes[ancestor].lineage_anchor:
                result._nodes[ancestor].lineage_anchor = ArcPointer(Int(0))
            if (
                result._nodes[ancestor].left < 0
                and result._nodes[ancestor].kind != SCAN_FRAME
            ):
                raise Error(
                    "Temporal producer requires in-memory original sources"
                )
            if result._nodes[ancestor].kind == SCAN_FRAME:
                for column in result._frames[
                    result._nodes[ancestor].offset
                ]._columns:
                    sources.append(column.copy())
            var signature = _prefix_signature(result, ancestor)
            retained = checked_add(
                retained, checked_mul(signature.byte_length() + 32, 2)
            )
            prefix.append(
                Prefix(
                    result._nodes[ancestor].lineage_anchor.value(), signature^
                )
            )
        var names = List[String]()
        var types = List[String]()
        for column in input._columns:
            names.append(column.name())
            types.append(column.dtype().name())
        for signature in signatures:
            retained = checked_add(
                retained, checked_mul(signature.byte_length() + 32, 2)
            )
        retained = checked_add(
            retained, checked_mul(bounds.capacity(), size_of[BoundExpr]())
        )
        retained = checked_add(
            retained, checked_mul(prefix.capacity(), size_of[Prefix]() + 8)
        )
        retained = checked_add(retained, 8)
        retained = checked_add(
            retained, checked_mul(sources.capacity(), size_of[Series]())
        )
        retained = checked_add(
            retained,
            checked_mul(
                names.capacity() + types.capacity() + signatures.capacity(),
                size_of[String](),
            ),
        )
        for bound in bounds:
            retained = checked_add(retained, _bound_bytes(bound))
        for name in names:
            retained = checked_add(
                retained, checked_mul(name.byte_length() + 32, 2)
            )
        for dtype in types:
            retained = checked_add(
                retained, checked_mul(dtype.byte_length() + 32, 2)
            )
        var geometry = _phase_geometry(operation)
        retained = checked_add(
            retained, checked_mul(geometry.byte_length() + 32, 2)
        )
        var phase = OBSERVED_IANA if temporal else OBSERVED_STRICT_CAST
        operation.producer_origin = ArcPointer(
            ProducerOrigin(
                phase,
                geometry,
                operation.lineage_anchor.value(),
                bounds^,
                signatures^,
                prefix^,
                sources^,
                names^,
                types^,
                "",
                retained,
            )
        )
        operation.text = _marker(phase)
    return result^


def bind_region(
    query: LazyFrame, index: Int, frame: DataFrame, source_task: Int
) raises -> ArcPointer[RegionProof]:
    # This is the explicit outer-DAG -> isolated-region bridge. Re-check the
    # complete prefix while original anchors and immutable sources exist.
    validate_prefix(query, index)
    var origin = query._nodes[index].producer_origin.value()
    if frame.width() != len(origin[].input_names):
        raise Error(
            "Observed temporal producer bridge input cardinality changed"
        )
    for at in range(frame.width()):
        if (
            frame._columns[at].name() != origin[].input_names[at]
            or frame._columns[at].dtype().name() != origin[].input_types[at]
        ):
            raise Error(
                "Observed temporal producer bridge logical input changed"
            )
    if source_task < 0:
        ref scan = query._nodes[query._nodes[index].left]
        if scan.kind != SCAN_FRAME:
            raise Error(
                "Observed temporal producer raw bridge requires its original scan"
            )
        var expected = query._frames[scan.offset].copy()
        if len(scan.names):
            expected = expected.select(scan.names)
        if frame.width() != expected.width():
            raise Error("Observed temporal producer raw bridge input changed")
        for at in range(frame.width()):
            if _source_layout(frame._columns[at]) != _source_layout(
                expected._columns[at]
            ):
                raise Error(
                    "Observed temporal producer raw bridge immutable source changed"
                )
    var names = List[String]()
    var types = List[String]()
    var raw_sources = List[Series]()
    var raw_signature = String("")
    var retained = size_of[RegionProof]()
    for column in frame._columns:
        names.append(column.name())
        types.append(column.dtype().name())
        if source_task < 0:
            var layout = _source_layout(column)
            raw_signature += _field(layout)
            raw_sources.append(column.copy())
        retained = checked_add(
            retained,
            checked_mul(
                column.name().byte_length()
                + column.dtype().name().byte_length()
                + 64,
                2,
            ),
        )
    retained = checked_add(
        retained,
        checked_mul(names.capacity() + types.capacity(), size_of[String]()),
    )
    retained = checked_add(
        retained, checked_mul(raw_sources.capacity(), size_of[Series]())
    )
    retained = checked_add(
        retained, checked_mul(raw_signature.byte_length() + 32, 2)
    )
    var failure = query._nodes[
        query._nodes[index].left
    ].source_failure if source_task < 0 else String("")
    retained = checked_add(retained, checked_mul(failure.byte_length() + 32, 2))
    return ArcPointer(
        RegionProof(
            origin,
            source_task,
            names^,
            types^,
            raw_sources^,
            raw_signature^,
            failure,
            retained,
        )
    )


def validate_region(query: LazyFrame, node: PlanNode) raises:
    if not node.producer_origin or not node.producer_region:
        raise Error(
            "Observed temporal producer requires a verified DAG region proof"
        )
    var proof = node.producer_region.value()
    if Int(Pointer(to=proof[].origin[])) != Int(
        Pointer(to=node.producer_origin.value()[])
    ):
        raise Error(
            "Observed temporal producer region belongs to another original AST"
        )
    if (
        len(query._nodes) != 2
        or node.left != 0
        or query._nodes[0].kind != SCAN_FRAME
        or query._nodes[0].offset != 0
        or query._nodes[0].source_failure != proof[].source_failure
    ):
        raise Error("Observed temporal producer region geometry changed")
    ref frame = query._frames[0]
    if frame.width() != len(proof[].input_names):
        raise Error(
            "Observed temporal producer region input cardinality changed"
        )
    var signature = String("")
    for at in range(frame.width()):
        ref column = frame._columns[at]
        if (
            column.name() != proof[].input_names[at]
            or column.dtype().name() != proof[].input_types[at]
        ):
            raise Error(
                "Observed temporal producer region logical input changed"
            )
        if proof[].source_task < 0:
            signature += _field(_source_layout(column))
    if proof[].source_task < 0 and signature != proof[].raw_signature:
        raise Error(
            "Observed temporal producer region immutable source changed"
        )


def has_candidates(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        if operation.kind != SELECT and operation.kind != WITH_COLUMNS:
            continue
        for expression in operation.exprs:
            for node in expression._nodes:
                if len(query._nodes) > 2 or operation.kind == WITH_COLUMNS:
                    if (
                        (node.op == CAST and node.integer)
                        or node.op == DT_REPLACE_TZ
                        or node.op == DT_TRUNCATE
                        or node.op == DT_OFFSET_BY
                    ):
                        return True
    return False
