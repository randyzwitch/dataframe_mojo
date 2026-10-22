"""Explicit CUT intervals through observed resident operands and text codes.

No source values are evaluated on the host. Literal label formatting reuses
CPU metadata formatting; every operand/output is an ordinary GPU row region.
QCUT and intermediate/partitioned Struct consumers are separate increments.
"""
from std.collections import Dict
from dataframe import Column, CsvSchema, DataFrame, DataType, Expr, Series
from dataframe.binding import BoundExpr, bind, ROWS
from dataframe.expr import (
    CUT,
    QCUT,
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_STRING,
    LIT_NULL,
    OVER,
    STRUCT_FIELD,
    STRUCT_PACK,
    SEP,
    Node,
    _node,
    col,
    lit,
    null,
    when,
    as_struct,
    subtree,
    split_names,
    is_reduction,
    is_string_op,
    STR_CONCAT,
)
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    SELECT,
    WITH_COLUMNS,
    FILTER,
    DROP,
    SCAN_FRAME,
    JOIN,
    _plan_node,
    _references,
)
from dataframe.join_type import JOIN_CROSS
from dataframe.parse import parse_float64
from dataframe.window import _bin_label
from .memory import checked_add
from .metadata import schema_frame
from .selectors import _live_nodes
from .aggregates import _in_memory_sources
from .qcuts import QCUT_TAG, stage_custom as stage_custom_qcut
from .partition_cuts import (
    TAG as PARTITION_CUT_TAG,
    stage as stage_partition_cut,
)

comptime CUT_TAG = "metal_explicit_cut_operand"
comptime CUT_ANCHOR_TAG = "metal_explicit_cut_scalar_anchor"
comptime CUT_ORDERED_SCALAR_TAG = "ordered_scalar_v1"


def has_cut_operations(query: LazyFrame) -> Bool:
    var live = _live_nodes(query)
    for index in range(len(query._nodes)):
        if not live[index]:
            continue
        for expression in query._nodes[index].exprs:
            for node in expression._nodes:
                if node.op == CUT or node.op == QCUT:
                    return True
    return False


def has_cut_materializations(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        if (
            operation.text == "metal_window_materialization"
            and len(operation.names2) >= 1
            and (
                operation.names2[0] == CUT_TAG
                or operation.names2[0] == QCUT_TAG
                or operation.names2[0] == PARTITION_CUT_TAG
            )
        ):
            return True
    return False


def staging_bytes(query: LazyFrame) raises -> Int:
    """One immutable Int8 literal payload per private scalar anchor source."""
    var seen = Dict[Int, Bool]()
    var total = 0
    for operation in query._nodes:
        if (
            operation.kind != SCAN_FRAME
            or len(operation.names2) != 1
            or operation.names2[0] != CUT_ANCHOR_TAG
        ):
            continue
        if operation.offset in seen:
            continue
        seen[operation.offset] = True
        ref frame = query._frames[operation.offset]
        if (
            frame.height() != 1
            or frame.width() != 1
            or frame._columns[0].dtype() != DataType.INT8
        ):
            raise Error("Metal CUT scalar anchor metadata differs")
        ref column = frame._columns[0]._data[Column[Int8]]
        total = checked_add(total, len(column._data[]))
        total = checked_add(total, len(column._bits[]))
    return total


def control_failure(query: LazyFrame) -> String:
    """A private CUT producer's failure is observed after its operand faults.

    The existing field is a scan decode failure only on SCAN nodes. On this
    private non-scan marker it is deferred control metadata, never a source
    decode error and never evaluated by CPU schema probes.
    """
    for operation in query._nodes:
        if (
            operation.text == "metal_window_materialization"
            and len(operation.names2) >= 1
            and operation.names2[0] == CUT_TAG
            and operation.source_failure
        ):
            return operation.source_failure
    return String()


def _name(mut used: Dict[String, Bool], serial: Int) -> String:
    var name = "__dataframe_cut_" + String(serial)
    while name in used:
        name += "_"
    used[name] = True
    return name^


def _append(mut nodes: List[Node], expression: Expr) -> Int:
    var start = len(nodes)
    for original in expression._nodes:
        var node = original.copy()
        if node.left >= 0:
            node.left += start
        if node.right >= 0:
            node.right += start
        if node.extra >= 0:
            node.extra += start
        nodes.append(node^)
    return len(nodes) - 1


def _controls(node: Node) raises -> Tuple[List[Float64], List[String], String]:
    var edges = List[Float64]()
    if node.text.byte_length():
        for value in node.text.split(SEP):
            edges.append(parse_float64(String(value)))
    var labels = List[String]()
    if node.text2.byte_length():
        labels = split_names(node.text2)
    else:
        for bin in range(len(edges) + 1):
            labels.append(_bin_label(edges, bin, (node.integer & 1) != 0))
    var failure = String()
    for index in range(1, len(edges)):
        if not edges[index] > edges[index - 1]:
            failure = "breaks must be unique and strictly increasing"
            break
    return (edges^, labels^, failure^)


def _values(
    name: String, node: Node, edges: List[Float64], labels: List[String]
) raises -> Expr:
    var category = lit(labels[len(labels) - 1])
    var breakpoint = lit(Float64(1) / Float64(0))
    for reverse in range(len(edges)):
        var index = len(edges) - 1 - reverse
        var cross = (
            col(name)
            >= lit(edges[index]) if (node.integer & 1) else col(name)
            > lit(edges[index])
        )
        category = when(cross).then(category).otherwise(lit(labels[index]))
        breakpoint = when(cross).then(breakpoint).otherwise(lit(edges[index]))
    # Ordered cross comparisons preserve the single-NaN-edge CPU contract.
    category = (
        when(col(name).is_nan()).then(null(DataType.STRING)).otherwise(category)
    )
    category = (
        when(col(name).is_not_null())
        .then(category)
        .otherwise(null(DataType.STRING))
    )
    breakpoint = (
        when(col(name).is_nan())
        .then(null(DataType.FLOAT64))
        .otherwise(breakpoint)
    )
    breakpoint = (
        when(col(name).is_not_null())
        .then(breakpoint)
        .otherwise(null(DataType.FLOAT64))
    )
    if node.integer & 2:
        return as_struct(
            [breakpoint.alias("breakpoint"), category.alias("category")], ""
        )
    return category^


def _scalar_tree(
    bound: BoundExpr, anchor: String, reductions: List[String]
) raises -> Tuple[Expr, List[String]]:
    """Clone occurrences in CPU left/right/extra postorder, retaining ranks.

    Shared checked children are evaluated once per AST occurrence by CPU. A
    topological node-index rank would conflate those visits, so clone each
    occurrence with an iterative traversal. Reduction children are already
    observed eager states and become infallible column leaves. New literal
    anchor wrappers have rank0; only original operator occurrences get ranks.
    """
    for node in bound.expr._nodes:
        if is_string_op(node.op) or node.op == STR_CONCAT:
            raise Error(
                "Metal unsupported [cut]: ordered scalar string lookup rank remapping is pending"
            )
    var pending = List[Tuple[Int, Bool]]()
    pending.append((len(bound.expr._nodes) - 1, False))
    var roots = List[Int]()
    var nodes = List[Node]()
    var ranks = List[String]()
    var next_rank = 1
    while len(pending):
        var visit = pending.pop()
        var original = visit[0]
        var node = bound.expr._nodes[original].copy()
        if not visit[1]:
            if is_reduction(node.op):
                if not reductions[original]:
                    raise Error("Metal CUT scalar reduction owner is missing")
                nodes.append(_node(COL, text=reductions[original]))
                ranks.append("0")
                roots.append(len(nodes) - 1)
                continue
            if (
                node.op == LIT_INT
                or node.op == LIT_FLOAT
                or node.op == LIT_BOOL
                or node.op == LIT_STRING
                or node.op == LIT_NULL
            ):
                # Lift the checked operand itself. Neither constant folding
                # nor the scalar admission guard can cross this COL/WHEN.
                var literal = Expr([node.copy()], "")
                var lifted = (
                    when(col(anchor).is_not_null())
                    .then(literal)
                    .otherwise(null(bound.dtypes[original]))
                )
                var before = len(nodes)
                roots.append(_append(nodes, lifted))
                for _ in range(len(nodes) - before):
                    ranks.append("0")
                continue
            pending.append((original, True))
            if node.extra >= 0:
                pending.append((node.extra, False))
            if node.right >= 0:
                pending.append((node.right, False))
            if node.left >= 0:
                pending.append((node.left, False))
            continue
        if node.extra >= 0:
            node.extra = roots.pop()
        if node.right >= 0:
            node.right = roots.pop()
        if node.left >= 0:
            node.left = roots.pop()
        if next_rank > Int(UInt32.MAX):
            raise Error("Metal CUT scalar visit rank capacity exceeded")
        ranks.append("0" if node.op == COL else String(next_rank))
        next_rank += 1
        nodes.append(node^)
        roots.append(len(nodes) - 1)
    if len(roots) != 1 or len(ranks) != len(nodes):
        raise Error("Metal CUT scalar occurrence traversal differs")
    return (Expr(nodes^, bound.expr._name), ranks^)


def _scalar_sibling(
    mut result: LazyFrame,
    aligned: Int,
    bound: BoundExpr,
    mut used: Dict[String, Bool],
    serial: Int,
) raises -> Tuple[Int, String, Int]:
    """Observe one original scalar expression, then broadcast its one row.

    The one-row anchor is immutable literal transport. Metadata scans clear it,
    so the ROWS-shaped final expression never evaluates unsafe scalar payloads
    during schema discovery. Actual empty inputs still evaluate the scalar
    branch because the LEN control and literal anchor each have one row.
    """
    var next_serial = serial
    var control = _name(used, next_serial)
    next_serial += 1
    # LEN requires a ROWS operand. A private broadcast field carries the
    # prefix dependency without assuming any public source column exists.
    result._nodes.append(
        _plan_node(
            WITH_COLUMNS,
            aligned,
            exprs=[lit(Int8(0)).alias(control)],
        )
    )
    result._nodes.append(
        _plan_node(
            SELECT,
            len(result._nodes) - 1,
            exprs=[col(control).len().alias(control)],
        )
    )
    var dependency = len(result._nodes) - 1
    var anchor = _name(used, next_serial)
    next_serial += 1
    result._frames.append(DataFrame([Series(anchor, Column[Int8]([0]))]))
    result._schemas.append(Optional[CsvSchema]())
    result._nodes.append(
        _plan_node(
            SCAN_FRAME, offset=len(result._frames) - 1, names2=[CUT_ANCHOR_TAG]
        )
    )
    result._nodes.append(
        _plan_node(
            JOIN,
            dependency,
            len(result._nodes) - 1,
            names2=["_right"],
            text="cross",
            how=JOIN_CROSS,
        )
    )
    var scalar_input = len(result._nodes) - 1
    var reductions = List[String](length=len(bound.expr._nodes), fill="")
    for n in range(len(bound.expr._nodes)):
        var node = bound.expr._nodes[n].copy()
        if is_reduction(node.op):
            var name = _name(used, next_serial)
            next_serial += 1
            # CPU prepares reduction states in node order, independently of
            # an outer conditional that may never select the reduction value.
            var leaf = subtree(bound.expr, n).alias(name)
            result._nodes.append(
                _plan_node(
                    SELECT,
                    aligned,
                    exprs=[leaf^],
                    text="metal_window_materialization",
                    names2=[CUT_TAG],
                )
            )
            var producer = len(result._nodes) - 1
            result._nodes.append(
                _plan_node(
                    JOIN,
                    scalar_input,
                    producer,
                    names2=["_right"],
                    text="cross",
                    how=JOIN_CROSS,
                )
            )
            scalar_input = len(result._nodes) - 1
            reductions[n] = name
    var scalar = _scalar_tree(bound, anchor, reductions)
    var value = scalar[0].copy()
    var predicate = col(anchor).is_not_null()
    var metadata: List[String] = [CUT_TAG, CUT_ORDERED_SCALAR_TAG, anchor]
    for _ in range(len(predicate._nodes)):
        metadata.append("0")
    metadata.extend(scalar[1].copy())
    var output = _name(used, next_serial)
    next_serial += 1
    value = (
        when(predicate)
        .then(value)
        .otherwise(null(bound.dtypes[len(bound.dtypes) - 1]))
        .alias(output)
    )
    # The outer typed-null leaf and WHEN are generated and cannot introduce
    # their own checked fault. Keep one rank entry per final expression node.
    metadata.append("0")
    metadata.append("0")
    if len(metadata) != len(value._nodes) + 3:
        raise Error("Metal CUT scalar rank metadata differs")
    result._nodes.append(
        _plan_node(
            SELECT,
            scalar_input,
            exprs=[value^],
            text="metal_window_materialization",
            names2=metadata,
        )
    )
    var produced = len(result._nodes) - 1
    result._nodes.append(
        _plan_node(
            JOIN,
            aligned,
            produced,
            names2=["_right"],
            text="cross",
            how=JOIN_CROSS,
        )
    )
    return (len(result._nodes) - 1, output^, next_serial)


def lower_cuts(query: LazyFrame) raises -> LazyFrame:
    """Expand explicit edges while keeping eager checked producers resident."""
    _in_memory_sources(query)
    var live = _live_nodes(query)
    var used = Dict[String, Bool]()
    for frame in query._frames:
        for name in frame.columns():
            used[name] = True
    for operation in query._nodes:
        for expression in operation.exprs:
            used[expression._name] = True
            for node in expression._nodes:
                if node.op == COL:
                    used[node.text] = True
    var result = query.copy()
    result._nodes = List[PlanNode]()
    var roots = List[Int]()
    var serial = 0
    for index in range(len(query._nodes)):
        var operation = query._nodes[index].copy()
        if operation.left >= 0:
            operation.left = roots[operation.left]
        if operation.right >= 0:
            operation.right = roots[operation.right]
        var changed = False
        if live[index]:
            for expression in operation.exprs:
                for node in expression._nodes:
                    changed |= node.op == CUT or node.op == QCUT
        if not changed:
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        if (
            operation.kind != SELECT
            and operation.kind != WITH_COLUMNS
            and operation.kind != FILTER
        ):
            raise Error(
                "Metal unsupported [cut]: requires row projection/filter consumer"
            )
        var schema = schema_frame(result, operation.left)
        # Bind all siblings before any resident evaluation, as CPU does.
        var bounds = List[BoundExpr]()
        for expression in operation.exprs:
            bounds.append(bind(expression, schema._columns))
        var required = Dict[String, Bool]()
        var all_columns = operation.kind != SELECT
        for expression in operation.exprs:
            var references = _references(expression)
            if not references:
                all_columns = True
            else:
                for name in references.value():
                    required[name] = True
        var columns = List[Expr]()
        for name in schema.columns():
            if all_columns or name in required:
                columns.append(col(name))
        result._nodes.append(
            _plan_node(
                SELECT,
                operation.left,
                exprs=columns,
                text="metal_shared_prefix",
            )
        )
        var aligned = len(result._nodes) - 1
        var outputs = List[Expr]()
        var temporary = List[String]()
        for expression_index in range(len(bounds)):
            ref bound = bounds[expression_index]
            var expression = bound.expr.copy()
            for node in expression._nodes:
                if node.op == STRUCT_PACK or node.op == STRUCT_FIELD:
                    raise Error(
                        "Metal unsupported [cut]: intermediate Struct consumers are pending"
                    )
            if bound.shape() != ROWS:
                if bound.dtypes[len(bound.dtypes) - 1].is_nested():
                    raise Error(
                        "Metal unsupported [cut]: nested scalar sibling transport is pending"
                    )
                var scalar = _scalar_sibling(
                    result, aligned, bound, used, serial
                )
                aligned = scalar[0]
                temporary.append(scalar[1])
                outputs.append(col(scalar[1]).alias(expression._name))
                serial = scalar[2]
                continue
            # CPU prepares OVER states eagerly, skipping their descendants
            # in the outer state walk. Retain original nodes for certificates;
            # never globally observe a partition operand as an ordinary row.
            var inside_over = List[Bool](
                length=len(expression._nodes), fill=False
            )
            for at in range(len(expression._nodes)):
                if expression._nodes[at].op != OVER:
                    continue
                var pending: List[Int] = [expression._nodes[at].left]
                while len(pending):
                    var child = pending.pop()
                    if child < 0 or inside_over[child]:
                        continue
                    inside_over[child] = True
                    ref original_child = expression._nodes[child]
                    if original_child.left >= 0:
                        pending.append(original_child.left)
                    if original_child.right >= 0:
                        pending.append(original_child.right)
                    if original_child.extra >= 0:
                        pending.append(original_child.extra)
            var positions = List[Int](length=len(expression._nodes), fill=-1)
            var rewritten = List[Node]()
            var packed = False
            var packed_value = Optional[Expr]()
            for n in range(len(expression._nodes)):
                var node = expression._nodes[n].copy()
                if inside_over[n]:
                    continue
                if node.op == OVER:
                    ref bins = expression._nodes[node.left]
                    var include_breaks = (
                        bins.op in [CUT, QCUT] and (bins.integer & 2) != 0
                    )
                    if include_breaks and (
                        n != len(expression._nodes) - 1
                        or operation.kind != SELECT
                        or index != len(query._nodes) - 1
                    ):
                        raise Error(
                            "Metal unsupported [cut]: include_breaks requires terminal projection"
                        )
                    var category = _name(used, serial)
                    serial += 1
                    var breakpoint = _name(used, serial)
                    serial += 1
                    temporary.append(category)
                    if include_breaks:
                        temporary.append(breakpoint)
                    var first_marker = len(result._nodes)
                    var replacement = stage_partition_cut(
                        result,
                        aligned,
                        schema,
                        bound,
                        n,
                        category,
                        breakpoint,
                        index,
                        True,
                    )
                    aligned = len(result._nodes) - 1
                    for marker in range(first_marker, len(result._nodes)):
                        for produced in result._nodes[marker].exprs:
                            if produced._name not in temporary:
                                temporary.append(produced._name)
                    if include_breaks:
                        packed_value = replacement^
                        packed = True
                    else:
                        positions[n] = _append(rewritten, replacement)
                elif node.op == CUT or node.op == QCUT:
                    if (node.integer & 2) and (
                        n != len(expression._nodes) - 1
                        or operation.kind != SELECT
                        or index != len(query._nodes) - 1
                    ):
                        raise Error(
                            "Metal unsupported [cut]: include_breaks requires terminal projection; intermediate Struct consumers are pending"
                        )
                    var prepared = Expr(rewritten.copy(), "")
                    var operand = subtree(prepared, positions[node.left]).cast(
                        DataType.FLOAT64
                    )
                    var controls = (List[Float64](), List[String](), String())
                    if node.op == CUT:
                        controls = _controls(node)
                    var name = _name(used, serial)
                    serial += 1
                    temporary.append(name)
                    result._nodes.append(
                        _plan_node(
                            WITH_COLUMNS,
                            aligned,
                            exprs=[operand.alias(name)],
                            text="metal_window_materialization",
                            names2=[CUT_TAG],
                            source_failure=controls[2],
                        )
                    )
                    aligned = len(result._nodes) - 1
                    var replacement: Expr
                    if node.op == CUT:
                        replacement = _values(
                            name, node, controls[0], controls[1]
                        )
                    else:
                        var category = _name(used, serial)
                        serial += 1
                        var breakpoint = _name(used, serial)
                        serial += 1
                        temporary.append(category)
                        if node.integer & 2:
                            temporary.append(breakpoint)
                        replacement = stage_custom_qcut(
                            result, aligned, name, category, breakpoint, node
                        )
                        aligned = len(result._nodes) - 1
                    if node.integer & 2:
                        # STRUCT_PACK child roots live in text2, not left/right;
                        # keep this terminal expression standalone and intact.
                        packed_value = replacement^
                        packed = True
                    else:
                        positions[n] = _append(rewritten, replacement)
                else:
                    if node.left >= 0:
                        node.left = positions[node.left]
                    if node.right >= 0:
                        node.right = positions[node.right]
                    if node.extra >= 0:
                        node.extra = positions[node.extra]
                    positions[n] = len(rewritten)
                    rewritten.append(node^)
            var value: Expr
            if packed:
                value = packed_value.take().alias(expression._name)
            else:
                var rewritten_expression = Expr(rewritten^, expression._name)
                value = subtree(
                    rewritten_expression, len(rewritten_expression._nodes) - 1
                ).alias(expression._name)
            if (
                not packed
                and not bound.dtypes[len(bound.dtypes) - 1].is_nested()
                and bound.shape() == ROWS
            ):
                # Finish siblings in CPU expression order, before evaluating a
                # later CUT's eager state. Final consumers only rename fields.
                var name = _name(used, serial)
                serial += 1
                temporary.append(name)
                result._nodes.append(
                    _plan_node(
                        WITH_COLUMNS,
                        aligned,
                        exprs=[value.alias(name)],
                        text="metal_window_materialization",
                        names2=[CUT_TAG],
                    )
                )
                aligned = len(result._nodes) - 1
                value = col(name).alias(expression._name)
            outputs.append(value^)
        operation.left = aligned
        operation.exprs = outputs^
        var kind = operation.kind
        result._nodes.append(operation^)
        if kind == FILTER or kind == WITH_COLUMNS:
            result._nodes.append(
                _plan_node(DROP, len(result._nodes) - 1, names=temporary)
            )
        roots.append(len(result._nodes) - 1)
    # Only rewritten expressions run in empty schema discovery. Original CUT
    # runtime validation is deferred until the marked operand visit above.
    _ = schema_frame(result, len(result._nodes) - 1)
    return result^
