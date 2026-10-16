"""Original public partition CUT/QCUT metadata, before private normalization.

This certificate captures binding and controls without reading source values.
It is deliberately separate from admission: the execution scope must also
prove eager OVER state order, shared WindowLayouts, and resident key codes.
"""
from std.memory import bitcast
from dataframe import (
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    lit,
    when,
    as_struct,
)
from dataframe.binding import BoundExpr, bind
from dataframe.expr import CUT, QCUT, OVER, COL, SEP, subtree, split_names
from dataframe.parse import parse_float64, parse_int64
from dataframe.lazy import LazyFrame, WITH_COLUMNS, _plan_node
from .implode_ordered import signature
from .partition_operands import certificate as operand_certificate
from .window_keys import WindowKey


@fieldwise_init
struct Capture(Copyable):
    var original: String
    var over_node: Int
    var operand: Expr
    var operand_signature: String
    var keys: List[WindowKey]
    var key_set: String
    var specifications: List[UInt64]
    var labels: List[String]
    var qcut: Bool
    var left_closed: Bool
    var include_breaks: Bool
    var allow_duplicates: Bool
    var numeric_state: Bool
    var nested: Bool
    var operand_dtype: DataType


def substitute(
    expression: Expr,
    root: Int,
    replacements: List[String],
    *,
    strip_over: Bool = False,
) raises -> Expr:
    from dataframe.expr import Node

    if len(replacements) != len(expression._nodes):
        raise Error("Partition original scope replacement geometry changed")
    var positions = List[Int](length=len(expression._nodes), fill=-1)
    var nodes = List[Node]()
    for at in range(root + 1):
        var node = expression._nodes[at].copy()
        if replacements[at]:
            if node.op != OVER:
                raise Error(
                    "Partition replacement is not an original OVER state"
                )
            positions[at] = len(nodes)
            nodes.append(col(replacements[at])._nodes[0].copy())
            continue
        if strip_over and node.op == OVER:
            positions[at] = positions[node.left]
            continue
        if node.left >= 0:
            node.left = positions[node.left]
        if node.right >= 0:
            node.right = positions[node.right]
        if node.extra >= 0:
            node.extra = positions[node.extra]
        positions[at] = len(nodes)
        nodes.append(node^)
    var rewritten = Expr(nodes^, "")
    return subtree(rewritten, positions[root])


def capture(
    original: BoundExpr, source: DataFrame, over_node: Int
) raises -> Capture:
    """Keep immutable controls and original AST/logical key interpretation.

    Bound siblings must be captured before lowering any occurrence. A key
    that is also a COL anywhere in the original expression uses decoded text
    hashing, matching CPU evaluate's whole-expression column preparation.
    """
    if over_node < 0 or over_node >= len(original.expr._nodes):
        raise Error("Partition CUT original OVER occurrence is missing")
    ref over = original.expr._nodes[over_node]
    if over.op != OVER or over.left < 0:
        raise Error("Partition CUT requires an original OVER occurrence")
    ref operation = original.expr._nodes[over.left]
    var numeric_state = operation.op != CUT and operation.op != QCUT
    var root = over.left if numeric_state else operation.left
    if root < 0:
        raise Error("Partition CUT original operand is missing")
    var operand = subtree(original.expr, root)
    var bound_operand = bind(operand, source._columns)
    var proof = signature(bound_operand)
    var nested = False
    var identity = List[String](length=len(original.expr._nodes), fill="")
    for node in operand._nodes:
        nested |= node.op == OVER
    # Replacing OVER by its numeric child here checks only supported value
    # geometry. Each original scope still executes through its own observer.
    var stripped = substitute(original.expr, root, identity, strip_over=True)
    _ = operand_certificate(stripped, source)
    var specifications = List[UInt64]()
    if not numeric_state and operation.text.byte_length():
        for part in operation.text.split(SEP):
            specifications.append(
                bitcast[DType.uint64](parse_float64(String(part)))
            )
    var labels = List[String]()
    if not numeric_state and operation.text2.byte_length():
        labels = split_names(operation.text2)
    if len(labels) and len(labels) != len(specifications) + 1:
        raise Error("Partition CUT literal label count changed after binding")
    var keys = List[WindowKey]()
    for name in split_names(over.text2):
        var field = source._index(name)
        var dtype = source._columns[field].dtype()
        var string_hash = dtype == DataType.STRING or dtype._is_bytes()
        if dtype.is_categorical():
            for node in original.expr._nodes:
                string_hash |= node.op == COL and node.text == name
        keys.append(WindowKey(name, dtype, string_hash))
    if not len(keys):
        raise Error("Partition CUT original key set is empty")
    return Capture(
        signature(original),
        over_node,
        operand^,
        proof^,
        keys^,
        over.text2,
        specifications^,
        labels^,
        not numeric_state and operation.op == QCUT,
        not numeric_state and (operation.integer & 1) != 0,
        not numeric_state and (operation.integer & 2) != 0,
        not numeric_state and (operation.integer & 4) != 0,
        numeric_state,
        nested,
        original.dtypes[root],
    )


def validate(saved: Capture, expression: Expr, source: DataFrame) raises:
    """Rebinding must preserve the entire expression before any row lifting."""
    var rebound = bind(expression, source._columns)
    if signature(rebound) != saved.original:
        raise Error("Partition CUT original expression metadata changed")
    var current = capture(rebound, source, saved.over_node)
    if (
        current.operand_signature != saved.operand_signature
        or current.key_set != saved.key_set
        or current.specifications != saved.specifications
        or current.labels != saved.labels
        or current.qcut != saved.qcut
        or current.left_closed != saved.left_closed
        or current.include_breaks != saved.include_breaks
        or current.allow_duplicates != saved.allow_duplicates
        or current.numeric_state != saved.numeric_state
        or current.nested != saved.nested
        or current.operand_dtype != saved.operand_dtype
        or len(current.keys) != len(saved.keys)
    ):
        raise Error("Partition CUT immutable control certificate changed")
    for at in range(len(current.keys)):
        if (
            current.keys[at].name != saved.keys[at].name
            or current.keys[at].dtype != saved.keys[at].dtype
            or current.keys[at].string_hash != saved.keys[at].string_hash
        ):
            raise Error("Partition CUT original logical key metadata changed")


comptime TAG = "metal_partition_cut_operand"
comptime VERSION = "partition_state_v1"


def is_marker(query: LazyFrame) -> Bool:
    if not len(query._nodes):
        return False
    ref node = query._nodes[len(query._nodes) - 1]
    return (
        node.kind == WITH_COLUMNS
        and node.text == "metal_window_materialization"
        and len(node.names2) == 8
        and node.names2[0] == TAG
        and node.names2[1] == VERSION
    )


def marker_capture(query: LazyFrame) raises -> Capture:
    if not is_marker(query):
        raise Error("Partition CUT private state marker is missing")
    ref operation = query._nodes[len(query._nodes) - 1]
    if len(operation.partition_exprs) != 2:
        raise Error("Partition CUT original expression certificate is missing")
    ref original = operation.partition_exprs[0]
    var source = DataFrame(operation.partition_columns.copy())
    var bound = bind(original, source._columns)
    if signature(bound) != operation.names2[5]:
        raise Error("Partition CUT original expression certificate changed")
    var saved = capture(bound, source, Int(parse_int64(operation.names2[4])))
    if len(operation.window_keys) != len(saved.keys):
        raise Error("Partition CUT original key certificate changed")
    for at in range(len(saved.keys)):
        if (
            operation.window_keys[at].name != saved.keys[at].name
            or operation.window_keys[at].dtype != saved.keys[at].dtype
            or operation.window_keys[at].string_hash
            != saved.keys[at].string_hash
        ):
            raise Error("Partition CUT original key certificate changed")
    ref over = original._nodes[saved.over_node]
    ref bins = original._nodes[over.left]
    var root = over.left if saved.numeric_state else bins.left
    var entering = len(operation.flags) == 2 and operation.flags[0]
    var rewritten = substitute(
        original, root, operation.names, strip_over=entering
    )
    var rewritten_bound = bind(rewritten, source._columns)
    var stored_bound = bind(operation.partition_exprs[1], source._columns)
    if signature(rewritten_bound) != signature(stored_bound):
        raise Error("Partition original nested-state substitution changed")
    saved.operand = rewritten^
    saved.operand_signature = operand_certificate(saved.operand, source)
    return saved^


def _marker(
    mut query: LazyFrame,
    left: Int,
    source: DataFrame,
    original: BoundExpr,
    saved: Capture,
    lowered: Expr,
    replacements: List[String],
    expressions: List[Expr],
    category: String,
    breakpoint: String,
    scope_id: Int,
    enter: Bool,
) raises:
    query._nodes.append(
        _plan_node(
            WITH_COLUMNS,
            left,
            exprs=expressions,
            names=replacements,
            flags=[enter, saved.numeric_state],
            text="metal_window_materialization",
            names2=[
                TAG,
                VERSION,
                category,
                breakpoint,
                String(saved.over_node),
                saved.original,
                String(scope_id),
                "1",
            ],
            window_keys=saved.keys,
            partition_exprs=[original.expr.copy(), lowered.copy()],
            partition_columns=source.clear()._columns,
        )
    )


def stage(
    mut query: LazyFrame,
    left: Int,
    source: DataFrame,
    original: BoundExpr,
    over_node: Int,
    category: String,
    breakpoint: String,
    scope_id: Int,
    cached_layout: Bool,
) raises -> Expr:
    var saved = capture(original, source, over_node)
    ref over = original.expr._nodes[over_node]
    ref bins = original.expr._nodes[over.left]
    var root = over.left if saved.numeric_state else bins.left
    var replacements = List[String](length=len(original.expr._nodes), fill="")
    var aligned = left
    var schema = source.copy()
    if saved.nested:
        # CPU chooses the outer fallback's cache before evaluating nested
        # states. Record that entry snapshot before any inner scope creates
        # another layout for the same key set.
        var identity = List[String](length=len(original.expr._nodes), fill="")
        var stripped = substitute(
            original.expr, root, identity, strip_over=True
        )
        _marker(
            query,
            aligned,
            schema,
            original,
            saved,
            stripped,
            identity,
            List[Expr](),
            category,
            breakpoint,
            scope_id,
            True,
        )
        aligned = len(query._nodes) - 1
        var top = List[Int]()
        var pending: List[Int] = [root]
        var seen = List[Bool](length=len(original.expr._nodes), fill=False)
        while len(pending):
            var at = pending.pop()
            if at < 0 or seen[at]:
                continue
            seen[at] = True
            ref node = original.expr._nodes[at]
            if node.op == OVER:
                top.append(at)
                continue
            if node.left >= 0:
                pending.append(node.left)
            if node.right >= 0:
                pending.append(node.right)
            if node.extra >= 0:
                pending.append(node.extra)
        sort(top)
        for at in top:
            var name = category + "_nested_" + String(at)
            while name in schema.columns():
                name += "_"
            var nested = stage(
                query,
                aligned,
                schema,
                original,
                at,
                name,
                name + "_breakpoint",
                scope_id,
                cached_layout,
            )
            aligned = len(query._nodes) - 1
            replacements[at] = name
            var columns = schema._columns.copy()
            columns.append(Series.full_null(name, original.dtypes[at], 0))
            schema = DataFrame(columns^)
    var lowered = substitute(original.expr, root, replacements)
    _ = operand_certificate(lowered, schema)
    var dependencies = List[String]()
    for node in lowered._nodes:
        if node.op == COL and node.text not in dependencies:
            dependencies.append(node.text)
    for key in saved.keys:
        if key.name not in dependencies:
            dependencies.append(key.name)
    var value: Expr
    if saved.numeric_state:
        value = col(dependencies[0]).cast(saved.operand_dtype)
    else:
        value = lit("") if not len(saved.labels) else lit(saved.labels[0])
        for at in range(1, len(saved.labels)):
            value = (
                when(col(dependencies[0]).is_not_null())
                .then(value)
                .otherwise(lit(saved.labels[at]))
            )
    # Harmless schema placeholders retain all original sources through pruning;
    # the dedicated runtime never interprets these row values.
    for dependency in dependencies:
        value = (
            when(col(dependency).is_not_null())
            .then(value.copy())
            .otherwise(value)
        )
    var expressions: List[Expr] = [value.alias(category)]
    if saved.include_breaks:
        expressions.append(
            when(col(dependencies[0]).is_not_null())
            .then(lit(Float64(0)))
            .otherwise(lit(Float64(0)))
            .alias(breakpoint)
        )
    _marker(
        query,
        aligned,
        schema,
        original,
        saved,
        lowered,
        replacements,
        expressions,
        category,
        breakpoint,
        scope_id,
        False,
    )
    if saved.include_breaks:
        return as_struct(
            [
                col(breakpoint).alias("breakpoint"),
                col(category).alias("category"),
            ],
            "",
        )
    return col(category)
