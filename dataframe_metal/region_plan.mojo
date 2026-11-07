"""Inline bounded row expressions while preserving all checked work.

WITH_COLUMNS definitions are parallel within a stage and sequential across
stages. Versioned bindings preserve overwrites and their dependencies. Every
definition must feed an output; filters must precede definitions.
"""
from dataframe import Expr, LazyFrame
from dataframe.expr import COL, SUM, COUNT, MEAN, MIN, MAX, WHEN, Node
from dataframe.lazy import SCAN_FRAME, FILTER, WITH_COLUMNS, SELECT, AGG
from .expression_plan import compile_expression
from .expression_format import MAX_CASE_SOURCE_NODES
from .conditional_plan import compile_conditional
from .typed_expression_plan import compile_typed_expression


def _expand(
    expression: Expr,
    names: List[String],
    versions: List[Int],
    definitions: List[Expr],
    mut dependencies: List[Int],
) -> Optional[Expr]:
    var nodes = List[Node]()
    var remap = List[Int]()
    for i in range(len(expression._nodes)):
        var node = expression._nodes[i].copy()
        var version = -1
        if node.op == COL:
            for j in range(len(names)):
                if names[j] == node.text:
                    version = versions[j]
        if version >= 0:
            if version not in dependencies:
                dependencies.append(version)
            var offset = len(nodes)
            for original in definitions[version]._nodes:
                var expanded = original.copy()
                if expanded.left >= 0:
                    expanded.left += offset
                if expanded.right >= 0:
                    expanded.right += offset
                if expanded.extra >= 0:
                    expanded.extra += offset
                nodes.append(expanded^)
        else:
            if node.left >= 0:
                if node.left >= i:
                    return None
                node.left = remap[node.left]
            if node.right >= 0:
                if node.right >= i:
                    return None
                node.right = remap[node.right]
            if node.extra >= 0:
                if node.extra >= i:
                    return None
                node.extra = remap[node.extra]
            nodes.append(node^)
        if len(nodes) > MAX_CASE_SOURCE_NODES + 1:
            return None
        remap.append(len(nodes) - 1)
    return Expr(nodes^, expression._name.copy())


def normalize_region(region: LazyFrame) raises -> Optional[LazyFrame]:
    var n = len(region._plan._nodes)
    var first = -1
    for i in range(n):
        if region._plan._nodes[i].kind == WITH_COLUMNS and first < 0:
            first = i
    if first < 0:
        return region.copy()
    if (
        first < 1
        or n < 3
        or region._plan._nodes[0].kind != SCAN_FRAME
        or (
            region._plan._nodes[n - 1].kind != SELECT
            and region._plan._nodes[n - 1].kind != AGG
        )
    ):
        return None
    for i in range(1, n):
        if region._plan._nodes[i].left != i - 1:
            return None
        if i < first and region._plan._nodes[i].kind != FILTER:
            return None
        if (
            i >= first
            and i < n - 1
            and region._plan._nodes[i].kind != WITH_COLUMNS
        ):
            return None
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    var names = List[String]()
    var versions = List[Int]()
    var definitions = List[Expr]()
    var edges = List[List[Int]]()
    for stage in range(first, n - 1):
        var added = List[Expr]()
        var stage_edges = List[List[Int]]()
        var stage_names = List[String]()
        for expression in region._plan._nodes[stage].exprs:
            if expression._name in stage_names:
                return None
            stage_names.append(expression._name.copy())
            var dependencies = List[Int]()
            var expanded = _expand(
                expression, names, versions, definitions, dependencies
            )
            if not expanded:
                return None
            # A prior WITH_COLUMNS definition ran for every row. Inlining it
            # into a CASE branch could hide a checked error in untaken rows.
            for node in expression._nodes:
                if node.op == WHEN and dependencies:
                    return None
            if (
                not compile_expression(expanded.value(), frame)
                and not compile_conditional(expanded.value(), frame)
                and not compile_typed_expression(expanded.value(), frame)
            ):
                return None
            added.append(expanded.take())
            stage_edges.append(dependencies^)
        if not added:
            return None
        # Publish bindings only after the entire parallel stage is expanded.
        for i in range(len(added)):
            var version = len(definitions)
            definitions.append(added[i].copy())
            edges.append(stage_edges[i].copy())
            var slot = -1
            for j in range(len(names)):
                if names[j] == stage_names[i]:
                    slot = j
            if slot < 0:
                names.append(stage_names[i].copy())
                versions.append(version)
            else:
                versions[slot] = version
    var used = List[Bool](length=len(definitions), fill=False)
    var pending = List[Int]()
    var outputs = List[Expr]()
    for expression in region._plan._nodes[n - 1].exprs:
        if not expression._nodes:
            return None
        var op = expression._nodes[len(expression._nodes) - 1].op
        if op != SUM and op != COUNT and op != MEAN and op != MIN and op != MAX:
            return None
        var output_dependencies = List[Int]()
        var expanded = _expand(
            expression, names, versions, definitions, output_dependencies
        )
        if not expanded:
            return None
        for node in expression._nodes:
            if node.op == WHEN and output_dependencies:
                return None
        for dependency in output_dependencies:
            pending.append(dependency)
        outputs.append(expanded.take())
    while pending:
        var version = pending.pop()
        if not used[version]:
            used[version] = True
            for dependency in edges[version]:
                pending.append(dependency)
    for present in used:
        if not present:
            return None
    var normalized = frame.lazy()
    for i in range(1, first):
        if len(region._plan._nodes[i].exprs) != 1:
            return None
        normalized = normalized.filter(region._plan._nodes[i].exprs[0])
    ref root = region._plan._nodes[n - 1]
    if root.kind == AGG:
        for key in root.group_keys():
            if key in names:
                return None
        return normalized.group_by(
            root.group_keys().copy(), maintain_order=root.maintain_order
        ).agg(outputs)
    return normalized.select_exprs(outputs)
