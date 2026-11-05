"""Recover supported regions from row expressions shared by the CPU planner.

Only row-local binary values immediately below the final projection are
inlined. Filters must already precede their evaluation; unused or dependent
with-columns work is declined so errors cannot be pruned by GPU fusion.
"""
from dataframe import Expr, LazyFrame
from dataframe.expr import COL, ADD, SUB, MUL, SUM, COUNT, MEAN
from dataframe.lazy import SCAN_FRAME, FILTER, WITH_COLUMNS, SELECT


def normalize_region(region: LazyFrame) raises -> Optional[LazyFrame]:
    var n = len(region._plan._nodes)
    if n < 3 or region._plan._nodes[n - 2].kind != WITH_COLUMNS:
        for node in region._plan._nodes:
            if node.kind == WITH_COLUMNS:
                return None
        return region.copy()
    if (
        region._plan._nodes[0].kind != SCAN_FRAME
        or region._plan._nodes[n - 1].kind != SELECT
    ):
        return None
    for i in range(1, n - 2):
        if (
            region._plan._nodes[i].kind != FILTER
            or region._plan._nodes[i].left != i - 1
        ):
            return None
    ref definitions = region._plan._nodes[n - 2].exprs
    if (
        not definitions
        or region._plan._nodes[n - 2].left != n - 3
        or region._plan._nodes[n - 1].left != n - 2
    ):
        return None
    var names = List[String]()
    for definition in definitions:
        if definition._name in names or len(definition._nodes) != 3:
            return None
        ref binary = definition._nodes[2]
        if (
            (binary.op != ADD and binary.op != SUB and binary.op != MUL)
            or binary.left != 0
            or binary.right != 1
        ):
            return None
        names.append(definition._name.copy())
    # Definitions are parallel expressions over the input. Reject apparent
    # dependencies/overwrites instead of accidentally substituting twice.
    for definition in definitions:
        for value in definition._nodes:
            if value.op == COL and value.text in names:
                return None
    var used = List[Bool](length=len(names), fill=False)
    var outputs = List[Expr]()
    for expression in region._plan._nodes[n - 1].exprs:
        var slot = -1
        for node in expression._nodes:
            if node.op == COL and node.text in names:
                if len(expression._nodes) != 2:
                    return None
                for i in range(len(names)):
                    if node.text == names[i]:
                        slot = i
        if slot < 0:
            outputs.append(expression.copy())
            continue
        ref aggregate = expression._nodes[1]
        if aggregate.left != 0 or (
            aggregate.op != SUM
            and aggregate.op != COUNT
            and aggregate.op != MEAN
        ):
            return None
        var expanded = definitions[slot].copy()
        var finish = aggregate.copy()
        finish.left = len(expanded._nodes) - 1
        expanded._nodes.append(finish^)
        expanded._name = expression._name.copy()
        outputs.append(expanded^)
        used[slot] = True
    for present in used:
        if not present:
            return None
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    var normalized = frame.lazy()
    for i in range(1, n - 2):
        if len(region._plan._nodes[i].exprs) != 1:
            return None
        normalized = normalized.filter(region._plan._nodes[i].exprs[0])
    return normalized.select_exprs(outputs)
