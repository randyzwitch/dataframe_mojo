"""Delay wide-column reads until a top-N has selected source rows.

Scan only the sort keys and row ids, keep the best rows, then gather the
requested payload columns and compute projections over that smaller set.
This avoids reading and copying payload values for rows discarded by the
limit. Row ids preserve the connection to the original source.
"""

from std.collections import Optional

from dataframe.aggregation.expressions import _row_local
from dataframe.expr import Expr, col
from dataframe.frame import DataFrame
from dataframe.planning.expression_inputs import _references
from dataframe.planning.model import (
    DROP,
    FILTER,
    PlanNode,
    SELECT,
    WITH_COLUMNS,
)
from dataframe.pipeline import ROW_ID_COLUMN
from dataframe.series import Series
from dataframe.storage.column import Column

# The largest top-N whose other columns are fetched afterwards by row id.
# Fetching is a random read per kept row, cheap for a few rows; for many rows,
# carrying the columns through the scan costs less. An empirical choice.
comptime LATE_MATERIALIZATION_MAX_ROWS = 50


@fieldwise_init
struct LatePlan(Copyable, Movable):
    """The top-N's side of the rewrite: the scan columns it reads (in the scan's
    order; the row id is added by the scan) and its operations, each projection
    cut to the columns used above it and carrying the row id up."""

    var columns: List[String]
    var operations: List[PlanNode]


def _add(mut names: List[String], found: List[String]):
    for name in found:
        if name not in names:
            names.append(name)


def late_materialization_plan(
    scan_columns: List[String],
    operations: List[PlanNode],
    order_names: List[String],
) -> Optional[LatePlan]:
    """The rewritten top-N side, or None when the chain holds another operator,
    an expression that is not row-local, a selector, or when every scan column
    is required anyway."""
    var needed = order_names.copy()
    var rewritten = List[PlanNode]()
    var i = len(operations)
    while i > 0:
        i -= 1
        ref node = operations[i]
        if node.kind == FILTER:
            if not _row_local(node.exprs):
                return None
            for e in node.exprs:
                var r = _references(e)
                if not r:
                    return None
                _add(needed, r.value())
            rewritten.append(node.copy())
        elif node.kind == SELECT or node.kind == WITH_COLUMNS:
            if not _row_local(node.exprs):
                return None
            var kept = List[Expr]()
            var below = List[String]()
            var outputs = List[String]()
            for e in node.exprs:
                outputs.append(e._name)
                if e._name not in needed:
                    continue
                var r = _references(e)
                if not r:
                    return None
                _add(below, r.value())
                kept.append(e.copy())
            if node.kind == WITH_COLUMNS:
                # Columns it does not write pass through.
                for name in needed:
                    if name not in outputs and name not in below:
                        below.append(name)
            else:
                kept.append(col(ROW_ID_COLUMN))
            var copy = node.copy()
            copy.exprs = kept^
            rewritten.append(copy^)
            needed = below^
        elif node.kind == DROP:
            rewritten.append(node.copy())
        else:
            return None
    for name in needed:
        if name not in scan_columns:
            return None
    if len(needed) >= len(scan_columns):
        # Every column the scan reads is required: nothing to defer.
        return None
    var columns = List[String]()
    for name in scan_columns:
        if name in needed:
            columns.append(name)
    rewritten.reverse()
    return LatePlan(columns^, rewritten^)


def late_fetch(
    scan: DataFrame, top_rows: DataFrame, operations: List[PlanNode]
) raises -> DataFrame:
    """The scan's rows with the top-N's row ids, in the top-N's order,
    through the original projections (the semi join and final projection;
    the filters already held for these rows)."""
    var ids = top_rows.column(ROW_ID_COLUMN).rechunk()
    ref values = ids._data[Column[Int64]]
    var rows = List[Int](capacity=len(ids))
    for r in range(len(ids)):
        rows.append(Int(values._get(r)))
    var frame = scan.take(rows)
    for node in operations:
        if node.kind == SELECT:
            frame = frame.select_exprs(node.exprs)
        elif node.kind == WITH_COLUMNS:
            frame = frame.with_columns(node.exprs)
        elif node.kind == DROP:
            frame = frame.drop(node.names)
    return frame^
