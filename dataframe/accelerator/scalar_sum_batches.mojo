"""Bound wide integer scalar reductions without a full-row derived matrix.

Original integer row expressions retain their bound types and checked faults.
Each GPU tile widens only the SUM input to Decimal128(scale=0), and a final
GPU reduction merges those exact partials before narrowing once. No query
expression or aggregate is evaluated on the host.
"""
from std.sys import size_of
from dataframe import DataFrame, DataType, Expr, Series, col
from dataframe.binding import bind, ROWS
from dataframe.expr import (
    SUM,
    COL,
    LIT_INT,
    LIT_NULL,
    CAST,
    ADD,
    SUB,
    MUL,
    NEG,
    ABS,
    subtree,
)
from dataframe.lazy import LazyFrame, SCAN_FRAME, SELECT
from .metadata import schema_frame
from .source_projection import project_sources
from .memory import checked_add, checked_mul

comptime TILE_ROWS = 65536


@fieldwise_init
struct SumBatches(Movable):
    var source: DataFrame
    var partials: List[Expr]
    var schema: DataFrame
    var batches: Int
    var reservation: Int


def prepare(query: LazyFrame) raises -> Optional[SumBatches]:
    if (
        len(query._nodes) != 2
        or query._nodes[0].kind != SCAN_FRAME
        or query._nodes[0].source_failure
        or query._nodes[1].kind != SELECT
        or query._nodes[1].left != 0
        or len(query._nodes[1].exprs) < 16
    ):
        return None
    var source = query._frames[query._nodes[0].offset].copy()
    if source.height() <= TILE_ROWS or source.height() > Int(Int32.MAX):
        return None
    var partials = List[Expr]()
    for expression in query._nodes[1].exprs:
        var bound = bind(expression, source._columns)
        var root = len(bound.expr._nodes) - 1
        ref reduction = bound.expr._nodes[root]
        if (
            reduction.op != SUM
            or reduction.min_count != 0
            or reduction.left < 0
            or bound.shapes[reduction.left] != ROWS
            or not bound.dtypes[root].is_integer()
        ):
            return None
        for index in range(root):
            var op = bound.expr._nodes[index].op
            if not bound.dtypes[index].is_integer() or (
                op != COL
                and op != LIT_INT
                and op != LIT_NULL
                and op != CAST
                and op != ADD
                and op != SUB
                and op != MUL
                and op != NEG
                and op != ABS
            ):
                return None
        partials.append(
            subtree(bound.expr, reduction.left)
            .cast(DataType.decimal(38, 0))
            .sum()
            .alias(expression._name)
        )
    var schema = schema_frame(query, 1).clear()
    var projected = project_sources(query)
    source = projected._frames[projected._nodes[0].offset].copy()
    var batches = (source.height() + TILE_ROWS - 1) // TILE_ROWS
    # Retain per-tile scalar frames and concatenation metadata until the final
    # device merge. This envelope includes values, validity, and Series owners.
    var reservation = checked_mul(
        checked_mul(batches, schema.width()),
        checked_add(64, checked_mul(3, size_of[Series]())),
    )
    return SumBatches(source^, partials^, schema^, batches, reservation)


def partial_query(plan: SumBatches, batch: Int) raises -> LazyFrame:
    if batch < 0 or batch >= plan.batches:
        raise Error("Metal scalar SUM tile is outside its source")
    return (
        plan.source.slice(batch * TILE_ROWS, TILE_ROWS)
        .lazy()
        .select_exprs(plan.partials)
    )


def merge_query(partials: DataFrame, plan: SumBatches) -> LazyFrame:
    var expressions = List[Expr]()
    for name in plan.schema.columns():
        expressions.append(col(name).sum().alias(name))
    return partials.lazy().select_exprs(expressions)


def narrow_query(merged: DataFrame, plan: SumBatches) -> LazyFrame:
    var expressions = List[Expr]()
    for field in plan.schema._columns:
        expressions.append(
            col(field.name()).cast(field.dtype()).alias(field.name())
        )
    return merged.lazy().select_exprs(expressions)
