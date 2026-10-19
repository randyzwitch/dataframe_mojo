"""Keep untouched wide result payloads out of bounded GPU row-selection plans.

Filters, ordering and slicing run on the GPU over their original operands and
an immutable source-row identity. Only terminal host result assembly gathers
untouched source values. This is not a CPU evaluation of the row-selection
query, and it does not claim GPU-only output materialization.
"""
from dataframe import DataFrame
from dataframe.lazy import (
    LazyFrame,
    SCAN_FRAME,
    FILTER,
    SORT,
    SLICE,
    _references,
)
from .metadata import schema_frame
from .memory import checked_add, checked_mul
from .opaque import _retained_frame_bytes


@fieldwise_init
struct PayloadSelection(Movable):
    var query: LazyFrame
    var source: DataFrame
    var identity: String
    var staging_bytes: Int
    var restoration_bytes: Int


def columns(query: LazyFrame) raises -> Optional[List[String]]:
    """Admit a single source, row-preserving schema and bounded final slice."""
    if len(query._nodes) < 2 or query._nodes[0].kind != SCAN_FRAME:
        return None
    ref terminal = query._nodes[len(query._nodes) - 1]
    if (
        terminal.kind != SLICE
        or terminal.offset < 0
        or terminal.length < 0
        or terminal.length > 4096
    ):
        return None
    var reads = List[String]()
    for i in range(1, len(query._nodes)):
        ref operation = query._nodes[i]
        if operation.left != i - 1 or operation.right >= 0:
            return None
        if operation.kind == FILTER:
            for expression in operation.exprs:
                var names = _references(expression)
                if not names:
                    return None
                for name in names.value():
                    if name not in reads:
                        reads.append(name)
        elif operation.kind == SORT:
            for name in operation.names:
                if name not in reads:
                    reads.append(name)
        elif operation.kind != SLICE or i != len(query._nodes) - 1:
            return None
    ref scan = query._nodes[0]
    var source = query._frames[scan.offset].copy()
    if len(scan.names):
        source = source.select(scan.names)
    if source.height() > Int(Int32.MAX):
        return None
    # Benefit is determined by plan geometry, never benchmark identifiers.
    if source.width() < 32 or source.width() <= 2 * (len(reads) + 1):
        return None
    for field in source._columns:
        if field.dtype().is_nested():
            return None
    # Validate the complete original schema, including fields not read by
    # predicates, before constructing any source-row identity payload.
    _ = schema_frame(query, len(query._nodes) - 1)
    return reads^


def prepare(
    query: LazyFrame, reads: List[String], budget: Int
) raises -> PayloadSelection:
    ref scan = query._nodes[0]
    var source = query._frames[scan.offset].copy()
    if len(scan.names):
        source = source.select(scan.names)
    var identity = "__dataframe_source_row"
    while identity in source.columns():
        identity += "_"
    var staging = checked_mul(source.height(), 8)
    # Whole-source payload is a deliberately conservative envelope for the
    # bounded output, including private chunk merging during final gathering.
    var restoration = checked_add(
        checked_mul(_retained_frame_bytes(source), 3),
        checked_mul(
            min(source.height(), query._nodes[len(query._nodes) - 1].length), 8
        ),
    )
    var reserve = checked_add(staging, restoration)
    if budget >= 0 and reserve > budget:
        raise Error("Metal late payload assembly exceeds memory budget")
    var selected = source.select(reads).with_row_index(identity)
    var rewritten = query.copy()
    rewritten._frames[scan.offset] = selected^
    rewritten._nodes[0].names = List[String]()
    rewritten = rewritten.select(List[String]([identity]))
    return PayloadSelection(
        rewritten^, source^, identity^, staging, restoration
    )


def restore(result: DataFrame, plan: PayloadSelection) raises -> DataFrame:
    if result.width() != 1 or result.columns()[0] != plan.identity:
        raise Error("Metal late payload row identity schema differs")
    var indices = List[Int](capacity=result.height())
    for row in range(result.height()):
        var value = result.item(row, plan.identity)
        if value.is_null():
            raise Error("Metal late payload row identity is null")
        var index = Int(value.int64())
        if index < 0 or index >= plan.source.height():
            raise Error("Metal late payload row identity is outside its source")
        indices.append(index)
    return plan.source.take(indices)
