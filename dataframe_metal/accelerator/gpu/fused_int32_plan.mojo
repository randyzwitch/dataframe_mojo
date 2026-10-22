"""Host-only certification for the first bounded native integer SUM family."""

from std.memory import bitcast
from dataframe import DataType
from dataframe_metal.accelerator.rows import RowPlan
from dataframe.expr import COL, LIT_INT, GT, ADD, MUL, CAST, SUM


@fieldwise_init
struct Program(Copyable):
    var filtered: Bool
    var arithmetic: Bool
    var threshold: Int32
    var addend: Int32
    var factor: Int32


def _node(plan: RowPlan, at: Int, op: Int, dtype: DataType) -> Bool:
    return plan.code[4 * at] == Int64(op) and plan.node_dtypes[at] == dtype


def _column(plan: RowPlan, at: Int) -> Bool:
    return _node(plan, at, COL, DataType.INT32) and plan.code[4 * at + 3] == 0


def _literal(plan: RowPlan, at: Int) -> Bool:
    var value = bitcast[DType.int64](plan.literals[at])
    return (
        _node(plan, at, LIT_INT, DataType.INT32)
        and value >= Int64(Int32.MIN)
        and value <= Int64(Int32.MAX)
    )


def _binary(
    plan: RowPlan, at: Int, op: Int, dtype: DataType, left: Int, right: Int
) -> Bool:
    return (
        _node(plan, at, op, dtype)
        and plan.code[4 * at + 1] == Int64(left)
        and plan.code[4 * at + 2] == Int64(right)
    )


def bind(plan: RowPlan) -> Optional[Program]:
    if (
        plan.source.width() != 1
        or plan.source.height() > Int(Int32.MAX)
        or plan.source._columns[0].dtype() != DataType.INT32
        or not plan.reductions
        or len(plan.outputs) != 1
        or plan.outputs[0].reduction != SUM
        or plan.outputs[0].dtype != DataType.INT64
        or plan.outputs[0].min_count < 0
        or plan.limit >= 0
        or plan.ordered_mode != 0
        or plan.decimal_feed_batch_rows
        or len(plan.source_text)
        or len(plan.output_text)
        or plan.code_domain
        or plan.source_domain
        or len(plan.iana_descriptors)
        or len(plan.zone_words)
        or plan.source_failure
        or len(plan.steps) < 1
        or len(plan.steps) > 2
        or len(plan.code) != len(plan.literals) * 4
        or len(plan.node_dtypes) != len(plan.literals)
        or len(plan.slot_dtypes) != plan.slots
    ):
        return None
    var filtered = len(plan.steps) == 2
    var start = 0
    var threshold = Int32(0)
    if filtered:
        ref predicate = plan.steps[0]
        if (
            not predicate.filter
            or predicate.order
            or predicate.grouped
            or predicate.start != 0
            or predicate.nodes != 3
            or predicate.slot != 1
            or predicate.gather_start != 0
            or predicate.gather_count != 1
            or len(plan.gathers) != 1
            or plan.gathers[0] != 0
            or len(plan.literals) < 3
            or plan.slots < 3
            or plan.slot_dtypes[1] != DataType.BOOL
        ):
            return None
        if (
            not _column(plan, 0)
            or not _literal(plan, 1)
            or not _binary(plan, 2, GT, DataType.BOOL, 0, 1)
        ):
            return None
        threshold = Int32(bitcast[DType.int64](plan.literals[1]))
        start = 3
    elif len(plan.gathers):
        return None
    ref projection = plan.steps[len(plan.steps) - 1]
    var slot = 2 if filtered else 1
    if (
        projection.filter
        or projection.order
        or projection.grouped
        or projection.start != start
        or (projection.nodes != 2 and projection.nodes != 6)
        or projection.gather_count != 1
        or projection.gather_start != len(plan.gathers)
        or projection.slot != slot
        or plan.slots != slot + 1
        or plan.outputs[0].slot != slot
        or plan.slot_dtypes[0] != DataType.INT32
        or plan.slot_dtypes[slot] != DataType.INT64
        or len(plan.literals) != start + projection.nodes
    ):
        return None
    if not _column(plan, start):
        return None
    var arithmetic = projection.nodes == 6
    var addend = Int32(0)
    var factor = Int32(1)
    if arithmetic:
        if (
            not _literal(plan, start + 1)
            or not _binary(plan, start + 2, ADD, DataType.INT32, 0, 1)
            or not _literal(plan, start + 3)
            or not _binary(plan, start + 4, MUL, DataType.INT32, 2, 3)
        ):
            return None
        addend = Int32(bitcast[DType.int64](plan.literals[start + 1]))
        factor = Int32(bitcast[DType.int64](plan.literals[start + 3]))
    var cast_at = start + projection.nodes - 1
    if (
        not _node(plan, cast_at, CAST, DataType.INT64)
        or plan.code[4 * cast_at + 1] != Int64(4 if arithmetic else 0)
        or (plan.code[4 * cast_at + 3] != 0 and plan.code[4 * cast_at + 3] != 1)
    ):
        return None
    return Program(filtered, arithmetic, threshold, addend, factor)
