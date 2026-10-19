"""Certify scalar integer SUM projections without row-matrix intermediates."""
from dataframe import DataType
from dataframe.accelerator.rows import RowPlan
from dataframe.expr import COL, LIT_INT, ADD, MUL, SUM
from std.memory import bitcast


@fieldwise_init
struct Program(Copyable):
    var add: Bool
    var multiply: Bool
    var addend: UInt64
    var factor: UInt64
    var min_count: Int64


def _node(plan: RowPlan, at: Int, op: Int, dtype: DataType) -> Bool:
    return (
        plan.code[4 * at] == Int64(op) and plan.node_dtypes[at] == dtype
        and plan.code[4 * at + 1] == -1 and plan.code[4 * at + 2] == -1
        and (op == COL or plan.code[4 * at + 3] == -1)
    )


def _binary(plan: RowPlan, at: Int, op: Int, left: Int, right: Int) -> Bool:
    return (
        plan.code[4 * at] == Int64(op)
        and plan.code[4 * at + 1] == Int64(left)
        and plan.code[4 * at + 2] == Int64(right)
        and plan.code[4 * at + 3] == -1
    )


def bind(plan: RowPlan) -> Optional[List[Program]]:
    if (
        plan.source.width() != 1
        or plan.source.height() > Int(Int32.MAX)
        or not plan.reductions
        or not len(plan.outputs)
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
        or len(plan.gathers)
        or len(plan.steps) != len(plan.outputs)
        or len(plan.code) != len(plan.literals) * 4
        or len(plan.node_dtypes) != len(plan.literals)
        or plan.slots != len(plan.outputs) + 1
        or len(plan.slot_dtypes) != plan.slots
    ):
        return None
    var dtype = plan.source._columns[0].dtype()
    if dtype != DataType.INT64 and dtype != DataType.UINT64:
        return None
    if plan.slot_dtypes[0] != dtype:
        return None
    var programs = List[Program](capacity=len(plan.outputs))
    var start = 0
    for output in range(len(plan.outputs)):
        ref step = plan.steps[output]
        ref field = plan.outputs[output]
        if (
            step.filter or step.order or step.grouped
            or step.start != start
            or (step.nodes != 1 and step.nodes != 3 and step.nodes != 5)
            or step.slot != output + 1
            or step.gather_start != 0
            or step.gather_count != 1
            or field.slot != step.slot
            or field.reduction != SUM
            or field.dtype != dtype
            or field.min_count < 0
            or plan.slot_dtypes[step.slot] != dtype
            or start + step.nodes > len(plan.literals)
        ):
            return None
        if not _node(plan, start, COL, dtype) or plan.code[4 * start + 3] != 0:
            return None
        var add = False
        var multiply = False
        var addend = UInt64(0)
        var factor = UInt64(1)
        if step.nodes >= 3:
            if not _node(plan, start + 1, LIT_INT, dtype):
                return None
            if plan.node_dtypes[start + 2] != dtype:
                return None
            if _binary(plan, start + 2, ADD, 0, 1):
                add = True
                addend = bitcast[DType.uint64](plan.literals[start + 1])
            elif step.nodes == 3 and _binary(plan, start + 2, MUL, 0, 1):
                multiply = True
                factor = bitcast[DType.uint64](plan.literals[start + 1])
            else:
                return None
        if step.nodes == 5:
            if (
                not _node(plan, start + 3, LIT_INT, dtype)
                or plan.node_dtypes[start + 4] != dtype
                or not _binary(plan, start + 4, MUL, 2, 3)
            ):
                return None
            multiply = True
            factor = bitcast[DType.uint64](plan.literals[start + 3])
        programs.append(Program(add, multiply, addend, factor, Int64(field.min_count)))
        start += step.nodes
    if start != len(plan.literals):
        return None
    return programs^
