"""Checked native-view descriptors for ordered scans and tuple grouping."""
from dataframe import DataType
from .ordered_plan import OrderedPlan, ordered_tag
from .group_layout import GroupLayout
from .value_program_metadata import predicate_words, write_predicate
from .accelerator.gpu.erased_input import NumericView


def ordered_scratch_words(layout: GroupLayout) raises -> Int:
    layout.check()
    return layout.words() + layout.capacity * layout.fields * 8


def ordered_metadata(
    plan: OrderedPlan,
    inputs: List[NumericView],
    keys: List[NumericView],
    layout: GroupLayout,
    rows: Int,
) raises -> List[UInt64]:
    layout.check()
    if (
        rows < 0
        or rows >= Int(Int32.MAX)
        or len(inputs) != len(plan.inputs)
        or layout.fields != len(plan.fields)
    ):
        raise Error("Metal ordered metadata does not match its native inputs")
    if (plan.keys and len(keys) != len(plan.keys.value().names)) or (
        not plan.keys and keys
    ):
        raise Error("Metal ordered metadata does not match its tuple keys")
    var start = 16 + len(plan.fields) * 4
    var words = List[UInt64](
        length=start + predicate_words(plan.predicate), fill=0
    )
    words[0] = UInt64(rows)
    words[1] = UInt64(layout.blocks * 256)
    words[2] = UInt64(layout.fields)
    words[3] = UInt64(layout.capacity)
    words[4] = UInt64(min(65536, layout.capacity // 2))
    words[5] = UInt64(start)
    words[6] = UInt64(layout.blocks)
    words[7] = UInt64(layout.global_state)
    words[11] = UInt64(len(keys))
    for i in range(len(keys)):
        ref key = keys[i]
        if key.rows != rows or key.tag not in [0, 1] or key.bit_offset < 0:
            raise Error("Metal ordered tuple key is not a native integer view")
        words[12 + i] = (
            UInt64(key.tag)
            | (UInt64(key.nullable) << 1)
            | (UInt64(key.bit_offset) << 2)
        )
    for field in range(len(plan.fields)):
        ref value = plan.fields[field]
        var slot = -1
        for i in range(len(plan.inputs)):
            if plan.inputs[i] == value.column:
                slot = i
        if (
            slot < 0
            or inputs[slot].rows != rows
            or inputs[slot].tag != ordered_tag(value.dtype)
            or inputs[slot].bit_offset < 0
        ):
            raise Error("Metal ordered value is not a matching native view")
        ref input = inputs[slot]
        var at = 16 + field * 4
        words[at] = UInt64(slot)
        words[at + 1] = (
            UInt64(input.bit_offset)
            | (UInt64(input.nullable) << 32)
            | (UInt64(input.tag) << 33)
        )
        words[at + 2] = UInt64(value.drop_nulls) | (
            UInt64(value.operations) << 8
        )
    write_predicate(words, start, plan.predicate, inputs)
    return words^
