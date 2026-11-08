"""Pack exact native distinct descriptors without copying input payloads."""
from .distinct_plan import DistinctPlan
from .distinct_layout import DistinctLayout
from .ordered_plan import ordered_tag
from .value_program_metadata import predicate_words, write_predicate
from .accelerator.gpu.erased_input import NumericView


def distinct_metadata(
    plan: DistinctPlan,
    inputs: List[NumericView],
    keys: List[NumericView],
    layout: DistinctLayout,
    rows: Int,
    blocks: Int,
) raises -> List[UInt64]:
    layout.check()
    if (
        rows < 0
        or rows >= Int(Int32.MAX)
        or len(inputs) != len(plan.inputs)
        or layout.fields != len(plan.fields)
        or blocks < 1
        or blocks > 128
    ):
        raise Error("Metal distinct metadata does not match its native inputs")
    if (plan.keys and len(keys) != len(plan.keys.value().names)) or (
        not plan.keys and keys
    ):
        raise Error("Metal distinct metadata does not match its tuple keys")
    var start = 16 + len(plan.fields) * 2
    var words = List[UInt64](
        length=start + predicate_words(plan.predicate), fill=0
    )
    words[0] = UInt64(rows)
    words[1] = UInt64(blocks * 256)
    words[2] = UInt64(layout.fields)
    words[3] = UInt64(layout.capacity)
    words[4] = UInt64(min(65536, layout.capacity // 2))
    words[5] = UInt64(start)
    words[6] = UInt64(layout.unique_capacity)
    words[11] = UInt64(len(keys))
    for i in range(len(inputs)):
        if (
            inputs[i].rows != rows
            or inputs[i].bit_offset < 0
            or inputs[i].bit_offset + rows > Int(UInt32.MAX)
        ):
            raise Error("Metal distinct bitmap span exceeds its index range")
    for i in range(len(keys)):
        ref key = keys[i]
        if (
            key.rows != rows
            or key.tag not in [0, 1]
            or key.bit_offset < 0
            or key.bit_offset + rows > Int(UInt32.MAX)
        ):
            raise Error(
                "Metal distinct tuple key is not a matching native view"
            )
        words[12 + i] = (
            UInt64(key.tag)
            | (UInt64(key.nullable) << 1)
            | (UInt64(key.bit_offset) << 2)
        )
    for field in range(len(plan.fields)):
        var slot = -1
        for i in range(len(plan.inputs)):
            if plan.inputs[i] == plan.fields[field].column:
                slot = i
        if slot < 0 or inputs[slot].tag != ordered_tag(
            plan.fields[field].dtype
        ):
            raise Error("Metal distinct value is not a matching native view")
        ref input = inputs[slot]
        words[16 + field * 2] = UInt64(slot)
        words[17 + field * 2] = (
            UInt64(input.bit_offset)
            | (UInt64(input.nullable) << 32)
            | (UInt64(input.tag) << 33)
            | (UInt64(input.value_bit_offset) << 35)
        )
    write_predicate(words, start, plan.predicate, inputs)
    return words^
