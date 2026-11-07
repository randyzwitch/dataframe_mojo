"""Scalar metadata for one contiguous bounded hash aggregation scan."""
from std.memory import bitcast
from dataframe import DataType
from dataframe.expr import COL, MIN, MAX
from .group_layout import GroupLayout
from .group_plan import GroupedComposition
from .expression_format import EXPRESSION_NODE_WORDS
from .value_program_metadata import (
    value_program_words,
    pack_value_program,
    predicate_words,
    write_predicate,
)
from .accelerator.gpu.erased_input import NumericView
from .accelerator.gpu.compound_predicate import (
    PREDICATE_HEADER_WORDS,
    PREDICATE_FIELD_WORDS,
)


def group_metadata(
    plan: GroupedComposition,
    inputs: List[NumericView],
    key: NumericView,
    blocks: Int,
) -> List[UInt64]:
    ref bundle = plan.selected.bundle
    ref compiled = plan.selected.program
    var predicate_at = 48
    var tree_at = predicate_at + predicate_words(compiled)
    var total = tree_at
    for field in bundle.fields:
        total += value_program_words(field.tree, field.conditions)
    var words = List[UInt64](length=total, fill=0)
    words[0] = UInt64(key.rows)
    words[1] = UInt64(blocks * 256)
    words[2] = UInt64(blocks)
    words[3] = UInt64(key.tag)
    words[4] = UInt64(key.bit_offset)
    words[5] = UInt64(key.nullable)
    words[6] = UInt64(len(bundle.fields))
    for value in bundle.fields:
        words[7] += UInt64(value.dtype == DataType.FLOAT32)
        words[8] += UInt64(value.dtype == DataType.INT32)
    words[10] = UInt64(predicate_at)
    for field in range(len(bundle.fields)):
        ref value = bundle.fields[field]
        var base = 16 + field * 8
        words[base] = UInt64(value.op)
        words[base + 1] = UInt64(value.left_input + 1)
        words[base + 2] = UInt64(value.right_input + 1)
        if value.left_input >= 0:
            ref input = inputs[value.left_input]
            words[base + 3] |= UInt64(input.nullable)
            words[base + 4] = UInt64(input.bit_offset)
        if value.right_input >= 0:
            ref input = inputs[value.right_input]
            words[base + 3] |= UInt64(input.nullable) << 1
            words[base + 5] = UInt64(input.bit_offset)
        words[base + 6] = UInt64(value.literal_side)
        words[base + 7] = UInt64(
            bitcast[DType.uint32](value.floating)
        ) if value.dtype.is_float() else bitcast[DType.uint64](value.integer)
        if value.needs_sum:
            words[9] |= UInt64(1) << UInt64(field)
        if value.tree:
            words[base + 6] = UInt64(tree_at)
            words[base + 7] = UInt64(len(value.tree))
            var packed = pack_value_program(
                value.tree, value.conditions, inputs
            )
            for word in packed:
                words[tree_at] = word
                tree_at += 1
    for output in bundle.outputs:
        if output.op == MIN or output.op == MAX:
            words[9] |= UInt64(1) << UInt64(
                output.field + (4 if output.op == MIN else 8)
            )
    write_predicate(words, predicate_at, compiled, inputs)
    return words^


def general_group_metadata(
    plan: GroupedComposition,
    inputs: List[NumericView],
    keys: List[NumericView],
    layout: GroupLayout,
) raises -> List[UInt64]:
    layout.check()
    if len(keys) != len(plan.keys.names) or not keys or len(keys) > 4:
        raise Error("Metal general group key bindings do not match the schema")
    var words = group_metadata(plan, inputs, keys[0], layout.blocks)
    if layout.fields != len(plan.selected.bundle.fields):
        raise Error("Metal general group field count does not match the schema")
    words[3] = UInt64(layout.capacity)
    words[4] = UInt64(layout.capacity // 2)
    words[5] = UInt64(layout.global_state)
    words[11] = UInt64(len(keys))
    for i in range(len(keys)):
        if keys[i].rows != keys[0].rows:
            raise Error("Metal general group key lengths do not match")
        var expected_tag = 0 if plan.keys.dtypes[i] == DataType.INT32 else 1
        if (
            keys[i].tag != expected_tag
            or keys[i].bit_offset < 0
            or keys[i].bit_offset > Int(UInt32.MAX)
        ):
            raise Error("Metal general group key has invalid native metadata")
        words[12 + i] = (
            UInt64(keys[i].tag)
            | (UInt64(keys[i].nullable) << 1)
            | (UInt64(keys[i].bit_offset) << 2)
        )
    return words^
