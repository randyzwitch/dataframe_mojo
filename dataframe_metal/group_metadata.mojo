"""Scalar metadata for one contiguous bounded hash aggregation scan."""
from std.memory import bitcast
from dataframe.expr import COL, MIN, MAX
from .group_plan import GroupedComposition
from .expression_format import EXPRESSION_NODE_WORDS
from .value_program_metadata import value_program_words, pack_value_program
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
    var tree_at = (
        predicate_at
        + PREDICATE_HEADER_WORDS
        + len(compiled.clauses) * PREDICATE_FIELD_WORDS
    )
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
    words[predicate_at] = UInt64(len(compiled.clauses))
    for i in range(8):
        words[predicate_at + 1 + i] = UInt64(compiled.truth[i])
    for i in range(len(compiled.clauses)):
        ref leaf = compiled.clauses[i]
        ref clause = leaf.bounds
        ref input = inputs[compiled.slots[i]]
        var base = (
            predicate_at + PREDICATE_HEADER_WORDS + i * PREDICATE_FIELD_WORDS
        )
        words[base] = UInt64(compiled.slots[i])
        words[base + 1] = (
            UInt64(input.bit_offset)
            | (UInt64(input.nullable) << 32)
            | (UInt64(input.tag) << 33)
        )
        words[base + 2] = bitcast[DType.uint64](clause.low)
        words[base + 3] = bitcast[DType.uint64](clause.high)
        words[base + 4] = UInt64(bitcast[DType.uint32](clause.low_float)) | (
            UInt64(bitcast[DType.uint32](clause.high_float)) << 32
        )
        if leaf.kind == 2:
            ref right = inputs[leaf.other_input]
            words[base + 2] = UInt64(leaf.other_input)
            words[base + 3] = (
                UInt64(right.bit_offset)
                | (UInt64(right.nullable) << 32)
                | (UInt64(right.tag) << 33)
            )
        words[base + 5] = (
            UInt64(clause.inclusive)
            | (UInt64(leaf.kind) << 2)
            | (UInt64(leaf.inverted) << 4)
            | (UInt64(leaf.comparison) << 5)
        )
    return words^
