"""Pack arithmetic/CASE instructions and their native predicate programs."""
from std.memory import bitcast
from dataframe.expr import COL
from .expression_plan import ValueNode
from .predicate_program import PredicateProgram
from .expression_format import CASE_TEST, EXPRESSION_NODE_WORDS
from .accelerator.gpu.erased_input import NumericView
from .accelerator.gpu.compound_predicate import (
    PREDICATE_HEADER_WORDS,
    PREDICATE_FIELD_WORDS,
)


def predicate_words(program: PredicateProgram) -> Int:
    return PREDICATE_HEADER_WORDS + len(program.clauses) * PREDICATE_FIELD_WORDS


def value_program_words(
    nodes: List[ValueNode], conditions: List[PredicateProgram]
) -> Int:
    var size = len(nodes) * EXPRESSION_NODE_WORDS
    for condition in conditions:
        size += predicate_words(condition)
    return size


def write_predicate(
    mut words: List[UInt64],
    at: Int,
    program: PredicateProgram,
    inputs: List[NumericView],
):
    words[at] = UInt64(len(program.clauses))
    for i in range(8):
        words[at + 1 + i] = UInt64(program.truth[i])
    for i in range(len(program.clauses)):
        ref leaf = program.clauses[i]
        ref clause = leaf.bounds
        ref input = inputs[program.slots[i]]
        var base = at + PREDICATE_HEADER_WORDS + i * PREDICATE_FIELD_WORDS
        words[base] = UInt64(program.slots[i])
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


def pack_value_program(
    nodes: List[ValueNode],
    conditions: List[PredicateProgram],
    inputs: List[NumericView],
) -> List[UInt64]:
    var words = List[UInt64](
        length=value_program_words(nodes, conditions), fill=0
    )
    var starts = List[Int]()
    var tail = len(nodes) * EXPRESSION_NODE_WORDS
    for condition in conditions:
        starts.append(tail)
        write_predicate(words, tail, condition, inputs)
        tail += predicate_words(condition)
    for i in range(len(nodes)):
        ref node = nodes[i]
        var at = i * EXPRESSION_NODE_WORDS
        words[at] = UInt64(node.op)
        if node.op == COL:
            ref input = inputs[node.input]
            words[at + 1] = UInt64(node.input)
            words[at + 2] = UInt64(input.bit_offset) | (
                UInt64(input.nullable) << 32
            )
        elif node.op == CASE_TEST:
            words[at + 1] = UInt64(starts[node.input])
        words[at + 3] = node.literal
    return words^
