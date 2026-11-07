"""Host proofs and metadata tags for the register CASE evaluator."""
from dataframe.expr import COL, LIT_INT, ADD, SUB, MUL
from .expression_plan import ValueNode
from .expression_format import CASE_HEADER, CASE_TEST, CASE_JUMP, CASE_NULL


def simple_case_layout(nodes: List[ValueNode]) -> Int:
    """Classify only CASE layouts implemented by the register evaluator."""
    if not nodes or nodes[0].op != CASE_HEADER:
        return -1
    for shape in range(4):
        var then_binary = Bool(shape & 1)
        var else_binary = Bool(shape & 2)
        var otherwise = 6 if then_binary else 4
        var jump = otherwise - 1
        var size = otherwise + (3 if else_binary else 1)
        if len(nodes) != size:
            continue
        if (
            nodes[1].op != CASE_TEST
            or nodes[1].literal != UInt64(otherwise)
            or nodes[jump].op != CASE_JUMP
            or nodes[jump].literal != UInt64(size)
        ):
            continue
        var eligible = True
        for at in [2, otherwise]:
            eligible = eligible and nodes[at].op in [COL, LIT_INT, CASE_NULL]
        if then_binary:
            eligible = eligible and nodes[3].op in [COL, LIT_INT, CASE_NULL]
            eligible = eligible and nodes[4].op in [ADD, SUB, MUL]
        if else_binary:
            eligible = eligible and nodes[otherwise + 1].op in [
                COL,
                LIT_INT,
                CASE_NULL,
            ]
            eligible = eligible and nodes[otherwise + 2].op in [ADD, SUB, MUL]
        if eligible:
            return shape
    return -1


def case_shape_tag(nodes: List[ValueNode]) -> UInt64:
    # Zero retains the legacy/general program format. Tags1..4 are written
    # only after proving the complete instruction shape on the host.
    return UInt64(simple_case_layout(nodes) + 1)
