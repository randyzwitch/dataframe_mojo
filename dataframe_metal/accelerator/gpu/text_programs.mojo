"""Private text literal provenance across resident dictionary unions."""
from std.memory import ArcPointer
from dataframe.lazy import LazyFrame, SCAN_FRAME
from dataframe.expr import LIT_INT
from dataframe_metal.accelerator.code_domains import code, TEXT_LITERAL
from .residency import TextDomain


def recode_region(mut query: LazyFrame, target: ArcPointer[TextDomain]) raises:
    """Update literal/control lookup metadata, never source row buffers.

    The SCAN's domain remains its original encoding; the consumer remap uses
    that identity. Non-scan nodes bind against the new sorted text union.
    """
    for at in range(len(query._nodes)):
        ref operation = query._nodes[at]
        if operation.kind != SCAN_FRAME:
            operation.code_domain = target[].dtype
        for output in range(len(operation.exprs)):
            ref expression = operation.exprs[output]
            for index in range(len(expression._nodes)):
                ref node = expression._nodes[index]
                if node.op == LIT_INT and node.min_count == TEXT_LITERAL:
                    node.integer = Int64(code(target[].dtype, node.text2))
