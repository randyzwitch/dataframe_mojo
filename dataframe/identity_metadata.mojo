"""Shared metadata provenance for bare primitive columns and identity casts."""
from .binding import BoundExpr
from .expr import COL, CAST


def identity_column_source(bound: BoundExpr, index: Int) -> Int:
    """Return the bound source only for an unchanged primitive column value.

    Every CAST must preserve the complete logical dtype (units, precision,
    scale and timezone included). Computed inputs and nested casts do not
    inherit source metadata; this helper grants no operation admission.
    """
    if bound.dtypes[index].is_nested():
        return -1
    var current = index
    while bound.expr._nodes[current].op == CAST:
        var child = bound.expr._nodes[current].left
        if child < 0 or bound.dtypes[current] != bound.dtypes[child]:
            return -1
        current = child
    if bound.expr._nodes[current].op != COL:
        return -1
    return bound.sources[current]
