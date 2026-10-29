"""Expressions whose row axis comes from frame height rather than a column."""
from .expr import Expr, ROW_LEN, INT_RANGE, _node, lit
from std.builtin.len import len as builtin_len


def len() -> Expr:
    """Number of input rows, including nulls, as an Int64 aggregate.

    Works on zero-column frames and restarts inside each over()/agg() group.
    Alias this import as `row_len` when also using the built-in len function.
    """
    return Expr([_node(ROW_LEN)], "len")


def int_range(start: Expr, end: Expr, step: Int = 1) raises -> Expr:
    """Int64 values from start to exclusive end, using a nonzero signed step.

    Bounds must each be one non-null integer scalar or aggregate. Use
    int_range(row_len()).over(key) for zero-based positions within groups.
    """
    if step == 0:
        raise Error("int_range step must be nonzero")
    var result = start._binary(end, INT_RANGE)
    result._nodes[builtin_len(result._nodes) - 1].integer = Int64(step)
    return result.alias("int_range")


def int_range(end: Expr, *, step: Int = 1) raises -> Expr:
    """Int64 values from zero to exclusive end, with a nonzero signed step."""
    return int_range(lit(Int64(0)), end, step)
