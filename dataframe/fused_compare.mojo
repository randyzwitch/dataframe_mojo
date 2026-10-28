"""The AND of a filter's column-versus-constant comparisons in one pass.

A filter like PDS-H q6's (a date range, a discount range and a quantity
bound: five comparisons on three columns) evaluated each comparison as its
own expression, each writing a full-length bitmap that the next AND read
back, about 2 ns a row a comparison. Here every comparison of a numeric
column against a constant of the column's own type runs over
cache-sized blocks of rows, and each block's bits are ANDed in a buffer that
stays in L1, so each column is read from memory once and one bitmap is
written. The combined bitmap is the selection for the next pipeline step.
Semantics are the comparison kernels' (`_compare_lanes`): IEEE for
floats, so a NaN compares false except with `!=`.
"""

from .expr import node_children
from .binding import bind
from dataframe.storage.column import Column
from .dtype import NUMERIC_DTYPES
from dataframe.execution import _numeric_literal, evaluate
from .expr import (
    AND,
    COL,
    EQ,
    GE,
    GT,
    IS_NOT_NULL,
    IS_NULL,
    LE,
    LIT_FLOAT,
    LIT_INT,
    LT,
    NE,
    OR,
    UNTYPED,
    Expr,
    subtree,
)
from dataframe.kernels.binary import _validity_bytes
from .series import Series

# Rows a block: each column's slice of a block stays in L1 (8 KB of an
# 8-byte column) while every comparison on it runs.
from dataframe.kernels.fused_compare import (
    FusedComparisons,
    SelectionTree,
    _ALL,
    _ANY,
    _BLOCK_ROWS,
    _COMPARE,
    _Comparison,
    _FusedJob,
    _MISSING,
    _TreeNode,
    _VALID,
    _and_valid,
    _apply,
    _block,
    _merged,
    _select_compare,
    _select_rows,
    _without,
    fused_mask,
    run_comparisons,
)


def _flip(op: Int) -> Int:
    if op == LT:
        return GT
    if op == LE:
        return GE
    if op == GT:
        return LT
    if op == GE:
        return LE
    return op


def _is_constant(expr: Expr, index: Int) -> Bool:
    var stack: List[Int] = [index]
    while len(stack) > 0:
        var i = stack.pop()
        ref node = expr._nodes[i]
        if node.op == COL:
            return False
        for child in node_children(node):
            stack.append(child)
    return True


def _comparison(
    part: Expr, columns: List[Series]
) raises -> Optional[_Comparison]:
    """The part as `column op constant`, when the column is numeric and
    the constant evaluates to one valid value of the column's exact
    dtype; None for any other part. A null row compares false, as a
    filter drops it."""
    var root = len(part._nodes) - 1
    ref node = part._nodes[root]
    var op = node.op
    if not (
        op == EQ or op == NE or op == LT or op == LE or op == GT or op == GE
    ):
        return None
    if node.left < 0 or node.right < 0:
        return None
    var column_at: Int
    var constant_at: Int
    if part._nodes[node.left].op == COL and _is_constant(part, node.right):
        column_at = node.left
        constant_at = node.right
    elif part._nodes[node.right].op == COL and _is_constant(part, node.left):
        column_at = node.right
        constant_at = node.left
        op = _flip(op)
    else:
        return None
    var name = part._nodes[column_at].text
    var found = -1
    for i in range(len(columns)):
        if columns[i].name() == name:
            found = i
    if found < 0:
        return None
    ref column = columns[found]
    if column.is_chunked():
        return None
    # A validity bitmap with no null in it (Parquet imports carry one) is
    # as good as none: every row's value is read.
    var numeric = False
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if column._data.isa[Column[Scalar[D]]]():
            numeric = True
    if not numeric:
        return None
    # A typed numeric literal (what the optimizer folds a constant cast
    # into) is read straight from its node; anything else is bound and
    # evaluated. Filters build these comparisons for every morsel.
    var value: Series
    ref constant = part._nodes[constant_at]
    if (
        (constant.op == LIT_INT or constant.op == LIT_FLOAT)
        and constant.text != UNTYPED
        and len(constant.dtypes) == 1
        and constant.dtypes[0]
    ):
        value = _numeric_literal(constant)
    else:
        var no_columns = List[Series]()
        value = evaluate(
            bind(subtree(part, constant_at), no_columns), no_columns, 1
        )
    if len(value) != 1 or value.is_chunked() or value.null_count() > 0:
        return None
    if value.dtype() != column.dtype():
        return None
    return Optional(
        _Comparison(column.copy(), op, value^, column.null_count() > 0)
    )


def fused_comparisons(
    columns: List[Series], height: Int, parts: List[Expr]
) raises -> FusedComparisons:
    """The AND of the parts that compare a numeric column without nulls
    with a constant of its dtype, as a bitmap of `height` rows, and which
    parts it answers. No part answered: an empty bitmap."""
    var used = List[Bool](length=len(parts), fill=False)
    var comparisons = List[_Comparison]()
    for k in range(len(parts)):
        var found = _comparison(parts[k], columns)
        if found:
            if len(found.value().column) != height:
                continue
            comparisons.append(found.value().copy())
            used[k] = True
    return run_comparisons(comparisons, height, used^)


def selection_tree(
    part: Expr, columns: List[Series]
) raises -> Optional[SelectionTree]:
    """`part` as a selection tree when every leaf is a comparison of a
    numeric column with a constant of its dtype, or a null test of a
    column; None otherwise."""
    var tree = SelectionTree()
    if not _add_tree(part, len(part._nodes) - 1, columns, tree):
        return None
    return Optional(tree^)


def _add_tree(
    part: Expr, at: Int, columns: List[Series], mut tree: SelectionTree
) raises -> Bool:
    """Append the subtree rooted at `at`, its root last; False when some
    leaf is unsupported."""
    ref node = part._nodes[at]
    if (node.op == AND or node.op == OR) and node.left >= 0 and node.right >= 0:
        if not _add_tree(part, node.left, columns, tree):
            return False
        var left = len(tree.nodes) - 1
        if not _add_tree(part, node.right, columns, tree):
            return False
        var right = len(tree.nodes) - 1
        tree.nodes.append(
            _TreeNode(_ALL if node.op == AND else _ANY, left, right, -1)
        )
        return True
    if (node.op == IS_NULL or node.op == IS_NOT_NULL) and node.left >= 0:
        ref child = part._nodes[node.left]
        if child.op != COL:
            return False
        for column in columns:
            if column.name() == child.text and not column.is_chunked():
                tree.tested.append(_validity_bytes(column, len(column)))
                tree.nodes.append(
                    _TreeNode(
                        _VALID if node.op == IS_NOT_NULL else _MISSING,
                        -1,
                        -1,
                        len(tree.tested) - 1,
                    )
                )
                return True
        return False
    var found = _comparison(subtree(part, at), columns)
    if not found:
        return False
    tree.comparisons.append(found.value().copy())
    tree.nodes.append(_TreeNode(_COMPARE, -1, -1, len(tree.comparisons) - 1))
    return True
