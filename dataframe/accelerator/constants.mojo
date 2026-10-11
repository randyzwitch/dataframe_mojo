"""Fold literal text parsing before accelerator dictionary encoding."""
from std.collections import Optional
from dataframe.column import Column
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.expr import (
    Expr,
    LIT_INT,
    LIT_FLOAT,
    LIT_STRING,
    LIT_NULL,
    CAST,
    DT_STRPTIME,
    is_string_op,
    is_dt_op,
    lit,
    null,
    subtree,
)
from dataframe.frame import DataFrame
from dataframe.lazy import LazyFrame
from dataframe.series import Series


def _constant(expression: Expr, root: Int) -> Bool:
    var fragment = subtree(expression, root)
    for node in fragment._nodes:
        if not (
            node.op == LIT_INT
            or node.op == LIT_FLOAT
            or node.op == LIT_STRING
            or node.op == LIT_NULL
            or node.op == CAST
            or is_string_op(node.op)
            or is_dt_op(node.op)
        ):
            return False
    return True


def has_literal_text_parsing(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        for expression in operation.exprs:
            for n in range(len(expression._nodes)):
                if expression._nodes[n].op == DT_STRPTIME and _constant(
                    expression, n
                ):
                    return True
    return False


def fold_literal_text_parsing(query: LazyFrame) raises -> LazyFrame:
    """Evaluate source-independent text literals, never input rows or columns.

    Invalid literal parsing remains unsupported, so preflight cannot expose a
    CPU parse error from an unobserved branch. Temporal logical dtypes survive
    replacement with their physical integer literal representation.
    """
    var result = query.copy()
    for o in range(len(result._nodes)):
        for e in range(len(result._nodes[o].exprs)):
            var expression = result._nodes[o].exprs[e].copy()
            var changed = False
            for n in range(len(expression._nodes)):
                if expression._nodes[n].op != DT_STRPTIME or not _constant(
                    expression, n
                ):
                    continue
                try:
                    var frame = DataFrame(List[Series]()).select_exprs(
                        [subtree(expression, n)]
                    )
                    if frame.height() != 1 or frame.width() != 1:
                        raise Error(
                            "literal parsing did not produce one scalar"
                        )
                    ref column = frame._columns[0]
                    var dtype = column.dtype()
                    var replacement = null(dtype)
                    var found = False
                    comptime for D in NUMERIC_DTYPES:
                        if dtype.physical() == DataType.of(D):
                            ref values = column._data[Column[Scalar[D]]]
                            if values.is_valid(0):
                                replacement = lit(values.value(0))
                                replacement._nodes[0].dtypes = [
                                    Optional[DataType](dtype)
                                ]
                            found = True
                    if not found:
                        raise Error("unsupported parsed literal dtype")
                    expression._nodes[n] = replacement._nodes[0].copy()
                    changed = True
                except error:
                    raise Error(
                        "Metal unsupported [literal text parsing]: "
                        + String(error)
                    )
            if changed:
                result._nodes[o].exprs[e] = subtree(
                    expression, len(expression._nodes) - 1
                )
    return result^
