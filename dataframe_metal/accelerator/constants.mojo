"""Fold source-independent scalar parsing and casts before dictionary encoding."""
from std.collections import Optional
from dataframe.column import Column
from dataframe.binding import bind
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.expr import (
    Expr,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
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
            or node.op == LIT_BOOL
            or node.op == LIT_STRING
            or node.op == LIT_NULL
            or node.op == CAST
            or is_string_op(node.op)
            or is_dt_op(node.op)
        ):
            return False
    return True


def _checked_temporal_literal(expression: Expr, root: Int) -> Bool:
    """Keep known overflowing physical temporal casts inside their masks.

    This is a static literal-word proof, not Series evaluation. Existing text
    parsing and datetime-operation folding keep their previous path. Ancestors
    of a deferred strict cast must not evaluate it indirectly either.
    """
    try:
        var fragment = subtree(expression, root)
        var temporal = False
        for node in fragment._nodes:
            if len(node.dtypes) and node.dtypes[0]:
                temporal = temporal or node.dtypes[0].value().is_temporal()
        if not temporal:
            return False
        var bound = bind(fragment, List[Series]())
        var count = len(bound.dtypes)
        var known = List[Bool](length=count, fill=False)
        var absent = List[Bool](length=count, fill=False)
        var words = List[Int128](length=count, fill=0)
        for index in range(count):
            ref node = bound.expr._nodes[index]
            var target = bound.dtypes[index]
            if node.op == LIT_NULL:
                known[index] = True
                absent[index] = True
                continue
            if node.op == LIT_INT and (
                target == DataType.INT64 or target.is_temporal()
            ):
                known[index] = True
                words[index] = Int128(node.integer)
                continue
            if node.op != CAST or node.left < 0 or not known[node.left]:
                continue
            var source = bound.dtypes[node.left]
            if not (
                (source == DataType.INT64 or source.is_temporal())
                and (target == DataType.INT64 or target.is_temporal())
                and (source.is_temporal() or target.is_temporal())
            ):
                continue
            if absent[node.left]:
                known[index] = True
                absent[index] = True
                continue
            var value = words[node.left]
            if (
                source == DataType.INT64
                or target == DataType.INT64
                or source == target
            ):
                pass
            elif source.is_date() and target.is_datetime():
                value *= Int128(86400) * Int128(target.per_second())
            elif (source.is_datetime() and target.is_datetime()) or (
                source.is_duration() and target.is_duration()
            ):
                var a = Int128(source.per_second())
                var b = Int128(target.per_second())
                value = value * (b // a) if b >= a else value // (a // b)
            elif (
                source.is_datetime()
                and (source.time_zone() == "" or source.time_zone() == "UTC")
                and (target.is_date() or target.is_time())
            ):
                var day = Int128(86400) * Int128(source.per_second())
                value = (
                    value
                    // day if target.is_date() else (value % day)
                    * (Int128(1000000000) // Int128(source.per_second()))
                )
            else:
                continue
            if value < Int128(Int64.MIN) or value > Int128(Int64.MAX):
                if node.integer:
                    return True
                absent[index] = True
            known[index] = True
            words[index] = value
    except:
        # Preserve the established literal parser's handling of unrelated
        # invalid expressions; this proof only defers known checked casts.
        pass
    return False


def _foldable(expression: Expr, root: Int) -> Bool:
    ref node = expression._nodes[root]
    if node.op == DT_STRPTIME:
        return _constant(expression, root) and not _checked_temporal_literal(
            expression, root
        )
    if node.op != CAST or len(node.dtypes) == 0 or not node.dtypes[0]:
        return False
    var dtype = node.dtypes[0].value()
    # Dictionary codes cannot represent numeric-to-string formatting, and
    # categorical codes need their dictionary rather than a bare integer.
    return (
        (dtype.is_numeric() or dtype.is_temporal())
        and _constant(expression, root)
        and not _checked_temporal_literal(expression, root)
    )


def has_literal_text_parsing(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        for expression in operation.exprs:
            for n in range(len(expression._nodes)):
                if _foldable(expression, n):
                    return True
    return False


def fold_literal_text_parsing(query: LazyFrame) raises -> LazyFrame:
    """Evaluate source-independent text literals, never input rows or columns.

    Invalid literal parsing remains unsupported, so preflight cannot expose a
    CPU parse error from an unobserved branch. Temporal and decimal logical dtypes
    survive replacement with their physical integer literal representation.
    """
    var result = query.copy()
    for o in range(len(result._nodes)):
        for e in range(len(result._nodes[o].exprs)):
            var expression = result._nodes[o].exprs[e].copy()
            var changed = False
            for n in range(len(expression._nodes)):
                if not _foldable(expression, n):
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
                    if dtype.is_decimal() and dtype.decimal_width() == 128:
                        ref values = column._data[Column[Int128]]
                        if values.is_valid(0):
                            var raw = values.value(0)
                            replacement = lit(raw.cast[DType.int64]())
                            replacement._nodes[0].text2 = String(
                                (raw >> 64).cast[DType.int64]()
                            )
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
