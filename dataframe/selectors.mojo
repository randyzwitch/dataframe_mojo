"""Selector expansion: one bound expression per matched column."""
from std.collections import Optional
from .dtype import DataType
from .expr import (
    COL,
    SELECT_ALL,
    SELECT_COLS,
    SELECT_DTYPE,
    SELECT_EXCLUDE,
    SELECTOR,
    SEP,
    Expr,
    _node,
)
from .series import Series


def _split(text: String) -> List[String]:
    var out = List[String]()
    if text.byte_length() == 0:
        return out^
    for part in text.split(SEP):
        out.append(String(part))
    return out^


def _matches(
    kind: Int,
    items: String,
    index: Int64,
    parsed: List[Optional[DataType]],
    columns: List[Series],
) raises -> List[String]:
    var names = List[String]()
    for column in columns:
        names.append(column.name())
    var result = List[String]()
    if kind == SELECT_ALL:
        return names^
    if kind == SELECT_COLS or kind == SELECT_EXCLUDE:
        var listed = _split(items)
        for name in listed:
            var found = False
            for existing in names:
                found = found or existing == name
            if not found:
                raise Error("Unknown selector column: " + name)
        if kind == SELECT_COLS:
            return listed^
        for name in names:
            var excluded = False
            for skip in listed:
                excluded = excluded or skip == name
            if not excluded:
                result.append(name)
        return result^
    if kind == SELECT_DTYPE:
        var dtypes = List[DataType]()
        var listed = _split(items)
        for i in range(len(listed)):
            if not parsed[i]:
                raise Error("Unknown selector dtype: " + listed[i])
            dtypes.append(parsed[i].value())
        for column in columns:
            for dtype in dtypes:
                if column.dtype() == dtype:
                    result.append(column.name())
                    break
        return result^
    var position = Int(index)
    if position < 0:
        position += len(names)
    if position < 0 or position >= len(names):
        raise Error("nth selector index " + String(index) + " is out of range")
    result.append(names[position])
    return result^


def expand(expression: Expr, columns: List[Series]) raises -> List[Expr]:
    """Replace a selector leaf with each matched column.

    Output names: an explicit alias wins; otherwise prefix + column name +
    suffix. At most one selector may appear in an expression.
    """
    var selector = -1
    for i in range(len(expression._nodes)):
        if expression._nodes[i].op == SELECTOR:
            if selector >= 0:
                raise Error("An expression may contain at most one selector")
            selector = i
    var result = List[Expr]()
    if selector < 0:
        result.append(expression.copy())
        return result^
    ref node = expression._nodes[selector]
    var parts = node.text.split(SEP)
    var prefix = String(parts[0])
    var suffix = String(parts[1])
    for name in _matches(
        node.min_count, node.text2, node.integer, node.dtypes, columns
    ):
        var nodes = expression._nodes.copy()
        nodes[selector] = _node(COL, text=name)
        var output = expression._name
        if output.byte_length() == 0:
            output = prefix + name + suffix
        result.append(Expr(nodes^, output))
    return result^


def expand_all(
    expressions: List[Expr], columns: List[Series]
) raises -> List[Expr]:
    var result = List[Expr]()
    for expression in expressions:
        for expanded in expand(expression, columns):
            result.append(expanded.copy())
    return result^
