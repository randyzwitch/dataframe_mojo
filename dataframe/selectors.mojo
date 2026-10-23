"""Selector expansion: one bound expression per matched column."""
from std.collections import Optional
from .dtype import DataType
from .expr import (
    split_names,
    node_children,
    _set_node_children,
    COL,
    H_ALL,
    H_ANY,
    HORIZONTAL,
    HORIZONTAL_ARGS,
    LIT_BOOL,
    horizontal_name,
    subtree,
    Node,
    _append_shifted,
    SELECT_ALL,
    SELECT_COLS,
    SELECT_REGEX,
    SELECT_DTYPE,
    SELECT_EXCLUDE,
    SELECTOR,
    Expr,
    _node,
)
from .series import Series
from .regex import Regex, Matcher


def _split(text: String) -> List[String]:
    var out = List[String]()
    if text.byte_length() == 0:
        return out^
    return split_names(text)


def _regex_names(pattern: String, names: List[String]) raises -> List[String]:
    var matcher = Matcher(Regex(pattern))
    var result = List[String]()
    for name in names:
        var bytes = Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=name.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
            length=name.byte_length(),
        )
        if matcher.is_match(bytes):
            result.append(name)
    return result^


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
    if kind == SELECT_REGEX:
        return _regex_names(items, names)
    if kind == SELECT_COLS or kind == SELECT_EXCLUDE:
        var listed = List[String]()
        var regex = False
        for item in _split(items):
            if item.startswith("^") and item.endswith("$"):
                regex = True
                listed.extend(_regex_names(item, names))
            else:
                listed.append(item)
        for name in listed:
            var found = False
            for existing in names:
                found = found or existing == name
            if not found:
                raise Error("Unknown selector column: " + name)
        if kind == SELECT_COLS:
            if not regex:
                return listed^
            # Polars treats a mixed regex/name list as a schema-ordered
            # union, while a list of exact names retains its explicit order.
            for name in names:
                if name in listed:
                    result.append(name)
            return result^
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


def _parents(nodes: List[Node]) -> List[Int]:
    var parents = List[Int](length=len(nodes), fill=-1)
    for i in range(len(nodes)):
        ref node = nodes[i]
        for child in node_children(node):
            parents[child] = i
    return parents^


def _emit(
    source: List[Node],
    at: Int,
    target: Int,
    replacement: List[Node],
    mut out: List[Node],
) -> Int:
    """Copy the subtree at `at` into `out`, with the subtree at `target`
    replaced by `replacement`, or dropped when that is empty: a dropped
    horizontal input leaves its sibling in its place. Returns the copy's
    root, or -1 when it was dropped."""
    if at == target:
        if len(replacement) == 0:
            return -1
        _append_shifted(out, replacement, len(out))
        return len(out) - 1
    var node = source[at].copy()
    if node.op == HORIZONTAL_ARGS:
        var left = _emit(source, node.left, target, replacement, out)
        var right = _emit(source, node.right, target, replacement, out)
        if left < 0:
            return right
        if right < 0:
            return left
        node.left = left
        node.right = right
    else:
        var children = node_children(node)
        for k in range(len(children)):
            children[k] = _emit(source, children[k], target, replacement, out)
        _set_node_children(node, children)
    out.append(node^)
    return len(out) - 1


def _replaced(expression: Expr, target: Int, replacement: List[Node]) -> Expr:
    var nodes = List[Node]()
    _ = _emit(
        expression._nodes,
        len(expression._nodes) - 1,
        target,
        replacement,
        nodes,
    )
    return Expr(nodes^, expression._name)


def _expand_inputs(expression: Expr, columns: List[Series]) raises -> Expr:
    """Expand each selector that is (part of) an input of a horizontal
    reduction into one input per matched column while keeping the
    expression as one expression. Other selectors are left alone."""
    var current = expression.copy()
    while True:
        var parents = _parents(current._nodes)
        var selector = -1
        var operand = -1
        var holder = -1
        for i in range(len(current._nodes)):
            if current._nodes[i].op != SELECTOR:
                continue
            var at = i
            while parents[at] >= 0:
                var op = current._nodes[parents[at]].op
                if op == HORIZONTAL or op == HORIZONTAL_ARGS:
                    selector = i
                    operand = at
                    holder = parents[at]
                    break
                at = parents[at]
            if selector >= 0:
                break
        if selector < 0:
            return current^
        ref node = current._nodes[selector]
        var parts = split_names(node.text)
        var names = _matches(
            node.min_count, node.text2, node.integer, node.dtypes, columns
        )
        var kind = Int(current._nodes[holder].integer)
        if len(names) == 0 and current._nodes[holder].op == HORIZONTAL:
            # The selector was the only input left.
            if kind != H_ANY and kind != H_ALL:
                raise Error(
                    horizontal_name(kind)
                    + " requires at least one input; the selector matched"
                    " no columns"
                )
            var literal = Expr(
                [_node(LIT_BOOL, integer=Int64(kind == H_ALL))],
                horizontal_name(kind),
            )
            var name = current._name
            current = _replaced(current, holder, literal._nodes)
            current._name = name if name.byte_length() > 0 else literal._name
            continue
        var input = subtree(current, operand)
        var at = -1
        for k in range(len(input._nodes)):
            if input._nodes[k].op == SELECTOR:
                if at >= 0:
                    raise Error(
                        "An expression may contain at most one selector"
                    )
                at = k
        var inputs = List[Node]()
        for k in range(len(names)):
            var copied = input._nodes.copy()
            copied[at] = _node(COL, text=names[k])
            var offset = len(inputs)
            _append_shifted(inputs, copied, offset)
            if k > 0:
                inputs.append(
                    _node(
                        HORIZONTAL_ARGS,
                        offset - 1,
                        len(inputs) - 1,
                        integer=Int64(kind),
                    )
                )
        if current._name.byte_length() == 0 and len(names) > 0:
            current._name = parts[0] + names[0] + parts[1]
        current = _replaced(current, operand, inputs)


def expand(expression: Expr, columns: List[Series]) raises -> List[Expr]:
    """Replace a selector leaf with each matched column.

    Output names: an explicit alias wins; otherwise prefix + column name +
    suffix. At most one selector may appear in an expression, apart from
    selectors among a horizontal reduction's inputs, which expand into
    more inputs of that one expression.
    """
    for node in expression._nodes:
        if node.op == HORIZONTAL:
            return _expanded(_expand_inputs(expression, columns), columns)
    return _expanded(expression, columns)


def _expanded(expression: Expr, columns: List[Series]) raises -> List[Expr]:
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
    var parts = split_names(node.text)
    var prefix = parts[0]
    var suffix = parts[1]
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
