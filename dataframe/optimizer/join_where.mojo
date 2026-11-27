"""Lower a filter over a cross join to a join that finds only the kept pairs.

`join_where` (and any cross join filtered on both inputs' columns) is a
cross join under a filter. Pairing every row with every row and then
filtering does work quadratic in the inputs, so the filter's comparisons
of a left expression with a right one choose a join instead:

- Equalities of a left and a right column of one type: a hash join on
  them, keeping both sides' columns as the cross join does. A float pair
  also stays in the filter: the hash table matches NaN with NaN (and -0.0
  with 0.0), while the comparison matches NaN with nothing.
- Otherwise up to two inequalities (<, <=, >, >=) of number or temporal
  keys: an inequality join (RANGE_JOIN), which finds the pairs from sorted
  keys. Two comparisons reading one key of a side make a range join; two
  with different keys on both sides make a two-key join.
- Every other comparison stays in the filter, over the join's output.

A comparison's operand may be an expression of one side: it is computed as
a hidden column of that side, and a selection above the join drops it.
Right columns are named in the filter as the cross join outputs them (a
name the left side has carries the suffix), and are renamed to the right
input's names below the join.
"""
from std.collections import Dict, Optional

from dataframe.dtype import DataType
from dataframe.expr import (
    AND,
    COL,
    EQ,
    GE,
    GT,
    LE,
    LT,
    Expr,
    col,
    subtree,
)
from dataframe.frame import DataFrame
from dataframe.join_type import JOIN_CROSS, JOIN_INNER
from dataframe.joins.range_join import (
    IE_GE,
    IE_GT,
    IE_LE,
    IE_LT,
    ie_less,
    ie_name,
    ie_swapped,
    range_key_type,
)
from dataframe.planning.context import QueryContext
from dataframe.planning.expression_inputs import _references
from dataframe.planning.graph import plan_reorder
from dataframe.planning.model import FILTER, JOIN, PlanNode
from dataframe.planning.node_builders import (
    _range_join_node,
    _select_node,
    _with_columns_node,
)
from dataframe.planning.properties import _boolean_parts, _expr_signature
from dataframe.trace import trace_path


@fieldwise_init
struct _Comparison(Copyable, Movable):
    """A comparison `left OP right` of a left-side expression with a
    right-side one (renamed to the right input's names), `op` an IE_ code,
    or -1 for an equality."""

    var left: Expr
    var right: Expr
    var op: Int
    var part: Int
    var left_float: Bool


def _ie_code(op: Int) -> Int:
    if op == LT:
        return IE_LT
    if op == LE:
        return IE_LE
    if op == GT:
        return IE_GT
    if op == GE:
        return IE_GE
    return -1


def _plain_column(expr: Expr) -> Optional[String]:
    """The column an operand only reads, unchanged (an operand's own name
    does not matter inside a predicate)."""
    if len(expr._nodes) == 1 and expr._nodes[0].op == COL:
        return expr._nodes[0].text
    return None


def _side(
    expr: Expr, left: Dict[String, Bool], right_out: Dict[String, String]
) -> Int:
    """0 when `expr` reads only left columns, 1 when it reads only right
    columns (by output name), -1 otherwise (both, neither or a
    selector)."""
    var reads = _references(expr)
    if not reads or len(reads.value()) == 0:
        return -1
    var side = -1
    for name in reads.value():
        var here = 0 if name in left else (1 if name in right_out else -1)
        if here < 0 or (side >= 0 and here != side):
            return -1
        side = here
    return side


def renamed_columns(expr: Expr, names: Dict[String, String]) -> Expr:
    """`expr` reading `names[c]` for each column c it reads."""
    var result = expr.copy()
    for i in range(len(result._nodes)):
        if result._nodes[i].op == COL:
            var renamed = names.get(result._nodes[i].text)
            if renamed:
                result._nodes[i].text = renamed.value()
    return result^


def _dtype_of(frame: DataFrame, expr: Expr) -> Optional[DataType]:
    """The type `expr` has over `frame`'s columns (no rows), if it binds."""
    try:
        var out = frame._select_exprs([expr.alias("__dtype")])
        if out.width() != 1:
            return None
        return out._columns[0].dtype()
    except:
        return None


def plan_root(self: QueryContext) -> Int:
    """The plan's output node: the last node no other node reads."""
    var read = List[Bool](length=len(self._nodes), fill=False)
    for node in self._nodes:
        if node.left >= 0:
            read[node.left] = True
        if node.right >= 0:
            read[node.right] = True
    var root = len(self._nodes) - 1
    for i in range(len(self._nodes)):
        if not read[i]:
            root = i
    return root


def _readers(self: QueryContext, index: Int) -> Int:
    var count = 0
    for node in self._nodes:
        if node.left == index:
            count += 1
        if node.right == index:
            count += 1
    return count


def plan_lower_join_where(mut self: QueryContext) raises:
    """Lower each filter directly over a cross join (see the module
    docstring)."""
    var count = len(self._nodes)
    var root = plan_root(self)
    var changed = False
    for f in range(count):
        if self._nodes[f].kind != FILTER or len(self._nodes[f].exprs) != 1:
            continue
        var c = self._nodes[f].left
        if (
            c < 0
            or self._nodes[c].kind != JOIN
            or self._nodes[c].how != JOIN_CROSS
            or _readers(self, c) != 1
        ):
            continue
        changed |= _lower(self, f, c)
    if changed:
        trace_path("lazy.lower_join_where")
        # Nodes added here come after the root; numbering from the root
        # keeps it last and drops the replaced cross joins.
        plan_reorder(self, root)


def _lower(mut self: QueryContext, f: Int, c: Int) raises -> Bool:
    var join = self._nodes[c].copy()
    var left_frame = self._execute(join.left, True)
    var right_frame = self._execute(join.right, True)
    var suffix = join.join_suffix()
    var left = Dict[String, Bool]()
    for name in left_frame.columns():
        left[name] = True
    # Right output name -> right input name.
    var right_out = Dict[String, String]()
    var output = left_frame.columns()
    for name in right_frame.columns():
        var shown = name + suffix if name in left else name
        right_out[shown] = name
        output.append(shown)
    var parts = _boolean_parts(self._nodes[f].exprs[0], AND)
    var equal = List[_Comparison]()
    var unequal = List[_Comparison]()
    for p in range(len(parts)):
        ref part = parts[p]
        ref root = part._nodes[len(part._nodes) - 1]
        if root.op != EQ and _ie_code(root.op) < 0:
            continue
        if root.left < 0 or root.right < 0:
            continue
        var a = subtree(part, root.left)
        var b = subtree(part, root.right)
        var side_a = _side(a, left, right_out)
        var side_b = _side(b, left, right_out)
        var op = -1 if root.op == EQ else _ie_code(root.op)
        if side_a == 1 and side_b == 0:
            var t = a.copy()
            a = b.copy()
            b = t^
            if op >= 0:
                op = ie_swapped(op)
        elif not (side_a == 0 and side_b == 1):
            continue
        var rb = renamed_columns(b, right_out)
        var ltype = _dtype_of(left_frame, a)
        var rtype = _dtype_of(right_frame, rb)
        if not ltype or not rtype:
            continue
        if op < 0:
            if (
                _plain_column(a)
                and _plain_column(rb)
                and ltype.value() == rtype.value()
                and not ltype.value().is_nested()
            ):
                equal.append(
                    _Comparison(
                        a.copy(), rb.copy(), op, p, ltype.value().is_float()
                    )
                )
        elif range_key_type(ltype.value(), rtype.value()):
            unequal.append(
                _Comparison(
                    a.copy(), rb.copy(), op, p, ltype.value().is_float()
                )
            )
    if len(equal) == 0 and len(unequal) == 0:
        return False
    var used = List[Bool](length=len(parts), fill=False)
    var joined: PlanNode
    var hidden = False
    if len(equal) > 0:
        # A hash join on the equalities, keeping both sides' columns.
        var left_keys = List[String]()
        var right_keys = List[String]()
        for e in equal:
            left_keys.append(_plain_column(e.left).value())
            right_keys.append(_plain_column(e.right).value())
            # A float pair stays in the filter as well: the hash table
            # matches NaN with NaN, the comparison matches NaN with
            # nothing.
            if not e.left_float:
                used[e.part] = True
        joined = join.copy()
        joined.how = JOIN_INNER
        joined.text = "inner"
        joined.names = left_keys^
        joined.right_keys = right_keys^
        joined.coalesce = False
        trace_path("lazy.join_where_hash")
    else:
        var chosen = _choose(unequal)
        var left_exprs = List[Expr]()
        var right_exprs = List[Expr]()
        var left_names = List[String]()
        var right_names = List[String]()
        var ops = List[String]()
        var left_hidden = Dict[String, String]()
        var right_hidden = Dict[String, String]()
        var taken = Dict[String, Bool]()
        for name in output:
            taken[name] = True
        for name in right_frame.columns():
            taken[name] = True
        for k in chosen:
            ref comparison = unequal[k]
            used[comparison.part] = True
            ops.append(ie_name(comparison.op))
            left_names.append(
                _key_name(
                    comparison.left,
                    "__ie_left_",
                    left_hidden,
                    left_exprs,
                    taken,
                )
            )
            right_names.append(
                _key_name(
                    comparison.right,
                    "__ie_right_",
                    right_hidden,
                    right_exprs,
                    taken,
                )
            )
        var left_input = join.left
        var right_input = join.right
        if len(left_exprs) > 0:
            self._nodes.append(
                _with_columns_node(left=left_input, exprs=left_exprs)
            )
            left_input = len(self._nodes) - 1
        if len(right_exprs) > 0:
            self._nodes.append(
                _with_columns_node(left=right_input, exprs=right_exprs)
            )
            right_input = len(self._nodes) - 1
        hidden = len(left_exprs) > 0 or len(right_exprs) > 0
        joined = _range_join_node(
            left_names,
            right_names,
            String(",").join(ops),
            suffix,
            left=left_input,
            right=right_input,
        )
        trace_path("lazy.join_where_range")
    var rest = Optional[Expr]()
    for p in range(len(parts)):
        if used[p]:
            continue
        if rest:
            rest = rest.value() & parts[p]
        else:
            rest = parts[p].copy()
    # The join takes the cross join's place; the filter keeps what is
    # left, above a selection of the cross join's columns when hidden keys
    # were added.
    self._nodes[c] = joined^
    var top = c
    if hidden:
        var keep = List[Expr]()
        for name in output:
            keep.append(col(name))
        self._nodes.append(_select_node(left=c, exprs=keep))
        top = len(self._nodes) - 1
    if rest:
        self._nodes[f].exprs = [rest.take()]
        self._nodes[f].left = top
    else:
        self._nodes[f] = self._nodes[top].copy()
    return True


def _key_name(
    expr: Expr,
    prefix: String,
    mut hidden: Dict[String, String],
    mut exprs: List[Expr],
    mut taken: Dict[String, Bool],
) raises -> String:
    """The column an inequality join compares for `expr`: the column it
    reads, or a hidden column computing it (one per distinct
    expression)."""
    var plain = _plain_column(expr)
    if plain:
        return plain.value()
    var signature = _expr_signature(expr)
    if signature in hidden:
        return hidden[signature]
    var name = prefix + String(len(exprs))
    while name in taken:
        name = "_" + name
    taken[name] = True
    hidden[signature] = name
    exprs.append(expr.alias(name))
    return name


def _choose(comparisons: List[_Comparison]) -> List[Int]:
    """The comparisons an inequality join uses: a pair bounding one key of
    a side from both directions (a range), else the first two, else the
    one."""
    for i in range(len(comparisons)):
        for j in range(i):
            ref a = comparisons[j]
            ref b = comparisons[i]
            if ie_less(a.op) == ie_less(b.op):
                continue
            var same_left = _expr_signature(a.left) == _expr_signature(b.left)
            var same_right = _expr_signature(a.right) == _expr_signature(
                b.right
            )
            if same_left or same_right:
                return [j, i]
    if len(comparisons) >= 2:
        return [0, 1]
    return [0]
