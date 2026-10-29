"""Expression and operation properties shared by planning and execution."""
from dataframe.planning.cost.eligibility import (
    _counts_distinct,
    _order_sensitive,
    _has_quantile,
    _row_steps_only,
    _filters,
    _key_read,
    _row_merge_reductions,
    _late_key_build,
)

from dataframe.planning.expression_inputs import _references
from dataframe.planning.cost.sampling import (
    _keys_repeat,
    _key_group_bound,
    _unique_key_column,
)

from dataframe.aggregation.hash_table import (
    reduction_supported,
)
from dataframe.binding import (
    AGGREGATE,
    ROWS,
    bind,
)
from dataframe.storage.column import (
    Column,
)

from dataframe.dtype import (
    DataType,
    NUMERIC_DTYPES,
)
from dataframe.expr import (
    ALL,
    AND,
    ANY,
    ARG_MAX,
    ARG_MIN,
    COL,
    CORR,
    COUNT,
    COV,
    EQ,
    Expr,
    FIRST,
    GE,
    GT,
    LAST,
    LE,
    LEN,
    LIT_BOOL,
    LIT_FLOAT,
    LIT_INT,
    LIT_STRING,
    LT,
    MAX,
    MEAN,
    MEDIAN,
    MIN,
    NE,
    NULL_COUNT,
    N_UNIQUE,
    Node,
    OR,
    OVER,
    QUANTILE,
    SELECTOR,
    STD,
    SUM,
    UNTYPED,
    VAR,
    col,
    is_reduction,
    is_sequence,
    is_window,
    lit,
    subtree,
)
from dataframe.frame import (
    DataFrame,
    concat,
)
from dataframe.aggregation.expressions import (
    _bind_all,
    _row_local,
)

from dataframe.join_order import (
    FilterEdge,
)
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_SEMI,
    join_type,
)
from dataframe.planning.model import (
    AGG,
    DROP,
    EXPLODE,
    FILL_NULL_STRATEGY,
    FILTER,
    GATHER_EVERY,
    GROUP_BY_DYNAMIC,
    INSERT_COLUMN,
    JOIN,
    PlanNode,
    REVERSE,
    ROLLING,
    ROW_INDEX,
    SCAN_CSV,
    SCAN_FRAME,
    SCAN_IPC,
    SCAN_NDJSON,
    SCAN_PARQUET,
    SELECT,
    SHIFT,
    TAIL,
    UNNEST,
    WITH_COLUMNS,
)
from dataframe.series import (
    Series,
)
from std.collections import (
    Dict,
    Optional,
)
from std.memory import (
    bitcast,
)


def _comparison_edge(
    expr: Expr, provider: Dict[String, Int]
) raises -> Optional[FilterEdge]:
    """A comparison whose two sides each read one column from different
    relations (`provider` maps each column to its relation), represented
    as a join-graph edge between those columns; None for anything else."""
    ref root = expr._nodes[len(expr._nodes) - 1]
    var op = root.op
    if not (
        op == EQ or op == NE or op == LT or op == LE or op == GT or op == GE
    ):
        return None
    if root.left < 0 or root.right < 0:
        return None
    var sides = List[String]()
    var relations = List[Int]()
    for child in [root.left, root.right]:
        var reads = _references(subtree(expr, child))
        if not reads:
            return None
        var names = reads.value().copy()
        if len(names) == 0:
            return None
        for name in names:
            if name != names[0]:
                return None
        var r = provider.get(names[0], -1)
        if r < 0:
            return None
        sides.append(names[0])
        relations.append(r)
    if relations[0] == relations[1]:
        return None
    return FilterEdge(relations[0], relations[1], sides[0], sides[1], op == EQ)


def _output_names(exprs: List[Expr]) -> List[String]:
    var names = List[String]()
    for e in exprs:
        names.append(e._name)
    return names^


def _produced_at(exprs: List[Expr], name: String) -> Int:
    """The last of `exprs` named `name` (the one a projection keeps), or
    -1."""
    var at = -1
    for k in range(len(exprs)):
        if exprs[k]._name == name:
            at = k
    return at


def _reads_input(exprs: List[Expr]) -> Bool:
    """Whether some expression reads an input column: a selection of
    literals alone yields one row whatever its input holds."""
    for e in exprs:
        for node in e._nodes:
            if node.op == SELECTOR:
                return False
            if node.op == COL:
                return True
    return False


def _plain_or_constant(expr: Expr) -> Bool:
    """A bare column (under any name) or a literal whose type does not
    depend on what it is compared with."""
    if len(expr._nodes) != 1:
        return False
    ref node = expr._nodes[0]
    if node.op == COL or node.op == LIT_STRING or node.op == LIT_BOOL:
        return True
    if node.op == LIT_INT or node.op == LIT_FLOAT:
        return node.text != UNTYPED
    return False


def _bare_column(expr: Expr) -> Bool:
    """A column read under its own name, with nothing applied."""
    return (
        len(expr._nodes) == 1
        and expr._nodes[0].op == COL
        and expr._nodes[0].text == expr._name
    )


def _schema_of(node: PlanNode, input: DataFrame) -> Optional[DataFrame]:
    """A filter's, select's or with_columns' result on a zero-row `input`,
    derived from the bound expression types. Return None when this cannot
    be known to match execution (a select of scalars has a row, or an
    expression does not bind cleanly), so the caller can execute the node."""
    try:
        if node.kind == FILTER:
            if len(node.exprs) != 1:
                return None
            for item in node.exprs[0]._nodes:
                if item.op == SELECTOR:
                    return None
            var bound = bind(node.exprs[0], input._columns)
            if bound.dtypes[len(bound.dtypes) - 1] != DataType.BOOL:
                return None
            return input.copy()
        if node.kind == AGG:
            return _aggregate_schema(node, input)
        if node.kind != SELECT and node.kind != WITH_COLUMNS:
            return None
        var bound = _bind_all(node.exprs, input._columns)
        var produced = List[Series](capacity=len(bound))
        for b in bound:
            if node.kind == SELECT and b.shape() != ROWS:
                return None
            var dtype = b.dtypes[len(b.dtypes) - 1]
            if dtype.is_nested() or dtype.is_categorical():
                return None
            produced.append(Series.full_null(b.expr._name, dtype, 0))
        if node.kind == SELECT:
            if len(produced) == 0:
                return None
            return DataFrame(produced^, height=0)
        var columns = input._columns.copy()
        for column in produced:
            var at = -1
            for k in range(len(columns)):
                if columns[k].name() == column.name():
                    at = k
                    break
            if at < 0:
                columns.append(column.copy())
            else:
                columns[at] = column.copy()
        return DataFrame(columns^, height=0)
    except:
        return None


def _aggregate_schema(
    node: PlanNode, input: DataFrame
) raises -> Optional[DataFrame]:
    """A group_by's result on a zero-row `input`: its key columns, then
    each aggregation's name and type bound from its expression. Running
    one on no rows
    sets up a whole grouping for nothing (TPC-DS q4: 170 us a group_by,
    each planned several times). None without keys (a global aggregation
    has one row), for keys or results of nested or categorical type, and
    for anything the aggregation would reject, which the caller then
    runs."""
    if len(node.names) == 0:
        return None
    var columns = List[Series](capacity=len(node.names) + len(node.exprs))
    for name in node.names:
        var key = input.column(name)
        var dtype = key.dtype()
        if dtype.is_nested() or dtype.is_categorical():
            return None
        columns.append(key.slice(0, 0))
    for b in _bind_all(node.exprs, input._columns):
        if b.shape() != AGGREGATE or b.expr._name in node.names:
            return None
        var dtype = b.dtypes[len(b.dtypes) - 1]
        if dtype.is_nested() or dtype.is_categorical():
            return None
        columns.append(Series.full_null(b.expr._name, dtype, 0))
    return DataFrame(columns^, height=0)


def _join_schema(
    node: PlanNode, left: DataFrame, right: DataFrame
) raises -> Optional[DataFrame]:
    """An inner, left, semi or anti join's result on zero-row inputs,
    derived from the input columns without executing the join. The schema
    contains the left columns, followed by non-coalesced right columns; a
    right name that collides with a left name receives the suffix.
    Running one on no
    rows prepares a hash join for nothing (TPC-DS q4: 60 us a join). None
    for other join types, keys of differing, nested or categorical types,
    and output names that still collide, which the caller then runs to
    raise or handle as it does."""
    var how = join_type(node.text)
    if not (
        how == JOIN_INNER
        or how == JOIN_LEFT
        or how == JOIN_SEMI
        or how == JOIN_ANTI
    ):
        return None
    if len(node.names) == 0 or len(node.names) != len(node.right_keys):
        return None
    var right_keys = List[Int]()
    var seen = Dict[String, Bool]()
    for i in range(len(node.names)):
        if node.names[i] in seen:
            return None
        seen[node.names[i]] = True
        var ltype = left.column(node.names[i]).dtype()
        var at = right._index(node.right_keys[i])
        var rtype = right._columns[at].dtype()
        if ltype != rtype or ltype.is_nested() or ltype.is_categorical():
            return None
        right_keys.append(at)
    if how == JOIN_SEMI or how == JOIN_ANTI:
        return left.copy()
    var columns = left._columns.copy()
    var names = Dict[String, Bool]()
    for column in left._columns:
        names[column.name()] = True
    var left_names = names.copy()
    for i in range(right.width()):
        if node.coalesce and i in right_keys:
            continue
        var name = right._columns[i].name()
        if name in left_names:
            name += node.join_suffix()
        if name in names:
            return None
        names[name] = True
        columns.append(right._columns[i].slice(0, 0).renamed(name))
    return DataFrame(columns^, height=0)


def _pinned(node: PlanNode) -> Bool:
    """A filter that reads a shared subplan and stays above it
    (`_filter_shared_subplans`)."""
    return node.kind == FILTER and node.filter_is_pinned()


def _through_projection(predicate: Expr, exprs: List[Expr]) -> Expr:
    """`predicate` over a projection's outputs, rewritten over its
    input: each column it reads that the projection produces becomes the
    input column or the constant producing it (`_plain_or_constant`)."""
    var result = predicate.copy()
    for i in range(len(result._nodes)):
        if result._nodes[i].op != COL:
            continue
        var at = _produced_at(exprs, result._nodes[i].text)
        if at < 0:
            continue
        result._nodes[i] = exprs[at]._nodes[0].copy()
    return result^


def _boolean_parts(expr: Expr, op: Int) -> List[Expr]:
    """The operands of a chain of `op` (AND or OR) at the root of expr, left
    to right; expr alone when its root is another operator."""
    var parts = List[Expr]()
    _collect_parts(expr, len(expr._nodes) - 1, op, parts)
    return parts^


def _collect_parts(expr: Expr, index: Int, op: Int, mut parts: List[Expr]):
    ref node = expr._nodes[index]
    if node.op == op and node.left >= 0 and node.right >= 0:
        _collect_parts(expr, node.left, op, parts)
        _collect_parts(expr, node.right, op, parts)
    else:
        parts.append(subtree(expr, index))


def _implied_filters(predicate: Expr) -> List[Expr]:
    """Filters that an OR of ANDs implies on fewer columns.

    `(a1 & b1) | (a2 & b2)` keeps a row only if some branch holds, and so
    only if `a1 | a2` holds, where a1 and a2 read one set of columns: that
    filter can move into the join input owning those columns. A branch
    with no part on those columns implies nothing.
    Under SQL's three-valued logic a true branch has every part true, so
    the derived filter keeps every row the predicate keeps.
    """
    var out = List[Expr]()
    var branches = _boolean_parts(predicate, OR)
    if len(branches) < 2:
        return out^
    var atoms = List[List[Expr]]()
    var reads = List[List[List[String]]]()
    for branch in branches:
        var parts = _boolean_parts(branch, AND)
        var part_reads = List[List[String]]()
        for part in parts:
            var r = _references(part)
            if not r:
                return out^
            part_reads.append(r.value().copy())
        atoms.append(parts^)
        reads.append(part_reads^)
    var tried = List[List[String]]()
    for a in range(len(atoms[0])):
        ref columns = reads[0][a]
        if len(columns) == 0:
            continue
        var seen = False
        for t in tried:
            seen = seen or (_covers(t, columns) and _covers(columns, t))
        if seen:
            continue
        tried.append(columns.copy())
        var implied = Expr(List[Node](), "")
        var found = True
        var partial = False
        for b in range(len(atoms)):
            var kept = Expr(List[Node](), "")
            for j in range(len(atoms[b])):
                if len(reads[b][j]) > 0 and _covers(columns, reads[b][j]):
                    kept = (
                        atoms[b][j].copy() if len(kept._nodes)
                        == 0 else kept & atoms[b][j]
                    )
                else:
                    partial = True
            if len(kept._nodes) == 0:
                found = False
                break
            implied = kept^ if len(implied._nodes) == 0 else implied | kept
        # A derived filter equal to the predicate adds nothing.
        if found and partial:
            out.append(implied^)
    return out^


def _stream_rows(node: PlanNode) -> Bool:
    if node.kind == FILTER or node.kind == WITH_COLUMNS:
        return _row_local(node.exprs)
    if node.kind == SELECT:
        if not _row_local(node.exprs):
            return False
        # An all-scalar select produces one row for the whole input.
        # Unbound selectors can match nothing and therefore have that shape;
        # batching them would produce one row per batch instead.
        var has_column = False
        for expression in node.exprs:
            for item in expression._nodes:
                if item.op == SELECTOR:
                    return False
                has_column |= item.op == COL
        return has_column
    return node.kind == DROP or node.kind == EXPLODE or node.kind == UNNEST


def _joined(parts: List[DataFrame]) raises -> DataFrame:
    if len(parts) == 1:
        return parts[0].copy()
    return concat(parts).rechunk()


def _same_names(a: List[String], b: List[String]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _order_free_reductions(keys: List[String], expressions: List[Expr]) -> Bool:
    """Whether a grouped reduction's results are the same whatever order
    its rows arrive in: the aggregate table's states, and distinct counts,
    medians and quantiles, which depend on each group's values only. Its
    input join may then build on either side; choosing the smaller input
    for the hash table does not change these order-independent results."""
    if len(keys) == 0 or len(expressions) == 0:
        return False
    for expression in expressions:
        for node in expression._nodes:
            if not is_reduction(node.op) or reduction_supported(node.op):
                continue
            if node.op not in [N_UNIQUE, MEDIAN, QUANTILE, ANY, ALL]:
                return False
    return True


def _distinct_inputs(expressions: List[Expr]) -> Optional[List[String]]:
    """The columns `n_unique` reductions read, or None when one reads an
    expression rather than a column."""
    var names = List[String]()
    for expression in expressions:
        ref nodes = expression._nodes
        for node in nodes:
            if node.op != N_UNIQUE:
                continue
            if node.left < 0 or nodes[node.left].op != COL:
                return None
            if nodes[node.left].text not in names:
                names.append(nodes[node.left].text)
    return names^


def _partitioned_order_free(exprs: List[Expr]) -> Bool:
    """Whether every row's values come out the same whatever order the rows
    arrive in: row-local steps, and aggregates (over partitions, `over`, or
    the whole input) that read no position. A window that does (cumulative,
    shift, rank, rolling) or a positional reduction does not."""
    for e in exprs:
        for node in e._nodes:
            if is_window(node.op) or is_sequence(node.op):
                return False
            if is_reduction(node.op) and node.op not in [
                SUM,
                COUNT,
                MIN,
                MAX,
                MEAN,
                STD,
                VAR,
                LEN,
                ANY,
                ALL,
                NULL_COUNT,
                N_UNIQUE,
                MEDIAN,
                QUANTILE,
                CORR,
                COV,
            ]:
                return False
    return True


def _order_insensitive(exprs: List[Expr]) -> Bool:
    """Whether aggregates give the same values whatever order their input
    rows arrive in: no window, and only reductions that read no position
    (so not first, last, arg_min, arg_max or a list of the rows)."""
    for e in exprs:
        for node in e._nodes:
            if is_window(node.op) or is_sequence(node.op) or node.op == OVER:
                return False
            if is_reduction(node.op) and node.op not in [
                SUM,
                COUNT,
                MIN,
                MAX,
                MEAN,
                STD,
                VAR,
                LEN,
                ANY,
                ALL,
                NULL_COUNT,
                N_UNIQUE,
                MEDIAN,
                QUANTILE,
                CORR,
                COV,
            ]:
                return False
    return True


def _chain_member(node: PlanNode) -> Bool:
    """A step `_order_chain` can move: an inner join with coalesced keys, a
    row-local filter, or a selection of plain columns."""
    if node.kind == JOIN:
        return node.how == JOIN_INNER and node.coalesce and len(node.names) > 0
    if node.kind == FILTER:
        return _row_local(node.exprs)
    if node.kind == SELECT:
        for e in node.exprs:
            if len(e._nodes) != 1 or e._nodes[0].op != COL:
                return False
            if e._nodes[0].text != e._name:
                return False
        return True
    return False


def _row_order_step(kind: Int) -> Bool:
    """Whether `kind` is a row-order operator (tail, reverse, shift,
    with_row_index): run whole, in input order, by its eager method."""
    return (
        kind == TAIL
        or kind == REVERSE
        or kind == SHIFT
        or kind == ROW_INDEX
        or kind == GATHER_EVERY
        or kind == INSERT_COLUMN
        or kind == FILL_NULL_STRATEGY
    )


def _merge_copy_needs(
    copy_of: List[Int],
    head: Int,
    mut needed: List[Optional[List[String]]],
    mut all_needed: List[Bool],
):
    """Give every copy of the repeated subplan `head` the union of what
    its copies' readers need, by merging the projections of all parents
    that read the shared subplan."""
    var whole = False
    var read = False
    var union = List[String]()
    for i in range(len(copy_of)):
        if copy_of[i] != head:
            continue
        whole |= all_needed[i]
        if needed[i]:
            read = True
            for name in needed[i].value():
                if name not in union:
                    union.append(name)
    if not whole and not read:
        return
    for i in range(len(copy_of)):
        if copy_of[i] != head:
            continue
        all_needed[i] = whole
        needed[i] = union.copy()


def _is_scan(kind: Int) -> Bool:
    return kind in [SCAN_FRAME, SCAN_CSV, SCAN_PARQUET, SCAN_IPC, SCAN_NDJSON]


def _set_members(node: PlanNode) -> List[List[String]]:
    """The keys each grouping set of a GROUPING_SETS node groups by."""
    var width = len(node.names)
    var sets = List[List[String]]()
    for s in range(len(node.grouping_membership()) // width):
        var members = List[String]()
        for k in range(width):
            if node.grouping_membership()[s * width + k]:
                members.append(node.names[k])
        sets.append(members^)
    return sets^


def _is_time_window(kind: Int) -> Bool:
    return kind == GROUP_BY_DYNAMIC or kind == ROLLING


def _time_window_reads(node: PlanNode) -> Optional[List[String]]:
    """The input columns a time window node reads: its index column, its
    `by` keys and the columns its aggregations reference. None
    when an aggregation reads columns through a selector."""
    var reads = node.names.copy()
    for e in node.exprs:
        var r = _references(e)
        if not r:
            return None
        for n in r.value():
            reads.append(n)
    return reads^


def _mix(h: UInt64, value: UInt64) -> UInt64:
    """One step of a plan node's hash (`_node_hash`): FNV-1a over a word."""
    return (h ^ value) * 0x100000001B3


def _mix_text(h: UInt64, text: String) -> UInt64:
    """`h` continued over the text's length and bytes, building no string."""
    var out = _mix(h, UInt64(text.byte_length()))
    for b in text.as_bytes():
        out = (out ^ UInt64(b)) * 0x100000001B3
    return out


def _mix_list(h: UInt64, items: List[String]) -> UInt64:
    var out = _mix(h, UInt64(len(items)))
    for item in items:
        out = _mix_text(out, item)
    return out


def _mix_expr(h: UInt64, expr: Expr) -> UInt64:
    """`h` continued over an expression node by node, the fields
    `_same_expr` compares (dtypes by count only)."""
    var out = _mix_text(h, expr._name)
    out = _mix(out, UInt64(len(expr._nodes)))
    for n in expr._nodes:
        out = _mix(out, UInt64(n.op))
        out = _mix(out, UInt64(n.left))
        out = _mix(out, UInt64(n.right))
        out = _mix_text(out, n.text)
        out = _mix(out, UInt64(n.integer))
        out = _mix(out, bitcast[DType.uint64](n.floating))
        out = _mix(out, UInt64(n.min_count))
        out = _mix(out, UInt64(n.extra))
        out = _mix_text(out, n.text2)
        out = _mix(out, UInt64(len(n.dtypes)))
    return out


def _same_expr(a: Expr, b: Expr) -> Bool:
    """Whether two expressions are the same node by node: every field, and
    dtypes by name (`_expr_signature`'s text equal)."""
    if a._name != b._name or len(a._nodes) != len(b._nodes):
        return False
    for k in range(len(a._nodes)):
        ref x = a._nodes[k]
        ref y = b._nodes[k]
        if (
            x.op != y.op
            or x.left != y.left
            or x.right != y.right
            or x.text != y.text
            or x.integer != y.integer
            or bitcast[DType.uint64](x.floating)
            != bitcast[DType.uint64](y.floating)
            or x.min_count != y.min_count
            or x.extra != y.extra
            or x.text2 != y.text2
            or len(x.dtypes) != len(y.dtypes)
        ):
            return False
        for d in range(len(x.dtypes)):
            if Bool(x.dtypes[d]) != Bool(y.dtypes[d]):
                return False
            if x.dtypes[d] and (
                x.dtypes[d].value().name() != y.dtypes[d].value().name()
            ):
                return False
    return True


def _field(text: String) -> String:
    """`text` prefixed by its byte length, for a signature."""
    return String(text.byte_length()) + ":" + text


def _field_list(items: List[String]) -> String:
    var out = String(len(items)) + "["
    for item in items:
        out += _field(item)
    return out + "]"


def _expr_signature(expr: Expr) -> String:
    """An expression as text, node by node, for `_signature`; free text
    with its length (`_field`)."""
    var text = _field(expr._name)
    for n in expr._nodes:
        text += (
            ";"
            + String(n.op)
            + ","
            + String(n.left)
            + ","
            + String(n.right)
            + ","
            + _field(n.text)
            + ","
            + String(n.integer)
            + ","
            + String(n.floating)
            + ","
            + String(n.min_count)
            + ","
            + String(n.extra)
            + ","
            + _field(n.text2)
        )
        for d in n.dtypes:
            text += "," + (d.value().name() if d else "_")
    return text^


def _joined(names: List[String]) -> String:
    var out = String()
    for i in range(len(names)):
        if i > 0:
            out += ", "
        out += names[i]
    return out^


def _key_set_filter(
    keys: Series, name: String, dtype: DataType
) raises -> Optional[Expr]:
    """A filter keeping the rows whose `name` is among `keys`' non-null
    values: the values themselves when there are at most 50, else their
    min-max range. None when the dtypes differ or the key is neither an
    integer (other than UInt64) nor a string."""
    if keys.dtype() != dtype:
        return None
    var distinct = keys.unique()
    if distinct.is_chunked():
        distinct = distinct.rechunk()
    if dtype == DataType.STRING:
        var values = List[String]()
        for i in range(len(distinct)):
            var cell = distinct.get(i)
            if not cell.is_null():
                values.append(cell.string())
        if len(values) > 50:
            return None
        if len(values) == 0:
            return Optional(lit(False))
        return Optional(col(name).is_in(values))
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        comptime if D.is_integral() and D != DType.uint64:
            if distinct._data.isa[Column[Scalar[D]]]():
                ref column = distinct._data[Column[Scalar[D]]]
                # Bounds first: a literal per value only for a short list
                # (PDS-H q21 built 729K literals for a range).
                var count = 0
                var low = Scalar[D].MAX
                var high = Scalar[D].MIN
                for i in range(len(column)):
                    if not column._valid(i):
                        continue
                    var value = column._get(i)
                    low = min(low, value)
                    high = max(high, value)
                    count += 1
                if count == 0:
                    return Optional(lit(False))
                if count > 50:
                    return Optional(
                        (col(name) >= lit(low)) & (col(name) <= lit(high))
                    )
                var literals = List[Expr](capacity=count)
                for i in range(len(column)):
                    if column._valid(i):
                        literals.append(lit(column._get(i)))
                return Optional(col(name).is_in(literals))
    return None


def _bound_filter(name: String, op: Int, bound: Int64, dtype: DataType) -> Expr:
    """`col(name) op bound`, the bound a literal of the column's integer
    type (Int64 or Int32)."""
    var value = lit(bound) if dtype == DataType.INT64 else lit(Int32(bound))
    if op == LT:
        return col(name) < value
    if op == LE:
        return col(name) <= value
    if op == GT:
        return col(name) > value
    return col(name) >= value


def _covers(columns: List[String], reads: List[String]) -> Bool:
    for name in reads:
        var found = False
        for c in columns:
            found = found or c == name
        if not found:
            return False
    return True


def _key_pairs(node: PlanNode) -> String:
    """`a` for keys named alike, `a = b` for differently named pairs."""
    var out = String()
    for i in range(len(node.names)):
        if i > 0:
            out += ", "
        out += node.names[i]
        if i < len(node.right_keys) and node.right_keys[i] != node.names[i]:
            out += " = " + node.right_keys[i]
    return out^


def _covers_exclusive(
    columns: List[String], other: List[String], reads: List[String]
) -> Bool:
    """Right-side names that the join output does not rename or shadow."""
    for name in reads:
        for c in other:
            if c == name:
                return False
    return _covers(columns, reads)


def _same_flags(a: List[Bool], b: List[Bool]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True
