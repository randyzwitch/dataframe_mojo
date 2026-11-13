"""Partial aggregation below a join (eager group-by).

A grouped aggregate over an equi-join that reads only one input's columns
in its aggregate functions can aggregate that input first, by its join
keys and its own grouping columns, and finish above the join. The join
then matches one row per partial group instead of every input row, and
the grouping above it sees as many rows as there are partial groups.

Why the result is the same: group the input of the aggregate side by
(join keys, that side's grouping columns). Every row of one partial
group meets the same rows of the other input, so every final group is
the union of whole partial groups, each repeated once per matching row
of the other input. For a decomposable aggregate, folding the partial
results gives the result over the union of their rows: a sum of partial
sums (a partial group repeated by a duplicate match adds its sum again,
as each of its rows would have), a sum of partial counts, a minimum of
partial minimums and a maximum of partial maximums.
"""
from dataframe.planning.node_builders import (
    _aggregate_node,
)
from dataframe.expr import (
    COUNT,
    Expr,
    LEN,
    MAX,
    MIN,
    SUM,
    col,
    subtree,
)
from dataframe.aggregation.expressions import (
    _row_local,
)
from dataframe.join_type import (
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.graph import (
    plan_columns_of,
    plan_reorder,
)
from dataframe.planning.model import (
    AGG,
    JOIN,
)
from dataframe.planning.expression_inputs import (
    _references,
)
from dataframe.planning.statistics import (
    PlanStatistics,
    plan_cardinality,
)
from dataframe.trace import (
    trace_path,
)
from std.collections import (
    Optional,
)

# Aggregating first pays only when it collapses many rows into few
# groups. Two estimates decide it, before anything runs. The join must
# keep most of the aggregate side's rows (at least 70%): a join that
# discards rows already shrinks the input, and the partial aggregate
# would hash rows the join throws away. And the aggregate side must hold
# at least 4 rows per row of the other input, the most groups its join
# keys can then have per matching row being small next to its rows.
comptime _MIN_AGGREGATE_TO_DIMENSION_RATIO = 4
comptime _MIN_JOIN_RETENTION = 0.7
# The partial groups: the join keys and at most one more grouping column
# of the aggregate side. Each further column multiplies the partial
# groups, and with them the rows the partial aggregate keeps.
comptime _MAX_EXTRA_LOWER_GROUPS = 1

comptime _PARTIAL_PREFIX = "__partial_aggregate_"


def plan_push_partial_aggregates(mut self: QueryContext) raises:
    """Split each grouped aggregate over a join whose aggregate functions
    read one input into a partial aggregate of that input below the join
    and a final aggregate above it (module docs). Run after join order is
    chosen, since the join directly under the aggregate decides the split.
    """
    var stats = PlanStatistics()
    var changed = False
    # Partial aggregates appended by a split are visited too: one may sit
    # on a further join and split again.
    var i = 0
    while i < len(self._nodes):
        if _push_partial_aggregate(self, stats, i):
            changed = True
            trace_path("lazy.partial_aggregate")
        i += 1
    if changed:
        plan_reorder(self)


def _parent_counts(self: QueryContext) -> List[Int]:
    var counts = List[Int](length=len(self._nodes), fill=0)
    for node in self._nodes:
        if node.left >= 0:
            counts[node.left] += 1
        if node.right >= 0:
            counts[node.right] += 1
    return counts^


def _side_of(
    name: String,
    left: List[String],
    right: List[String],
    shared: List[String],
) -> Int:
    """0 when `name` is a column of the join's left input, 1 of its right
    input, -1 when neither. A name in `shared` is a key pair the join
    coalesces into one column, the same value on either side: -2."""
    if name in shared:
        return -2
    if name in left:
        return 0
    if name in right:
        return 1
    return -1


def _partial_function(expr: Expr) -> Int:
    """The aggregate function at the root of `expr` when it can be split
    into a partial and a final step, else -1: a sum, a count of values or
    of rows, a minimum or a maximum over a row-local input."""
    var root = len(expr._nodes) - 1
    if root < 1:
        return -1
    ref node = expr._nodes[root]
    if (
        node.op != SUM
        and node.op != COUNT
        and node.op != LEN
        and node.op != MIN
        and node.op != MAX
    ):
        return -1
    if node.left < 0 or node.right >= 0:
        return -1
    if not _row_local([subtree(expr, node.left)]):
        return -1
    return node.op


def _final_expr(expr: Expr, name: String) -> Expr:
    """The final step over the partial results in column `name`: partial
    sums and counts add up, minimums and maximums fold again. A sum keeps
    its minimum count of valid values: a final group with no valid value
    has only partial sums that are null themselves."""
    ref node = expr._nodes[len(expr._nodes) - 1]
    var partial = col(name)
    if node.op == SUM:
        return partial.sum(min_count=node.min_count).alias(expr._name)
    if node.op == COUNT or node.op == LEN:
        return partial.sum().alias(expr._name)
    if node.op == MIN:
        return partial.min().alias(expr._name)
    return partial.max().alias(expr._name)


def _push_partial_aggregate(
    mut self: QueryContext, mut stats: PlanStatistics, index: Int
) raises -> Bool:
    var agg = self._nodes[index].copy()
    if (
        agg.kind != AGG
        or agg.maintain_order
        or len(agg.flags) > 0
        or len(agg.names) == 0
        or len(agg.exprs) == 0
        or agg.left < 0
    ):
        return False
    var at = agg.left
    var join = self._nodes[at].copy()
    if (
        join.kind != JOIN
        or join.how not in [JOIN_INNER, JOIN_LEFT, JOIN_RIGHT]
        or len(join.names) == 0
        or len(join.names) != len(join.right_keys)
        or join.left < 0
        or join.right < 0
    ):
        return False
    # The join must belong to this aggregate alone: it is replaced.
    var parents = _parent_counts(self)
    if parents[at] != 1:
        return False
    var left = plan_columns_of(self, join.left)
    var right = plan_columns_of(self, join.right)
    # A name both inputs hold is renamed by the join unless it is a key
    # pair under one name that the join coalesces; keep to joins that
    # rename nothing, so each output column names its input column.
    var shared_keys = List[String]()
    for name in right:
        if name not in left:
            continue
        var pair = -1
        for k in range(len(join.names)):
            if join.names[k] == name and join.right_keys[k] == name:
                pair = k
        if pair < 0 or not join.coalesce or join.how != JOIN_INNER:
            return False
        shared_keys.append(name)
    # The aggregate side: the input every aggregate function reads.
    var side = -1
    var functions = List[Int]()
    for expr in agg.exprs:
        var function = _partial_function(expr)
        if function < 0:
            return False
        functions.append(function)
        var reads = _references(expr)
        if not reads or len(reads.value()) == 0:
            return False
        for name in reads.value():
            var s = _side_of(name, left, right, shared_keys)
            if s == -2:
                # A coalesced key reads the same value on both sides.
                continue
            if s < 0 or (side >= 0 and s != side):
                return False
            side = s
    if side < 0:
        return False
    # Every row of the aggregate side must reach the aggregate: an outer
    # join may add rows from the other side only.
    if (join.how == JOIN_LEFT and side != 0) or (
        join.how == JOIN_RIGHT and side != 1
    ):
        return False
    var keys = join.names.copy() if side == 0 else join.right_keys.copy()
    var ours = left.copy() if side == 0 else right.copy()
    # Partial groups: the join keys, then the aggregate side's grouping
    # columns. The final aggregate needs a grouping column of the other
    # side, or nothing above the partial aggregate would group further.
    var lower = List[String]()
    for key in keys:
        if key not in lower:
            lower.append(key)
    var key_count = len(lower)
    var other_groups = 0
    for name in agg.names:
        var s = _side_of(name, left, right, shared_keys)
        if s == -1:
            return False
        if s == side or s == -2:
            if name not in lower and name in ours:
                lower.append(name)
        else:
            other_groups += 1
    if other_groups == 0:
        return False
    if (
        len(lower) >= len(agg.names)
        or len(lower) > key_count + _MAX_EXTRA_LOWER_GROUPS
    ):
        return False
    if not _collapses(self, stats, at, side):
        return False
    # Names for the partial results, free in both inputs.
    var partial_names = List[String]()
    var n = 0
    while len(partial_names) < len(agg.exprs):
        var name = _PARTIAL_PREFIX + String(n)
        n += 1
        if name not in left and name not in right:
            partial_names.append(name)
    var partial_exprs = List[Expr]()
    var final_exprs = List[Expr]()
    for j in range(len(agg.exprs)):
        partial_exprs.append(agg.exprs[j].alias(partial_names[j]))
        final_exprs.append(_final_expr(agg.exprs[j], partial_names[j]))
    var before = self._execute(index, True)
    var saved = agg.copy()
    var aggregate_input = join.left if side == 0 else join.right
    self._nodes.append(
        _aggregate_node(keys=lower, exprs=partial_exprs, left=aggregate_input)
    )
    var partial = len(self._nodes) - 1
    var rejoined = join.copy()
    if side == 0:
        rejoined.left = partial
    else:
        rejoined.right = partial
    self._nodes.append(rejoined^)
    var final_node = saved.copy()
    final_node.left = len(self._nodes) - 1
    final_node.exprs = final_exprs^
    self._nodes[index] = final_node^
    # The final step must give back every column as the aggregate did,
    # in name, order and type; a function whose final step would change
    # its type keeps the plan as it was.
    var after = self._execute(index, True)
    var same = after.width() == before.width()
    if same:
        var names = before.columns()
        var now = after.columns()
        for c in range(len(names)):
            if (
                names[c] != now[c]
                or before.column(names[c]).dtype()
                != after.column(now[c]).dtype()
            ):
                same = False
                break
    if not same:
        self._nodes[index] = saved^
        return False
    return True


def _collapses(
    self: QueryContext, mut stats: PlanStatistics, join: Int, side: Int
) raises -> Bool:
    """Whether the estimates say aggregating the `side` input of `join`
    first collapses many rows (the constants above). Missing estimates
    do not decline: the split is right whatever the sizes."""
    ref node = self._nodes[join]
    var aggregate_input = node.left if side == 0 else node.right
    var other_input = node.right if side == 0 else node.left
    var rows = plan_cardinality(self, stats, aggregate_input)
    var joined = plan_cardinality(self, stats, join)
    if rows < 0 or joined < 0:
        return True
    if joined > rows:
        return True
    if Float64(joined) < _MIN_JOIN_RETENTION * Float64(rows):
        return False
    var other = plan_cardinality(self, stats, other_input)
    if other < 0:
        return True
    return rows >= _MIN_AGGREGATE_TO_DIMENSION_RATIO * other
