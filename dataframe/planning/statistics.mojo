"""Estimate plan cardinalities and column domains without executing queries.

Start from stored table statistics and carry counts through projections,
filters, group keys and joins. Track whether distinct counts came from
observed values, bounds or fallback row counts, so estimates can prefer
reliable evidence. Join estimates divide the product of input rows by the
shared key domain; unknown inputs retain unknown estimates rather than
executing data to obtain them.
"""
from std.collections import Dict, Optional
from std.math import exp

from dataframe.expr import COL, Expr, conjuncts
from dataframe.frame import DataFrame
from dataframe.join_order import SEMI_ANTI_SELECTIVITY
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_INNER,
    JOIN_MARK,
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
)
from dataframe.kernels.segment_statistics import series_statistics
from dataframe.planning.context import QueryContext
from dataframe.planning.graph import plan_names_of
from dataframe.planning.model import (
    AGG,
    ASOF,
    CONCAT,
    DROP,
    EXPLODE,
    FILTER,
    GROUPING_SETS,
    JOIN,
    PlanNode,
    REVERSE,
    ROW_INDEX,
    SCAN_FRAME,
    SELECT,
    SHIFT,
    SLICE,
    SORT,
    TAIL,
    UNIQUE,
    UNNEST,
    WITH_COLUMNS,
)
from dataframe.planning.properties import _references, _stream_rows
from dataframe.storage.statistics import (
    COMPARE_EQUAL,
    FILTER_ALWAYS_FALSE,
    FILTER_ALWAYS_TRUE,
    FILTER_FALSE_OR_NULL,
    NUMERIC_STATS,
)
from dataframe.table_filter import (
    CONJUNCTION_AND,
    CONSTANT_COMPARISON,
    FilterNode,
    check_statistics,
    table_filters,
)

# The share of rows a filter keeps when nothing measures it: a guess, low enough
# that a filtered input looks smaller than an unfiltered one (filters usually
# remove most rows) without assuming it is tiny. An empirical choice.
comptime DEFAULT_SELECTIVITY = 0.2
# Where a distinct count came from: the row count or the min/max span (upper
# bounds), a sketch of the values (an estimate), or an exact count.
comptime COUNT_CARDINALITY = 0
comptime COUNT_MIN_MAX = 1
comptime COUNT_HLL = 2
comptime COUNT_EXACT = 3


@fieldwise_init
struct DistinctCount(Copyable, ImplicitlyCopyable, Movable):
    """A column's distinct count and where it came from; `count` is -1 when the column has none (no such
    column, or a relation of unknown rows)."""

    var count: Int
    var source: Int

    @staticmethod
    def unknown() -> Self:
        return Self(-1, COUNT_CARDINALITY)

    def known(self) -> Bool:
        return self.count >= 0

    def reliable(self) -> Bool:
        """Whether the count measured the values: a HyperLogLog or exact count,
        not a bound."""
        return self.source == COUNT_HLL or self.source == COUNT_EXACT


struct PlanStatistics(Movable):
    """The statistics of a plan's nodes, filled as they are asked for:
    each node's estimated rows (-1 when unknown) and each (node, column)
    pair's distinct count. Valid while the nodes it read stay as they
    are: a pass that rewrites nodes starts a new one."""

    var rows: Dict[Int, Int]
    var counts: Dict[String, DistinctCount]
    # Each node's output column names, derived once: a join's distinct
    # counts ask which side holds a column at every join below it.
    var names: Dict[Int, List[String]]

    def __init__(out self):
        self.rows = Dict[Int, Int]()
        self.counts = Dict[String, DistinctCount]()
        self.names = Dict[Int, List[String]]()


def estimate_distinct_cardinality(
    counts: List[Int], input_cardinality: Int
) -> Int:
    """The groups so many distinct values per key give over `input_cardinality`
    rows. The keys' counts multiply, each key after the first taking 5% off
    because keys grouped together are usually correlated (an empirical
    discount). Drawing `rows` values at random from `product` possible groups
    hits product * (1 - e^(-rows / product)) distinct ones on average, at most
    the rows; half the rows with no counts, a guess."""
    if len(counts) == 0:
        return input_cardinality // 2
    var product = 1.0
    for count in counts:
        product *= Float64(max(count, 1))
    product *= 0.95 ** Float64(len(counts) - 1)
    var multiplier = 1.0 - exp(-Float64(input_cardinality) / product)
    var estimate = (
        Float64(input_cardinality) if multiplier
        == 0.0 else product * multiplier
    )
    var result = Int(min(estimate, Float64(input_cardinality)))
    return max(result, 1) if input_cardinality > 0 else 0


def _filtered(rows: Int) -> Int:
    """A filter's rows over `rows` input rows (`LOGICAL_FILTER`)."""
    if rows <= 0:
        return rows
    return max(Int(Float64(rows) * DEFAULT_SELECTIVITY), 1)


def _scan_under_filters(self: QueryContext, index: Int) -> Int:
    """The table scan under the chain of filters ending at `index`, or -1
    when the chain ends elsewhere: the filters a scan's table filters
    come from."""
    var cursor = index
    while cursor >= 0 and self._nodes[cursor].kind == FILTER:
        cursor = self._nodes[cursor].left
    if cursor >= 0 and self._nodes[cursor].kind == SCAN_FRAME:
        return cursor
    return -1


def _equality_rows(
    nodes: List[FilterNode], at: Int, cardinality: Int, count: DistinctCount
) -> Int:
    """The rows node `at` of a table filter keeps: an equality with a constant
    keeps the rows over the column's distinct count, a conjunction the fewest
    any part keeps, anything else every row."""
    ref node = nodes[at]
    if node.kind == CONJUNCTION_AND:
        var rows = cardinality
        for child in node.children:
            rows = min(rows, _equality_rows(nodes, child, cardinality, count))
        return rows
    if node.kind != CONSTANT_COMPARISON or node.op != COMPARE_EQUAL:
        return cardinality
    if count.source == COUNT_CARDINALITY or count.count <= 0:
        return cardinality
    return (cardinality + count.count - 1) // count.count


def _filter_chain_rows(
    self: QueryContext, mut stats: PlanStatistics, index: Int, scan: Int
) raises -> Int:
    """Rows the chain of filters at `index` keeps of the table scan `scan` under
    it: each part reading one of its columns is a table filter, the rest one
    filter above the scan."""
    var table = self._frames[self._nodes[scan].source_slot()].copy()
    var base = table.height()
    var parts = List[Expr]()
    var cursor = index
    while cursor != scan:
        for expr in self._nodes[cursor].exprs:
            for part in conjuncts(expr):
                parts.append(part.copy())
        cursor = self._nodes[cursor].left
    var columns = table.columns()
    var single = List[Expr]()
    var residual = False
    for part in parts:
        var reads = _references(part)
        var one = False
        if reads:
            ref names = reads.value()
            one = len(names) > 0
            for name in names:
                one = one and name == names[0] and name in columns
        if one:
            single.append(part.copy())
        else:
            residual = True
    var rows = base
    if len(single) > 0:
        var built = table_filters(single, table._columns)
        var narrowing = len(single)
        for filter in built.filters:
            var column = table.column(filter.column)
            var bounds = series_statistics(column)
            if bounds and len(filter.nodes) > 0:
                var proven = check_statistics(filter, bounds.value())
                # A filter the statistics prove keeps no row
                # (FILTER_ALWAYS_FALSE or FILTER_FALSE_OR_NULL: a null result
                # drops the row too) makes the scan an empty result.
                if (
                    proven == FILTER_ALWAYS_FALSE
                    or proven == FILTER_FALSE_OR_NULL
                ):
                    return 0
                # One they prove keeps every row is no filter
                # (`FILTER_ALWAYS_TRUE`).
                if proven == FILTER_ALWAYS_TRUE:
                    narrowing -= 1
                    continue
            var count = plan_distinct(self, stats, scan, filter.column)
            rows = min(
                rows,
                _equality_rows(
                    filter.nodes, len(filter.nodes) - 1, base, count
                ),
            )
        if rows == base and narrowing > 0:
            rows = max(Int(Float64(base) * DEFAULT_SELECTIVITY), 1)
        if base == 0:
            rows = 0
    if residual:
        rows = _filtered(rows)
    return rows


def plan_cardinality(
    self: QueryContext, mut stats: PlanStatistics, index: Int
) raises -> Int:
    """Rows node `index` yields, estimated from statistics before anything runs;
    -1 when an input has none (a file scan, an operator with no estimate)."""
    if index < 0:
        return -1
    if index in stats.rows:
        return stats.rows[index]
    var rows = _cardinality(self, stats, index)
    stats.rows[index] = rows
    return rows


def _cardinality(
    self: QueryContext, mut stats: PlanStatistics, index: Int
) raises -> Int:
    ref node = self._nodes[index]
    var kind = node.kind
    if kind == SCAN_FRAME:
        return self._frames[node.source_slot()].height()
    if kind == FILTER:
        var scan = _scan_under_filters(self, index)
        if scan >= 0:
            return _filter_chain_rows(self, stats, index, scan)
        return _filtered(plan_cardinality(self, stats, node.left))
    if (
        kind == DROP
        or kind == WITH_COLUMNS
        or kind == REVERSE
        or kind == SHIFT
        or kind == ROW_INDEX
        or kind == EXPLODE
        or kind == UNNEST
        or (kind == SELECT and _stream_rows(node))
    ):
        # Projections and windows keep their input's rows; so do unnests, for
        # lack of a measure of the lists' lengths.
        return plan_cardinality(self, stats, node.left)
    if kind == SORT or kind == SLICE or kind == TAIL:
        # A limit keeps at most its count (`LOGICAL_LIMIT`).
        var rows = plan_cardinality(self, stats, node.left)
        var limit = node.sort_limit()
        if kind == TAIL:
            limit = node.tail_count()
        elif kind == SLICE:
            limit = node.slice_length()
        if rows < 0 or limit < 0:
            return rows
        return min(rows, limit)
    if kind == CONCAT:
        var first = plan_cardinality(self, stats, node.left)
        var second = plan_cardinality(self, stats, node.right)
        if first < 0 or second < 0:
            return -1
        return first + second
    if kind == AGG or kind == GROUPING_SETS or kind == UNIQUE:
        var rows = plan_cardinality(self, stats, node.left)
        if rows < 0:
            return -1
        var keys = _group_keys(self, node)
        if kind == AGG and len(keys) == 0:
            return 1
        var counts = List[Int]()
        for key in keys:
            var count = plan_distinct(self, stats, node.left, key)
            if not count.known():
                return -1
            counts.append(max(count.count, 1))
        return estimate_distinct_cardinality(counts, rows)
    if kind == ASOF:
        # An as-of join keeps its left input's rows.
        return plan_cardinality(self, stats, node.left)
    if kind != JOIN:
        return -1
    var left = plan_cardinality(self, stats, node.left)
    if left < 0:
        return -1
    if node.how == JOIN_MARK:
        return left
    if node.how == JOIN_RIGHT_SEMI or node.how == JOIN_RIGHT_ANTI:
        # The build side is the output (the left input probes).
        return plan_cardinality(self, stats, node.right)
    if node.how == JOIN_SEMI or node.how == JOIN_ANTI:
        # A fifth of the left rows (SEMI_ANTI_SELECTIVITY), with nothing
        # measuring how many find a match.
        return (
            max(Int(Float64(left) / SEMI_ANTI_SELECTIVITY), 1) if left
            > 0 else 0
        )
    var right = plan_cardinality(self, stats, node.right)
    if right < 0:
        return -1
    if node.how == JOIN_CROSS:
        return left * right
    if node.how != JOIN_INNER or len(node.names) == 0:
        return max(left, right)
    if left == 0 or right == 0:
        return 0
    var domain = 1.0
    for i in range(len(node.names)):
        var a = plan_distinct(self, stats, node.left, node.names[i])
        var b = plan_distinct(self, stats, node.right, node.right_keys[i])
        domain = max(domain, Float64(_total_domain(a, b)))
    return max(Int(Float64(left) * Float64(right) / domain), 1)


def _total_domain(a: DistinctCount, b: DistinctCount) -> Int:
    """The distinct keys two joined columns can share, for estimating the join's
    rows: the larger reliable count, else the larger bounds count, else the
    smaller fallback count; 1 when neither side has one."""
    var reliable = -1
    var bounded = -1
    var fallback = -1
    for count in [a, b]:
        if not count.known():
            continue
        if count.reliable():
            reliable = max(reliable, count.count)
        elif count.source == COUNT_MIN_MAX:
            bounded = max(bounded, count.count)
        elif fallback < 0 or count.count < fallback:
            fallback = count.count
    if reliable >= 0:
        return reliable
    if bounded >= 0:
        return bounded
    return max(fallback, 1)


def _group_keys(self: QueryContext, node: PlanNode) raises -> List[String]:
    """The columns an aggregate groups by: an aggregate's keys, the largest
    grouping set's, a unique's subset or every input column."""
    if node.kind == AGG:
        return node.names.copy()
    if node.kind == UNIQUE:
        if len(node.names) > 0:
            return node.names.copy()
        var names = plan_names_of(self, node.left)
        if names:
            return names.take()
        return self._execute(node.left, True).columns()
    # A grouping set per row of flags, a flag per key.
    var keys = len(node.names)
    var best = List[String]()
    if keys == 0:
        return best^
    for s in range(len(node.grouping_membership()) // keys):
        var members = List[String]()
        for k in range(keys):
            if node.grouping_membership()[s * keys + k]:
                members.append(node.names[k])
        if len(members) > len(best):
            best = members^
    return best^


def _has_column(
    self: QueryContext, mut stats: PlanStatistics, index: Int, name: String
) raises -> Bool:
    """Whether node `index` outputs column `name`."""
    var known = stats.names.get(index)
    if known:
        return name in known.value()
    var names = plan_names_of(self, index)
    var columns = names.take() if names else self._execute(
        index, True
    ).columns()
    var found = name in columns
    stats.names[index] = columns^
    return found


def plan_distinct(
    self: QueryContext, mut stats: PlanStatistics, index: Int, name: String
) raises -> DistinctCount:
    """Trace a column's distinct-count estimate through projections, filters,
    grouping keys and named join outputs. Return unknown if the node does
    not output that column or its relation has unknown cardinality."""
    if index < 0:
        return DistinctCount.unknown()
    var key = String(index) + "/" + name
    if key in stats.counts:
        return stats.counts[key]
    var count = _distinct(self, stats, index, name)
    stats.counts[key] = count
    return count


def _scan_distinct(
    self: QueryContext, index: Int, name: String
) raises -> DistinctCount:
    """A table column's distinct count: its HyperLogLog count, else its integer
    bounds' span (no more values fit between them), else the table's rows."""
    var slot = self._nodes[index].source_slot()
    ref table = self._frames[slot]
    if name not in table.columns():
        return DistinctCount.unknown()
    var rows = table.height()
    var count = table._distinct_count(name)
    if count < 0 and slot in self._materialized_distinct:
        # A shared subplan result retains its estimated distinct counts,
        # avoiding a fresh scan merely to plan its consumers.
        count = self._materialized_distinct[slot].get(name, -1)
    if count > 0:
        return DistinctCount(count, COUNT_HLL)
    if rows > 0:
        var bounds = series_statistics(table.column(name))
        if bounds:
            ref b = bounds.value()
            if b.stats_type == NUMERIC_STATS and b.has_min_max:
                if b.int_max >= b.int_min:
                    var span = b.int_max - b.int_min + 1
                    if span <= Int128(rows):
                        return DistinctCount(Int(span), COUNT_MIN_MAX)
                    return DistinctCount(rows, COUNT_MIN_MAX)
    return DistinctCount(rows, COUNT_CARDINALITY)


def _distinct(
    self: QueryContext, mut stats: PlanStatistics, index: Int, name: String
) raises -> DistinctCount:
    ref node = self._nodes[index]
    var kind = node.kind
    if kind == SCAN_FRAME:
        return _scan_distinct(self, index, name)
    if kind == FILTER:
        var count = plan_distinct(self, stats, node.left, name)
        if (
            count.known()
            and count.source == COUNT_CARDINALITY
            and _scan_under_filters(self, index) >= 0
        ):
            # A table column with no statistics counts the rows left
            # after the table's filters.
            return DistinctCount(
                plan_cardinality(self, stats, index), COUNT_CARDINALITY
            )
        return count
    if (
        kind == DROP
        or kind == SORT
        or kind == SLICE
        or kind == TAIL
        or kind == REVERSE
        or kind == SHIFT
        or kind == EXPLODE
        or kind == UNNEST
        or kind == CONCAT
    ):
        return plan_distinct(self, stats, node.left, name)
    if kind == UNIQUE:
        var count = plan_distinct(self, stats, node.left, name)
        var rows = plan_cardinality(self, stats, index)
        if count.known() and rows >= 0:
            count.count = min(count.count, rows)
        return count
    if kind == ROW_INDEX:
        if name == node.row_index_name():
            return _relation_count(self, stats, index)
        return plan_distinct(self, stats, node.left, name)
    if kind == SELECT or kind == WITH_COLUMNS:
        var at = -1
        for i in range(len(node.exprs)):
            if node.exprs[i]._name == name:
                at = i
        if at < 0:
            if kind == WITH_COLUMNS:
                return plan_distinct(self, stats, node.left, name)
            return DistinctCount.unknown()
        ref expr = node.exprs[at]
        if len(expr._nodes) == 1 and expr._nodes[0].op == COL:
            return plan_distinct(self, stats, node.left, expr._nodes[0].text)
        var reads = _references(expr)
        if reads and len(reads.value()) == 0:
            # A constant (`expression_is_constant`): one value.
            var rows = plan_cardinality(self, stats, index)
            if rows < 0:
                return DistinctCount.unknown()
            return DistinctCount(min(rows, 1), COUNT_EXACT)
        return _relation_count(self, stats, index)
    if kind == AGG or kind == GROUPING_SETS:
        if name in node.names:
            return plan_distinct(self, stats, node.left, name)
        return _relation_count(self, stats, index)
    if kind != JOIN:
        return DistinctCount.unknown()
    # A join's column keeps its side's count: a join repeats or drops values but
    # makes no new ones.
    if node.how == JOIN_RIGHT_SEMI or node.how == JOIN_RIGHT_ANTI:
        return plan_distinct(self, stats, node.right, name)
    if _has_column(self, stats, node.left, name):
        return plan_distinct(self, stats, node.left, name)
    if node.how == JOIN_SEMI or node.how == JOIN_ANTI:
        return DistinctCount.unknown()
    if _has_column(self, stats, node.right, name):
        return plan_distinct(self, stats, node.right, name)
    var suffix = node.join_suffix()
    if suffix.byte_length() > 0 and name.endswith(suffix):
        var base = String(
            name[byte = 0 : name.byte_length() - suffix.byte_length()]
        )
        return plan_distinct(self, stats, node.right, base)
    return DistinctCount.unknown()


def _relation_count(
    self: QueryContext, mut stats: PlanStatistics, index: Int
) raises -> DistinctCount:
    """A computed column's count: its relation's rows."""
    var rows = plan_cardinality(self, stats, index)
    if rows < 0:
        return DistinctCount.unknown()
    return DistinctCount(rows, COUNT_CARDINALITY)
