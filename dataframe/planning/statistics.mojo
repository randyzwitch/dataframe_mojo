"""Plan statistics estimated before anything runs (DuckDB 2.0's
`RelationStatisticsHelper` and `RelationStatsExtractor`, in
`optimizer/relation_statistics/`).

Each plan node has a cardinality and, for each output column, a distinct
count with its source (`DistinctCount`): a table's HyperLogLog count
(`HLL`), the span of its integer bounds (`MIN_MAX`), a constant's one value
(`EXACT`), or, with nothing known, the rows of the relation the column
came from (`CARDINALITY`). Nothing is sampled or run:

- A table scan has its table's rows. The filters right above it are its
  table filters (`ExtractGetStats`): an equality with a constant on a
  column of known distinct count keeps the rows over that count, a
  conjunction the fewest any part keeps, a filter the column's statistics
  prove always true nothing; when no part narrows the estimate it keeps
  `DEFAULT_SELECTIVITY` of the rows. A part reading several columns is a
  filter above the scan, as any other filter: a fifth of its input.
- Projections, windows and sorts keep their input's rows (a top-N at most
  its limit); a projected column keeps its source column's count, a
  constant has one value, a computed column its relation's rows.
- An aggregate's groups come from its keys' counts
  (`EstimateDistinctCardinality`), and so do a unique's rows.
- A semi or anti join keeps its left input's rows, an outer join its
  larger input's, a cross join both multiplied, and a union both added.
  An inner join is DuckDB's estimate of joining two relations
  (`CardinalityEstimator`): their rows multiplied, divided by the key
  pairs' largest total domain, each pair's domain its most reliable
  count (`DomainEstimate`: the larger HyperLogLog count, else the larger
  bounds count, else the smaller relation's rows). A column of a join
  keeps its side's count.

A node's statistics are computed once per `PlanStatistics`, the memo a
planning pass keeps (DuckDB's extractor cache), and column counts only
for the columns asked for.
"""
from std.collections import Dict, Optional
from std.math import exp

from dataframe.expr import COL, Expr, conjuncts
from dataframe.frame import DataFrame
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_INNER,
    JOIN_MARK,
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
    FILTER_ALWAYS_TRUE,
    NUMERIC_STATS,
)
from dataframe.table_filter import (
    CONJUNCTION_AND,
    CONSTANT_COMPARISON,
    FilterNode,
    check_statistics,
    table_filters,
)

# RelationStatisticsHelper::DEFAULT_SELECTIVITY: the rows a filter keeps
# when nothing measures it.
comptime DEFAULT_SELECTIVITY = 0.2
# DistinctCountSource.
comptime COUNT_CARDINALITY = 0
comptime COUNT_MIN_MAX = 1
comptime COUNT_HLL = 2
comptime COUNT_EXACT = 3


@fieldwise_init
struct DistinctCount(Copyable, ImplicitlyCopyable, Movable):
    """A column's distinct count and where it came from (DuckDB's
    `DistinctCount`); `count` is -1 when the column has none (no such
    column, or a relation of unknown rows)."""

    var count: Int
    var source: Int

    @staticmethod
    def unknown() -> Self:
        return Self(-1, COUNT_CARDINALITY)

    def known(self) -> Bool:
        return self.count >= 0

    def reliable(self) -> Bool:
        """`IsReliableDistinctCount`: a HyperLogLog or exact count."""
        return self.source == COUNT_HLL or self.source == COUNT_EXACT


struct PlanStatistics(Movable):
    """The statistics of a plan's nodes, filled as they are asked for:
    each node's estimated rows (-1 when unknown) and each (node, column)
    pair's distinct count. Valid while the nodes it read stay as they
    are: a pass that rewrites nodes starts a new one."""

    var rows: Dict[Int, Int]
    var counts: Dict[String, DistinctCount]

    def __init__(out self):
        self.rows = Dict[Int, Int]()
        self.counts = Dict[String, DistinctCount]()


def estimate_distinct_cardinality(
    counts: List[Int], input_cardinality: Int
) -> Int:
    """`EstimateDistinctCardinality`: the groups so many distinct values per
    key give over `input_cardinality` rows. The keys' counts multiply, each
    key after the first taking 5% off for correlation, and that many
    values drawn from the rows give product * (1 - e^(-rows / product)),
    at most the rows; half the rows with no counts."""
    if len(counts) == 0:
        return input_cardinality // 2
    var product = 1.0
    for count in counts:
        product *= Float64(max(count, 1))
    product *= 0.95 ** Float64(len(counts) - 1)
    var multiplier = 1.0 - exp(-Float64(input_cardinality) / product)
    var estimate = (
        Float64(input_cardinality) if multiplier == 0.0 else product
        * multiplier
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
    come from (DuckDB pushes them into the get)."""
    var cursor = index
    while cursor >= 0 and self._nodes[cursor].kind == FILTER:
        cursor = self._nodes[cursor].left
    if cursor >= 0 and self._nodes[cursor].kind == SCAN_FRAME:
        return cursor
    return -1


def _equality_rows(
    nodes: List[FilterNode], at: Int, cardinality: Int, count: DistinctCount
) -> Int:
    """`InspectTableFilter` over node `at` of a table filter: an equality
    with a constant keeps the rows over the column's distinct count, a
    conjunction the fewest any part keeps, anything else every row."""
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
    """Rows the chain of filters at `index` keeps of the table scan `scan`
    under it (`ExtractGetStats`): each part reading one of its columns is
    a table filter, the rest one filter above the scan."""
    var table = self._frames[self._nodes[scan].source_slot()].copy()
    var base = table.height()
    var parts = List[Expr]()
    var cursor = index
    while cursor != scan:
        for expr in self._nodes[cursor].exprs:
            for part in conjuncts(expr):
                parts.append(part^)
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
                # A filter the statistics prove keeps every row is no
                # filter (`FILTER_ALWAYS_TRUE`).
                if check_statistics(filter, bounds.value()) == (
                    FILTER_ALWAYS_TRUE
                ):
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
    """Rows node `index` yields, estimated from statistics before anything
    runs (`RelationStatisticsHelper::ExtractOperatorStats`); -1 when an
    input has none (a file scan, an operator with no estimate)."""
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
        # Projections and windows keep their input's rows; so do unnests
        # (`ExtractUnnestStats`).
        return plan_cardinality(self, stats, node.left)
    if kind == SORT or kind == SLICE or kind == TAIL:
        # A limit keeps at most its count (`LOGICAL_LIMIT`).
        var rows = plan_cardinality(self, stats, node.left)
        var limit = node.offset if kind == TAIL else node.length
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
    if node.how == JOIN_SEMI or node.how == JOIN_ANTI or node.how == JOIN_MARK:
        return left
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
    """`DomainEstimate::GetDistinctCount` over a key pair: the larger
    reliable count, else the larger bounds count, else the smaller
    fallback count; 1 when neither side has one."""
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
    """The columns an aggregate groups by: an aggregate's keys, the
    largest grouping set's (`ExtractAggregationStats`), a unique's subset
    or every input column (`ExtractDistinctStats`)."""
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
    for s in range(len(node.flags) // keys):
        var members = List[String]()
        for k in range(keys):
            if node.flags[s * keys + k]:
                members.append(node.names[k])
        if len(members) > len(best):
            best = members^
    return best^


def _has_column(self: QueryContext, index: Int, name: String) raises -> Bool:
    """Whether node `index` outputs column `name`."""
    var names = plan_names_of(self, index)
    if names:
        return name in names.value()
    return name in self._execute(index, True).columns()


def plan_distinct(
    self: QueryContext, mut stats: PlanStatistics, index: Int, name: String
) raises -> DistinctCount:
    """Column `name`'s distinct count at node `index`, with its source,
    carried from the table it reads through projections, filters,
    aggregate keys and joins by name, as DuckDB carries a column's count
    by its binding (`RelationColumnStatsLookup`). Unknown for a column
    the node does not output or a relation of unknown rows."""
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
    """`GetDistinctCountFromStats` for a table column: its HyperLogLog
    count, else its integer bounds' span (`GetMinMaxDistinctCount`), else
    the table's rows."""
    var slot = self._nodes[index].source_slot()
    ref table = self._frames[slot]
    if name not in table.columns():
        return DistinctCount.unknown()
    var rows = table.height()
    var count = table._distinct_count(name)
    if count < 0 and slot in self._origin_tables:
        # A shared subplan's result reads its table's counts, as DuckDB
        # carries a column's base-table count into a materialized CTE.
        ref names = self._origin_names[slot]
        if name in names:
            count = self._origin_tables[slot]._distinct_count(names[name])
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
    # A join's column keeps its side's count (`ProjectChildStats`).
    if _has_column(self, node.left, name):
        return plan_distinct(self, stats, node.left, name)
    if node.how == JOIN_SEMI or node.how == JOIN_ANTI:
        return DistinctCount.unknown()
    if _has_column(self, node.right, name):
        return plan_distinct(self, stats, node.right, name)
    var suffix = node.join_suffix() if len(node.names2) > 0 else String("")
    if suffix.byte_length() > 0 and name.endswith(suffix):
        var base = String(name[byte = 0 : name.byte_length() - suffix.byte_length()])
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
