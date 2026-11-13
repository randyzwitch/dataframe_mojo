"""Join graphs ordered by estimated cost (`join_order.mojo`). The estimator
divides the relations' rows by each
spanning edge's total domain, the fewest table rows among the relations
its keys reach; the cheapest tree is found exactly below twelve relations
and greedily from twelve; a disconnected graph is left alone; a filter
comparing two relations' columns is an edge of the estimate; semi, anti
and left joins are operators whose conflict rules say where they go (a
semi join before inner joins its keys do not read, an inner join on a
left join's right columns above it). A plan's
result is the same joins over separately collected inputs: a snowflake
written large-table-last, a nested chain met from its far side (its keys
swapped, the dropped ones read from their partners), a filter between
joins, and a graph with no consumer above it, ordered like any other since
a join promises no row order."""
# contract-group: planner.relational
from std.collections import Dict
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col, lit
from dataframe.kernels.distinct_statistics import distinct_estimate
from dataframe.join_order import (
    FilterEdge,
    JoinEdge,
    JoinGraph,
    JoinOperator,
    OPERATOR_INNER,
    OPERATOR_LEFT,
    OPERATOR_SEMI,
    build_conflicts,
    order_joins,
)
from dataframe.planning.model import SCAN_FRAME, SELECT
from tests.support.coverage_compare import (
    assert_frame,
    assert_frame_unordered,
)
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def test_estimates_divide_by_each_spanning_edges_total_domain() raises:
    # Three relations in a chain, 0 filtered to a tenth of its table.
    var rows: List[Float64] = [10.0, 1000.0, 10000.0]
    var domains: List[Float64] = [100.0, 1000.0, 10000.0]
    var edges: List[JoinEdge] = [
        JoinEdge(0, 1, ["a"], ["b0"]),
        JoinEdge(1, 2, ["b1"], ["c"]),
    ]
    var graph = JoinGraph(rows, domains, edges)
    # 10 x 1000 rows over the domain 100 (relation 0's whole table).
    assert_equal(graph.estimate(0b011), 100.0)
    assert_equal(graph.estimate(0b110), 10000.0)
    # Edges by descending domain: 1000, then 100.
    assert_equal(graph.estimate(0b111), 1000.0)
    # The cheapest tree joins the filtered relation first.
    var found = order_joins(rows, domains, edges)
    var tree = found.take()
    assert_equal(tree.mask[3], 0b011)
    assert_equal(tree.mask[4], 0b111)


def test_twelve_relations_join_greedily_and_disconnected_graphs_stay() raises:
    var rows = List[Float64]()
    var domains = List[Float64]()
    var edges = List[JoinEdge]()
    for r in range(12):
        rows.append(Float64(100 * (r + 1)))
        domains.append(Float64(100 * (r + 1)))
        if r > 0:
            edges.append(
                JoinEdge(r - 1, r, ["k" + String(r)], ["k" + String(r)])
            )
    var found = order_joins(rows, domains, edges)
    var tree = found.take()
    assert_equal(len(tree.mask), 23)
    assert_equal(tree.mask[22], (1 << 12) - 1)
    var apart: List[JoinEdge] = [JoinEdge(0, 1, ["x"], ["y"])]
    assert_true(
        not order_joins([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], apart),
        "a disconnected graph was ordered",
    )


def ints(name: String, values: List[Int64]) raises -> Series:
    return Series(name, Column[Int64](values.copy()))


def tables() raises -> List[LazyFrame]:
    """A fact table of 40K rows, its 2K products, their 20 categories, and
    4K stores in 40 regions."""
    var f_product = List[Int64]()
    var f_store = List[Int64]()
    var f_amount = List[Int64]()
    for row in range(40_000):
        f_product.append(Int64((row * 7919) % 2000))
        f_store.append(Int64((row * 104729) % 4000))
        f_amount.append(Int64(row % 97))
    var p_id = List[Int64]()
    var p_category = List[Int64]()
    for p in range(2000):
        p_id.append(Int64(p))
        p_category.append(Int64(p % 20))
    var c_id = List[Int64]()
    var c_group = List[Int64]()
    for c in range(20):
        c_id.append(Int64(c))
        c_group.append(Int64(c % 4))
    var s_id = List[Int64]()
    var s_region = List[Int64]()
    for s in range(4000):
        s_id.append(Int64(s))
        s_region.append(Int64(s % 40))
    return [
        DataFrame(
            [
                ints("f_product", f_product),
                ints("f_store", f_store),
                ints("f_amount", f_amount),
            ]
        ).lazy(),
        DataFrame([ints("p_id", p_id), ints("p_category", p_category)]).lazy(),
        DataFrame([ints("c_id", c_id), ints("c_group", c_group)]).lazy(),
        DataFrame([ints("s_id", s_id), ints("s_region", s_region)]).lazy(),
    ]


def test_a_snowflake_and_a_nested_chain_join_cheapest_first() raises:
    var t = tables()
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    # Written large table last, with the category chain nested on the
    # right and keyed at its far table (products), and a filter between
    # joins on a column the fact table supplies.
    var categories = (
        t[2]
        .filter(col("c_group") == lit(Int64(1)))
        .join(t[1], left_on=["c_id"], right_on=["p_category"])
    )
    var query = (
        t[3]
        .filter(col("s_region") < lit(Int64(10)))
        .join(t[0], left_on=["s_id"], right_on=["f_store"])
        .filter(col("f_amount") > lit(Int64(10)))
        .join(categories, left_on=["f_product"], right_on=["p_id"])
        .group_by("c_id")
        .agg(
            [
                col("f_amount").sum().alias("total"),
                col("s_id").len().alias("rows"),
                col("c_group").max().alias("category"),
            ]
        )
        .sort("c_id")
    )
    var observed = StrategyTrace()
    var ordered = query.collect()
    var took = "dataframe-path: lazy.cost_join_order\n" in observed.text()
    _ = observed^
    assert_true(took, "the join graph was not ordered by cost")
    var eager = (
        t[3]
        .filter(col("s_region") < lit(Int64(10)))
        .collect()
        .join(t[0].collect(), left_on=["s_id"], right_on=["f_store"])
        .filter(col("f_amount") > lit(Int64(10)))
        .join(categories.collect(), left_on=["f_product"], right_on=["p_id"])
        .group_by(["c_id"])
        .agg(
            [
                col("f_amount").sum().alias("total"),
                col("s_id").len().alias("rows"),
                col("c_group").max().alias("category"),
            ]
        )
        .sort(["c_id"])
    )
    assert_true(eager.height() > 0)
    assert_frame(ordered, eager)
    _ = trace^


def test_a_join_graph_with_no_consumer_is_ordered_by_cost() raises:
    # A join promises no row order, so a graph collected as it is, with no
    # aggregation or sort above it, is ordered by cost like any other.
    var t = tables()
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var query = (
        t[3]
        .filter(col("s_region") < lit(Int64(2)))
        .join(t[0], left_on=["s_id"], right_on=["f_store"])
        .join(t[1], left_on=["f_product"], right_on=["p_id"])
    )
    var observed = StrategyTrace()
    var rows = query.collect()
    assert_true(
        "dataframe-path: lazy.cost_join_order\n" in observed.text(),
        "a join graph with no consumer was not ordered by cost",
    )
    _ = observed^
    _ = trace^
    var eager = (
        t[3]
        .filter(col("s_region") < lit(Int64(2)))
        .collect()
        .join(t[0].collect(), left_on=["s_id"], right_on=["f_store"])
        .join(t[1].collect(), left_on=["f_product"], right_on=["p_id"])
    )
    assert_frame_unordered(rows, eager)


def test_counted_domains_take_the_largest_count_and_charge_unused_edges() raises:
    # A triangle: 0-1 and 1-2 join on one key set, 0-2 on another. With
    # counts, a set's domain is the largest count among its columns (not
    # the fewest table rows), and an edge met once the set is connected
    # adds one to the denominator's multiplier when its domain is counted.
    var rows: List[Float64] = [100.0, 1000.0, 1000.0]
    var domains: List[Float64] = [100.0, 1000.0, 1000.0]
    var edges: List[JoinEdge] = [
        JoinEdge(0, 1, ["a"], ["b"]),
        JoinEdge(1, 2, ["b"], ["c"]),
        JoinEdge(0, 2, ["x"], ["y"]),
    ]
    var uncounted = JoinGraph(rows, domains, edges)
    # {a, b, c} has domain 100 (relation 0's rows); {x, y} also 100.
    assert_equal(uncounted.estimate(0b011), 1000.0)
    # Two spanning edges of domain 100; the third, not counted, adds
    # nothing: 1e8 / (100 * 100).
    assert_equal(uncounted.estimate(0b111), 10000.0)
    var distinct = Dict[String, Float64]()
    distinct["0/a"] = 50.0
    distinct["1/b"] = 500.0
    distinct["2/y"] = 1000.0
    var counted = JoinGraph(rows, domains, edges, distinct)
    # {a, b, c}: the largest count, 500; {x, y}: 1000.
    assert_equal(counted.estimate(0b011), 200.0)
    # Spanning by descending domain: 0-2 (1000), then 0-1 (500); 1-2
    # (500, counted) is left over: 1e8 / (1000 * 500 * 2).
    assert_equal(counted.estimate(0b111), 100.0)


def test_a_filter_comparing_two_relations_is_an_edge_of_the_estimate() raises:
    # Totals (0) by state against state averages (1), kept where a total
    # exceeds its state's average; customers (2); addresses (3) filtered
    # to 5% of their table. Every compared column has a distinct count.
    var rows: List[Float64] = [13000.0, 50.0, 100000.0, 2500.0]
    var domains: List[Float64] = [13000.0, 50.0, 100000.0, 50000.0]
    var edges: List[JoinEdge] = [
        JoinEdge(0, 1, ["state"], ["avg_state"]),
        JoinEdge(0, 2, ["customer"], ["c_id"]),
        JoinEdge(2, 3, ["c_address"], ["a_id"]),
    ]
    var distinct = Dict[String, Float64]()
    distinct["0/state"] = 50.0
    distinct["1/avg_state"] = 50.0
    distinct["0/total"] = 12500.0
    distinct["1/average"] = 50.0
    distinct["2/c_id"] = 100000.0
    distinct["2/c_address"] = 45000.0
    distinct["3/a_id"] = 50000.0
    # Without the filter, totals join their states to 13000 rows, and
    # customers join the few addresses first (5000 rows).
    var plain = JoinGraph(rows, domains, edges, distinct)
    assert_equal(plain.estimate(0b0011), 13000.0)
    var unordered = order_joins(rows, domains, edges, distinct)
    var unfiltered = unordered.take()
    assert_equal(unfiltered.mask[4], 0b1100)
    # The comparison is an edge: its domain, the
    # larger count 12500, is met first and divides by 12500^(2/3); the
    # state edge is then left over and doubles the denominator.
    var filters: List[FilterEdge] = [
        FilterEdge(0, 1, "total", "average", False)
    ]
    var filtered = JoinGraph(rows, domains, edges, distinct, filters)
    var expected = 13000.0 * 50.0 / (12500.0 ** (2.0 / 3.0) * 2.0)
    assert_true(abs(filtered.estimate(0b0011) - expected) < 1e-6)
    # So the totals meet their averages first, as written.
    var found = order_joins(rows, domains, edges, distinct, filters)
    var ordered = found.take()
    assert_equal(ordered.mask[4], 0b0011)


def test_distinct_estimates_sample_rows_and_skip_booleans() raises:
    var n = 100_000
    var few = List[Int64]()
    var unique = List[Int64]()
    var mostly_null = List[Bool]()
    for row in range(n):
        few.append(Int64(row % 5))
        unique.append(Int64(row))
        mostly_null.append(row % 1000 == 0)
    # Five values: every sampled window holds all five (the sketch is
    # coarse, but this few values land in distinct registers).
    var small = distinct_estimate(ints("few", few))
    assert_true(small >= 4 and small <= 6)
    # A unique column comes out near its row count (a coarse sketch).
    var estimate = distinct_estimate(ints("unique", unique))
    assert_true(estimate > n // 3 and estimate <= n)
    # Null rows count as sampled but are never hashed.
    var sparse = Series("sparse", Column[Int64](few.copy(), mostly_null))
    assert_true(distinct_estimate(sparse) <= 5)
    var flags = List[Bool]()
    for row in range(10):
        flags.append(row % 2 == 0)
    assert_equal(distinct_estimate(Series("flag", Column[Bool](flags^))), -1)


def test_table_statistics_are_shared_by_copies_and_redone_for_new_columns() raises:
    var values = List[Int64]()
    for row in range(5000):
        values.append(Int64(row % 7))
    var plain = DataFrame([ints("k", values)])
    assert_equal(plain._distinct_count("k"), -1)
    var table = plain^._with_statistics()
    var copy = table.copy()
    assert_equal(copy._distinct_count("k"), 7)
    assert_equal(table._distinct_count("k"), 7)
    assert_equal(len(table._statistics.value()[].counts), 1)
    # A copy whose column is replaced is counted again, not read back.
    var other = List[Int64]()
    for row in range(5000):
        other.append(Int64(row % 3))
    copy._columns[0] = ints("k", other)
    assert_equal(copy._distinct_count("k"), 3)
    assert_equal(table._distinct_count("k"), 7)
    assert_equal(table._distinct_count("missing"), -1)


def test_joins_over_tables_with_statistics_match_the_written_order() raises:
    var t = tables()
    var fact = t[0].collect()._with_statistics().lazy()
    var products = t[1].collect()._with_statistics().lazy()
    var stores = t[3].collect()._with_statistics().lazy()
    var query = (
        stores.filter(col("s_region") < lit(Int64(2)))
        .join(fact, left_on=["s_id"], right_on=["f_store"])
        .join(products, left_on=["f_product"], right_on=["p_id"])
        .group_by(["p_category"])
        .agg([col("f_amount").sum().alias("total")])
        .sort(["p_category"])
    )
    var eager = (
        t[3]
        .filter(col("s_region") < lit(Int64(2)))
        .collect()
        .join(t[0].collect(), left_on=["s_id"], right_on=["f_store"])
        .join(t[1].collect(), left_on=["f_product"], right_on=["p_id"])
        .group_by(["p_category"])
        .agg([col("f_amount").sum().alias("total")])
        .sort(["p_category"])
    )
    assert_true(eager.height() > 0)
    assert_frame(query.collect(), eager)


def test_join_estimates_follow_join_kind_rules() raises:
    # Verify the estimator's join-kind rules: outer joins use the larger input
    # estimate, semi/anti joins retain a fifth of the left estimate, and inner
    # joins divide input-row products by the shared key domain.
    var t = tables()
    var left = t[0].join(
        t[1], left_on=["f_product"], right_on=["p_id"], how="left"
    )
    assert_equal(left._estimate(len(left._plan._nodes) - 1), 40_000)
    var full = t[1].join(
        t[0], left_on=["p_id"], right_on=["f_product"], how="full"
    )
    assert_equal(full._estimate(len(full._plan._nodes) - 1), 40_000)
    var semi = t[1].join(
        t[0], left_on=["p_id"], right_on=["f_product"], how="semi"
    )
    assert_equal(semi._estimate(len(semi._plan._nodes) - 1), 400)
    var anti = t[3].join(
        t[0], left_on=["s_id"], right_on=["f_store"], how="anti"
    )
    assert_equal(anti._estimate(len(anti._plan._nodes) - 1), 800)
    var inner = t[0].join(t[1], left_on=["f_product"], right_on=["p_id"])
    assert_equal(inner._estimate(len(inner._plan._nodes) - 1), 40_000)


def test_an_aggregate_over_joins_is_estimated_from_its_keys_counts() raises:
    # An aggregate over joined tables with statistics: its keys keep their
    # tables' HyperLogLog counts through the joins, and its groups are
    # estimated from them over the joined rows.
    var t = tables()
    var fact = t[0].collect()._with_statistics()
    var products = t[1].collect()._with_statistics()
    var query = (
        fact.lazy()
        .join(products.lazy(), left_on=["f_product"], right_on=["p_id"])
        .group_by(["p_category"])
        .agg([col("f_amount").sum().alias("total")])
    )
    var top = len(query._plan._nodes) - 1
    var categories = products._distinct_count("p_category")
    assert_true(categories > 0)
    var count = query._distinct(top, "p_category")
    assert_equal(count.count, categories)
    var groups = query._estimate(top)
    assert_true(groups > 0 and groups <= categories, String(groups))
    # A filter keeps a fifth of the rows when nothing measures it.
    var filtered = fact.lazy().filter(col("f_amount") < lit(Int64(10)))
    assert_equal(
        filtered._estimate(len(filtered._plan._nodes) - 1), 40_000 // 5
    )
    # An equality keeps the rows over the column's distinct count.
    var equal = fact.lazy().filter(col("f_amount") == lit(Int64(10)))
    var amounts = fact._distinct_count("f_amount")
    assert_equal(
        equal._estimate(len(equal._plan._nodes) - 1),
        (40_000 + amounts - 1) // amounts,
    )
    # With a table's segment statistics, a filter they prove keeps no row
    # is an empty relation.
    var stored = t[0].collect()._with_statistics()._with_storage_statistics()
    var none = stored.lazy().filter(col("f_amount") > lit(Int64(1000)))
    assert_equal(none._estimate(len(none._plan._nodes) - 1), 0)
    var some = stored.lazy().filter(col("f_amount") > lit(Int64(10)))
    assert_equal(some._estimate(len(some._plan._nodes) - 1), 40_000 // 5)


def test_union_all_estimates_sum_known_inputs_and_preserve_unknowns() raises:
    var left = DataFrame([Series("k", Column[Int64]([1, 2]))]).lazy()
    var right = DataFrame([Series("k", Column[Int64]([3, 4, 5]))]).lazy()
    var empty = DataFrame([Series("k", Column[Int64](List[Int64]()))]).lazy()
    for query in [
        left.concat(right),
        left.concat(right).concat(left),
        left.concat(empty),
    ]:
        assert_equal(
            query._estimate(len(query._plan._nodes) - 1),
            query.collect(optimize=False, engine="cpu").height(),
        )
    var unknown = left.concat(left.select(col("k").sum()))
    assert_equal(unknown._estimate(len(unknown._plan._nodes) - 1), -1)


def test_a_join_graph_over_a_left_join_orders_without_running_it() raises:
    # The left join is one relation of the graph: its rows are estimated,
    # the graph is ordered, and the rows equal the written joins'.
    var t = tables()
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var sales = t[0].join(
        t[3], left_on=["f_store"], right_on=["s_id"], how="left"
    )
    var query = (
        sales.join(
            t[1].filter(col("p_category") < lit(Int64(3))),
            left_on=["f_product"],
            right_on=["p_id"],
        )
        .join(t[2], left_on=["p_category"], right_on=["c_id"])
        .group_by(["c_group"])
        .agg([col("f_amount").sum().alias("total")])
        .sort(["c_group"])
    )
    var observed = StrategyTrace()
    var rows = query.collect()
    assert_true(
        "dataframe-path: lazy.cost_join_order\n" in observed.text(),
        "the graph over a left join was not ordered",
    )
    _ = observed^
    _ = trace^
    var eager = (
        t[0]
        .collect()
        .join(
            t[3].collect(), left_on=["f_store"], right_on=["s_id"], how="left"
        )
        .join(
            t[1].filter(col("p_category") < lit(Int64(3))).collect(),
            left_on=["f_product"],
            right_on=["p_id"],
        )
        .join(t[2].collect(), left_on=["p_category"], right_on=["c_id"])
        .group_by(["c_group"])
        .agg([col("f_amount").sum().alias("total")])
        .sort(["c_group"])
    )
    assert_true(eager.height() > 0)
    assert_frame(rows, eager)


def test_join_key_distinct_counts_follow_renames_to_their_table_column() raises:
    # A renamed join key retains its source distinct-count estimate through
    # the projection. Losing it would substitute a foreign-key domain and
    # change estimated join cardinality.
    var table = tables()[1].collect()._with_statistics()
    var renamed = (
        table.lazy()
        .filter(col("p_category") < lit(Int64(5)))
        .select_exprs([col("p_id").alias("pk"), col("p_category")])
    )
    var top = len(renamed._plan._nodes) - 1
    assert_equal(
        renamed._distinct(top, "pk").count, table._distinct_count("p_id")
    )
    assert_equal(
        renamed._distinct(top, "p_category").count,
        table._distinct_count("p_category"),
    )
    # Without a measured distinct count, a computed column falls back to its
    # relation's row count.
    var computed = table.lazy().with_columns(
        [(col("p_id") + lit(Int64(1))).alias("pk")]
    )
    var made = computed._distinct(len(computed._plan._nodes) - 1, "pk")
    assert_equal(made.count, table.height())
    assert_true(not made.reliable())


def test_a_semi_join_runs_before_inner_joins_its_keys_do_not_read() raises:
    # Customers joined to addresses and demographics,
    # then a semi join on the customer key. The semi join's conflict
    # rules ask only for the customers (inner joins below it associate),
    # so it can run first and the joins after it see a fifth of the rows.
    var rows: List[Float64] = [100_000.0, 50_000.0, 1_920_000.0, 300_000.0]
    var edges: List[JoinEdge] = [
        JoinEdge(0, 1, ["c_addr"], ["ca_sk"]),
        JoinEdge(0, 2, ["c_demo"], ["cd_sk"]),
    ]
    var ops: List[JoinOperator] = [
        JoinOperator(OPERATOR_INNER, 0b0001, 0b0010, 0b0011, [], []),
        JoinOperator(OPERATOR_INNER, 0b0011, 0b0100, 0b0101, [0], []),
        JoinOperator(OPERATOR_SEMI, 0b0111, 0b1000, 0b1001, [0, 1], []),
    ]
    build_conflicts(ops)
    assert_true(not ops[0].must_apply and not ops[1].must_apply)
    assert_true(ops[2].must_apply)
    assert_equal(ops[2].left_total, 0b0001)
    assert_equal(ops[2].right_total, 0b1000)
    var graph = JoinGraph(
        rows, rows, edges, Dict[String, Float64](), List[FilterEdge](), ops
    )
    # A semi join keeps a fifth of its left input's rows.
    assert_equal(graph.estimate(0b1001), 20_000.0)
    assert_equal(graph.candidate(0b0001, 0b1000), 2)
    # Its right relation joins nothing else.
    assert_equal(graph.candidate(0b0010, 0b1000), -2)
    var found = order_joins(
        rows, rows, edges, Dict[String, Float64](), List[FilterEdge](), ops
    )
    var tree = found.take()
    assert_equal(tree.mask[4], 0b1001)
    assert_equal(tree.operator[4], 2)
    assert_equal(tree.mask[6], 0b1111)


def test_inner_joins_on_a_left_joins_right_columns_stay_above_it() raises:
    # The left join cannot cross an inner join that removes its preserved
    # rows: doing so would introduce unmatched rows absent from the original
    # expression. Verify the conflict rules reject that split.
    var rows: List[Float64] = [1000.0, 100.0, 10.0]
    var ops: List[JoinOperator] = [
        JoinOperator(OPERATOR_LEFT, 0b001, 0b010, 0b011, [], []),
        JoinOperator(OPERATOR_INNER, 0b011, 0b100, 0b110, [0], []),
    ]
    build_conflicts(ops)
    assert_true(ops[1].must_apply)
    assert_equal(ops[1].left_total, 0b011)
    assert_equal(ops[1].right_total, 0b100)
    var edges: List[JoinEdge] = [
        JoinEdge(0, 1, ["a"], ["b"], 0),
        JoinEdge(1, 2, ["b2"], ["c"], 1),
    ]
    var graph = JoinGraph(
        rows, rows, edges, Dict[String, Float64](), List[FilterEdge](), ops
    )
    assert_equal(graph.candidate(0b010, 0b100), -2)
    assert_equal(graph.candidate(0b001, 0b010), 0)
    assert_equal(graph.candidate(0b011, 0b100), 1)
    # A left join keeps at least its left input's rows.
    assert_true(graph.estimate(0b011) >= 1000.0)
    var found = order_joins(
        rows, rows, edges, Dict[String, Float64](), List[FilterEdge](), ops
    )
    var tree = found.take()
    assert_equal(tree.mask[3], 0b011)
    assert_equal(tree.operator[4], 1)


def test_semi_anti_and_left_joins_are_ordered_with_the_inner_joins() raises:
    # Inner joins, then a semi, a left and an anti join, and a filter on
    # the left join's right column, which must stay above that join.
    var t = tables()
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var regions = t[3].filter(col("s_region") < lit(Int64(5)))
    var later = (
        t[3]
        .select_exprs(
            [col("s_id").alias("l_id"), col("s_region").alias("l_region")]
        )
        .filter(col("l_region") < lit(Int64(20)))
    )
    var excluded = (
        t[1]
        .filter(col("p_category") == lit(Int64(7)))
        .select_exprs([col("p_id").alias("x_id")])
    )
    var query = (
        t[0]
        .join(t[1], left_on=["f_product"], right_on=["p_id"])
        .join(t[2], left_on=["p_category"], right_on=["c_id"])
        .join(regions, left_on=["f_store"], right_on=["s_id"], how="semi")
        .join(later, left_on=["f_store"], right_on=["l_id"], how="left")
        .join(excluded, left_on=["f_product"], right_on=["x_id"], how="anti")
        .filter(col("l_region").is_null() | (col("c_group") == lit(Int64(1))))
        .group_by(["c_group"])
        .agg(
            [
                col("f_amount").sum().alias("total"),
                col("f_amount").len().alias("rows"),
                col("l_region").max().alias("region"),
            ]
        )
        .sort(["c_group"])
    )
    var observed = StrategyTrace()
    var rows = query.collect()
    assert_true(
        "dataframe-path: lazy.cost_join_order\n" in observed.text(),
        "the graph with semi, anti and left joins was not ordered",
    )
    _ = observed^
    _ = trace^
    var written = query.collect(optimize=False, engine="cpu")
    assert_true(written.height() > 0)
    assert_frame(rows, written)


def test_composite_keys_cap_the_join_domain_at_the_smaller_relation() raises:
    # For two equality keys joining the same relations, cap the estimated key
    # domain at the smaller relation's rows when that count is within eight
    # times the observed domain.
    var edges: List[JoinEdge] = [JoinEdge(0, 1, ["a", "b"], ["c", "d"])]
    var distinct = Dict[String, Float64]()
    distinct["0/a"] = 100.0
    distinct["1/c"] = 100.0
    distinct["0/b"] = 50.0
    distinct["1/d"] = 50.0
    var graph = JoinGraph([1000.0, 500.0], [1000.0, 500.0], edges, distinct)
    # 1000 * 500 / min(1000, 500)
    assert_equal(graph.estimate(3), 1000.0)
    # Beyond eight times the domain the cap is implausible.
    var far = JoinGraph(
        [100000.0, 90000.0], [100000.0, 90000.0], edges, distinct
    )
    assert_equal(far.estimate(3), 100000.0 * 90000.0 / 100.0)


def test_one_table_slices_are_not_shared() raises:
    # Two readers share a source table but keep their own scan projections. A
    # filter/projection over a single table does not justify materializing a
    # separate shared frame, and source statistics remain available.
    var day = List[Int64]()
    var year = List[Int64]()
    for d in range(7000):
        day.append(Int64(d))
        year.append(Int64(d // 365))
    var dates = DataFrame(
        [ints("d_sk", day), ints("d_year", year)]
    )._with_statistics()
    var shared = dates.lazy().select_exprs(
        [col("d_sk").alias("day"), col("d_year")]
    )
    var channels = List[LazyFrame]()
    for c in range(2):
        var sold = List[Int64]()
        var amount = List[Int64]()
        for row in range(5000):
            sold.append(Int64((row * 31 + c) % 200))
            amount.append(Int64(row % 11))
        channels.append(
            DataFrame([ints("f_day", sold), ints("f_amount", amount)])
            .lazy()
            .join(shared, left_on=["f_day"], right_on=["day"])
        )
    var query = channels[0].concat(channels[1])
    var plan = query._optimized()
    var nodes = len(plan._plan._nodes)
    plan._share_subplans(True, 65536)
    assert_equal(len(plan._plan._nodes), nodes)
    for i in range(len(plan._plan._nodes)):
        ref node = plan._plan._nodes[i]
        if node.kind != SELECT:
            continue
        assert_equal(
            plan._distinct(i, "day").count, dates._distinct_count("d_sk")
        )
    # The shared result joins as the written query does.
    assert_frame_unordered(
        query.collect(),
        query.collect(optimize=False, engine="cpu"),
    )


def test_a_filtered_relation_streaming_into_a_build_is_not_run_ahead() raises:
    # Filtered customers, orders and lines. Planning
    # estimates every relation from statistics and runs none of them
    # ahead: each filter runs once, in the pipeline the plan puts it in.
    var customer = List[Int64]()
    var segment = List[Int64]()
    for c in range(2000):
        customer.append(Int64(c))
        segment.append(Int64(c % 5))
    var order = List[Int64]()
    var buyer = List[Int64]()
    var day = List[Int64]()
    for o in range(20_000):
        order.append(Int64(o))
        buyer.append(Int64((o * 7919) % 2000))
        day.append(Int64(o % 100))
    var line_order = List[Int64]()
    var shipped = List[Int64]()
    var price = List[Int64]()
    for l in range(80_000):
        line_order.append(Int64((l * 104729) % 20_000))
        shipped.append(Int64(l % 100))
        price.append(Int64(l % 997))
    var query = (
        DataFrame([ints("c_id", customer), ints("c_segment", segment)])
        .lazy()
        .filter(col("c_segment") == lit(Int64(1)))
        .join(
            DataFrame(
                [
                    ints("o_id", order),
                    ints("o_buyer", buyer),
                    ints("o_day", day),
                ]
            )
            .lazy()
            .filter(col("o_day") < lit(Int64(50))),
            left_on=["c_id"],
            right_on=["o_buyer"],
        )
        .join(
            DataFrame(
                [
                    ints("l_order", line_order),
                    ints("l_shipped", shipped),
                    ints("l_price", price),
                ]
            )
            .lazy()
            .filter(col("l_shipped") > lit(Int64(50))),
            left_on=["o_id"],
            right_on=["l_order"],
        )
        .group_by(["o_id", "o_day"])
        .agg([col("l_price").sum().alias("revenue")])
    )
    var report = query.profile(engine="cpu")[1].copy()
    var filters = 0
    for r in range(report.height()):
        if report.item(r, "operator").string().startswith("FILTER"):
            filters += 1
    assert_equal(filters, 3)
    assert_frame_unordered(
        query.collect(), query.collect(optimize=False, engine="cpu")
    )


def test_a_graph_with_an_empty_relation_is_its_empty_result() raises:
    # Stored statistics prove one inner-join input empty. Skip the remaining
    # graph work while retaining the full result schema.
    var t = tables()
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var none = List[Int64]()
    var query = (
        t[0]
        .join(t[1], left_on=["f_product"], right_on=["p_id"])
        .join(
            DataFrame([ints("c_id", none), ints("c_group", none)])
            .lazy()
            .filter(col("c_group") == lit(Int64(99))),
            left_on=["p_category"],
            right_on=["c_id"],
        )
        .join(t[3], left_on=["f_store"], right_on=["s_id"], how="semi")
    )
    var observed = StrategyTrace()
    var rows = query.collect()
    assert_true(
        "dataframe-path: lazy.empty_join_graph\n" in observed.text(),
        "the graph with an empty relation ran",
    )
    _ = observed^
    _ = trace^
    var written = query.collect(optimize=False, engine="cpu")
    assert_equal(rows.height(), 0)
    assert_equal(written.height(), 0)
    assert_equal(len(rows.columns()), len(written.columns()))
    for i in range(len(rows.columns())):
        assert_equal(rows.columns()[i], written.columns()[i])
        var name = rows.columns()[i]
        assert_true(rows.column(name).dtype() == written.column(name).dtype())


def test_a_filter_rejecting_nulls_makes_a_left_join_inner() raises:
    # A filter on right columns that rejects nulls removes every unmatched
    # left-join row. Convert to an inner join and push the eligible filter
    # into the right input.
    var t = tables()
    var products = t[1].filter(col("p_id") < lit(Int64(1500)))
    var rejecting = (
        t[0]
        .join(products, left_on=["f_product"], right_on=["p_id"], how="left")
        .filter(col("p_category") > lit(Int64(5)))
    )
    var keeping = (
        t[0]
        .join(products, left_on=["f_product"], right_on=["p_id"], how="left")
        .filter(
            col("p_category").is_null() | (col("p_category") > lit(Int64(5)))
        )
    )
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var rows = rejecting.collect()
    assert_true(
        "dataframe-path: lazy.left_join_to_inner\n" in observed.text(),
        "the left join under a null-rejecting filter stayed a left join",
    )
    _ = observed^
    var kept_trace = StrategyTrace()
    var kept = keeping.collect()
    assert_true(
        "dataframe-path: lazy.left_join_to_inner\n" not in kept_trace.text(),
        "a filter keeping null rows made the left join inner",
    )
    _ = kept_trace^
    _ = trace^
    assert_true(rows.height() > 0)
    assert_frame_unordered(
        rows, rejecting.collect(optimize=False, engine="cpu")
    )
    var written = keeping.collect(optimize=False, engine="cpu")
    assert_true(written.height() > rows.height())
    assert_frame_unordered(kept, written)


def test_an_empty_build_ends_its_stream_with_the_rows_above_it() raises:
    # The products' filter keeps nothing. Planning runs nothing ahead; the
    # stream builds the products, finds them empty, and probes no fact
    # row. A count above the inner
    # join still yields its one row.
    var t = tables()
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var query = t[0].join(
        t[1].filter(col("p_category") == lit(Int64(99))),
        left_on=["f_product"],
        right_on=["p_id"],
    )
    var counted = query.select_exprs([col("f_amount").count().alias("rows")])
    var observed = StrategyTrace()
    var rows = query.collect()
    var count = counted.collect()
    assert_true(
        "dataframe-path: stream.empty_build\n" in observed.text(),
        "the empty build did not end its stream",
    )
    _ = observed^
    _ = trace^
    assert_equal(rows.height(), 0)
    assert_equal(
        len(rows.columns()),
        len(query.collect(optimize=False, engine="cpu").columns()),
    )
    assert_frame(count, counted.collect(optimize=False, engine="cpu"))
