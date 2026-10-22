"""Join graphs ordered by estimated cost (DuckDB's join order optimizer,
`join_order.mojo`). The estimator divides the relations' rows by each
spanning edge's total domain, the fewest table rows among the relations
its keys reach; the cheapest tree is found exactly below twelve relations
and greedily from twelve; a disconnected graph is left alone; a filter
comparing two relations' columns is an edge of the estimate. A plan's
result is the same joins over separately collected inputs: a snowflake
written large-table-last, a nested chain met from its far side (its keys
swapped, the dropped ones read from their partners), a filter between
joins, and a plan whose rows' order shows, which keeps its written order."""
# contract-group: planner.relational
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col, lit
from dataframe.distinct_statistics import distinct_estimate
from dataframe.join_order import FilterEdge, JoinEdge, JoinGraph, order_joins
from tests.support.coverage_compare import assert_frame
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


def test_a_join_graph_whose_row_order_shows_keeps_its_order() raises:
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
        "dataframe-path: lazy.cost_join_order\n" not in observed.text(),
        "an order-visible join graph was reordered",
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
    assert_frame(rows, eager)


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
    # The comparison is an edge (`CreateHyperGraphEdges`): its domain, the
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


def test_distinct_estimates_sample_like_duckdb_and_skip_booleans() raises:
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


def test_joins_the_order_cannot_reorder_are_estimated_not_run() raises:
    # DuckDB's estimate for an input the join order cannot reorder: a
    # left, right or full join yields its larger input's rows, a semi or
    # anti join its left input's.
    var t = tables()
    var left = t[0].join(
        t[1], left_on=["f_product"], right_on=["p_id"], how="left"
    )
    assert_equal(left._relation_estimate(len(left._nodes) - 1), 40_000)
    var full = t[1].join(
        t[0], left_on=["p_id"], right_on=["f_product"], how="full"
    )
    assert_equal(full._relation_estimate(len(full._nodes) - 1), 40_000)
    var semi = t[1].join(
        t[0], left_on=["p_id"], right_on=["f_product"], how="semi"
    )
    assert_equal(semi._relation_estimate(len(semi._nodes) - 1), 2000)
    var anti = t[3].join(
        t[0], left_on=["s_id"], right_on=["f_store"], how="anti"
    )
    assert_equal(anti._relation_estimate(len(anti._nodes) - 1), 4000)
    # An inner join is part of the graph itself: no estimate here.
    var inner = t[0].join(t[1], left_on=["f_product"], right_on=["p_id"])
    assert_equal(inner._relation_estimate(len(inner._nodes) - 1), -1)


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
