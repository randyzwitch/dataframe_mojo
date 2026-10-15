"""Join graphs ordered by estimated cost (DuckDB's join order optimizer,
`join_order.mojo`). The estimator divides the relations' rows by each
spanning edge's total domain, the fewest table rows among the relations
its keys reach; the cheapest tree is found exactly below twelve relations
and greedily from twelve; a disconnected graph is left alone. A plan's
result is the same joins over separately collected inputs: a snowflake
written large-table-last, a nested chain met from its far side (its keys
swapped, the dropped ones read from their partners), a filter between
joins, and a plan whose rows' order shows, which keeps its written order."""
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col, lit
from dataframe.join_order import JoinEdge, JoinGraph, order_joins
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
