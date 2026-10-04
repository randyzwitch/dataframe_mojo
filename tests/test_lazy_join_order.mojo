"""Join order in lazy chains of inner joins: filters an OR of ANDs implies
on one input, joins pushed into the input that holds their keys, and the
most selective joins first. Results must equal the eager plan as written,
row for row once sorted.
"""
from std.testing import TestSuite, assert_equal, assert_true
from std.memory import ArcPointer
from dataframe.lazy import SCAN_FRAME, JOIN

from dataframe import Column, DataFrame, LazyFrame, Series, col, lit

comptime FACT = 20_000
comptime ORDERS = 5_000
comptime CUSTOMERS = 1_000
comptime SUPPLIERS = 200
comptime NATIONS = 25


def tables() raises -> List[DataFrame]:
    """Fact rows, orders, customers, suppliers and nations (one null
    name)."""
    var f_order = List[Int64](capacity=FACT)
    var f_supp = List[Int64](capacity=FACT)
    var f_value = List[Float64](capacity=FACT)
    for i in range(FACT):
        f_order.append(Int64((i * 7919) % ORDERS))
        # Rows of one order (i apart by ORDERS) have different suppliers.
        f_supp.append(Int64((i * 13 + i // ORDERS) % SUPPLIERS))
        f_value.append(Float64(i % 101))
    var o_key = List[Int64](capacity=ORDERS)
    var o_cust = List[Int64](capacity=ORDERS)
    var o_status = List[String](capacity=ORDERS)
    for i in range(ORDERS):
        o_key.append(Int64(i))
        o_cust.append(Int64((i * 31) % CUSTOMERS))
        o_status.append("F" if i % 3 == 0 else "O")
    var c_key = List[Int64](capacity=CUSTOMERS)
    var c_nation = List[Int64](capacity=CUSTOMERS)
    for i in range(CUSTOMERS):
        c_key.append(Int64(i))
        c_nation.append(Int64(i % NATIONS))
    var s_key = List[Int64](capacity=SUPPLIERS)
    var s_nation = List[Int64](capacity=SUPPLIERS)
    var s_name = List[String](capacity=SUPPLIERS)
    for i in range(SUPPLIERS):
        s_key.append(Int64(i))
        s_nation.append(Int64((i * 7) % NATIONS))
        s_name.append("supplier" + String(i % 40))
    var n_key = List[Int64](capacity=NATIONS)
    var n_name = List[String](capacity=NATIONS)
    var n_valid = List[Bool](capacity=NATIONS)
    for i in range(NATIONS):
        n_key.append(Int64(i))
        n_name.append("nation" + String(i))
        n_valid.append(i != 4)
    return [
        DataFrame(
            [
                Series("f_orderkey", Column[Int64](f_order^)),
                Series("f_suppkey", Column[Int64](f_supp^)),
                Series("value", Column[Float64](f_value^)),
            ]
        ),
        DataFrame(
            [
                Series("o_orderkey", Column[Int64](o_key^)),
                Series("o_custkey", Column[Int64](o_cust^)),
                Series("o_status", Column[String](o_status^)),
            ]
        ),
        DataFrame(
            [
                Series("c_custkey", Column[Int64](c_key^)),
                Series("c_nationkey", Column[Int64](c_nation^)),
            ]
        ),
        DataFrame(
            [
                Series("s_suppkey", Column[Int64](s_key^)),
                Series("s_nationkey", Column[Int64](s_nation^)),
                Series("s_name", Column[String](s_name^)),
            ]
        ),
        DataFrame(
            [
                Series("n_nationkey", Column[Int64](n_key^)),
                Series("n_name", Column[String](n_name^, n_valid^)),
            ]
        ),
    ]


def same(lazy: LazyFrame) raises:
    """The optimized plan's result against the plan run as written."""
    var got = lazy.collect()
    var want = lazy.collect(optimize=False, streaming=False)
    var names = want.columns()
    assert_equal(got.columns(), names)
    assert_equal(got.height(), want.height())
    assert_true(want.height() > 0, "the query should keep rows")
    assert_true(
        got.sort(names).equals(want.sort(names)),
        "optimized and unoptimized results differ",
    )


def nations(t: List[DataFrame], key: String, name: String) raises -> LazyFrame:
    return (
        t[4]
        .lazy()
        .select_exprs(
            [col("n_nationkey").alias(key), col("n_name").alias(name)]
        )
    )


def test_or_of_ands_narrows_both_nations() raises:
    """A filter pairing two nations' names, as PDS-H q7 does: each nation
    input gets the names its side allows, customers and suppliers are
    joined to them first, and the answer is unchanged."""
    var t = tables()
    var query = (
        t[0]
        .lazy()
        .filter(col("value") > lit(Float64(5)))
        .join(t[3].lazy(), left_on=["f_suppkey"], right_on=["s_suppkey"])
        .join(t[1].lazy(), left_on=["f_orderkey"], right_on=["o_orderkey"])
        .join(t[2].lazy(), left_on=["o_custkey"], right_on=["c_custkey"])
        .join(nations(t, "s_nationkey", "supp_nation"), "s_nationkey")
        .join(nations(t, "c_nationkey", "cust_nation"), "c_nationkey")
        .filter(
            (
                (col("supp_nation") == "nation7")
                & (col("cust_nation") == "nation0")
            )
            | (
                (col("supp_nation") == "nation23")
                & (col("cust_nation") == "nation14")
            )
        )
    )
    var plan = query.explain()
    # The predicate stays on top, and one implied filter sits on each
    # nation input, below the joins.
    var filters = 0
    var at = plan.find("FILTER")
    while at >= 0:
        filters += 1
        at = plan.find("FILTER", at + 1)
    assert_equal(filters, 4, plan)
    same(query)


def test_implied_filter_keeps_null_semantics() raises:
    """A branch reading a null name is not true, so the implied filter may
    drop that nation; a branch whose other part is on the fact side keeps
    the implied filter weaker than the predicate."""
    var t = tables()
    var query = (
        t[0]
        .lazy()
        .join(t[3].lazy(), left_on=["f_suppkey"], right_on=["s_suppkey"])
        .join(nations(t, "s_nationkey", "supp_nation"), "s_nationkey")
        .filter(
            (
                (col("supp_nation") == "nation4")
                & (col("value") > lit(Float64(3)))
            )
            | (col("supp_nation").is_null() & (col("value") < lit(Float64(50))))
            | (
                (col("supp_nation") == "nation7")
                & (col("value") > lit(Float64(90)))
            )
        )
    )
    same(query)


def test_selective_dimension_joins_first() raises:
    """Grouped inputs, a filter between joins, and a filtered nation that
    belongs to the supplier join, as in PDS-H q21."""
    var t = tables()
    var per_order = (
        t[0]
        .lazy()
        .group_by(["f_orderkey"])
        .agg([col("f_suppkey").n_unique().alias("suppliers")])
    )
    var query = (
        t[0]
        .lazy()
        .filter(col("value") > lit(Float64(20)))
        .join(per_order, "f_orderkey")
        .filter(col("suppliers") > lit(Int64(1)))
        .join(
            t[1].lazy().filter(col("o_status") == "F"),
            left_on=["f_orderkey"],
            right_on=["o_orderkey"],
        )
        .join(t[3].lazy(), left_on=["f_suppkey"], right_on=["s_suppkey"])
        .join(
            t[4].lazy().filter(col("n_name") == "nation3"),
            left_on=["s_nationkey"],
            right_on=["n_nationkey"],
        )
        .group_by(["s_name"])
        .agg([col("s_name").len().alias("count")])
    )
    same(query)
    # Without the final aggregation the chain's columns keep their order.
    var rows = (
        t[0]
        .lazy()
        .join(t[1].lazy(), left_on=["f_orderkey"], right_on=["o_orderkey"])
        .join(t[3].lazy(), left_on=["f_suppkey"], right_on=["s_suppkey"])
        .join(
            t[4].lazy().filter(col("n_name") == "nation3"),
            left_on=["s_nationkey"],
            right_on=["n_nationkey"],
        )
    )
    same(rows)


def test_shared_column_names_keep_the_written_order() raises:
    """Two inputs bring a column of one name; a join suffixes the second,
    so the chain runs as written."""
    var t = tables()
    var named = (
        t[4]
        .lazy()
        .select_exprs(
            [
                col("n_nationkey").alias("s_nationkey"),
                col("n_name").alias("s_name"),
            ]
        )
    )
    var query = (
        t[0]
        .lazy()
        .join(t[3].lazy(), left_on=["f_suppkey"], right_on=["s_suppkey"])
        .join(
            t[1].lazy().filter(col("o_status") == "F"),
            left_on=["f_orderkey"],
            right_on=["o_orderkey"],
        )
        .join(named.filter(col("s_name") == "nation0"), "s_nationkey")
    )
    same(query)


def test_date_literal_filters_move_below_joins() raises:
    """A date literal's dtype is not a column the filter reads."""
    var t = tables()
    var dated = (
        t[1]
        .with_column(
            Series(
                "o_date",
                Column[String](List[String](length=ORDERS, fill="2020-05-01")),
            )
        )
        .lazy()
        .with_columns([col("o_date").str().to_date().alias("o_date")])
    )
    var query = (
        t[0]
        .lazy()
        .join(dated, left_on=["f_orderkey"], right_on=["o_orderkey"])
        .filter(col("o_date") > lit("2020-01-01").str().to_date())
    )
    var plan = query.explain()
    var join_at = plan.find("JOIN")
    assert_true(join_at >= 0 and plan.find("FILTER") > join_at, plan)
    same(query)


def attach_source_counters(mut plan: LazyFrame) -> List[ArcPointer[Int]]:
    var counters = List[ArcPointer[Int]]()
    for i in range(len(plan._nodes)):
        if plan._nodes[i].kind == SCAN_FRAME:
            var counter = ArcPointer(Int(0))
            plan._nodes[i]._executions = counter.copy()
            counters.append(counter^)
    return counters^


def measured_star(reorder: Bool) raises -> LazyFrame:
    var base = DataFrame(
        [
            Series("a", Column[Int64]([1, 2, 3])),
            Series("b", Column[Int64]([1, 2, 3])),
        ]
    )
    var a = DataFrame(
        [
            Series("a", Column[Int64]([1, 2, 3])),
            Series("v1", Column[Int64]([1, 2, 3])),
        ]
    )
    var b = DataFrame(
        [
            Series("b", Column[Int64]([1, 2, 3])),
            Series("v2", Column[Int64]([1, 2, 3])),
        ]
    )
    return (
        base.lazy()
        .join(a.lazy().filter(col("v1") < lit(Int64(3 if reorder else 2))), "a")
        .join(b.lazy().filter(col("v2") < lit(Int64(2 if reorder else 3))), "b")
    )


def test_measured_inputs_execute_once_with_or_without_reordering() raises:
    for reorder in [False, True]:
        for streaming in [False, True]:
            var query = measured_star(reorder)
            var expected = query.collect(optimize=False, streaming=False)
            var counters = attach_source_counters(query)
            var plan = query._optimized()
            plan._order_joins(streaming, 2)
            # Each filtered dimension was measured; the base is still lazy.
            assert_equal(counters[0][], 0)
            assert_equal(counters[1][], 1)
            assert_equal(counters[2][], 1)
            # Only the base and the two cached results remain retained.
            assert_equal(len(plan._frames), 3)
            assert_equal(len(plan._schemas), 3)
            var joins = List[String]()
            for node in plan._nodes:
                if node.kind == JOIN:
                    joins.append(node.names[0])
            assert_equal(joins[0], "b" if reorder else "a")
            var actual = plan._execute(
                len(plan._nodes) - 1, False, streaming, 2
            )
            assert_true(actual.equals(expected))
            for counter in counters:
                assert_equal(counter[], 1)
            # A second planning pass neither remeasures nor retains copies.
            plan._order_joins(streaming, 2)
            assert_equal(len(plan._frames), 3)
            for counter in counters:
                assert_equal(counter[], 1)


def test_rejected_push_candidate_reuses_measured_input() raises:
    var a = List[Int64]()
    var b = List[Int64]()
    for i in range(100):
        a.append(Int64(i))
        b.append(Int64(i % 10))
    var base = DataFrame([Series("a", Column[Int64](a.copy()))])
    var host = DataFrame(
        [
            Series("a", Column[Int64](a^)),
            Series("b", Column[Int64](b^)),
        ]
    )
    var small = DataFrame(
        [
            Series("b", Column[Int64]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
            Series("v", Column[Int64]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
        ]
    )
    for streaming in [False, True]:
        # Eligible by table sizes, but measuring 70% selectivity rejects the
        # candidate. Its keys require the host, so greedy ordering stays put.
        var query = (
            base.lazy()
            .join(host.lazy(), "a")
            .join(small.lazy().filter(col("v") < lit(Int64(7))), "b")
        )
        var expected = query.collect(optimize=False, streaming=False)
        var counters = attach_source_counters(query)
        var plan = query._optimized()
        plan._order_joins(streaming, 16)
        assert_equal(counters[0][], 0)
        assert_equal(counters[1][], 0)
        assert_equal(counters[2][], 1)
        assert_equal(len(plan._frames), 3)
        var actual = plan._execute(len(plan._nodes) - 1, False, streaming, 16)
        assert_true(actual.sort("a").equals(expected.sort("a")))
        for counter in counters:
            assert_equal(counter[], 1)


def test_nested_measured_candidates_do_not_become_the_plan_root() raises:
    # Two pushed joins are measured in turn. The older speculative node is
    # then unreachable but has a higher index than the original plan root.
    var a = List[Int64]()
    var b = List[Int64]()
    for i in range(10_000):
        a.append(Int64(i))
        b.append(Int64(i % 1000))
    var base = DataFrame(
        [
            Series("a", Column[Int64](a.copy())),
            Series("z", Column[Int64](b.copy())),
        ]
    )
    var first = DataFrame(
        [
            Series("a", Column[Int64](a^)),
            Series("b", Column[Int64](b^)),
        ]
    )
    var bs = List[Int64]()
    var cs = List[Int64]()
    for i in range(1000):
        bs.append(Int64(i))
        cs.append(Int64(i % 100))
    var second = DataFrame(
        [
            Series("b", Column[Int64](bs.copy())),
            Series("c", Column[Int64](cs^)),
        ]
    )
    var independent = DataFrame(
        [
            Series("z", Column[Int64](bs.copy())),
            Series("tag", Column[Int64](bs^)),
        ]
    )
    var cs2 = List[Int64]()
    for i in range(100):
        cs2.append(Int64(i))
    var third = DataFrame([Series("c", Column[Int64](cs2^))])
    for streaming in [False, True]:
        var query = (
            base.lazy()
            .join(first.lazy(), "a")
            .join(second.lazy(), "b")
            .join(third.lazy().filter(col("c") < lit(Int64(10))), "c")
            .join(independent.lazy().filter(col("tag") < lit(Int64(200))), "z")
        )
        var expected = query.collect(optimize=False, streaming=False)
        assert_equal(expected.height(), 200)
        var counters = attach_source_counters(query)
        var plan = query._optimized()
        plan._order_joins(streaming, 256)
        var actual = plan._execute(len(plan._nodes) - 1, False, streaming, 256)
        assert_true(actual.sort("a").equals(expected.sort("a")))
        for counter in counters:
            assert_equal(counter[], 1)
        # Only the base and the final two dimension results remain live.
        assert_equal(len(plan._frames), 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
