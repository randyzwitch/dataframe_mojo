"""Chains of inner joins start from the input that saves the most hashing.

A lazy chain streams its first input through hash tables built on every
other input. Written as `small.join(large)`, it would hash the large
table; the planner re-roots the chain at the larger input when the first
one supplies only join keys and the rows' order cannot show. Every result
must equal the plan as written.
"""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Expr, LazyFrame, Series, col, lit

comptime FACT = 60_000
comptime ORDERS = 9_000
comptime CUSTOMERS = 700


def facts() raises -> DataFrame:
    """Many rows per order, some with a null key; orders are not unique in
    it, so building on it needs a hash table."""
    var order = List[Int64](capacity=FACT)
    var valid = List[Bool](capacity=FACT)
    var value = List[Float64](capacity=FACT)
    var tag = List[String](capacity=FACT)
    for i in range(FACT):
        order.append(Int64((i * 7919) % (ORDERS + 500)))
        valid.append(i % 97 != 0)
        value.append(Float64(i % 101))
        tag.append("t" + String(i % 7))
    return DataFrame(
        [
            Series("f_order", Column[Int64](order^, valid^)),
            Series("f_value", Column[Float64](value^)),
            Series("f_tag", Column[String](tag^)),
        ]
    )


def orders() raises -> DataFrame:
    var key = List[Int64](capacity=ORDERS)
    var customer = List[Int64](capacity=ORDERS)
    var status = List[String](capacity=ORDERS)
    for i in range(ORDERS):
        # Shuffled, so the key is no arithmetic progression.
        key.append(Int64((i * 4_001) % ORDERS))
        customer.append(Int64((i * 31) % (CUSTOMERS + 40)))
        status.append("F" if i % 3 == 0 else "O")
    return DataFrame(
        [
            Series("o_key", Column[Int64](key^)),
            Series("o_customer", Column[Int64](customer^)),
            Series("o_status", Column[String](status^)),
        ]
    )


def customers() raises -> DataFrame:
    var key = List[Int64](capacity=CUSTOMERS)
    var segment = List[String](capacity=CUSTOMERS)
    for i in range(CUSTOMERS):
        key.append(Int64((i * 389) % CUSTOMERS))
        segment.append("s" + String(i % 5))
    return DataFrame(
        [
            Series("c_key", Column[Int64](key^)),
            Series("c_segment", Column[String](segment^)),
        ]
    )


def planned(plan: LazyFrame) raises -> String:
    """The plan as it runs: optimized, then join-ordered."""
    var ready = plan._optimized()
    ready._order_joins(True, 65536)
    var text = String()
    ready._describe(len(ready._nodes) - 1, 0, text, True)
    return text^


def same(plan: LazyFrame, by: List[String]) raises:
    """Optimized, unoptimized and non-streaming runs agree once sorted."""
    var want = plan.collect(optimize=False).sort(by)
    var got = plan.collect().sort(by)
    assert_equal(got.columns(), want.columns())
    assert_true(got.equals(want), "optimized result differs")
    assert_true(
        plan.collect(streaming=False).sort(by).equals(want),
        "materializing result differs",
    )


def test_small_first_input_moves_behind_the_large_one() raises:
    var chosen = customers().lazy().filter(col("c_segment") == "s1")
    var open_orders = orders().lazy().filter(col("o_status") == "O")
    var plan = (
        chosen.join(open_orders, left_on=["c_key"], right_on=["o_customer"])
        .join(facts().lazy(), left_on=["o_key"], right_on=["f_order"])
        .group_by(["o_key", "f_tag"])
        .agg([col("f_value").sum().alias("total"), col("f_value").len()])
    )
    var text = planned(plan)
    # The fact table now probes; its join reads the keys the other way.
    assert_true("f_order = o_key" in text, text)
    same(plan, ["o_key", "f_tag"])


def test_a_dropped_key_is_still_available_under_its_name() raises:
    # c_key is the first input's join key and also an output column. Once
    # the chain starts elsewhere, its values come from o_customer.
    var plan = (
        customers()
        .lazy()
        .select(["c_key"])
        .join(orders().lazy(), left_on=["c_key"], right_on=["o_customer"])
        .join(facts().lazy(), left_on=["o_key"], right_on=["f_order"])
        .group_by(["c_key"])
        .agg([col("f_value").sum().alias("total")])
    )
    assert_true("f_order = o_key" in planned(plan))
    same(plan, ["c_key"])
    # The same key reused by a later join, and a shared key name.
    var extra = DataFrame(
        [
            Series("c_key", Column[Int64]([Int64(3), 5, 8, 600, 699])),
            Series("bonus", Column[Float64]([1.0, 2.0, 3.0, 4.0, 5.0])),
        ]
    )
    var reused = (
        customers()
        .lazy()
        .select(["c_key"])
        .join(orders().lazy(), left_on=["c_key"], right_on=["o_customer"])
        .join(facts().lazy(), left_on=["o_key"], right_on=["f_order"])
        .join(extra.lazy(), ["c_key"])
        .group_by(["c_key"])
        .agg(
            [
                col("f_value").sum().alias("total"),
                col("bonus").max().alias("bonus"),
            ]
        )
    )
    same(reused, ["c_key"])


def test_two_key_joins_and_null_keys() raises:
    var pairs = List[Int64]()
    var kinds = List[String]()
    for i in range(40):
        pairs.append(Int64(i * 200))
        kinds.append("t" + String(i % 7))
    var wanted = DataFrame(
        [
            Series("w_order", Column[Int64](pairs^)),
            Series("w_tag", Column[String](kinds^)),
        ]
    )
    var plan = (
        wanted.lazy()
        .join(
            facts().lazy(),
            left_on=["w_order", "w_tag"],
            right_on=["f_order", "f_tag"],
        )
        .group_by(["w_order"])
        .agg([col("f_value").mean().alias("mean"), col("f_value").len()])
    )
    assert_true("f_order = w_order" in planned(plan))
    same(plan, ["w_order"])


def test_order_stays_as_written_where_it_can_show() raises:
    var chosen = customers().lazy().select(["c_key"])
    var joined = chosen.join(
        orders().lazy(), left_on=["c_key"], right_on=["o_customer"]
    ).join(facts().lazy(), left_on=["o_key"], right_on=["f_order"])
    # No aggregation above: rows come back in the written order.
    assert_true("f_order = o_key" not in planned(joined))
    assert_true(joined.collect().equals(joined.collect(optimize=False)))
    # An aggregate that reads row position, or keeps input order.
    var first = joined.group_by(["c_key"]).agg(
        [col("f_value").first().alias("first")]
    )
    assert_true("f_order = o_key" not in planned(first))
    same(first, ["c_key"])
    var ordered = joined.group_by(["c_key"], maintain_order=True).agg(
        [col("f_value").sum().alias("total")]
    )
    assert_true("f_order = o_key" not in planned(ordered))
    assert_true(ordered.collect().equals(ordered.collect(optimize=False)))


def test_first_input_that_supplies_columns_stays_first() raises:
    # c_segment is carried to the output: as the first input it streams
    # as slices, as a right input it would be gathered per joined row.
    var plan = (
        customers()
        .lazy()
        .join(orders().lazy(), left_on=["c_key"], right_on=["o_customer"])
        .join(facts().lazy(), left_on=["o_key"], right_on=["f_order"])
        .group_by(["c_segment"])
        .agg([col("f_value").sum().alias("total")])
    )
    assert_true("f_order = o_key" not in planned(plan))
    same(plan, ["c_segment"])


def test_a_progression_key_needs_no_hash_table_so_nothing_moves() raises:
    # The large right input is keyed by 0, 1, 2, ...: it is looked up by
    # position, so building on it costs nothing and the chain stays.
    var ids = List[Int64](capacity=FACT)
    var amounts = List[Float64](capacity=FACT)
    for i in range(FACT):
        ids.append(Int64(i))
        amounts.append(Float64(i % 13))
    var dimension = DataFrame(
        [
            Series("d_id", Column[Int64](ids^)),
            Series("d_amount", Column[Float64](amounts^)),
        ]
    )
    var few = DataFrame(
        [Series("k", Column[Int64]([Int64(5), 77, 59_999, 77, 123_456]))]
    )
    var plan = (
        few.lazy()
        .join(dimension.lazy(), left_on=["k"], right_on=["d_id"])
        .group_by(["k"])
        .agg([col("d_amount").sum().alias("total")])
    )
    assert_true("d_id = k" not in planned(plan))
    same(plan, ["k"])


def test_ungrouped_aggregate_and_filtered_large_input() raises:
    var chosen = customers().lazy().filter(col("c_segment") == "s2")
    var plan = (
        chosen.select(["c_key"])
        .join(orders().lazy(), left_on=["c_key"], right_on=["o_customer"])
        .join(
            facts().lazy().filter(col("f_value") > lit(Float64(20))),
            left_on=["o_key"],
            right_on=["f_order"],
        )
        .select(col("f_value").sum().alias("total"))
    )
    assert_true("f_order = o_key" in planned(plan))
    var want = plan.collect(optimize=False)
    var got = plan.collect()
    assert_equal(
        got.column("total").float64()._get(0),
        want.column("total").float64()._get(0),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
