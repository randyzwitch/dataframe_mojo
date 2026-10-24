"""A stream builds a join on its left side when that side, estimated, is the
smaller: the join's result keeps the left's row order
and every row the written join gives, and a join whose left cannot be the
smaller keeps building on its right."""
# contract-group: planner.relational
from std.testing import assert_true
from dataframe import Column, DataFrame, Expr, LazyFrame, Series, col, lit
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def ints(name: String, values: List[Int64]) raises -> Series:
    return Series(name, Column[Int64](values.copy()))


def frames() raises -> List[LazyFrame]:
    # 4K parts, a few hundred kept by the filter; 60K stock rows with three
    # payload columns, several per part.
    var p_key = List[Int64]()
    var p_size = List[Int64]()
    for p in range(4000):
        p_key.append(Int64(p))
        p_size.append(Int64(p % 37))
    var s_part = List[Int64]()
    var s_cost = List[Int64]()
    var s_qty = List[Int64]()
    var s_site = List[Int64]()
    for row in range(60_000):
        s_part.append(Int64((row * 7919) % 4000))
        s_cost.append(Int64(row % 991))
        s_qty.append(Int64(row % 13))
        s_site.append(Int64(row % 7))
    return [
        DataFrame([ints("p_key", p_key), ints("p_size", p_size)]).lazy(),
        DataFrame(
            [
                ints("s_part", s_part),
                ints("s_cost", s_cost),
                ints("s_qty", s_qty),
                ints("s_site", s_site),
            ]
        ).lazy(),
    ]


def test_a_small_left_builds_and_keeps_its_row_order() raises:
    var t = frames()
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var query = (
        t[0]
        .filter(col("p_size") == lit(Int64(3)))
        .join(t[1], left_on=["p_key"], right_on=["s_part"])
        .filter(col("s_qty") > lit(Int64(2)))
    )
    var observed = StrategyTrace()
    var rows = query.collect()
    assert_true(
        "dataframe-path: stream.smaller_left_build\n" in observed.text(),
        "the smaller left side was not built",
    )
    _ = observed^
    _ = trace^
    var eager = (
        t[0]
        .filter(col("p_size") == lit(Int64(3)))
        .collect()
        .join(t[1].collect(), left_on=["p_key"], right_on=["s_part"])
        .filter(col("s_qty") > lit(Int64(2)))
    )
    assert_true(eager.height() > 0)
    # Same rows in the same order: the written join's left-major order.
    assert_frame(rows, eager)


def test_a_left_that_cannot_be_smaller_builds_on_the_right() raises:
    var t = frames()
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var query = t[1].join(
        t[0].filter(col("p_size") == lit(Int64(3))),
        left_on=["s_part"],
        right_on=["p_key"],
    )
    var observed = StrategyTrace()
    var rows = query.collect()
    assert_true(
        "dataframe-path: stream.smaller_left_build\n" not in observed.text(),
        "a large left side was built",
    )
    _ = observed^
    _ = trace^
    assert_frame(
        rows,
        t[1]
        .collect()
        .join(
            t[0].filter(col("p_size") == lit(Int64(3))).collect(),
            left_on=["s_part"],
            right_on=["p_key"],
        ),
    )


def test_string_keys_from_one_build_group_by_its_row_number_and_merge_equal_values() raises:
    # 6,000 dimension rows whose names repeat (rows i and i + 4,000 hold the
    # same values) with a null name and a null region; 90K fact rows, some of them
    # without a dimension row (a left join). Grouped by two dimension strings
    # and a fact column, the sink groups by the dimension's row number and the
    # groups of equal values merge afterwards: the result must be the eager
    # join's, whose rows the sums, minimums and maximums are taken over.
    var d_key = List[Int64]()
    var d_name = List[String]()
    var d_name_valid = List[Bool]()
    var d_region = List[String]()
    var d_region_valid = List[Bool]()
    for i in range(6000):
        d_key.append(Int64(i))
        d_name.append("name " + String(i % 4000))
        d_name_valid.append(i % 4000 != 7)
        d_region.append("region " + String(i % 5))
        d_region_valid.append(i % 4000 != 11)
    var f_key = List[Int64]()
    var f_year = List[Int64]()
    var f_amount = List[Float64]()
    var f_amount_valid = List[Bool]()
    for row in range(90_000):
        f_key.append(Int64((row * 7) % 6500))
        f_year.append(Int64(2000 + row % 3))
        f_amount.append(Float64(row % 97) / 4.0)
        f_amount_valid.append(row % 11 != 0)
    var dimension = DataFrame(
        [
            ints("d_key", d_key),
            Series("d_name", Column[String](d_name^, d_name_valid^)),
            Series("d_region", Column[String](d_region^, d_region_valid^)),
        ]
    )
    var facts = DataFrame(
        [
            ints("f_key", f_key),
            ints("f_year", f_year),
            Series("f_amount", Column[Float64](f_amount^, f_amount_valid^)),
        ]
    )
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var keys: List[String] = ["d_name", "d_region", "f_year"]
    var reductions: List[Expr] = [
        col("f_amount").sum(min_count=1).alias("total"),
        col("f_amount").min().alias("low"),
        col("f_amount").max().alias("high"),
    ]
    for how in ["inner", "left"]:
        var rows = (
            facts.lazy()
            .join(
                dimension.lazy(),
                left_on=["f_key"],
                right_on=["d_key"],
                how=how,
            )
            .group_by(keys)
            .agg(reductions)
            .collect()
            .sort(keys)
        )
        var expected = (
            facts.join(
                dimension, left_on=["f_key"], right_on=["d_key"], how=how
            )
            .group_by(keys)
            .agg(reductions)
            .sort(keys)
        )
        assert_frame(rows, expected)
    assert_true(
        "dataframe-path: lazy.row_number_keys\n" in observed.text(),
        "the keys were not grouped by the build's row number",
    )
    _ = observed^
    _ = trace^


def test_a_small_left_of_a_left_join_builds_in_the_pipeline_and_emits_unmatched_rows_last() raises:
    # 3,000 customers (one with a null key) and 60K orders that reach only
    # the first 1,500 customer keys, some of them several times: under an
    # aggregation the left join builds on the customers, the orders stream
    # through them, and the customers no order matched come out as a final
    # morsel with null order columns.
    var c_key = List[Int64]()
    var c_valid = List[Bool]()
    var c_val = List[Int64]()
    for c in range(3000):
        c_key.append(Int64(c))
        c_valid.append(c != 17)
        c_val.append(Int64(c % 11))
    var o_cust = List[Int64]()
    var o_id = List[Int64]()
    for row in range(60_000):
        o_cust.append(Int64((row * 37) % 1500))
        o_id.append(Int64(row))
    var customers = DataFrame(
        [
            Series("c_key", Column[Int64](c_key^, c_valid^)),
            ints("c_val", c_val),
        ]
    )
    var orders = DataFrame([ints("o_cust", o_cust), ints("o_id", o_id)])
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var counts = (
        customers.lazy()
        .join(
            orders.lazy(),
            left_on=["c_key"],
            right_on=["o_cust"],
            how="left",
        )
        .group_by(["c_key"])
        .agg(
            [
                col("o_id").count().alias("n"),
                col("c_val").sum().alias("v"),
            ]
        )
        .collect()
        .sort(["c_key"])
    )
    var text = observed.text()
    assert_true(
        "dataframe-path: lazy.swap_left_join_sides\n" in text,
        "the left join did not build on the left",
    )
    assert_true(
        "dataframe-path: pipeline.join_unmatched\n" in text,
        "the unmatched left rows were not emitted by the pipeline",
    )
    _ = observed^
    _ = trace^
    var expected = (
        customers.join(
            orders, left_on=["c_key"], right_on=["o_cust"], how="left"
        )
        .group_by(["c_key"])
        .agg(
            [
                col("o_id").count().alias("n"),
                col("c_val").sum().alias("v"),
            ]
        )
        .sort(["c_key"])
    )
    assert_true(counts.height() == 3000)
    assert_frame(counts, expected)
