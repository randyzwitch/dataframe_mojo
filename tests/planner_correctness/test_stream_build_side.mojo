"""A stream builds a join on its left side when that side, estimated, is the
smaller, flipping the join inside the pipeline (a join promises no row
order): the result has every row the written join gives, and a join whose
left cannot be the smaller keeps building on its right."""
# contract-group: planner.relational
from std.testing import assert_true
from dataframe import Column, DataFrame, Expr, LazyFrame, Series, col, lit
from tests.support.coverage_compare import (
    assert_frame,
    assert_frame_unordered,
)
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


def test_a_small_left_builds_inside_the_pipeline() raises:
    # The join promises no row order, so it flips inside the pipeline:
    # the small left is hashed and the stock rows stream through it.
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
        "dataframe-path: lazy.swap_join_sides\n" in observed.text(),
        "the smaller left side was not built inside the pipeline",
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
    assert_frame_unordered(rows, eager)


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
    assert_frame_unordered(
        rows,
        t[1]
        .collect()
        .join(
            t[0].filter(col("p_size") == lit(Int64(3))).collect(),
            left_on=["s_part"],
            right_on=["p_key"],
        ),
    )


def _dimension(rows: Int, repeat: Int) raises -> DataFrame:
    # Names repeat every `repeat` rows (rows i and i + repeat hold the same
    # values), with a null name and a null region among them.
    var key = List[Int64]()
    var name = List[String]()
    var name_valid = List[Bool]()
    var region = List[String]()
    var region_valid = List[Bool]()
    var city = List[String]()
    var country = List[String]()
    for i in range(rows):
        key.append(Int64(i))
        name.append("name " + String(i % repeat))
        name_valid.append(i % repeat != 7)
        region.append("region " + String(i % 5))
        region_valid.append(i % repeat != 11)
        city.append("a city called " + String(i % repeat) + " of the region")
        country.append("country " + String(i % 3))
    return DataFrame(
        [
            ints("d_key", key),
            Series("d_name", Column[String](name^, name_valid)),
            Series("d_region", Column[String](region^, region_valid)),
            Series("d_city", Column[String](city^)),
            Series("d_country", Column[String](country^)),
        ]
    )._with_statistics()


def _facts(rows: Int, spread: Int, step: Int) raises -> DataFrame:
    var key = List[Int64]()
    var year = List[Int64]()
    var amount = List[Float64]()
    var valid = List[Bool]()
    for row in range(rows):
        key.append(Int64((row * step) % spread))
        year.append(Int64(2000 + row % 3))
        amount.append(Float64(row % 97) / 4.0)
        valid.append(row % 11 != 0)
    return DataFrame(
        [
            ints("f_key", key),
            ints("f_year", year),
            Series("f_amount", Column[Float64](amount^, valid)),
        ]
    )._with_statistics()


def _row_number_trace(
    dimension: DataFrame, facts: DataFrame, means: Bool = False
) raises -> Bool:
    """Whether the row-number keys ran, after checking that the lazy
    grouped result is the eager join's, for an inner and a left join."""
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var keys: List[String] = [
        "d_name",
        "d_region",
        "d_city",
        "d_country",
        "f_year",
    ]
    var reductions: List[Expr] = [
        col("f_amount").sum(min_count=1).alias("total"),
        col("f_amount").min().alias("low"),
        col("f_amount").max().alias("high"),
    ]
    if means:
        reductions.append(col("f_amount").mean().alias("average"))
        reductions.append(col("f_amount").count().alias("seen"))
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
    var taken = "dataframe-path: lazy.row_number_keys\n" in observed.text()
    _ = observed^
    _ = trace^
    return taken


def test_string_keys_from_one_build_group_by_its_row_number_when_groups_are_fewer_than_rows() raises:
    # 6,000 dimension rows with four string columns whose values repeat, and
    # 400K fact rows, some without a dimension row (a left join): about
    # 18,000 groups for some 370,000 joined rows, so the sink groups by the
    # dimension's row number and the groups of equal values merge afterwards,
    # to the eager join's result.
    assert_true(
        _row_number_trace(_dimension(6000, 4000), _facts(400_000, 6500, 7)),
        "the keys were not grouped by the build's row number",
    )


def test_string_keys_stay_when_every_joined_row_is_its_own_group() raises:
    # Each of 40K fact rows meets its own dimension row, so the groups are
    # the rows: the regroup would repeat the work the row numbers save.
    assert_true(
        not _row_number_trace(
            _dimension(40_000, 40_000), _facts(40_000, 40_000, 1)
        ),
        "keys grouped by row number although the groups are about the rows",
    )


def test_unique_build_values_group_by_the_row_number_for_any_reduction() raises:
    # No two dimension rows hold the same values, so the groups of the row
    # numbers are the final groups and a mean or a count (which do not merge)
    # are taken over them directly.
    assert_true(
        _row_number_trace(
            _dimension(6000, 6000), _facts(400_000, 6500, 7), means=True
        ),
        "unique build values were not grouped by the build's row number",
    )


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
            Series("c_key", Column[Int64](c_key^, c_valid)),
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
