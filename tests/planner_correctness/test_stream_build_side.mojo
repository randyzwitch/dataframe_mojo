"""A stream builds a join on its left side when that side, estimated, is the
smaller (DuckDB's build side): the join's result keeps the left's row order
and every row the written join gives, and a join whose left cannot be the
smaller keeps building on its right."""
# contract-group: planner.relational
from std.testing import assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col, lit
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
