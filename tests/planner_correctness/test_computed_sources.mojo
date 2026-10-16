"""A stream whose source is computed (a union, an aggregate) runs that
source once, as an in-memory table, through the same pipelines a table
takes: grouped reductions over a union and joins probed by an aggregate
give what the eager steps give."""
# contract-group: planner.relational
from dataframe import Column, DataFrame, LazyFrame, Series, col, lit
from tests.support.coverage_compare import assert_frame

comptime ROWS = 60_000


def _part(shift: Int) raises -> DataFrame:
    var keys = List[Int64]()
    var values = List[Int64]()
    var valid = List[Bool]()
    var labels = List[String]()
    for row in range(ROWS):
        keys.append(Int64((row * 31 + shift) % 97))
        values.append(Int64(row % 1_000) - 500)
        valid.append((row + shift) % 13 != 4)
        labels.append("g" + String((row + shift) % 5))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^, valid^)),
            Series("label", Column[String](labels^)),
        ]
    )


def _channel(frame: DataFrame, name: String) raises -> LazyFrame:
    return (
        frame.lazy()
        .filter(col("v") > lit(Int64(-400)))
        .select_exprs(
            [lit(name).alias("channel"), col("label"), col("k"), col("v")]
        )
    )


def test_grouped_reduction_over_a_union_matches_the_eager_steps() raises:
    var a = _part(0)
    var b = _part(7)
    var c = _part(11)
    var union = (
        _channel(a, "store")
        .concat(_channel(b, "web"))
        .concat(_channel(c, "catalog"))
    )
    var lazy = (
        union.group_by(["channel", "label"])
        .agg([col("v").sum().alias("total"), col("v").len().alias("n")])
        .sort(["channel", "label"])
        .collect()
    )
    var eager = (
        union.collect()
        .group_by(["channel", "label"])
        .agg([col("v").sum().alias("total"), col("v").len().alias("n")])
        .sort(["channel", "label"])
    )
    assert_frame(lazy, eager)


def test_an_aggregate_probing_joins_matches_the_eager_steps() raises:
    var a = _part(0)
    var b = _part(5)
    var totals = (
        a.lazy().group_by(["k", "label"]).agg([col("v").sum().alias("a_total")])
    )
    var others = (
        b.lazy().group_by(["k", "label"]).agg([col("v").sum().alias("b_total")])
    )
    var lazy = (
        totals.join(others, ["k", "label"], how="left")
        .filter(col("a_total") > col("b_total"))
        .sort(["k", "label"])
        .collect()
    )
    var eager = (
        totals.collect()
        .join(others.collect(), on=["k", "label"], how="left")
        .filter(col("a_total") > col("b_total"))
        .sort(["k", "label"])
    )
    assert_frame(lazy, eager)
