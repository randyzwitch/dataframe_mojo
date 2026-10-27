"""A UNION ALL under a join and an aggregation or a materialize sink gives
the eager result under every executor setting: its legs are the pipeline's
sources, each running its own row-local steps first. Legs are sorted on the
join key, so the join's key range skips whole morsels of a leg; the second
leg names its columns in another order; one leg filters its own rows; and a
three-way union takes one leg from an aggregation rather than a scan."""
# contract-group: lazy.row_order
from std.testing import assert_equal
from dataframe import (
    Column,
    DataFrame,
    DataType,
    LazyFrame,
    Series,
    col,
    concat,
    lit,
)
from tests.support.coverage_compare import assert_frame

comptime ROWS = 150_000


def leg(n: Int, shift: Int, reorder: Bool) raises -> DataFrame:
    """`n` rows ascending in `k`, a measure and a label; the columns in
    another order when `reorder`."""
    var keys = List[Int64](capacity=n)
    var values = List[Int64](capacity=n)
    var labels = List[String](capacity=n)
    for row in range(n):
        keys.append(Int64(row // 3 + shift))
        values.append(Int64((row * 7) % 101))
        labels.append(String(row % 5))
    var k = Series("k", Column[Int64](keys^))
    var v = Series("v", Column[Int64](values^))
    var s = Series("s", Column[String](labels^))
    if reorder:
        return DataFrame([v^, s^, k^])
    return DataFrame([k^, v^, s^])


def dimension() raises -> DataFrame:
    """A few hundred keys inside the legs' range, with a group label."""
    var keys = List[Int64]()
    var groups = List[String]()
    for key in range(20_000, 20_600):
        keys.append(Int64(key))
        groups.append("g" + String(key % 7))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("g", Column[String](groups^)),
        ]
    )


def check(plan: LazyFrame, expected: DataFrame, order: List[String]) raises:
    for optimize in [True, False]:
        for streaming in [True, False]:
            for batch_size in [65536, 1000]:
                var out = plan.collect(
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=batch_size,
                )
                if len(order) > 0:
                    out = out.sort(order)
                assert_frame(out, expected)


def test_union_legs_through_a_join_into_each_sink() raises:
    var a = leg(ROWS, 0, False)
    var b = leg(ROWS // 2, 15_000, True)
    var d = dimension()
    var order: List[String] = ["k", "v", "s"]
    # Eager reference: the union stacked, then joined.
    var stacked = concat([a.copy(), b.select(order)])
    var joined = stacked.join(d, "k")
    var groups: List[String] = ["g"]
    var sums = (
        joined.group_by(groups)
        .agg([col("v").sum().alias("total"), col("v").len().alias("rows")])
        .sort(groups)
    )
    var union = a.lazy().concat(b.lazy())
    # Grouped sink.
    check(
        union.join(d.lazy(), "k")
        .group_by(groups)
        .agg([col("v").sum().alias("total"), col("v").len().alias("rows")]),
        sums,
        groups,
    )
    # Materialize sink. An inner join's row order is not specified, so
    # both sides are sorted on every column.
    var rows: List[String] = ["k", "v", "s", "g"]
    check(union.join(d.lazy(), "k"), joined.sort(rows), rows)
    # A leg with a filter and a computed column of its own.
    var filtered = (
        a.lazy().filter(col("v") > 50).with_columns([(col("v") * 2).alias("v")])
    )
    var eager_filtered = a.filter(col("v") > 50).with_columns(
        [(col("v") * 2).alias("v")]
    )
    var mixed = concat([eager_filtered^, b.select(order)]).join(d, "k")
    check(filtered.concat(b.lazy()).join(d.lazy(), "k"), mixed.sort(rows), rows)


def test_three_legs_one_from_an_aggregation() raises:
    var a = leg(ROWS, 0, False)
    var b = leg(ROWS // 2, 15_000, True)
    var d = dimension()
    var order: List[String] = ["k", "v", "s"]
    var keys: List[String] = ["k", "s"]
    var summed = a.lazy().group_by(keys).agg([col("v").sum().alias("v")])
    var summed_eager = a.group_by(keys).agg([col("v").sum().alias("v")])
    var three = (
        a.lazy()
        .concat(b.lazy())
        .concat(summed.select(order))
        .join(d.lazy(), "k")
    )
    var expected = (
        concat([a.copy(), b.select(order), summed_eager.select(order)])
        .join(d, "k")
        .group_by(["g", "s"])
        .agg([col("v").sum().alias("total")])
        .sort(["g", "s"])
    )
    check(
        three.group_by(["g", "s"]).agg([col("v").sum().alias("total")]),
        expected,
        ["g", "s"],
    )
