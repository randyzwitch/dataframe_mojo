"""Semi-join reduction: when a stream's inner or semi joins build on
aggregates grouped by the join key, the probe side runs first and its
distinct keys filter each aggregate's input. Every result equals the same
joins over the separately collected inputs: two aggregates on one key above
a selective join chain (PDS-H q21's shape, distinct counts included), a semi
join, and an aggregate on another key, which is left alone."""
from std.testing import assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col, lit
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime LINES = 120_000


def lines() raises -> DataFrame:
    var orders = List[Int64]()
    var suppliers = List[Int64]()
    var late = List[Int64]()
    for row in range(LINES):
        orders.append(Int64(row // 4))
        suppliers.append(Int64((row * 7919) % 97))
        late.append(Int64(row % 3))
    return DataFrame(
        [
            Series("o", Column[Int64](orders^)),
            Series("s", Column[Int64](suppliers^)),
            Series("late", Column[Int64](late^)),
        ]
    )


def suppliers() raises -> DataFrame:
    var keys = List[Int64]()
    var nation = List[Int64]()
    for s in range(97):
        keys.append(Int64(s))
        nation.append(Int64(s % 11))
    return DataFrame(
        [
            Series("sk", Column[Int64](keys^)),
            Series("nation", Column[Int64](nation^)),
        ]
    )


def test_aggregates_built_on_take_only_the_probe_keys() raises:
    var line = lines()
    var supp = suppliers()
    var per_order = (
        line.lazy()
        .group_by("o")
        .agg([col("s").n_unique().alias("n"), col("late").sum().alias("t")])
        .filter(col("n") > lit(Int64(1)))
        .select_exprs([col("o"), col("n"), col("t")])
    )
    var late_per_order = (
        line.lazy()
        .filter(col("late") == lit(Int64(0)))
        .group_by("o")
        .agg([col("s").n_unique().alias("ln")])
    )
    # The selective side: lines of one nation's suppliers.
    var chosen = (
        line.lazy()
        .join(
            supp.lazy().filter(col("nation") == lit(Int64(3))),
            left_on=["s"],
            right_on=["sk"],
        )
        .select_exprs([col("o"), col("s")])
    )
    var query = (
        chosen.join(per_order, ["o"])
        .join(late_per_order, ["o"])
        .group_by("s")
        .agg([col("n").sum().alias("n"), col("ln").sum().alias("ln")])
        .sort("s")
    )
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var reduced = query.collect()
    var took = "dataframe-path: stream.probe_key_reduction\n" in observed.text()
    _ = observed^
    _ = trace^
    assert_true(took, "the aggregates were not reduced by the probe keys")
    var eager = (
        chosen.collect()
        .join(per_order.collect(), on=["o"])
        .join(late_per_order.collect(), on=["o"])
        .group_by(["s"])
        .agg([col("n").sum().alias("n"), col("ln").sum().alias("ln")])
        .sort(["s"])
    )
    assert_frame(reduced, eager)


def test_semi_joins_and_other_keys_keep_their_results() raises:
    var line = lines()
    var supp = suppliers()
    var chosen = (
        line.lazy()
        .join(
            supp.lazy().filter(col("nation") == lit(Int64(5))),
            left_on=["s"],
            right_on=["sk"],
        )
        .select_exprs([col("o"), col("s")])
    )
    var busy = (
        line.lazy()
        .group_by("o")
        .agg([col("late").sum().alias("t")])
        .filter(col("t") > lit(Int64(2)))
        .select_exprs([col("o")])
    )
    var semi = (
        chosen.join(busy, ["o"], how="semi")
        .group_by("s")
        .agg([col("o").count().alias("c")])
        .sort("s")
    )
    var eager_semi = (
        chosen.collect()
        .join(busy.collect(), on=["o"], how="semi")
        .group_by(["s"])
        .agg([col("o").count().alias("c")])
        .sort(["s"])
    )
    assert_frame(semi.collect(), eager_semi)
    # Grouped on the supplier, joined on it: the key is the group key.
    # Grouped on the order but joined on the supplier: left alone.
    var by_order = line.lazy().group_by("o").agg([col("s").max().alias("s")])
    var other = (
        chosen.join(by_order, ["s"])
        .group_by("s")
        .agg([col("o_right").count().alias("c")])
        .sort("s")
    )
    var eager_other = (
        chosen.collect()
        .join(by_order.collect(), on=["s"])
        .group_by(["s"])
        .agg([col("o_right").count().alias("c")])
        .sort(["s"])
    )
    assert_frame(other.collect(), eager_other)
