"""Verify partial aggregation below a join. A grouped aggregate whose
functions read only the fact side of a join, grouped by a dimension
column, aggregates the facts by join key first and finishes above the
join. Results must equal the unsplit plan: duplicate dimension keys repeat
partial results, all-null groups keep a null sum, and outer joins keep
unmatched facts."""
# contract-group: planner.relational
from std.testing import assert_equal

from dataframe import (
    Column,
    DataFrame,
    Expr,
    LazyFrame,
    Series,
    col,
)
from tests.support.coverage_compare import assert_frame_unordered
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime DIMENSIONS = 400
comptime REPEATED = 50
comptime FACT_ROWS = 20_000
comptime FACT_KEYS = 450
comptime SOLO = 7


def dimension() raises -> DataFrame:
    """Keys 0..399, the first 50 twice under another region; key 7 alone
    in region "solo"; every 11th region null."""
    var keys = List[Int64]()
    var regions = List[String]()
    var valid = List[Bool]()
    var segments = List[Int64]()
    for d in range(DIMENSIONS + REPEATED):
        var key = d if d < DIMENSIONS else d - DIMENSIONS
        keys.append(Int64(key))
        if key == SOLO and d < DIMENSIONS:
            regions.append("solo")
        elif d >= DIMENSIONS:
            regions.append("repeat " + String(key % 3))
        else:
            regions.append("region " + String(key % 13))
        valid.append(key == SOLO or d % 11 != 4)
        segments.append(Int64(key % 5))
    return DataFrame(
        [
            Series("d", Column[Int64](keys^)),
            Series("region", Column[String](regions^, valid)),
            Series("seg", Column[Int64](segments^)),
        ]
    )


def facts() raises -> DataFrame:
    """Keys 0..449, so 400..449 match nothing; values null for key 7 and
    every 17th row."""
    var keys = List[Int64]()
    var values = List[Int64]()
    var value_valid = List[Bool]()
    var floats = List[Float64]()
    var years = List[Int64]()
    for row in range(FACT_ROWS):
        var key = (row * 7) % FACT_KEYS
        keys.append(Int64(key))
        values.append(Int64((row * 31) % 1000 - 300))
        value_valid.append(key != SOLO and row % 17 != 3)
        floats.append(Float64((row * 13) % 997) / 8.0)
        years.append(Int64(2000 + row % 3))
    return DataFrame(
        [
            Series("fd", Column[Int64](keys^)),
            Series("v", Column[Int64](values^, value_valid)),
            Series("f", Column[Float64](floats^)),
            Series("y", Column[Int64](years^)),
        ]
    )


def splittable() -> List[Expr]:
    return [
        col("v").sum(min_count=1).alias("total"),
        col("v").sum().alias("total0"),
        (col("v") * 2 + 1).sum().alias("computed"),
        col("v").count().alias("values"),
        col("v").len().alias("rows"),
        col("f").min().alias("low"),
        col("f").max().alias("high"),
    ]


def split_once(paths: String) -> Bool:
    return "dataframe-path: lazy.partial_aggregate\n" in paths


def run(
    query: LazyFrame, expect_split: Bool, threads: String = "8"
) raises -> None:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", threads)
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var expected = query.collect(optimize=False)
    var observed = StrategyTrace()
    var actual = query.collect()
    var paths = observed.text()
    _ = observed^
    assert_equal(split_once(paths), expect_split)
    assert_frame_unordered(actual, expected)
    _ = trace^
    _ = workers^


def test_inner_join_splits_with_duplicate_keys_and_null_groups() raises:
    """Grouped by a dimension column and a fact column: the facts group by
    (key, year) below the join. Repeated dimension keys add a partial
    result once per match; region "solo" holds only null values, so its
    min_count=1 sum stays null while the plain sum is 0."""
    for threads in ["1", "8"]:
        run(
            facts()
            .lazy()
            .join(dimension().lazy(), left_on=["fd"], right_on=["d"])
            .group_by(["region", "y"])
            .agg(splittable()),
            True,
            threads,
        )


def test_dimension_on_the_left_splits_its_right_input() raises:
    run(
        dimension()
        .lazy()
        .join(facts().lazy(), left_on=["d"], right_on=["fd"])
        .group_by(["seg", "region"])
        .agg(splittable()),
        True,
    )


def test_left_join_keeps_unmatched_facts() raises:
    """Facts on the preserved side: keys 400..449 match nothing and form
    the null region's groups."""
    run(
        facts()
        .lazy()
        .join(
            dimension().lazy(), left_on=["fd"], right_on=["d"], how="left"
        )
        .group_by(["region"])
        .agg(splittable()),
        True,
    )


def test_unsplittable_functions_and_groups_keep_the_plan() raises:
    """A mean has no partial form here; grouping by fact columns alone
    leaves nothing to finish above the join; aggregating the dimension
    side of a left join would lose its unmatched rows."""
    var joined = facts().lazy().join(
        dimension().lazy(), left_on=["fd"], right_on=["d"]
    )
    run(
        joined.group_by(["region"]).agg([col("v").mean().alias("mean")]),
        False,
    )
    run(
        joined.group_by(["y"]).agg([col("v").sum().alias("total")]),
        False,
    )
    run(
        facts()
        .lazy()
        .join(
            dimension().lazy(), left_on=["fd"], right_on=["d"], how="left"
        )
        .group_by(["y"])
        .agg([col("seg").sum().alias("segments")]),
        False,
    )
