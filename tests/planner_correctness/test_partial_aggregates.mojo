"""Verify partial aggregation below a join. A grouped aggregate whose
functions read only the fact side of a join, grouped by a dimension
column, aggregates the facts by join key first and finishes above the
join. Results must equal the unsplit plan: duplicate dimension keys repeat
partial results, all-null groups keep a null sum and mean, and outer joins
keep unmatched facts. The partial aggregate evaluates rows the join drops,
so only aggregate inputs that cannot raise split: an overflow in a dropped
row must not raise, and an overflow in a group the query later discards
still raises."""
# contract-group: planner.relational
# contract-group: planner.errors_schema
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    lit,
)
from dataframe.row_expressions import len as row_len
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


def facts(overflow_key: Int64 = -1) raises -> DataFrame:
    """Keys 0..449, so 400..449 match nothing; `v` and `f` null for key 7
    and every 17th row. Floats are multiples of 1/8, so every sum of them
    is exact whatever the order. With `overflow_key`, two more rows under
    that key hold Int64.MAX in `v`."""
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
    if overflow_key >= 0:
        for _ in range(2):
            keys.append(overflow_key)
            values.append(Int64.MAX)
            value_valid.append(True)
            floats.append(1.0)
            years.append(2000)
    return DataFrame(
        [
            Series("fd", Column[Int64](keys^)),
            Series("v", Column[Int64](values^, value_valid)),
            Series("f", Column[Float64](floats^, value_valid)),
            Series("y", Column[Int64](years^)),
        ]
    )


def splittable() -> List[Expr]:
    """Functions with a partial form over inputs that cannot raise."""
    return [
        col("f").sum(min_count=1).alias("total"),
        col("f").sum().alias("total0"),
        (col("f") * 2.0 + 1.0).sum().alias("computed"),
        col("f").mean().alias("mean"),
        col("v").count().alias("values"),
        col("v").len().alias("rows"),
        row_len().alias("all_rows"),
        col("v").min().alias("low"),
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


def _overflow(plan: LazyFrame, optimize: Bool) raises -> Bool:
    """Whether collecting `plan` raises an overflow error."""
    try:
        _ = plan.collect(optimize=optimize, streaming=False, engine="cpu")
    except error:
        if String(error).find("overflow") < 0:
            raise error
        return True
    return False


def test_inner_join_splits_with_duplicate_keys_and_null_groups() raises:
    """Grouped by two dimension columns and a fact column: the facts group
    by (key, year) below the join. Repeated dimension keys add a partial
    result once per match; region "solo" holds only null values, so its
    min_count=1 sum and its mean stay null while the plain sum is 0."""
    for threads in ["1", "8"]:
        run(
            facts()
            .lazy()
            .join(dimension().lazy(), left_on=["fd"], right_on=["d"])
            .group_by(["region", "seg", "y"])
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
        .join(dimension().lazy(), left_on=["fd"], right_on=["d"], how="left")
        .group_by(["region", "seg"])
        .agg(splittable()),
        True,
    )


def test_unsplittable_functions_and_groups_keep_the_plan() raises:
    """A median has no partial form; an integer sum can raise; grouping by
    fact columns alone leaves nothing to finish above the join; partial
    groups (key, year) as many as the final groups (region, year) save
    nothing; aggregating the dimension side of a left join would lose its
    unmatched rows."""
    var joined = (
        facts().lazy().join(dimension().lazy(), left_on=["fd"], right_on=["d"])
    )
    run(
        joined.group_by(["region"]).agg([col("f").median().alias("mid")]),
        False,
    )
    run(
        joined.group_by(["region", "seg"]).agg([col("v").sum().alias("total")]),
        False,
    )
    run(
        joined.group_by(["y"]).agg([col("f").sum().alias("total")]),
        False,
    )
    run(
        joined.group_by(["region", "y"]).agg([col("f").sum().alias("total")]),
        False,
    )
    run(
        facts()
        .lazy()
        .join(dimension().lazy(), left_on=["fd"], right_on=["d"], how="left")
        .group_by(["y"])
        .agg([col("seg").max().alias("segments")]),
        False,
    )


def test_overflow_in_a_dropped_row_does_not_raise() raises:
    """Two rows of Int64.MAX under key 450, which the join drops: the
    unsplit plan never sums them.
    An integer sum stays above the join and does not raise; a sum of the
    values cast to float cannot raise, splits, and does not raise either.
    """
    var joined = (
        facts(overflow_key=450)
        .lazy()
        .join(dimension().lazy(), left_on=["fd"], right_on=["d"])
    )
    var integer = joined.group_by(["region", "seg", "y"]).agg(
        [col("v").sum().alias("total"), col("v").count().alias("n")]
    )
    for optimize in [False, True]:
        assert_true(not _overflow(integer, optimize))
    run(integer, False)
    run(
        joined.group_by(["region", "seg", "y"]).agg(
            [
                col("v").cast(DataType.FLOAT64).sum().alias("total"),
                col("v").count().alias("n"),
            ]
        ),
        True,
    )


def test_overflow_in_a_discarded_group_still_raises() raises:
    """Two rows of Int64.MAX under key 1 overflow the integer sum of region
    "region 1", and a later filter discards that region: the overflow still raises, split or not."""
    var plan = (
        facts(overflow_key=1)
        .lazy()
        .join(dimension().lazy(), left_on=["fd"], right_on=["d"])
        .group_by(["region", "y"])
        .agg([col("v").sum().alias("total"), col("v").count().alias("n")])
        .filter(col("region") != lit("region 1"))
    )
    for optimize in [False, True]:
        assert_true(_overflow(plan, optimize))
