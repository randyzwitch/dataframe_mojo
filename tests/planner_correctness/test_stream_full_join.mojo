"""A full join streams: its left rows probe the right's index in morsels as a
left join does, each worker flags the build rows its morsels matched, and the
build rows none matched come out as a final morsel with the left columns
null (DuckDB's FULL OUTER probe and its scan of the unmatched build rows).
The rows are the eager join's, on two-column keys with duplicates and null
keys on both sides, with and without coalescing the keys."""
# contract-group: planner.relational
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Expr, Series, col, lit, when
from tests.support.coverage_compare import (
    assert_frame,
    assert_frame_unordered,
)
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def _side(
    prefix: String, rows: Int, spread: Int, step: Int, null_every: Int
) raises -> DataFrame:
    var a = List[Int64]()
    var a_valid = List[Bool]()
    var b = List[Int64]()
    var v = List[Int64]()
    for row in range(rows):
        a.append(Int64((row * step) % spread))
        a_valid.append(row % null_every != 0)
        b.append(Int64(row % 3))
        v.append(Int64(row % 101))
    return DataFrame(
        [
            Series(prefix + "_a", Column[Int64](a^, a_valid)),
            Series(prefix + "_b", Column[Int64](b^)),
            Series(prefix + "_v", Column[Int64](v^)),
        ]
    )


def _count_if(condition: Expr) -> Expr:
    return (
        when(condition)
        .then(lit(Int64(1)))
        .otherwise(lit(Int64(0)))
        .sum(min_count=1)
    )


def test_a_full_join_under_an_aggregation_emits_unmatched_build_rows_last() raises:
    # 12K left rows over 5K key values and 8K right rows over 9K values (a
    # step of 7), so rows match once or several times or not at all on
    # either side, as TPC-DS q97 counts its store-only, catalog-only and
    # shared customer-item pairs.
    var left = _side("l", 12_000, 5_000, 1, 97)
    var right = _side("r", 8_000, 9_000, 7, 89)
    var left_seen = col("l_a").is_not_null()
    var right_seen = col("r_a").is_not_null()
    var reductions: List[Expr] = [
        _count_if(left_seen & ~right_seen).alias("left_only"),
        _count_if(~left_seen & right_seen).alias("right_only"),
        _count_if(left_seen & right_seen).alias("both"),
        col("l_v").sum().alias("lv"),
        col("r_v").sum().alias("rv"),
        col("r_v").count().alias("rn"),
    ]
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var counts = (
        left.lazy()
        .join(
            right.lazy(),
            left_on=["l_a", "l_b"],
            right_on=["r_a", "r_b"],
            how="full",
            coalesce=False,
        )
        .select_exprs(reductions)
        .collect()
    )
    assert_true(
        "dataframe-path: pipeline.join_unmatched\n" in observed.text(),
        "the unmatched build rows were not emitted by the pipeline",
    )
    _ = observed^
    _ = trace^
    var expected = (
        left.join(
            right,
            left_on=["l_a", "l_b"],
            right_on=["r_a", "r_b"],
            how="full",
            coalesce=False,
        )
        .lazy()
        .select_exprs(reductions)
        .collect()
    )
    assert_frame(counts, expected)


def test_a_grouped_full_join_with_coalesced_keys_matches_the_eager_join() raises:
    # Coalesced keys: a build row nobody matched brings its own key values
    # into the left key columns.
    var left = _side("k", 12_000, 5_000, 1, 97)
    var names = Dict[String, String]()
    names["k_v"] = "r_v"
    var right = _side("k", 8_000, 9_000, 7, 89).rename(names)
    var reductions: List[Expr] = [
        col("k_v").sum().alias("lv"),
        col("r_v").sum().alias("rv"),
        col("k_v").count().alias("ln"),
        col("r_v").count().alias("rn"),
    ]
    var rows = (
        left.lazy()
        .join(
            right.lazy(),
            left_on=["k_a", "k_b"],
            right_on=["k_a", "k_b"],
            how="full",
        )
        .group_by(["k_b"])
        .agg(reductions)
        .collect()
        .sort(["k_b"])
    )
    var expected = (
        left.join(
            right, left_on=["k_a", "k_b"], right_on=["k_a", "k_b"], how="full"
        )
        .group_by(["k_b"])
        .agg(reductions)
        .sort(["k_b"])
    )
    assert_frame(rows, expected)


def test_a_collected_full_join_has_the_eager_rows() raises:
    var left = _side("l", 3_000, 1_200, 1, 97)
    var right = _side("r", 2_000, 2_500, 7, 89)
    for coalesce in [False, True]:
        var rows = (
            left.lazy()
            .join(
                right.lazy(),
                left_on=["l_a", "l_b"],
                right_on=["r_a", "r_b"],
                how="full",
                coalesce=coalesce,
            )
            .filter(col("r_v") != lit(Int64(5)))
            .collect()
        )
        var expected = left.join(
            right,
            left_on=["l_a", "l_b"],
            right_on=["r_a", "r_b"],
            how="full",
            coalesce=coalesce,
        ).filter(col("r_v") != lit(Int64(5)))
        assert_frame_unordered(rows, expected)


def test_an_empty_left_side_keeps_every_right_row() raises:
    var left = _side("l", 0, 10, 1, 97)
    var right = _side("r", 50, 40, 7, 9)
    for how in ["full", "right"]:
        var joined = left.join(
            right,
            left_on=["l_a", "l_b"],
            right_on=["r_a", "r_b"],
            how=how,
            coalesce=False,
        )
        assert_equal(joined.height(), 50)
        assert_equal(joined.column("l_v").null_count(), 50)
        assert_frame(
            joined.select(["r_a", "r_b", "r_v"]),
            right,
        )


def test_two_full_joins_over_many_build_morsels_match_the_eager_join() raises:
    # 150K and 120K build rows: each join's unmatched rows span many
    # morsels, which the workers scan for them each on its own share. The
    # second join's build rows that the first one's unmatched rows match
    # are flagged during that round, before its own round reads them.
    var left = _side("l", 60_000, 40_000, 1, 97)
    var middle = _side("m", 150_000, 90_000, 7, 89)
    var right = _side("r", 120_000, 70_000, 3, 83)
    var reductions: List[Expr] = [
        _count_if(col("l_a").is_not_null()).alias("ln"),
        _count_if(col("m_a").is_not_null()).alias("mn"),
        _count_if(col("r_a").is_not_null()).alias("rn"),
        col("l_v").sum().alias("lv"),
        col("m_v").sum().alias("mv"),
        col("r_v").sum().alias("rv"),
        col("m_v").count().alias("mc"),
    ]
    var counts = (
        left.lazy()
        .join(
            middle.lazy(),
            left_on=["l_a", "l_b"],
            right_on=["m_a", "m_b"],
            how="full",
            coalesce=False,
        )
        .join(
            right.lazy(),
            left_on=["m_a", "m_b"],
            right_on=["r_a", "r_b"],
            how="full",
            coalesce=False,
        )
        .select_exprs(reductions)
        .collect()
    )
    var expected = (
        left.join(
            middle,
            left_on=["l_a", "l_b"],
            right_on=["m_a", "m_b"],
            how="full",
            coalesce=False,
        )
        .join(
            right,
            left_on=["m_a", "m_b"],
            right_on=["r_a", "r_b"],
            how="full",
            coalesce=False,
        )
        .lazy()
        .select_exprs(reductions)
        .collect()
    )
    assert_frame(counts, expected)
