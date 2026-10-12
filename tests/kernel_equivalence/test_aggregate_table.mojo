"""The aggregate table (DuckDB's radix-partitioned hash aggregate) against a
scalar model of each group, through both routes that dispatch to it: the
eager group-by over many groups, and a pipeline's grouped sink. Keys cover
short, long, empty, Unicode and null strings, null integers, -0.0/0.0 and NaN
floats and null Booleans; layouts cover contiguous, offset and unequally
chunked columns and strings carrying Parquet dictionary codes beside plain
ones. Worker counts cover tables that clear and keep their rows (three or
more) and tables that grow (two)."""
from std.testing import assert_true

from dataframe import DataFrame, Expr, col, concat, lit
from tests.support.aggregate_table_reference import (
    aggregate_frame,
    assert_groups,
    coded_strings,
    group_models,
)
from tests.support.coverage_generators import chunk_layout, offset_layout
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def reductions() -> List[Expr]:
    return [
        col("x").sum(min_count=1).alias("x_sum"),
        col("x").mean().alias("x_mean"),
        col("x").min().alias("x_min"),
        col("x").max().alias("x_max"),
        col("x").count().alias("x_count"),
        col("w").sum(min_count=1).alias("w_sum"),
        col("w").null_count().alias("w_nulls"),
        col("x").len().alias("rows"),
        col("y").mean().alias("y_mean"),
        col("y").std().alias("y_std"),
    ]


def eager_route(threads: Int) raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 72_000
    var groups = 6_000
    var raw = aggregate_frame(rows, groups)
    var models = group_models(rows, groups)
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        var observed = StrategyTrace()
        var actual = frame.group_by(["s", "i", "f", "b"]).agg(reductions())
        assert_true(
            "dataframe-path: group_by.aggregate_table\n" in observed.text(),
            "the eager group-by did not reach the aggregate table",
        )
        _ = observed^
        assert_groups(actual, models, True)
    _ = trace^
    _ = workers^


def test_eager_route_with_tables_that_clear_matches_scalar_groups_on_every_layout() raises:
    eager_route(8)


def test_eager_route_with_tables_that_grow_matches_scalar_groups_on_every_layout() raises:
    eager_route(2)


def test_pipeline_sink_after_a_filter_matches_scalar_groups_on_every_layout() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 72_000
    var groups = 6_000
    var start = 9_001
    var raw = aggregate_frame(rows, groups)
    var models = group_models(rows, groups, start)
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        var observed = StrategyTrace()
        var actual = (
            frame.lazy()
            .filter(col("r") >= lit(Int64(start)))
            .group_by(["s", "i", "f", "b"])
            .agg(reductions())
            .collect()
        )
        assert_true(
            "dataframe-path: pipeline.group_table\n" in observed.text(),
            "the pipeline did not sink into the aggregate table",
        )
        _ = observed^
        assert_groups(actual, models, True)
    _ = trace^
    _ = workers^


def test_dictionary_coded_and_plain_string_chunks_form_the_same_groups() raises:
    # The first half's keys carry dictionary codes and are hashed through
    # their dictionary's entries; the second half's are plain strings. A
    # group with rows in both halves must come back once.
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var rows = 72_000
    var groups = 6_000
    var raw = aggregate_frame(rows, groups)
    var half = rows // 2
    var first = raw.slice(0, half).rechunk()
    var columns = first._columns.copy()
    columns[0] = coded_strings(first.column("s"))
    var coded = DataFrame(columns^, height=half)
    var mixed = concat([coded.copy(), raw.slice(half, rows - half).rechunk()])
    var models = group_models(rows, groups)
    assert_groups(
        mixed.group_by(["s", "i", "f", "b"]).agg(reductions()), models, True
    )
    assert_groups(
        mixed.lazy()
        .filter(col("r") >= lit(Int64(0)))
        .group_by(["s", "i", "f", "b"])
        .agg(reductions())
        .collect(),
        models,
        True,
    )
    _ = workers^


def test_tables_that_fill_many_times_merge_every_group_once() raises:
    # 60K groups over three workers: each worker's sink table fills,
    # grows or clears many times, and a group's rows end up in several
    # workers' partitions before the finish merges them.
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "3")
    var rows = 240_000
    var groups = 60_000
    var frame = aggregate_frame(rows, groups)
    assert_groups(
        frame.group_by(["s", "i"]).agg(reductions()),
        group_models(rows, groups),
        False,
    )
    _ = workers^
