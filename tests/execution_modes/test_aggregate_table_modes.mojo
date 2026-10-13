"""A pipeline sinking into the aggregate table gives the scalar groups for
every worker count and batch size, and the same rows in the same order on
every run: groups come back by hash partition and then by where each group's
first row sits in the input, never by which worker took which morsel. That
order comes from merging the workers' rows, each in first-row order, without
a sort.

Rounding policy: integer states and keys are exact on every run. A floating
state sums each worker's partial sums, and which rows a worker reduced
depends on scheduling, so a repeated run may differ in the last bits; it
must agree within a relative 1e-12."""
from std.testing import assert_equal, assert_true

from dataframe import DataFrame, DataType, Expr, col, lit
from tests.support.aggregate_table_reference import (
    aggregate_frame,
    assert_groups,
    group_models,
)
from tests.support.aggregate_table_reference import assert_close
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


def assert_same_run(actual: DataFrame, first: DataFrame) raises:
    """Same columns, rows and order; floats within the rounding policy."""
    assert_equal(actual.columns(), first.columns())
    assert_equal(actual.height(), first.height())
    for name in first.columns():
        var floating = first.column(name).dtype() == DataType.FLOAT64
        for r in range(first.height()):
            var a = actual.item(r, name)
            var b = first.item(r, name)
            assert_equal(a.is_null(), b.is_null())
            if a.is_null():
                continue
            if floating:
                assert_close(a.float64(), b.float64(), name)
            else:
                assert_true(a == b, name + " row " + String(r))


def test_every_worker_count_and_batch_size_gives_the_scalar_groups() raises:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 36_000
    var groups = 5_000
    var start = 1_001
    var frame = aggregate_frame(rows, groups)
    var models = group_models(rows, groups, start)
    for threads in [1, 2, 3, 8]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        for batch in [997, 4096, 65536]:
            var observed = StrategyTrace()
            var actual = (
                frame.lazy()
                .filter(col("r") >= lit(Int64(start)))
                .group_by(["s", "i", "f", "b"])
                .agg(reductions())
                .collect(batch_size=batch)
            )
            assert_true(
                "dataframe-path: pipeline.group_table\n" in observed.text(),
                "no aggregate table sink at "
                + String(threads)
                + " workers, batch "
                + String(batch),
            )
            # Each worker's rows arrive in first-row order, so the finish
            # merges them into that order without sorting the groups.
            assert_true(
                "dataframe-path: aggregate_table.sorted_groups\n"
                not in observed.text(),
                "groups needed a sort at "
                + String(threads)
                + " workers, batch "
                + String(batch),
            )
            _ = observed^
            assert_groups(actual, models, True)
        _ = workers^
    _ = trace^


def test_repeated_runs_return_the_same_rows_in_the_same_order() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var rows = 72_000
    var groups = 9_000
    var frame = aggregate_frame(rows, groups)
    var plan = (
        frame.lazy()
        .filter(col("r") >= lit(Int64(7)))
        .group_by(["s", "i"])
        .agg(reductions())
    )
    var first = plan.collect(batch_size=1024)
    assert_groups(first, group_models(rows, groups, 7), False)
    for _ in range(4):
        assert_same_run(plan.collect(batch_size=1024), first)
    var eager = frame.filter(col("r") >= lit(Int64(7)))
    var grouped = eager.group_by(["s", "i"]).agg(reductions())
    for _ in range(4):
        assert_same_run(eager.group_by(["s", "i"]).agg(reductions()), grouped)
    _ = workers^
