"""A pipeline sinking into the aggregate table gives the scalar groups for
every worker count and batch size. Without maintain_order the groups come
back in no particular order (as Polars' group_by and DuckDB's GROUP BY give
them), so runs are compared as sets of groups. With maintain_order they come
back in first-occurrence order on every run, whichever worker took which
morsel: the finish merges the workers' rows, each in first-row order,
without a sort.

Rounding policy: integer states and keys are exact on every run. A floating
state sums each worker's partial sums, and which rows a worker reduced
depends on scheduling, so a repeated run may differ in the last bits; it
must agree within a relative 1e-12."""
# contract-group: execution.modes
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
            for ordered in [False, True]:
                var observed = StrategyTrace()
                var actual = (
                    frame.lazy()
                    .filter(col("r") >= lit(Int64(start)))
                    .group_by(["s", "i", "f", "b"], maintain_order=ordered)
                    .agg(reductions())
                    .collect(batch_size=batch)
                )
                var label = (
                    String(" at ")
                    + String(threads)
                    + " workers, batch "
                    + String(batch)
                )
                assert_true(
                    "dataframe-path: pipeline.group_table\n" in observed.text(),
                    "no aggregate table sink" + label,
                )
                # Groups in no order are never sorted; ordered ones arrive
                # in first-row order from each worker, so the finish merges
                # them into that order without sorting them either.
                assert_true(
                    "dataframe-path: aggregate_table.sorted_groups\n"
                    not in observed.text(),
                    "groups needed a sort" + label,
                )
                _ = observed^
                assert_groups(actual, models, True)
        _ = workers^
    _ = trace^


def test_repeated_runs_return_the_same_rows() raises:
    # The same groups on every run; in the same order only under
    # maintain_order.
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var rows = 72_000
    var groups = 9_000
    var frame = aggregate_frame(rows, groups)
    var eager = frame.filter(col("r") >= lit(Int64(7)))
    for ordered in [False, True]:
        var plan = (
            frame.lazy()
            .filter(col("r") >= lit(Int64(7)))
            .group_by(["s", "i"], maintain_order=ordered)
            .agg(reductions())
        )
        var first = plan.collect(batch_size=1024)
        assert_groups(first, group_models(rows, groups, 7), False)
        var grouped = eager.group_by(["s", "i"], maintain_order=ordered).agg(
            reductions()
        )
        if ordered:
            assert_same_run(grouped, first)
        else:
            first = first.sort(["s", "i"])
            assert_same_run(grouped.sort(["s", "i"]), first)
        for _ in range(4):
            var again = plan.collect(batch_size=1024)
            var eager_again = eager.group_by(
                ["s", "i"], maintain_order=ordered
            ).agg(reductions())
            if not ordered:
                again = again.sort(["s", "i"])
                eager_again = eager_again.sort(["s", "i"])
            assert_same_run(again, first)
            assert_same_run(eager_again, first)
    _ = workers^


def test_repartitioned_sinks_keep_the_ordered_groups_in_order() raises:
    # Partitions start at one per worker and multiply as tables clear: 60K
    # groups fill every table at three and eight workers, so how many
    # partitions the sinks reach depends on which worker took which
    # morsel. Ordered groups still come back in first-occurrence order on
    # every run and at every worker count; the others are the same groups.
    # One worker copies its rows without merging them.
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 240_000
    var groups = 60_000
    var frame = aggregate_frame(rows, groups)
    var models = group_models(rows, groups)
    var plan = (
        frame.lazy()
        .filter(col("r") >= lit(Int64(0)))
        .group_by(["s", "i"])
        .agg(reductions())
    )
    var ordered_plan = (
        frame.lazy()
        .filter(col("r") >= lit(Int64(0)))
        .group_by(["s", "i"], maintain_order=True)
        .agg(reductions())
    )
    var reference = frame.clear()
    var sorted_reference = frame.clear()
    for threads in [1, 2, 3, 8]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        var observed = StrategyTrace()
        var first = plan.collect()
        if threads == 1:
            assert_true(
                "dataframe-path: aggregate_table.single_sink_finish\n"
                in observed.text(),
                "one worker's rows were merged rather than copied",
            )
            reference = ordered_plan.collect()
            assert_groups(reference, models, False)
            sorted_reference = reference.sort(["s", "i"])
        _ = observed^
        assert_groups(first, models, False)
        assert_same_run(first.sort(["s", "i"]), sorted_reference)
        for _ in range(3):
            assert_same_run(plan.collect().sort(["s", "i"]), sorted_reference)
            assert_same_run(ordered_plan.collect(), reference)
        _ = workers^
    _ = trace^


def test_steps_over_the_groups_run_in_the_finish() raises:
    # A filter on the aggregate's outputs alone (SQL's HAVING) runs in the
    # table's finish before the groups' keys are read out; one reading a
    # key too (in a disjunction, which no pass moves below the aggregate),
    # and the projections above, run on each worker's part of the groups. The rows are those of the same steps run over the whole
    # eager result, a filter keeping none of them included.
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 72_000
    var groups = 9_000
    var frame = aggregate_frame(rows, groups)
    var predicates: List[Expr] = [
        col("rows") > lit(Int64(7)),
        col("rows") > lit(Int64(1_000_000)),
        (col("rows") > lit(Int64(7))) | (col("i") > lit(Int64(100))),
    ]
    for p in range(len(predicates)):
        var eager = (
            frame.filter(col("r") >= lit(Int64(7)))
            .group_by(["s", "i"])
            .agg(reductions())
            .filter(predicates[p])
            .with_columns([(col("x_count") + lit(Int64(1))).alias("next")])
            .select(["s", "i", "next", "w_sum"])
            .sort(["s", "i"])
        )
        if p == 1:
            assert_equal(eager.height(), 0)
        else:
            assert_true(eager.height() > 0)
        for threads in [1, 3, 8]:
            var workers = ScopedEnvironment(
                "DATAFRAME_THREADS", String(threads)
            )
            var observed = StrategyTrace()
            var actual = (
                frame.lazy()
                .filter(col("r") >= lit(Int64(7)))
                .group_by(["s", "i"])
                .agg(reductions())
                .filter(predicates[p])
                .with_columns([(col("x_count") + lit(Int64(1))).alias("next")])
                .select(["s", "i", "next", "w_sum"])
                .collect(batch_size=4096)
            )
            var text = observed.text()
            _ = observed^
            var label = String(" at ") + String(threads) + " for " + String(p)
            assert_equal(
                "dataframe-path: pipeline.aggregate_finish_filter\n" in text,
                p < 2,
                "the filter's place in the finish" + label,
            )
            if p != 1:
                assert_true(
                    "dataframe-path: pipeline.aggregate_finish_steps\n" in text,
                    "the steps ran after the finish" + label,
                )
            assert_same_run(actual.sort(["s", "i"]), eager)
            _ = workers^
    _ = trace^
