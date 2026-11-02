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


def test_repartitioned_sinks_return_the_same_order_on_every_run() raises:
    # Partitions start at one per worker and multiply as tables clear: 60K
    # groups fill every table at three and eight workers, so how many
    # partitions the sinks reach depends on which worker took which
    # morsel. Groups still come back in one order on every run with the
    # same workers. One worker copies its rows without merging them, in
    # first-row order.
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
            # With one partition, that is first-occurrence order.
            assert_same_run(
                frame.lazy()
                .filter(col("r") >= lit(Int64(0)))
                .group_by(["s", "i"], maintain_order=True)
                .agg(reductions())
                .collect(),
                first,
            )
        _ = observed^
        assert_groups(first, models, False)
        for _ in range(3):
            assert_same_run(plan.collect(), first)
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
