"""Independent inputs run at once, each from its own copy of the plan: a
union's inputs (stacked in input order, nested unions included), a join's
two computed inputs, and a stream's build sides (filtered or aggregated)
and breaker source, which run ahead of it as scheduled pipelines. Results match the inputs computed one by one, and an error in
any input reaches the caller."""
# contract-group: planner.relational
from std.testing import assert_equal, assert_raises, assert_true
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
from dataframe.frame import concat
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 200_000


def source() raises -> DataFrame:
    var keys = List[Int64]()
    var values = List[Int64]()
    var valid = List[Bool]()
    var text = List[String]()
    for row in range(ROWS):
        keys.append(Int64((row * 7919) % 997))
        values.append(Int64(row % 1_000) - 500)
        valid.append(row % 23 != 5)
        text.append("r" + String(row % 50))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^, valid)),
            Series("s", Column[String](text^)),
        ]
    )


def part(frame: DataFrame, shift: Int) raises -> LazyFrame:
    """One grouped input, distinct per `shift`."""
    return (
        frame.lazy()
        .filter((col("k") % lit(Int64(3))) == lit(Int64(shift)))
        .group_by("k")
        .agg([col("v").sum().alias("total"), col("v").count().alias("n")])
        .sort("k")
    )


def test_union_inputs_stack_in_order() raises:
    var frame = source()
    var inputs = List[LazyFrame]()
    for shift in range(3):
        inputs.append(part(frame, shift))
    var union = inputs[0].concat(inputs[1]).concat(inputs[2])
    var nested = inputs[0].concat(inputs[1].concat(inputs[2]))
    var expected = concat(
        [inputs[0].collect(), inputs[1].collect(), inputs[2].collect()]
    )
    assert_frame(union.collect(), expected)
    assert_frame(nested.collect(), expected)
    assert_equal(union.collect().height(), 997)


def test_an_error_in_either_input_reaches_the_caller() raises:
    var frame = source()
    var good = part(frame, 0)
    # Strings that are not integers fail a strict cast when it runs.
    var bad = frame.lazy().select_exprs(
        [col("s").cast(DataType.INT64).alias("k"), col("v").alias("total")]
    )
    with assert_raises():
        _ = good.select_exprs([col("k"), col("total")]).concat(bad).collect()
    with assert_raises():
        _ = bad.concat(good.select_exprs([col("k"), col("total")])).collect()


def totals(frame: DataFrame, shift: Int, name: String) raises -> LazyFrame:
    """Per-key totals of one third of the rows, under its own name."""
    return (
        frame.lazy()
        .filter((col("k") % lit(Int64(2))) == lit(Int64(shift % 2)))
        .group_by("k")
        .agg([col("v").sum().alias(name)])
    )


def test_join_inputs_run_at_once_and_join_as_computed() raises:
    """A join of two aggregates (both inputs computed, so both run at
    once) and a stream of left joins over three aggregates (each build
    and the source end in an aggregate, so all three
    run ahead of the stream) give what joining the separately collected
    inputs gives."""
    var frame = source()
    var a = totals(frame, 0, "a")
    var b = totals(frame, 1, "b")
    var c = frame.lazy().group_by("k").agg([col("v").count().alias("c")])
    var inner = a.join(b, ["k"]).sort("k").collect()
    var eager_inner = a.collect().join(b.collect(), on=["k"]).sort(["k"])
    assert_frame(inner, eager_inner)
    var chained = (
        c.join(a, ["k"], how="left")
        .join(b, ["k"], how="left")
        .group_by("c")
        .agg([col("a").sum().alias("a"), col("b").sum().alias("b")])
        .sort("c")
        .collect()
    )
    var eager_chained = (
        c.collect()
        .join(a.collect(), on=["k"], how="left")
        .join(b.collect(), on=["k"], how="left")
        .group_by(["c"])
        .agg([col("a").sum().alias("a"), col("b").sum().alias("b")])
        .sort(["c"])
    )
    assert_frame(chained, eager_chained)


def test_uncoalesced_right_join_inputs_run_ahead_and_match() raises:
    """A stream over a right join that keeps both key columns
    (coalesce=False), one aggregate building and another probing: the
    stream runs this join as a step, so both inputs run ahead of it, as
    for its inner and left joins. The grouped sums are what joining the
    separately collected inputs gives."""
    var frame = source()
    var a = totals(frame, 0, "a")
    var b = frame.lazy().group_by("k").agg([col("v").count().alias("c")])
    var streamed = (
        a.join(
            b,
            left_on=["k"],
            right_on=["k"],
            how="right",
            suffix="_b",
            coalesce=False,
        )
        .group_by("c")
        .agg([col("a").sum().alias("a"), col("k_b").count().alias("n")])
        .sort("c")
        .collect()
    )
    var eager = (
        a.collect()
        .join(
            b.collect(),
            left_on=["k"],
            right_on=["k"],
            how="right",
            suffix="_b",
            coalesce=False,
        )
        .group_by(["c"])
        .agg([col("a").sum().alias("a"), col("k_b").count().alias("n")])
        .sort(["c"])
    )
    assert_frame(streamed, eager)


def test_prefetched_builds_hash_their_keys_and_match() raises:
    """Two aggregates built on, by two keys and by one, under a third
    that probes (all three computed ahead of the stream): each build's
    job also hashes its keys, and the stream probes with those indexes.
    The sums are what joining the separately collected inputs gives."""
    var frame = source()
    var pairs = (
        frame.lazy()
        .filter(col("v") > lit(Int64(0)))
        .group_by(["k", "s"])
        .agg([col("v").sum().alias("p")])
    )
    var singles = totals(frame, 1, "b")
    var probe = (
        frame.lazy().group_by(["k", "s"]).agg([col("v").count().alias("c")])
    )
    var streamed = (
        probe.join(pairs, ["k", "s"], how="left")
        .join(singles, ["k"], how="left")
        .group_by("c")
        .agg([col("p").sum().alias("p"), col("b").sum().alias("b")])
        .sort("c")
        .collect()
    )
    var eager = (
        probe.collect()
        .join(pairs.collect(), on=["k", "s"], how="left")
        .join(singles.collect(), on=["k"], how="left")
        .group_by(["c"])
        .agg([col("p").sum().alias("p"), col("b").sum().alias("b")])
        .sort(["c"])
    )
    assert_frame(streamed, eager)


def test_an_error_in_a_prefetched_input_reaches_the_caller() raises:
    var frame = source()
    var good = totals(frame, 0, "a")
    var bad = (
        frame.lazy()
        .with_columns([col("s").cast(DataType.INT64).alias("x")])
        .group_by("k")
        .agg([col("x").sum().alias("x")])
    )
    var counts = frame.lazy().group_by("k").agg([col("v").count().alias("c")])
    with assert_raises():
        _ = counts.join(good, ["k"], how="left").join(bad, ["k"]).collect()


def test_whole_input_selections_cross_joined_run_ahead_and_match() raises:
    """One-row statistics of filtered bands, cross joined side by side:
    each band's selection reads its whole input, so
    each is a breaker computed ahead of the stream, and the row is what
    the separately collected bands give."""
    var frame = source()
    var row = Optional[LazyFrame]()
    var eager = Optional[DataFrame]()
    for band in range(4):
        var tag = "b" + String(band)
        var part = (
            frame.lazy()
            .filter((col("k") % lit(Int64(4))) == lit(Int64(band)))
            .select_exprs(
                [
                    col("v").mean().alias(tag + "_mean"),
                    col("v").count().alias(tag + "_n"),
                    col("v").n_unique().alias(tag + "_distinct"),
                ]
            )
        )
        var alone = part.collect()
        eager = Optional(
            eager.value().join(alone, how="cross") if eager else alone^
        )
        row = Optional(row.value().join(part, how="cross") if row else part^)
    var plan = row.take()
    var top = len(plan._plan._nodes) - 1
    assert_true(plan._ends_in_breaker(plan._plan._nodes[top].right))
    assert_frame(plan.collect(), eager.take())


def test_filtered_build_sides_run_ahead_as_scheduled_pipelines() raises:
    """Two filtered build sides, neither a breaker, are scheduled as
    sibling join-build pipelines before the stream probes them; the
    stream takes both results and their indexes, and the sums are what
    joining the separately collected inputs gives. Unoptimized, so the
    join order pass runs no input ahead itself."""
    var frame = source()
    var evens = (
        frame.lazy()
        .filter((col("k") % lit(Int64(2))) == lit(Int64(0)))
        .select(["k"])
        .unique()
        .with_columns([lit(Int64(1)).alias("even")])
    )
    var small = (
        frame.lazy()
        .filter(col("k") < lit(Int64(400)))
        .select(["k"])
        .unique()
        .with_columns([lit(Int64(2)).alias("small")])
        .filter(col("small") > lit(Int64(0)))
    )
    var dims_a = evens.collect()
    var dims_b = small.collect()
    var a = dims_a.lazy().filter(col("k") > lit(Int64(10)))
    var b = dims_b.lazy().filter(col("k") > lit(Int64(20)))
    var probe = frame.lazy().filter(col("v") > lit(Int64(-400)))
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var streamed = (
        probe.join(a, ["k"])
        .join(b, ["k"])
        .group_by("s")
        .agg([col("v").sum().alias("v"), col("small").sum().alias("n")])
        .sort("s")
        .collect(optimize=False)
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "stream.scheduled_inputs" in paths, "the builds were not scheduled"
    )
    var eager = (
        probe.collect()
        .join(a.collect(), on=["k"])
        .join(b.collect(), on=["k"])
        .group_by(["s"])
        .agg([col("v").sum().alias("v"), col("small").sum().alias("n")])
        .sort(["s"])
    )
    assert_frame(streamed, eager)


def test_in_memory_build_sides_run_ahead_as_scheduled_pipelines() raises:
    """Build sides that are tables already in memory are scheduled too:
    each finishes into its hash index as a sibling join-build pipeline,
    since every join's build side was chosen before scheduling. The sums
    are what joining the tables eagerly gives."""
    var frame = source()
    var dims_a = (
        frame.lazy()
        .filter((col("k") % lit(Int64(2))) == lit(Int64(0)))
        .select(["k"])
        .unique()
        .with_columns([lit(Int64(1)).alias("even")])
        .collect()
    )
    var dims_b = (
        frame.lazy()
        .filter(col("k") < lit(Int64(400)))
        .select(["k"])
        .unique()
        .with_columns([lit(Int64(2)).alias("small")])
        .collect()
    )
    var probe = frame.lazy().filter(col("v") > lit(Int64(-400)))
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var streamed = (
        probe.join(dims_a.lazy(), ["k"])
        .join(dims_b.lazy(), ["k"])
        .group_by("s")
        .agg([col("v").sum().alias("v"), col("small").sum().alias("n")])
        .sort("s")
        .collect(optimize=False)
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "stream.scheduled_inputs" in paths, "the builds were not scheduled"
    )
    var eager = (
        probe.collect()
        .join(dims_a, on=["k"])
        .join(dims_b, on=["k"])
        .group_by(["s"])
        .agg([col("v").sum().alias("v"), col("small").sum().alias("n")])
        .sort(["s"])
    )
    assert_frame(streamed, eager)


def test_operator_build_sides_schedule_their_own_inputs() raises:
    """A build side that is an operator over whole inputs (here a sort of
    a grouped stream) is a pipeline of the probe's schedule, and so is
    the stream it reads: no schedule runs inside another's task. The sums
    are what joining the inputs eagerly gives."""
    var frame = source()
    var first = part(frame, 0)
    var second = part(frame, 1)
    var probe = frame.lazy().filter(col("v") > lit(Int64(-400)))
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var streamed = (
        probe.join(first, ["k"], how="left")
        .join(second, ["k"], how="left")
        .group_by("s")
        .agg(
            [
                col("total").sum().alias("t0"),
                col("total_right").sum().alias("t1"),
            ]
        )
        .sort("s")
        .collect(optimize=False)
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "stream.operator_inputs" in paths,
        "the operator's inputs were not scheduled",
    )
    var eager = (
        probe.collect()
        .join(first.collect(), on=["k"], how="left")
        .join(second.collect(), on=["k"], how="left")
        .group_by(["s"])
        .agg(
            [
                col("total").sum().alias("t0"),
                col("total_right").sum().alias("t1"),
            ]
        )
        .sort(["s"])
    )
    assert_frame(streamed, eager)


def test_aggregate_build_sides_finish_with_the_steps_above_them() raises:
    """A build side that filters and projects an aggregate's groups is an
    aggregate input of the schedule whose finish runs those steps: the
    stream reading it takes that result as its own, and no schedule runs
    inside another's task. The sums are what joining the inputs eagerly
    gives."""
    var frame = source()
    var groups = (
        frame.lazy()
        .group_by("k")
        .agg([col("v").sum().alias("total"), col("v").count().alias("n")])
        .filter(col("total") > lit(Int64(0)))
        .select(["k", "total"])
    )
    var probe = frame.lazy().filter(col("v") > lit(Int64(-400)))
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var streamed = (
        probe.join(groups, ["k"])
        .group_by("s")
        .agg([col("v").sum().alias("v"), col("total").sum().alias("t")])
        .sort("s")
        .collect(optimize=False)
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "pipeline.aggregate_finish_filter" in paths,
        "the aggregate's finish did not filter its groups",
    )
    assert_true(
        "schedule.nested" not in paths, "a schedule ran inside another's task"
    )
    var eager = (
        probe.collect()
        .join(groups.collect(), on=["k"])
        .group_by(["s"])
        .agg([col("v").sum().alias("v"), col("total").sum().alias("t")])
        .sort(["s"])
    )
    assert_frame(streamed, eager)


def test_a_declined_aggregate_input_streams_the_rows_below_it() raises:
    """A build side whose aggregate the stream declines once its inputs
    are known (an ordered grouping over a right join, whose rows come in
    its build side's order, which no pipeline keeps) still runs in the
    schedule:
    its pipeline makes the joined rows, and the aggregate runs over them
    as an operator, with no schedule inside another's task. The result
    is what the eager steps give."""
    var frame = source()
    var weights = List[Int64]()
    var keys = List[Int64]()
    for k in range(997):
        keys.append(Int64(k))
        weights.append(Int64(k % 13))
    var dim = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("w", Column[Int64](weights^)),
        ]
    ).lazy()
    var probe = frame.lazy().filter(col("v") > lit(Int64(-400)))
    var grouped = (
        probe.join(
            dim,
            left_on=["k"],
            right_on=["k"],
            how="right",
            suffix="_d",
            coalesce=False,
        )
        .group_by("s", maintain_order=True)
        .agg([col("v").median().alias("m"), col("w").sum().alias("t")])
    )
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var streamed = (
        frame.lazy()
        .select(["s", "k"])
        .join(grouped, ["s"])
        .group_by("s")
        .agg([col("m").max().alias("m"), col("t").max().alias("t")])
        .sort("s")
        .collect(optimize=False)
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "stream.declined_rows_below" in paths,
        "the declined aggregate's input rows were not streamed",
    )
    assert_true(
        "schedule.nested" not in paths, "a schedule ran inside another's task"
    )
    var eager_grouped = (
        frame.filter(col("v") > lit(Int64(-400)))
        .join(
            dim.collect(),
            left_on=["k"],
            right_on=["k"],
            how="right",
            suffix="_d",
            coalesce=False,
        )
        .group_by(["s"], maintain_order=True)
        .agg([col("v").median().alias("m"), col("w").sum().alias("t")])
    )
    var eager = (
        frame.select(["s", "k"])
        .join(eager_grouped, on=["s"])
        .group_by(["s"])
        .agg([col("m").max().alias("m"), col("t").max().alias("t")])
        .sort(["s"])
    )
    assert_frame(streamed, eager)


def test_ungrouped_distinct_counts_finish_as_rounds_of_the_schedule() raises:
    """An ungrouped stream's distinct counts finish as rounds of tasks of
    the query's schedule, every count's tables merged in its own rounds:
    no task starts parallel work of its own. The counts are what the
    eager reductions give."""
    var frame = source()
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var streamed = (
        frame.lazy()
        .filter(col("v") > lit(Int64(-400)))
        .select_exprs(
            [
                col("k").n_unique().alias("keys"),
                col("s").n_unique().alias("texts"),
                col("v").sum().alias("total"),
            ]
        )
        .collect(optimize=False)
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "parallel.jobs_in_task" not in paths,
        "a task started parallel work of its own",
    )
    var eager = frame.filter(col("v") > lit(Int64(-400))).select_exprs(
        [
            col("k").n_unique().alias("keys"),
            col("s").n_unique().alias("texts"),
            col("v").sum().alias("total"),
        ]
    )
    assert_frame(streamed, eager)


def test_a_profiled_query_runs_its_events_in_the_one_schedule() raises:
    """With an execution report, the query's schedule runs one event at a
    time, each event's tasks on every thread, rather than its probes'
    workers and finish rounds as parallel work inside one task. The rows
    are what collecting the query gives."""
    var frame = source()
    var first = part(frame, 0)
    var probe = frame.lazy().filter(col("v") > lit(Int64(-400)))
    var query = (
        probe.join(first, ["k"])
        .group_by("s")
        .agg([col("v").sum().alias("v"), col("total").sum().alias("t")])
        .sort("s")
    )
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var profiled = query.profile()
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "schedule.nested" not in paths, "a schedule ran inside another's task"
    )
    assert_true(profiled[1].height() > 0, "the report recorded no node")
    assert_frame(profiled[0], query.collect())


def windows() -> List[Expr]:
    """Windows that all partition by `k`, alone or with `s`: ranks, sums,
    running sums, shifts, a string and a boolean result, nulls in `v`."""
    return [
        col("v").rank("min").over("k").alias("rank"),
        col("v").sum().over("k").alias("total"),
        col("v").cum_sum().over(["k", "s"], order_by="v").alias("running"),
        col("v").shift(1).over("k").alias("before"),
        col("s").first().over("k").alias("first_s"),
        (col("v").mean().over("k") > lit(Float64(0.0))).alias("above"),
        (col("v") * lit(Int64(2))).alias("double"),
    ]


def test_windows_over_shared_keys_run_as_rounds_of_the_schedule() raises:
    """A projection whose windows share a partition key, computed as an
    input of the query's schedule (here the source a filter streams), runs
    as rounds of tasks of that schedule: rows are split into bins by that
    key, each bin evaluated by one task, and the results written back to
    the rows they came from. No task starts parallel work of its own, and
    every column equals the eager projection's, row for row."""
    var frame = source().head(20_000).with_row_index("row")
    var keep = col("row") >= lit(Int64(0))
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var added = (
        frame.lazy()
        .with_columns(windows())
        .filter(keep)
        .collect(optimize=False)
        .sort("row")
    )
    var exprs = windows()
    exprs.insert(0, col("row"))
    var selected = (
        frame.lazy()
        .select_exprs(exprs)
        .filter(keep)
        .collect(optimize=False)
        .sort("row")
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "window.partitioned_sink" in paths,
        "the windows did not run as a partitioned sink",
    )
    assert_true(
        "parallel.jobs_in_task" not in paths,
        "a task started parallel work of its own",
    )
    var eager_added = frame.with_columns(windows())
    var eager_selected = frame.select_exprs(exprs)
    assert_equal(added.columns(), eager_added.columns())
    assert_equal(selected.columns(), eager_selected.columns())
    for name in eager_added.columns():
        assert_true(
            added[name].equals(eager_added[name]), "column differs: " + name
        )
    for name in eager_selected.columns():
        assert_true(
            selected[name].equals(eager_selected[name]),
            "column differs: " + name,
        )


def test_windows_without_a_shared_key_keep_the_whole_input() raises:
    """Windows with no key in common (or over all rows) run over the whole
    input, as the eager projection does."""
    var frame = source().head(20_000)
    var exprs: List[Expr] = [
        col("v").sum().over("k").alias("by_k"),
        col("v").sum().over("s").alias("by_s"),
        col("v").cum_sum().over(order_by="k").alias("all"),
    ]
    var lazy = frame.lazy().with_columns(exprs).collect(optimize=False)
    var eager = frame.with_columns(exprs)
    for name in eager.columns():
        assert_true(lazy[name].equals(eager[name]), "column differs: " + name)


def test_a_filter_on_windows_runs_them_as_rounds_of_the_schedule() raises:
    """A filter whose predicate holds windows computes the predicate as a
    column of a projection below it, so the windows run as rounds of the
    query's schedule like any projection's, and the filter reads that
    column. The rows kept are the eager filter's, in the same order."""
    var frame = source().head(20_000).with_row_index("row")
    var predicate = (
        col("v").cast(DataType.FLOAT64)
        > lit(Float64(0.5)) * col("v").mean().over("k")
    ) | (col("v") == col("v").min().over(["k", "s"]))
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var lazy = frame.lazy().filter(predicate).collect().sort("row")
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "lazy.lower_whole_input_filter" in paths,
        "the filter's windows were not computed as a column",
    )
    assert_true(
        "window.partitioned_sink" in paths,
        "the windows did not run as a partitioned sink",
    )
    assert_true(
        "parallel.jobs_in_task" not in paths,
        "a task started parallel work of its own",
    )
    assert_frame(lazy, frame.filter(predicate))


def test_windows_without_a_shared_key_split_by_keys() raises:
    """A projection whose windows share no key runs as one projection per
    set of windows that do, each as rounds of the query's schedule, and
    the projection reads their results; windows over all rows run in one
    more projection. Every column equals the eager projection's."""
    var frame = source().head(20_000).with_row_index("row")
    var keep = col("row") >= lit(Int64(0))
    var keyed: List[Expr] = [
        col("v").sum().over("k").alias("by_k"),
        col("v").max().over(["s", "k"]).alias("by_s_k"),
        (col("v").sum().over("s") - col("v").min().over("s")).alias("by_s"),
        (col("v").rank("min").over("k") + col("row")).alias("rank_row"),
    ]
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var added = (
        frame.lazy().with_columns(keyed).filter(keep).collect().sort("row")
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "lazy.split_windows" in paths, "the windows were not split by keys"
    )
    assert_true(
        "window.partitioned_sink" in paths,
        "the windows did not run as a partitioned sink",
    )
    assert_true(
        "parallel.jobs_in_task" not in paths,
        "a task started parallel work of its own",
    )
    assert_frame(added, frame.with_columns(keyed))
    var mixed = keyed.copy()
    mixed.append(col("v").cum_sum().over(order_by="row").alias("all"))
    mixed.insert(0, col("row"))
    var selected = (
        frame.lazy().select_exprs(mixed).filter(keep).collect().sort("row")
    )
    assert_frame(selected, frame.select_exprs(mixed))


def test_reductions_beside_row_values_run_as_an_aggregate() raises:
    """A projection or filter whose expressions reduce every input row
    inside a value per row computes the reductions in an aggregate of the
    input, joined to every row, so no task evaluates them as parallel
    work of its own. Rows and values equal the eager ones, in order."""
    var frame = source().head(20_000).with_row_index("row")
    var keep = col("v") > lit(Int64(-480))
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var input = frame.lazy().filter(keep)
    var exprs: List[Expr] = [
        (col("v") - col("v").min()).alias("above_min"),
        (col("v").cast(DataType.FLOAT64) / col("v").mean()).alias("share"),
        (col("row") - col("v").count()).alias("n"),
    ]
    var added = input.with_columns(exprs).collect()
    var highest = input.filter(col("v") == col("v").max()).collect()
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "lazy.lower_reductions" in paths,
        "the reductions were not computed in an aggregate",
    )
    assert_true(
        "lazy.lower_whole_input_filter" in paths,
        "the filter's reduction was not computed as a column",
    )
    assert_true(
        "parallel.jobs_in_task" not in paths,
        "a task started parallel work of its own",
    )
    var kept = frame.filter(keep)
    assert_frame(added, kept.with_columns(exprs))
    assert_frame(highest, kept.filter(col("v") == col("v").max()))


def test_a_sort_runs_as_rounds_of_the_schedule() raises:
    """A sort of every row, computed as an input of the query's schedule
    (here the source a filter streams), runs as rounds of tasks of that
    schedule: runs of sorted keys, their merge boundaries, the merged
    partitions, then the rows gathered in that order. No task starts
    parallel work of its own, and the rows equal the eager sort's."""
    var frame = source().with_row_index("row")
    var keep = col("row") >= lit(Int64(0))
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var lazy = (
        frame.lazy()
        .sort(
            ["s", "v"],
            descending=[False, True],
            nulls_last=[True, False],
            maintain_order=True,
        )
        .filter(keep)
        .collect(optimize=False)
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true("sort.rounds" in paths, "the sort did not run as rounds")
    assert_true(
        "parallel.jobs_in_task" not in paths,
        "a task started parallel work of its own",
    )
    var eager = frame.sort(
        ["s", "v"],
        descending=[False, True],
        nulls_last=[True, False],
        maintain_order=True,
    )
    assert_equal(lazy.columns(), eager.columns())
    for name in eager.columns():
        assert_true(lazy[name].equals(eager[name]), "column differs: " + name)


def test_ranks_over_every_row_run_as_rounds_of_the_schedule() raises:
    """A projection of ranks over all rows, computed as an input of the
    query's schedule, runs as rounds of tasks of that schedule: the sort,
    peer boundaries, range summaries and ranks written. No task starts
    parallel work of its own, and the ranks equal the eager ones for
    every method, both directions, with nulls."""
    var frame = source().with_row_index("row")
    var keep = col("row") >= lit(Int64(0))
    var exprs: List[Expr] = [
        col("row"),
        col("v").rank("min").alias("min"),
        col("v").rank("dense", descending=True).alias("dense"),
        col("v").rank("ordinal").alias("ordinal"),
        col("k").rank("average").alias("average"),
        col("s").rank("max", descending=True).alias("max"),
    ]
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var lazy = (
        frame.lazy()
        .select_exprs(exprs)
        .filter(keep)
        .collect(optimize=False)
        .sort("row")
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "window.rank_rounds" in paths, "the ranks did not run as rounds"
    )
    assert_true(
        "parallel.jobs_in_task" not in paths,
        "a task started parallel work of its own",
    )
    var eager = frame.select_exprs(exprs)
    assert_equal(lazy.columns(), eager.columns())
    for name in eager.columns():
        assert_true(lazy[name].equals(eager[name]), "column differs: " + name)


def test_top_rows_run_as_rounds_of_the_schedule() raises:
    """A sort limited to its first rows, computed as an input of the
    query's schedule, runs as rounds of tasks of that schedule: a heap
    per range of rows, combined, then the rows gathered. No task starts
    parallel work of its own, and the rows equal the eager ones."""
    var frame = source().with_row_index("row")
    var keep = col("row") >= lit(Int64(0))
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var lazy = (
        frame.lazy()
        .sort(
            ["s", "v"],
            descending=[False, True],
            nulls_last=[True, False],
            maintain_order=True,
        )
        .head(1000)
        .filter(keep)
        .collect()
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true("sort.rounds" in paths, "the top rows did not run as rounds")
    assert_true(
        "parallel.jobs_in_task" not in paths,
        "a task started parallel work of its own",
    )
    var eager = frame.sort(
        ["s", "v"],
        descending=[False, True],
        nulls_last=[True, False],
        maintain_order=True,
    ).head(1000)
    assert_equal(lazy.columns(), eager.columns())
    for name in eager.columns():
        assert_true(lazy[name].equals(eager[name]), "column differs: " + name)
