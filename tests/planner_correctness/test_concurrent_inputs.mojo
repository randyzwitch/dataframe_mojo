"""Independent inputs run at once, each from its own copy of the plan: a
union's inputs (stacked in input order, nested unions included), a join's
two computed inputs, and a stream's build sides (filtered or aggregated)
and breaker source, which run ahead of it as scheduled pipelines. Results match the inputs computed one by one, and an error in
any input reaches the caller."""
# contract-group: planner.relational
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, DataType, LazyFrame, Series, col, lit
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
    once) and a stream of left joins over three aggregates (TPC-DS q78's
    shape: each build and the source end in an aggregate, so all three
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
    """One-row statistics of filtered bands, cross joined side by side
    (TPC-DS q28's shape): each band's selection reads its whole input, so
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
