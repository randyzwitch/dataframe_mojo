"""Independent inputs run at once, each from its own copy of the plan: a
union's inputs (stacked in input order, nested unions included), a join's
two computed inputs, and a stream's aggregate builds and source, which run
ahead of it. Results match the inputs computed one by one, and an error in
any input reaches the caller."""
# contract-group: planner.relational
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, DataType, LazyFrame, Series, col, lit
from dataframe.frame import concat
from tests.support.coverage_compare import assert_frame

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
    var top = len(plan._nodes) - 1
    assert_true(plan._ends_in_breaker(plan._nodes[top].right))
    assert_frame(plan.collect(), eager.take())
