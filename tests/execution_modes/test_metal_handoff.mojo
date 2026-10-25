"""Regions the CPU executor hands to Metal: which shapes Metal accepts, and
results equal to the CPU's for the ones it runs."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    LazyFrame,
    Series,
    col,
    concat,
    lit,
)
from dataframe_metal.handoff import plan_region
from dataframe_metal.metal import MetalRuntime
from tests.support.metal_outcomes import require_metal


def _frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("i", Column[Int32]([1, 5, 9])),
            Series("l", Column[Int64]([1, 5, 9])),
            Series("f", Column[Float32]([1.5, 5, 9])),
            Series("d", Column[Float64]([1.5, 5, 9])),
            Series("s", Column[String](["a", "b", "c"])),
        ]
    )


def test_metal_accepts_filtered_sums_of_int32_int64_and_float32() raises:
    var frame = _frame()
    for name in ["i", "l", "f"]:
        var plain = plan_region(frame.lazy().select(col(name).sum()))
        assert_true(Bool(plain))
        assert_false(plain.value().filtered)
        var filtered = plan_region(
            frame.lazy()
            .filter(col(name) > 2)
            .filter((col(name) <= 9) & (col(name) >= 1))
            .select(col(name).sum().alias("total"))
        )
        assert_true(Bool(filtered))
        assert_equal(filtered.value().output, "total")
        assert_true(filtered.value().filtered)
    var ranged = (
        plan_region(
            frame.lazy()
            .filter((col("i") > 2) & (col("i") < 9) & (col("i") >= 3))
            .select(col("i").sum())
        )
        .value()
        .copy()
    )
    assert_equal(ranged.low, 3)
    assert_equal(ranged.high, 8)
    var floats = (
        plan_region(frame.lazy().filter(col("f") > 2.5).select(col("f").sum()))
        .value()
        .copy()
    )
    assert_equal(floats.low_float, Float32(2.5))
    assert_equal(floats.inclusive & 1, UInt32(0))
    assert_equal(floats.inclusive & 2, UInt32(2))


def test_metal_declines_float64_strings_and_other_shapes() raises:
    var frame = _frame()
    var declined: List[LazyFrame] = [
        frame.lazy().select(col("d").sum()),
        frame.lazy().select(col("i").mean()),
        frame.lazy().select_exprs([col("i").sum(), col("l").sum()]),
        frame.lazy().filter(col("l") > 2).select(col("i").sum()),
        frame.lazy().filter(col("i") != 2).select(col("i").sum()),
        frame.lazy().filter(col("i") > lit(Int64(2))).select(col("i").sum()),
        frame.lazy().filter(col("i") > 2.5).select(col("i").sum()),
        frame.lazy().filter(col("f") > 0.1).select(col("f").sum()),
        frame.lazy().filter(col("s") == "a").select(col("i").sum()),
        frame.lazy().select((col("i") + 1).sum()),
        frame.lazy().group_by("s").agg([col("i").sum()]),
    ]
    for plan in declined:
        assert_false(Bool(plan_region(plan)))


def _close(actual: DataFrame, expected: DataFrame) raises:
    assert_equal(actual.columns(), expected.columns())
    var a = actual.item(0, actual.columns()[0])
    var e = expected.item(0, expected.columns()[0])
    assert_equal(a.is_null(), e.is_null())
    if e.is_null():
        return
    assert_true(a.dtype() == e.dtype())
    if e.dtype() == DataType.FLOAT32:
        var x = Float64(a.float32())
        var y = Float64(e.float32())
        assert_true(abs(x - y) <= 1e-6 + 1e-5 * abs(y), String(x, " ", y))
    else:
        assert_true(actual.equals(expected))


def _columns(rows: Int, nulls: Bool) raises -> DataFrame:
    var ints = List[Int32](capacity=rows)
    var longs = List[Int64](capacity=rows)
    var floats = List[Float32](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for row in range(rows):
        var value = (row * 7919) % 2001 - 1000
        ints.append(Int32(value))
        longs.append(Int64(value) * 4_000_000_000)
        floats.append(Float32(value) / 8)
        valid.append(not nulls or row % 11 != 3)
    return DataFrame(
        [
            Series("i", Column[Int32](ints^, valid.copy())),
            Series("l", Column[Int64](longs^, valid.copy())),
            Series("f", Column[Float32](floats^, valid^)),
        ]
    )


# requires: metal
def test_metal_handoff_sums_equal_cpu_results() raises:
    var runtime = require_metal()
    for rows in [0, 1, 255, 257, 100_003, 2_000_003]:
        for nulls in [False, True]:
            var frame = _columns(rows, nulls)
            for name in ["i", "l", "f"]:
                var plans: List[LazyFrame] = [
                    frame.lazy().select(col(name).sum()),
                    frame.lazy()
                    .filter((col(name) > -100) & (col(name) <= 700))
                    .select(col(name).sum().alias("kept")),
                    frame.lazy()
                    .filter(col(name) > 1_000_000_000)
                    .select(col(name).sum(min_count=1)),
                ]
                for plan in plans:
                    _close(
                        plan.collect(accelerator=runtime),
                        plan.collect(engine="cpu"),
                    )


# requires: metal
def test_metal_handoff_reads_sliced_and_chunked_columns() raises:
    # A slice starts inside a page, so the in-place view keeps an offset;
    # a chunked column is copied instead of wrapped.
    var runtime = require_metal()
    var whole = _columns(300_001, True)
    var parts: List[DataFrame] = [
        whole.slice(5, 200_000),
        whole.slice(1, 7),
        concat([whole.slice(0, 1000), whole.slice(5000, 100_000)]),
    ]
    for frame in parts:
        for name in ["i", "l", "f"]:
            var plan = (
                frame.lazy().filter(col(name) >= -500).select(col(name).sum())
            )
            _close(
                plan.collect(accelerator=runtime), plan.collect(engine="cpu")
            )


# requires: metal
def test_metal_handoff_overflow_raises_the_cpu_error() raises:
    var runtime = require_metal()
    var frame = DataFrame(
        [Series("i", Column[Int32](List[Int32](length=3, fill=Int32.MAX)))]
    )
    var plan = frame.lazy().select(col("i").sum())
    with assert_raises(contains="int32 expression sum overflow"):
        _ = plan.collect(engine="cpu")
    with assert_raises(contains="int32 expression sum overflow"):
        _ = plan.collect(accelerator=runtime)
    var extremes = DataFrame(
        [Series("l", Column[Int64]([Int64.MAX, Int64.MIN, Int64.MAX, -1]))]
    )
    var exact = extremes.lazy().select(col("l").sum())
    assert_true(
        exact.collect(accelerator=runtime).equals(exact.collect(engine="cpu"))
    )


# requires: metal
def test_metal_handoff_float32_nan_and_explain_and_profile() raises:
    var runtime = require_metal()
    var nan = Float32(0) / Float32(0)
    var frame = DataFrame(
        [
            Series(
                "f", Column[Float32]([1, nan, 3, 4], [True, True, True, False])
            )
        ]
    )
    for plan in [
        frame.lazy().select(col("f").sum()),
        frame.lazy().filter(col("f") > 1).select(col("f").sum()),
        frame.lazy().filter(col("f") >= 1).select(col("f").sum()),
    ]:
        var expected = plan.collect(engine="cpu")
        var actual = plan.collect(accelerator=runtime)
        var e = expected.item(0, "f").float32()
        var a = actual.item(0, "f").float32()
        assert_equal(e != e, a != a)
        if e == e:
            assert_equal(a, e)
    var summed = (
        _columns(1000, False).lazy().filter(col("i") > 0).select(col("i").sum())
    )
    assert_true("HANDOFF metal" in summed.explain(accelerator=runtime))
    var profiled = summed.profile(accelerator=runtime)
    var report = profiled[1].copy()
    var handed = False
    for row in range(report.height()):
        if report.item(row, "algorithm").string() == "handoff":
            handed = True
    assert_true(handed)
    var declined = _columns(10, False).lazy().select(col("i").mean())
    assert_true("supports no node" in declined.explain(accelerator=runtime))
    assert_true(
        declined.collect(accelerator=runtime).equals(
            declined.collect(engine="cpu")
        )
    )
