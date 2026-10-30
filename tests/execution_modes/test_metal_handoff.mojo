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
from tests.support.group_strategy_reference import ScopedEnvironment
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


def _calibrate(path: String, cpu_row_ns: Float64, accel_row_ns: Float64) raises:
    """A calibration for the filtered Int32 sum alone; other keys are left
    unmeasured, so they stay on the CPU."""
    var text = String(
        '{"operators": {"filter_sum:int32:filtered": {"cpu": {"fixed_ns":'
        ' 1000, "row_ns": '
    )
    text += String(cpu_row_ns) + '}, "accel": {"fixed_ns": 2000000,'
    text += ' "row_ns": ' + String(accel_row_ns) + "}}}}"
    with open(path, "w") as target:
        target.write(text)


# requires: metal
def test_metal_automatic_placement_follows_measured_costs() raises:
    _ = require_metal()
    var frame = _columns(100_000, False)
    var plan = frame.lazy().filter(col("i") > 0).select(col("i").sum())
    var expected = plan.collect(engine="cpu")
    # Without a calibration, engine="auto" places no node by hand-off
    # ("PLACE " lines; per-pipeline placement prints "PLACEMENT").
    var unset = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", "")
    assert_false("\nPLACE " in plan.explain(engine="auto"))
    _ = unset^
    for accel_cheaper in [True, False]:
        # Metal at 0.1 ns/row against the CPU's 50 is estimated faster from
        # about 40,000 rows; at 100 ns/row it never is. A process reads a
        # calibration once per path, so each costing has its own file.
        var path = String("build/test-handoff-calibration-") + (
            "cheap.json" if accel_cheaper else "dear.json"
        )
        _calibrate(path, 50.0, 0.1 if accel_cheaper else 100.0)
        var calibrated = ScopedEnvironment(
            "DATAFRAME_HANDOFF_CALIBRATION", path
        )
        var shown = plan.explain(engine="auto")
        assert_true("PLACE" in shown)
        assert_true(("accel from" in shown) == accel_cheaper, shown)
        var profiled = plan.profile(engine="auto")
        assert_true(profiled[0].equals(expected))
        var report = profiled[1].copy()
        var placed = String()
        for row in range(report.height()):
            var algorithm = report.item(row, "algorithm").string()
            if algorithm.startswith("handoff") or algorithm.startswith(
                "placed cpu"
            ):
                placed = algorithm
        assert_true(
            placed.startswith(
                "handoff (accel"
            ) if accel_cheaper else placed.startswith("placed cpu (accel"),
            placed,
        )
        assert_true(plan.collect(engine="auto").equals(expected))
        # An operator the calibration does not cover runs on the CPU.
        var plain = frame.lazy().select(col("i").sum())
        assert_true("no measured cost" in plain.explain(engine="auto"))
        _ = calibrated^
