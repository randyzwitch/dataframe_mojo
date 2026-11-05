"""Regions the CPU executor hands to Metal: which shapes Metal accepts, and
results equal to the CPU's for the ones it runs."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.math import isinf
from std.memory import bitcast
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    concat,
    lit,
)
from dataframe_metal.handoff import execute_timed, operator_key, plan_region
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.multi_plan import plan_bundle, bundle_key
from dataframe_metal.arithmetic_plan import plan_arithmetic
from dataframe_metal.accelerator.gpu.arithmetic_sum import integer_binary
from dataframe.expr import ADD, SUB, MUL
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
        frame.lazy().filter(col("i") != 2).select(col("i").sum()),
        frame.lazy().filter(col("i") > lit(Int64(2))).select(col("i").sum()),
        frame.lazy().filter(col("i") > 2.5).select(col("i").sum()),
        frame.lazy().filter(col("f") > 0.1).select(col("f").sum()),
        frame.lazy().filter(col("s") == "a").select(col("i").sum()),
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


def _cross_columns(rows: Int, nulls: Bool) raises -> DataFrame:
    var ints = List[Int32]()
    var longs = List[Int64]()
    var floats = List[Float32]()
    var iv = List[Bool]()
    var lv = List[Bool]()
    var fv = List[Bool]()
    for row in range(rows):
        ints.append(Int32((row * 7) % 19 - 9))
        longs.append(Int64((row * 13) % 23 - 11) * 4_000_000_000)
        floats.append(Float32((row * 17) % 29 - 14) / 8)
        iv.append(not nulls or row % 11 != 3)
        lv.append(not nulls or row % 13 != 5)
        fv.append(not nulls or row % 17 != 7)
    return DataFrame(
        [
            Series("i", Column[Int32](ints^, iv)),
            Series("l", Column[Int64](longs^, lv)),
            Series("f", Column[Float32](floats^, fv)),
        ]
    )


def test_metal_accepts_cross_column_ranges_and_separates_cost_keys() raises:
    var frame = _frame()
    for value in ["i", "l", "f"]:
        for predicate in ["i", "l", "f"]:
            var plan = (
                frame.lazy().filter(col(predicate) > 2).select(col(value).sum())
            )
            var accepted = plan_region(plan)
            assert_true(Bool(accepted))
            assert_equal(accepted.value().column, value)
            assert_equal(accepted.value().filter_column, predicate)
            assert_equal(
                accepted.value().filter_dtype, frame.column(predicate).dtype()
            )
            assert_true(
                operator_key(accepted.value()).startswith(
                    "filter_sum:" if value == predicate else "cross_filter_sum:"
                )
            )
    assert_false(
        Bool(
            plan_region(
                frame.lazy()
                .filter((col("i") > 0) & (col("l") > 0))
                .select(col("f").sum())
            )
        )
    )


# requires: metal
def test_metal_cross_column_types_nulls_and_unrelated_chunk_boundaries() raises:
    var runtime = require_metal()
    for nulls in [False, True]:
        var whole = _cross_columns(13_001, nulls)
        var chunked = List[Series]()
        for name in ["i", "l", "f"]:
            var split = 19 if name == "i" else (2503 if name == "l" else 101)
            chunked.append(
                Series._from_chunks(
                    [
                        whole.column(name).slice(5, split),
                        whole.column(name).slice(5 + split, 7),
                        whole.column(name).slice(12 + split, 10003 - split - 7),
                    ]
                )
            )
        for frame in [whole.slice(5, 10003), DataFrame(chunked^)]:
            for predicate in ["i", "l", "f"]:
                for value in ["i", "l", "f"]:
                    for threshold in [0, 1_000_000_000]:
                        var plan = (
                            frame.lazy()
                            .filter(col(predicate) > threshold)
                            .select(col(value).sum(min_count=1).alias("total"))
                        )
                        assert_true(
                            "HANDOFF metal" in plan.explain(accelerator=runtime)
                        )
                        var expected = plan.collect(engine="cpu")
                        _close(plan.collect(accelerator=runtime), expected)
                        _close(execute_timed(plan)[0], expected)
                        _close(
                            execute_timed(plan, zero_copy=False)[0], expected
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
def test_metal_chunk_batches_preserve_independent_owners_and_bitmap_offsets() raises:
    var runtime = require_metal()
    var parts = List[DataFrame]()
    # Separate allocations, mixed absent/present bitmaps and every bit offset.
    # No source chunk may be released before the batched device work completes.
    for i in range(64):
        var source = _columns(1200 + i, i % 3 != 0)
        parts.append(source.slice(i % 8 + 1, 1003 + i))
    var frame = concat(parts)
    assert_equal(frame.column("i").n_chunks(), 64)
    for name in ["i", "l", "f"]:
        var empty_bound = (
            9_000_000_000_000_000 if name == "l" else 1_000_000_000
        )
        var plans: List[LazyFrame] = [
            frame.lazy().select(col(name).sum()),
            frame.lazy()
            .filter((col(name) > -100) & (col(name) <= 700))
            .select(col(name).sum(min_count=1)),
            frame.lazy()
            .filter(col(name) > empty_bound)
            .select(col(name).sum(min_count=1)),
        ]
        for plan in plans:
            assert_true("HANDOFF metal" in plan.explain(accelerator=runtime))
            var expected = plan.collect(engine="cpu")
            _close(plan.collect(accelerator=runtime), expected)
            _close(execute_timed(plan)[0], expected)
            _close(execute_timed(plan, zero_copy=False)[0], expected)


def test_metal_bundle_shares_states_and_accepts_mixed_native_types() raises:
    var frame = _frame()
    var plan = frame.lazy().select_exprs(
        [
            col("i").sum().alias("sum_i"),
            col("i").count().alias("count_i"),
            col("l").mean().alias("mean_l"),
            col("f").mean().alias("mean_f"),
        ]
    )
    var bundle = plan_bundle(plan)
    assert_true(Bool(bundle))
    assert_equal(len(bundle.value().fields), 3)
    assert_equal(len(bundle.value().outputs), 4)
    assert_true(bundle.value().precise_float)
    assert_equal(bundle.value().fields[0].column, "f")
    assert_equal(bundle.value().fields[1].column, "i")
    assert_equal(bundle.value().fields[2].column, "l")
    assert_equal(bundle.value().outputs[0].field, 1)
    assert_equal(bundle.value().outputs[1].field, 1)
    assert_equal(bundle.value().outputs[2].field, 2)
    assert_equal(bundle.value().outputs[3].field, 0)
    assert_true(bundle_key(bundle.value()).endswith(":exact_float:typed_v1"))
    var filtered = plan_bundle(
        frame.lazy()
        .filter(col("i") > 0)
        .select_exprs([col("i").sum().alias("s"), col("i").count().alias("c")])
    )
    var plain = plan_bundle(
        frame.lazy().select_exprs(
            [col("i").sum().alias("s"), col("i").count().alias("c")]
        )
    )
    var external = plan_bundle(
        frame.lazy()
        .filter(col("l") > 0)
        .select_exprs([col("i").sum().alias("s"), col("i").count().alias("c")])
    )
    assert_false(bundle_key(plain.value()) == bundle_key(filtered.value()))
    assert_true(":filtered:external:" in bundle_key(external.value()))
    assert_true(":filtered:shared0:" in bundle_key(filtered.value()))


def _close_bundle(actual: DataFrame, expected: DataFrame) raises:
    assert_equal(actual.columns(), expected.columns())
    assert_equal(actual.height(), 1)
    for name in expected.columns():
        assert_equal(actual.column(name).dtype(), expected.column(name).dtype())
        var a = actual.item(0, name)
        var e = expected.item(0, name)
        assert_equal(a.is_null(), e.is_null())
        if e.is_null():
            continue
        if e.dtype() == DataType.FLOAT64 or e.dtype() == DataType.FLOAT32:
            var x = a.float64() if e.dtype() == DataType.FLOAT64 else Float64(
                a.float32()
            )
            var y = e.float64() if e.dtype() == DataType.FLOAT64 else Float64(
                e.float32()
            )
            if x != x or y != y:
                assert_true(x != x and y != y)
            elif isinf(x) or isinf(y):
                assert_equal(x, y)
            elif x != y:
                var tolerance = 1e-12 if e.dtype() == DataType.FLOAT64 else 1e-5
                assert_true(abs(x - y) <= tolerance + tolerance * abs(y))
        else:
            assert_true(actual.column(name).equals(expected.column(name)))


# requires: metal
def test_metal_fused_sum_count_mean_share_predicate_with_mixed_nulls() raises:
    var runtime = require_metal()
    for nulls in [False, True]:
        var base = _cross_columns(13001, nulls)
        var chunked = List[Series]()
        for name in ["i", "l", "f"]:
            var split = 19 if name == "i" else (2503 if name == "l" else 101)
            chunked.append(
                Series._from_chunks(
                    [
                        base.column(name).slice(5, split),
                        base.column(name).slice(5 + split, 10003 - split),
                    ]
                )
            )
        for frame in [base.slice(5, 10003), DataFrame(chunked^)]:
            for predicate in ["", "i", "l", "f"]:
                var query = frame.lazy()
                if predicate:
                    query = query.filter(col(predicate) > 0)
                var plan = query.select_exprs(
                    [
                        col("i").sum().alias("sum_i"),
                        col("i").count().alias("count_i"),
                        col("i").mean().alias("mean_i"),
                        col("l").sum().alias("sum_l"),
                        col("l").count().alias("count_l"),
                        col("l").mean().alias("mean_l"),
                        col("f").sum().alias("sum_f"),
                        col("f").count().alias("count_f"),
                        col("f").mean().alias("mean_f"),
                    ]
                )
                assert_true(
                    "HANDOFF metal" in plan.explain(accelerator=runtime)
                )
                var expected = plan.collect(engine="cpu")
                _close_bundle(plan.collect(accelerator=runtime), expected)
                _close_bundle(execute_timed(plan)[0], expected)
                _close_bundle(execute_timed(plan, zero_copy=False)[0], expected)


# requires: metal
def test_metal_bundle_empty_min_count_and_integer_mean_without_sum_overflow() raises:
    var runtime = require_metal()
    var frame = DataFrame([Series("i", Column[Int32]([Int32.MAX, Int32.MAX]))])
    var mean = frame.lazy().select_exprs(
        [col("i").mean().alias("m"), col("i").count().alias("c")]
    )
    _close_bundle(execute_timed(mean)[0], mean.collect(engine="cpu"))
    for empty in [frame.clear(), frame.copy()]:
        var plan = (
            empty.lazy()
            .filter(col("i") < 0)
            .select_exprs(
                [
                    col("i").sum().alias("zero"),
                    col("i").sum(min_count=1).alias("null"),
                    col("i").mean().alias("mean"),
                    col("i").count().alias("count"),
                ]
            )
        )
        _close_bundle(execute_timed(plan)[0], plan.collect(engine="cpu"))
    var overflow = frame.lazy().select_exprs(
        [col("i").sum().alias("s"), col("i").count().alias("c")]
    )
    with assert_raises(contains="int32 expression sum overflow"):
        _ = overflow.collect(accelerator=runtime)


# requires: metal
def test_metal_float32_mean_preserves_range_specials_and_cpu_order_adjudication() raises:
    var runtime = require_metal()
    var maximum = Float32.MAX_FINITE
    var nan = Float32(0) / Float32(0)
    for values in [
        [maximum, maximum],
        [nan, Float32(1)],
        [Float32(1) / Float32(0), Float32(1)],
    ]:
        var frame = DataFrame(
            [Series("f", Column[Float32](List[Float32](values)))]
        )
        var plan = frame.lazy().select_exprs(
            [col("f").mean().alias("m"), col("f").count().alias("c")]
        )
        _close_bundle(execute_timed(plan)[0], plan.collect(engine="cpu"))
        var sums = frame.lazy().select_exprs(
            [col("f").sum().alias("s"), col("f").count().alias("c")]
        )
        _close_bundle(
            sums.collect(accelerator=runtime), sums.collect(engine="cpu")
        )
        with assert_raises(contains="CPU non-finite accumulation"):
            _ = execute_timed(sums)
    var wide = DataFrame([Series("f", Column[Float32]([maximum, 1, -maximum]))])
    var plan = wide.lazy().select_exprs(
        [col("f").mean().alias("m"), col("f").count().alias("c")]
    )
    _close_bundle(plan.collect(accelerator=runtime), plan.collect(engine="cpu"))
    with assert_raises(contains="CPU accumulation order"):
        _ = execute_timed(plan)
    # Subnormals retry the full exponent window without changing the result.
    var tiny = bitcast[DType.float32](UInt32(1))
    var small = DataFrame([Series("f", Column[Float32]([tiny, tiny * 2]))])
    var small_mean = small.lazy().select_exprs(
        [col("f").mean().alias("m"), col("f").count().alias("c")]
    )
    assert_equal(
        execute_timed(small_mean)[0].item(0, "m").float64(),
        Float64(tiny) * 1.5,
    )


# requires: metal
def test_metal_bundle_count_only_and_all_supported_field_arities() raises:
    var runtime = require_metal()
    var frame = _frame().with_column(Series("j", Column[Int32]([7, 11, 13])))
    var plans: List[LazyFrame] = [
        frame.lazy().select(col("f").count()),
        frame.lazy().select_exprs(
            [col("i").sum().alias("i"), col("f").sum().alias("f")]
        ),
        frame.lazy().select_exprs(
            [
                col("i").sum().alias("i"),
                col("l").mean().alias("l"),
                col("f").mean().alias("f"),
                col("j").count().alias("j"),
            ]
        ),
    ]
    for plan in plans:
        _close_bundle(execute_timed(plan)[0], plan.collect(engine="cpu"))
        _close_bundle(
            plan.collect(accelerator=runtime), plan.collect(engine="cpu")
        )
    # Every compact float-prefix specialization, with integer-first output
    # ordering and duplicate SUM/COUNT/MEAN states, must preserve CPU results.
    var typed = frame.with_column(
        Series("g", Column[Float32]([3, -2, 1], [True, False, True]))
    )
    typed = typed.with_column(Series("h", Column[Float32]([-4, 2, 8])))
    typed = typed.with_column(
        Series("k", Column[Float32]([1, 0, -1], [False, True, True]))
    )
    var integers = [String("i"), String("l"), String("j")]
    var floats = [String("f"), String("g"), String("h"), String("k")]
    for arity in range(1, 5):
        for float_count in range(1, arity + 1):
            var expressions = List[Expr]()
            var names = List[String]()
            for i in range(arity - float_count):
                names.append(integers[i].copy())
            for i in range(float_count):
                names.append(floats[i].copy())
            for name in names:
                expressions.append(col(name).mean().alias("m" + name))
                expressions.append(col(name).count().alias("c" + name))
                expressions.append(col(name).sum(min_count=3).alias("s" + name))
            var plan = typed.lazy().select_exprs(expressions)
            var expected = plan.collect(engine="cpu")
            _close_bundle(execute_timed(plan)[0], expected)
            _close_bundle(execute_timed(plan, zero_copy=False)[0], expected)
            var wide = typed.copy()
            for i in range(float_count):
                wide = wide.with_column(
                    Series(
                        floats[i],
                        Column[Float32](
                            List[Float32](length=3, fill=Float32.MAX_FINITE)
                        ),
                    )
                )
            var wide_plan = wide.lazy().select_exprs(expressions)
            _close_bundle(
                execute_timed(wide_plan)[0], wide_plan.collect(engine="cpu")
            )
    # The exact Int64 total requires cancellation across distinct chunks.
    var extremes = concat(
        [
            DataFrame([Series("l", Column[Int64]([Int64.MIN, 0]))]),
            DataFrame([Series("l", Column[Int64]([Int64.MAX, 0]))]),
        ]
    )
    var plan = extremes.lazy().select(col("l").sum())
    _close(plan.collect(accelerator=runtime), plan.collect(engine="cpu"))
    _close(execute_timed(plan)[0], plan.collect(engine="cpu"))


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
    var declined = _columns(10, False).lazy().select(col("i").median())
    assert_true("supports no node" in declined.explain(accelerator=runtime))
    assert_true(
        declined.collect(accelerator=runtime).equals(
            declined.collect(engine="cpu")
        )
    )


def _calibrate(
    path: String,
    cpu_row_ns: Float64,
    accel_row_ns: Float64,
    plan: LazyFrame,
    min_rows: Int = 1,
    max_rows: Int = 1_000_000_000,
) raises:
    from dataframe.json.value import json_quote
    from dataframe_metal.calibration import (
        identity_json,
        shape_key,
        pipeline_key,
    )
    from dataframe_metal.handoff import region_key

    var key = shape_key(plan, region_key(plan))
    var pipeline = pipeline_key(plan)
    var cpu = String('{"fixed_ns":1000,"row_ns":') + String(cpu_row_ns) + "}"
    var text = String('{"schema_version":2,"identity":') + identity_json()
    text += (
        ',"operators":{'
        + json_quote(pipeline)
        + ':{"cpu":'
        + cpu
        + '}},"regions":{'
        + json_quote(key)
        + ':{"pipeline_key":'
        + json_quote(pipeline)
        + ',"cpu":'
        + cpu
        + ',"min_rows":'
        + String(min_rows)
        + ',"max_rows":'
        + String(max_rows)
        + ',"accel":{"fixed_ns":2000000,"row_ns":'
        + String(accel_row_ns)
        + "}}}}"
    )
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
    from dataframe_metal.accelerator.gpu.identity import runtime_build_id

    if not runtime_build_id().startswith("dataframe-mojo-metal-v2;cpu="):
        var path = String("build/test-handoff-development-calibration.json")
        _calibrate(path, 50.0, 0.1, plan)
        var rejected = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", path)
        assert_true("no measured cost" in plan.explain(engine="auto"))
        assert_false("accel from" in plan.explain(engine="auto"))
        assert_true(plan.collect(engine="auto").equals(expected))
        _ = rejected^
        return
    for accel_cheaper in [True, False]:
        # Metal at 0.1 ns/row against the CPU's 50 is estimated faster from
        # about 40,000 rows; at 100 ns/row it never is. A process reads a
        # calibration once per path, so each costing has its own file.
        var path = String("build/test-handoff-calibration-") + (
            "cheap.json" if accel_cheaper else "dear.json"
        )
        _calibrate(path, 50.0, 0.1 if accel_cheaper else 100.0, plan)
        var calibrated = ScopedEnvironment(
            "DATAFRAME_HANDOFF_CALIBRATION", path
        )
        var shown = plan.explain(engine="auto")
        assert_true("PIPELINE " in shown)
        assert_true(("; on metal:0" in shown) == accel_cheaper, shown)
        var trace_path = String("build/test-handoff-pipeline-trace.jsonl")
        with open(trace_path, "w") as target:
            target.write("")
        var trace = ScopedEnvironment(
            "DATAFRAME_METAL_PLACEMENT_TRACE", trace_path
        )
        assert_true(plan.collect(engine="auto").equals(expected))
        var traced = String()
        with open(trace_path, "r") as source:
            traced = source.read()
        assert_true(('"engine":"metal"' in traced) == accel_cheaper, traced)
        var plain = frame.lazy().select(col("i").sum())
        assert_false("; on metal:0" in plain.explain(engine="auto"))
        _ = trace^
        _ = calibrated^


def test_metal_arithmetic_matcher_accepts_native_pairs_and_declines_wider_work() raises:
    var frame = _frame().with_column(Series("j", Column[Int32]([3, 4, 5])))
    for expression in [
        col("i") + col("j"),
        col("i") - 2,
        2 - col("i"),
        col("f") * lit(Float32(2)),
        col("l") * col("l"),
    ]:
        var plan = (
            frame.lazy()
            .filter(col("l") > 0)
            .select(expression.sum().alias("answer"))
        )
        assert_true(Bool(plan_arithmetic(plan)))
    for expression in [
        col("i") + col("l"),
        col("d") * col("d"),
        col("i") / col("i"),
        (col("i") + 1) * 2,
        col("i") + lit(Int64(2)),
    ]:
        assert_false(
            Bool(plan_arithmetic(frame.lazy().select(expression.sum())))
        )


# requires: metal
def test_metal_arithmetic_sums_match_cpu_with_literals_nulls_and_chunks() raises:
    var runtime = require_metal()
    for nulls in [False, True]:
        var base = _cross_columns(13001, nulls)
        var columns = List[Series]()
        for name in ["i", "l", "f"]:
            var split = 19 if name == "i" else (2503 if name == "l" else 101)
            columns.append(
                Series._from_chunks(
                    [
                        base.column(name).slice(5, split),
                        base.column(name).slice(5 + split, 10003 - split),
                    ]
                )
            )
        for frame in [base.slice(5, 10003), DataFrame(columns^)]:
            for name in ["i", "l", "f"]:
                var input = frame.with_column(
                    base.column(name).slice(0, 10003).renamed("other")
                )
                var expressions: List[Expr] = [
                    col(name) + col("other"),
                    col(name) - col("other"),
                    col(name) * 2,
                    2 - col(name),
                ]
                if name == "f":
                    expressions.append(col(name) * col("other"))
                else:
                    expressions.append(col(name) * 0)
                for expression in expressions:
                    for predicate in ["", "i", "l", "f"]:
                        var query = input.lazy()
                        if predicate:
                            query = query.filter(col(predicate) > 0)
                        var plan = query.select(
                            expression.sum().alias("answer")
                        )
                        assert_true(
                            "HANDOFF metal" in plan.explain(accelerator=runtime)
                        )
                        var expected = plan.collect(engine="cpu")
                        _close_bundle(execute_timed(plan)[0], expected)
                        _close_bundle(
                            execute_timed(plan, zero_copy=False)[0], expected
                        )
                        _close_bundle(
                            plan.collect(accelerator=runtime), expected
                        )


# requires: metal
def test_metal_arithmetic_checks_intermediate_overflow_before_cancelling_sum() raises:
    var runtime = require_metal()
    for dtype in [DataType.INT32, DataType.INT64]:
        var frame: DataFrame
        if dtype == DataType.INT32:
            frame = DataFrame(
                [
                    Series("a", Column[Int32]([Int32.MAX, Int32.MAX])),
                    Series("b", Column[Int32]([2, -2])),
                    Series("p", Column[Int32]([0, 0])),
                ]
            )
        else:
            frame = DataFrame(
                [
                    Series("a", Column[Int64]([Int64.MAX, Int64.MAX])),
                    Series("b", Column[Int64]([2, -2])),
                    Series("p", Column[Int32]([0, 0])),
                ]
            )
        for expression in [
            col("a") * col("b"),
            col("a") + col("a"),
            col("b") - col("a"),
            col("a") * 1,
        ]:
            var plan = frame.lazy().select(expression.sum())
            with assert_raises(contains="overflow"):
                _ = execute_timed(plan)
            with assert_raises(contains="overflow"):
                _ = plan.collect(accelerator=runtime)
            var excluded = (
                frame.lazy()
                .filter(col("p") > 0)
                .select(expression.sum(min_count=1))
            )
            _close_bundle(
                execute_timed(excluded)[0], excluded.collect(engine="cpu")
            )
    var extremes = DataFrame(
        [
            Series("a", Column[Int64]([Int64.MIN, Int64.MIN, Int64.MAX])),
            Series("b", Column[Int64]([1, -1, 1])),
            Series("p", Column[Int32]([1, 0, 1])),
        ]
    )
    var valid = (
        extremes.lazy().filter(col("p") > 0).select((col("a") * col("b")).sum())
    )
    assert_equal(execute_timed(valid)[0].item().int64(), Int64(-1))
    var invalid = extremes.lazy().select((col("a") * col("b")).sum())
    with assert_raises(contains="overflow"):
        _ = invalid.collect(accelerator=runtime)
    var skipped = DataFrame(
        [
            Series("a", Column[Int32]([Int32.MAX, Int32.MIN], [False, False])),
            Series("b", Column[Int32]([2, -1])),
        ]
    )
    var nulls = skipped.lazy().select((col("a") * col("b")).sum(min_count=1))
    _close_bundle(execute_timed(nulls)[0], nulls.collect(engine="cpu"))
    for frame in [skipped.copy(), skipped.clear()]:
        var empty = frame.lazy().select((col("a") * col("b")).sum())
        _close_bundle(execute_timed(empty)[0], empty.collect(engine="cpu"))
    var large_float = DataFrame(
        [Series("f", Column[Float32]([Float32.MAX_FINITE]))]
    )
    var nonfinite = large_float.lazy().select((col("f") * 2).sum())
    _close_bundle(
        nonfinite.collect(accelerator=runtime), nonfinite.collect(engine="cpu")
    )
    with assert_raises(contains="CPU non-finite accumulation"):
        _ = execute_timed(nonfinite)


def test_metal_arithmetic_overflow_flags_match_wide_integer_oracle() raises:
    var values: List[Int64] = [
        Int64.MIN,
        Int64.MIN + 1,
        -4_000_000_000,
        -3_037_000_500,
        -2,
        -1,
        0,
        1,
        2,
        3_037_000_500,
        4_000_000_000,
        Int64.MAX - 1,
        Int64.MAX,
    ]
    for a in values:
        for b in values:
            for op in [ADD, SUB, MUL]:
                var x = Int128(a)
                var y = Int128(b)
                var expected = x + y if op == ADD else (
                    x - y if op == SUB else x * y
                )
                var computed = integer_binary(a, b, UInt32(op), UInt32(1))
                var overflow = expected < Int128(
                    Int64.MIN
                ) or expected > Int128(Int64.MAX)
                assert_equal(computed[1], overflow)
                if not overflow:
                    assert_equal(computed[0], expected.cast[DType.int64]())


def test_metal_composition_shares_whole_expressions_and_preserves_outputs() raises:
    from dataframe_metal.composition_plan import (
        plan_composition,
        composition_key,
    )

    var frame = _frame()
    var value = col("f") * col("f")
    var plan = frame.lazy().select_exprs(
        [
            col("i").sum().alias("integer"),
            value.sum().alias("sum"),
            value.count().alias("count"),
            value.mean().alias("mean"),
            (col("f") + 1).sum().alias("other"),
        ]
    )
    var matched = plan_composition(plan)
    assert_true(Bool(matched))
    assert_equal(len(matched.value().fields), 3)
    assert_equal(len(matched.value().inputs), 2)
    assert_equal(matched.value().outputs[0].field, 2)
    for output in range(1, 4):
        assert_equal(matched.value().outputs[output].field, 0)
    assert_true(composition_key(matched.value()).endswith(":exact_float"))
    assert_false(Bool(plan_composition(frame.lazy().select(value.sum()))))
    assert_false(
        Bool(
            plan_composition(frame.lazy().select((col("i") + col("l")).mean()))
        )
    )


# requires: metal
def test_metal_composition_matches_cpu_for_shared_arithmetic_nulls_and_chunks() raises:
    var runtime = require_metal()
    for nulls in [False, True]:
        var base = _cross_columns(13001, nulls)
        var columns = List[Series]()
        for name in ["i", "l", "f"]:
            var split = 19 if name == "i" else (2503 if name == "l" else 101)
            columns.append(
                Series._from_chunks(
                    [
                        base.column(name).slice(5, split),
                        base.column(name).slice(5 + split, 10003 - split),
                    ]
                )
            )
        for frame in [base.slice(5, 10003), DataFrame(columns^)]:
            for name in ["i", "l", "f"]:
                var input = frame.with_column(
                    base.column(name).slice(0, 10003).renamed("other")
                )
                for expression in [
                    col(name) + col("other"),
                    col(name) - col("other"),
                    col(name) * 2,
                    2 - col(name),
                ]:
                    for predicate in ["", "i", "l", "f", "other"]:
                        var query = input.lazy()
                        if predicate:
                            query = query.filter(col(predicate) > 0)
                        var plan = query.select_exprs(
                            [
                                expression.sum().alias("sum"),
                                expression.count().alias("count"),
                                expression.mean().alias("mean"),
                                col("i").count().alias("bare_count"),
                                col("f").sum().alias("bare_sum"),
                            ]
                        )
                        assert_true(
                            "HANDOFF metal" in plan.explain(accelerator=runtime)
                        )
                        var expected = plan.collect(engine="cpu")
                        _close_bundle(execute_timed(plan)[0], expected)
                        _close_bundle(
                            execute_timed(plan, zero_copy=False)[0], expected
                        )
                        _close_bundle(
                            plan.collect(accelerator=runtime), expected
                        )


# requires: metal
def test_metal_composition_uses_eight_value_inputs_and_independent_predicate() raises:
    _ = require_metal()
    var columns = List[Series]()
    for i in range(9):
        columns.append(
            Series(
                String(i),
                Column[Int32]([1, 2, 3, 4], [True, i % 2 == 0, True, True]),
            )
        )
    var frame = DataFrame(columns^).slice(1, 3)
    var expressions = List[Expr]()
    for i in range(4):
        expressions.append(
            (col(String(i * 2)) * col(String(i * 2 + 1))).sum().alias(String(i))
        )
    var plan = frame.lazy().filter(col("8") > 0).select_exprs(expressions)
    _close_bundle(execute_timed(plan)[0], plan.collect(engine="cpu"))
    _close_bundle(
        execute_timed(plan, zero_copy=False)[0], plan.collect(engine="cpu")
    )


# requires: metal
def test_metal_composition_checks_overflow_empty_and_exact_float_retry() raises:
    var runtime = require_metal()
    for wide in [False, True]:
        var frame: DataFrame
        if wide:
            frame = DataFrame(
                [
                    Series("a", Column[Int64]([Int64.MAX, Int64.MAX])),
                    Series("b", Column[Int64]([2, -2])),
                ]
            )
        else:
            frame = DataFrame(
                [
                    Series("a", Column[Int32]([Int32.MAX, Int32.MAX])),
                    Series("b", Column[Int32]([2, -2])),
                ]
            )
        var value = col("a") * col("b")
        for expression in [value.count(), value.mean()]:
            var plan = frame.lazy().select(expression)
            with assert_raises(contains="overflow"):
                _ = execute_timed(plan)
            with assert_raises(contains="overflow"):
                _ = plan.collect(accelerator=runtime)
        var mean = frame.lazy().select((col("a") * 1).mean())
        _close_bundle(execute_timed(mean)[0], mean.collect(engine="cpu"))
        var sum = frame.lazy().select_exprs(
            [(col("a") * 1).sum().alias("s"), col("b").count().alias("c")]
        )
        with assert_raises(contains="overflow"):
            _ = execute_timed(sum)
        var empty = (
            frame.clear()
            .lazy()
            .select_exprs(
                [
                    value.sum(min_count=1).alias("s"),
                    value.count().alias("c"),
                    value.mean().alias("m"),
                ]
            )
        )
        _close_bundle(execute_timed(empty)[0], empty.collect(engine="cpu"))
    var frame = DataFrame(
        [Series("f", Column[Float32]([Float32.MAX_FINITE, Float32.MAX_FINITE]))]
    )
    var value = col("f") * 1
    var exact = frame.lazy().select_exprs(
        [value.mean().alias("m"), value.count().alias("c")]
    )
    _close_bundle(execute_timed(exact)[0], exact.collect(engine="cpu"))
    var nonfinite = frame.lazy().select_exprs(
        [(col("f") * 2).sum().alias("s"), col("f").count().alias("c")]
    )
    with assert_raises(contains="CPU non-finite accumulation"):
        _ = execute_timed(nonfinite)
    _close_bundle(
        nonfinite.collect(accelerator=runtime), nonfinite.collect(engine="cpu")
    )


def test_metal_calibration_validates_all_identity_dimensions_and_layouts() raises:
    from dataframe.json.value import JsonValue
    from dataframe_metal.calibration import (
        validate_identity,
        shape_key,
        pipeline_key,
    )
    from dataframe_metal.handoff import region_key

    var current = JsonValue(
        '{"runtime_build_id":"dataframe-mojo-metal-v2;cpu=a;metal=b","device":"M1","host":"h;thermal=0","workers":8}'
    )
    var valid = JsonValue(
        '{"schema_version":2,"identity":' + current.canonical() + "}"
    )
    assert_equal(validate_identity(valid, current), String())
    for identity in [
        '{"runtime_build_id":"changed","device":"M1","host":"h;thermal=0","workers":8}',
        '{"runtime_build_id":"dataframe-mojo-metal-v2;cpu=a;metal=b","device":"M2","host":"h;thermal=0","workers":8}',
        '{"runtime_build_id":"dataframe-mojo-metal-v2;cpu=a;metal=b","device":"M1","host":"h;thermal=1","workers":8}',
        '{"runtime_build_id":"dataframe-mojo-metal-v2;cpu=a;metal=b","device":"M1","host":"h;thermal=0","workers":4}',
    ]:
        var wrong = JsonValue(
            '{"schema_version":2,"identity":' + identity + "}"
        )
        assert_true(Bool(validate_identity(wrong, current)))
    assert_true(Bool(validate_identity(JsonValue('{"operators":{}}'), current)))
    var frame = _frame()
    var plain = frame.lazy().select(col("i").sum())
    var key = shape_key(plain, region_key(plain))
    assert_true(pipeline_key(plain).startswith("pipeline:select"))
    var nullable = (
        frame.with_column(
            Series("i", Column[Int32]([1, 5, 9], [True, False, True]))
        )
        .lazy()
        .select(col("i").sum())
    )
    assert_false(key == shape_key(nullable, region_key(nullable)))
    var chunked = (
        DataFrame(
            [
                Series._from_chunks(
                    [
                        frame.column("i").slice(0, 1),
                        frame.column("i").slice(1, 2),
                    ]
                )
            ]
        )
        .lazy()
        .select(col("i").sum())
    )
    assert_false(key == shape_key(chunked, region_key(chunked)))


# requires: metal
def test_metal_calibration_rejects_overwritten_cached_models_and_outside_rows() raises:
    _ = require_metal()
    from dataframe_metal.accelerator.gpu.identity import runtime_build_id
    from dataframe_metal.calibration import measured_cost, shape_key
    from dataframe_metal.handoff import region_key, estimate

    if not runtime_build_id().startswith("dataframe-mojo-metal-v2;cpu="):
        return
    var plan = (
        _columns(100_000, False)
        .lazy()
        .filter(col("i") > 0)
        .select(col("i").sum())
    )
    var path = String("build/test-handoff-cache-coherence.json")
    _calibrate(path, 50, 0.1, plan, 100_000, 100_000)
    var configured = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", path)
    var key = shape_key(plan, region_key(plan))
    assert_true(measured_cost(key, 100_000) > 0)
    assert_equal(measured_cost(key, 0), Float64(-1))
    assert_equal(measured_cost(key, 1_000_000_001), Float64(-1))
    assert_equal(estimate(plan, 0)[1], Float64(-1))
    assert_equal(estimate(plan, 1_000_000)[1], Float64(-1))
    var outside = (
        _columns(100, False).lazy().filter(col("i") > 0).select(col("i").sum())
    )
    assert_false("; on metal:0" in outside.explain(engine="auto"))
    _calibrate(path, 0.01, 100, plan, 100_000, 100_000)
    assert_equal(measured_cost(key, 100_000), Float64(-1))
    assert_true("no measured cost" in plan.explain(engine="auto"))
    _ = configured^


# requires: metal
def test_metal_pipeline_adjudicates_changed_float_values_on_cpu() raises:
    _ = require_metal()
    from dataframe_metal.accelerator.gpu.identity import runtime_build_id

    if not runtime_build_id().startswith("dataframe-mojo-metal-v2;cpu="):
        return
    var tiny = bitcast[DType.float32](UInt32(1))
    var frame = DataFrame([Series("f", Column[Float32]([tiny, tiny]))])
    var value = col("f") * lit(Float32(2))
    var plan = frame.lazy().select_exprs(
        [
            value.sum().alias("s"),
            value.count().alias("c"),
            value.mean().alias("m"),
        ]
    )
    var expected = plan.collect(engine="cpu")
    assert_equal(
        bitcast[DType.uint32](expected.item(0, "s").float32()), UInt32(4)
    )
    var path = String("build/test-handoff-changed-values.json")
    _calibrate(path, 50_000_000, 0.1, plan)
    var configured = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", path)
    assert_true("; on metal:0" in plan.explain(engine="auto"))
    var trace_path = String("build/test-handoff-adjudication-trace.jsonl")
    with open(trace_path, "w") as target:
        target.write("")
    var trace = ScopedEnvironment("DATAFRAME_METAL_PLACEMENT_TRACE", trace_path)
    var actual = plan.collect(engine="auto")
    assert_true(actual.equals(expected))
    assert_equal(
        bitcast[DType.uint32](actual.item(0, "s").float32()), UInt32(4)
    )
    var traced = String()
    with open(trace_path, "r") as source:
        traced = source.read()
    assert_true('"engine":"cpu-adjudication"' in traced, traced)
    assert_false('"engine":"metal"' in traced, traced)
    _ = trace^
    _ = configured^


def test_compact_float_pair_loads_unaligned_windows_and_guards_tail() raises:
    from dataframe_metal.accelerator.gpu.float_composed_reduce import _pair
    from dataframe_metal.accelerator.gpu.multi_reduce import Bytes

    var values = List[Float32]([Float32(1), Float32(2), Float32(3)])
    var bytes = Bytes(unsafe_from_address=Int(values.unsafe_ptr()))
    var middle = _pair(bytes, 1, 3)
    assert_equal(middle[0], Float32(2))
    assert_equal(middle[1], Float32(3))
    var tail = _pair(bytes, 2, 3)
    assert_equal(tail[0], Float32(3))
    assert_equal(tail[1], Float32(0))
    _ = values^


def test_metal_calibration_corpus_checks_shared_arithmetic_mean_and_both_null_masks() raises:
    from benchmarks.bench_metal_calibration import (
        _table,
        _chunks,
        _reference,
        _expressions,
    )
    from dataframe_metal.calibration_capture import _check
    from dataframe_metal.composition_plan import plan_composition

    for nulls in [False, True]:
        var example = _reference(32, nulls, 4)
        assert_equal(
            example.item(0, "count").int64(), Int64(16 if nulls else 24)
        )
        assert_equal(
            example.item(0, "sales").float32(), Float32(-1 if nulls else 2)
        )
        assert_equal(
            example.item(0, "mean").float64(), -0.0625 if nulls else 1.0 / 12
        )
        for rows in [32, 65, 127]:
            var source = _table(rows, nulls)
            for chunks in [1, 8]:
                var frame = _chunks(source, chunks)
                var plan = (
                    frame.lazy()
                    .filter(col("p") > 0)
                    .select_exprs(_expressions(4))
                )
                var matched = plan_composition(plan)
                assert_true(Bool(matched))
                assert_equal(len(matched.value().fields), 2)
                assert_true(matched.value().precise_float)
                _check(plan.collect(engine="cpu"), _reference(rows, nulls, 4))


def _typed_composition_cases() raises -> List[LazyFrame]:
    var columns = List[Series]()
    var valid = List[Bool]([True, False, True])
    for i in range(4):
        columns.append(Series("i" + String(i), Column[Int32]([1, 2, 3], valid)))
        columns.append(
            Series(
                "l" + String(i),
                Column[Int64](
                    [
                        (Int64(1) << 40) + 1,
                        (Int64(1) << 40) + 2,
                        (Int64(1) << 40) + 3,
                    ],
                    valid,
                ),
            )
        )
        columns.append(
            Series("f" + String(i), Column[Float32]([1, 2, 3], valid))
        )
    columns.append(Series("p", Column[Int32]([1, 1, 1])))
    var frame = DataFrame(columns^)
    var result = List[LazyFrame]()
    for arity in range(1, 5):
        for floats in range(arity + 1):
            for narrow in range(arity - floats + 1):
                var names = List[String]()
                for i in range(floats):
                    names.append("f" + String(i))
                for i in range(narrow):
                    names.append("i" + String(i))
                for i in range(arity - floats - narrow):
                    names.append("l" + String(i))
                for means in [False, True]:
                    var expressions = List[Expr]()
                    for i in range(arity - 1, -1, -1):
                        var name = names[i].copy()
                        var value = col(name) + 1
                        if name.startswith("l"):
                            value = col(name) + lit(Int64(1))
                        elif name.startswith("f"):
                            value = col(name) + lit(Float32(1))
                        expressions.append(value.sum().alias("s_" + name))
                        expressions.append(value.count().alias("c_" + name))
                        if means:
                            expressions.append(value.mean().alias("m_" + name))
                    result.append(
                        frame.lazy()
                        .filter(col("p") > 0)
                        .select_exprs(expressions)
                    )
    return result^


def test_composed_schema_packs_native_types_and_remaps_all_output_aliases() raises:
    from dataframe_metal.composition_plan import (
        plan_composition,
        single_float_composition,
        composition_key,
    )

    var cases = _typed_composition_cases()
    assert_equal(len(cases), 68)
    for query in cases:
        var matched = plan_composition(query)
        assert_true(Bool(matched))
        ref plan = matched.value()
        var rank = -1
        for field in plan.fields:
            var current = 0 if field.dtype == DataType.FLOAT32 else (
                1 if field.dtype == DataType.INT32 else 2
            )
            assert_true(current >= rank)
            rank = current
        ref expressions = query._plan._nodes[len(query._plan._nodes) - 1].exprs
        for i in range(len(plan.outputs)):
            assert_equal(
                plan.fields[plan.outputs[i].field].left,
                expressions[i]._nodes[0].text,
            )
        if not single_float_composition(plan):
            assert_true(
                composition_key(plan).startswith(
                    "composed_reductions:v2:typed_fields:"
                )
            )


# requires: metal
def test_composed_typed_native_fields_match_cpu_across_all_arities() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    for query in _typed_composition_cases():
        var expected = query.collect(engine="cpu")
        _close_bundle(execute_timed(query)[0], expected)
        _close_bundle(execute_timed(query, zero_copy=False)[0], expected)
        _close_bundle(query.collect(accelerator=runtime), expected)


# requires: metal
def test_composed_typed_native_float_sums_adjudicate_underflow_without_affecting_counts() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var tiny = bitcast[DType.float32](UInt32(1))
    var frame = DataFrame(
        [
            Series("i", Column[Int32]([1, 2])),
            Series("f", Column[Float32]([tiny, tiny])),
            Series("l", Column[Int64]([3, 4])),
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            (col("i") + 1).sum().alias("i"),
            col("f").sum().alias("f"),
            col("l").count().alias("n"),
        ]
    )
    with assert_raises(contains="gradual underflow"):
        _ = execute_timed(query)
    var actual = query.collect(accelerator=runtime)
    assert_true(actual.equals(query.collect(engine="cpu")))
    assert_equal(
        bitcast[DType.uint32](actual.item(0, "f").float32()), UInt32(2)
    )
    var count = frame.lazy().select_exprs(
        [
            (col("i") + 1).sum().alias("i"),
            col("f").count().alias("f"),
            col("l").count().alias("n"),
        ]
    )
    assert_true(execute_timed(count)[0].equals(count.collect(engine="cpu")))
    var mean = frame.lazy().select_exprs(
        [
            (col("i") + 1).sum().alias("i"),
            col("f").mean().alias("f"),
            col("l").count().alias("n"),
        ]
    )
    assert_true(execute_timed(mean)[0].equals(mean.collect(engine="cpu")))
