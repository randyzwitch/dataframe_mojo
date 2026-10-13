"""Eligibility and automatic placement are tested before any device submission."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, LazyFrame, Series, col, lit
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows, RowPlan
from dataframe.accelerator.memory import (
    row_memory,
    bitmap_bytes,
    checked_add,
    checked_mul,
)
from dataframe.accelerator.cost import (
    CalibrationKey,
    PlanEstimate,
    cost_feature_names,
    select_auto,
)
from tests.support.calibration_fixture import (
    calibration_features,
    calibration_vector,
    calibration_record,
    calibration_table,
)
from tests.support.coverage_generators import chunk_layout


def _rejection(
    query: LazyFrame,
    fragment: String,
    capabilities: RowCapabilities = RowCapabilities(),
    *,
    terminal: Bool = True,
) raises:
    var failed = False
    try:
        _ = lower_rows(query, capabilities, terminal_boundary=terminal)
    except error:
        if String(error).find(fragment) < 0:
            raise error
        failed = True
    assert_true(failed, "unsupported region must reject before submission")


def test_row_lowering_uses_schema_and_retains_source_owners_without_evaluation() raises:
    var source = DataFrame(
        [
            Series(
                "i",
                Column[Int64]([Int64.MAX, Int64.MIN, 7], [True, False, True]),
            ),
            Series(
                "unused",
                Column[String](
                    ["not numeric", "", "long external storage beyond inline"]
                ),
            ),
        ]
    )
    var query = source.lazy().select((col("i") + 1).alias("v"))
    # Executing this expression would overflow. Lowering may inspect metadata
    # and literal bits, but must not evaluate data values.
    var lowered = lower_rows(query)
    assert_equal(lowered.source.columns(), ["i"])
    assert_equal(lowered.source.height(), 3)
    assert_equal(len(lowered.outputs), 1)
    assert_equal(lowered.outputs[0].name, "v")
    assert_true(lowered.outputs[0].dtype == DataType.INT64)
    source = DataFrame(List[Series]())
    query = source.lazy()
    assert_equal(lowered.source.item(0, "i").int64(), Int64.MAX)
    assert_true(lowered.source.item(1, "i").is_null())
    assert_equal(lowered.source.item(2, "i").int64(), Int64(7))
    var huge = (
        DataFrame([Series("i", Column[Int64]([0]))])
        .lazy()
        .select((col("i") + lit(Int64(9007199254740993))).alias("v"))
    )
    var exact = lower_rows(huge)
    from std.memory import bitcast

    var found = False
    for value in exact.literals:
        found = found or bitcast[DType.int64](value) == Int64(9007199254740993)
    assert_true(found, "integer literal table must preserve bits above 2^53")


def test_row_lowering_rejects_unsupported_storage_types_and_unobservable_errors() raises:
    var integers = DataFrame([Series("i", Column[Int64]([1, 2, 3]))])
    var checked = integers.lazy().select((col("i") + 1).alias("v"))
    _rejection(checked.head(1), "terminal projections without head")
    _rejection(checked, "terminal projections without head", terminal=False)
    _rejection(chunk_layout(integers).lazy().select(col("i")), "chunked input")
    var text = DataFrame([Series("s", Column[String](["1", "2"]))])
    _rejection(text.lazy().select(col("s")), "[dtype]")
    _rejection(integers.lazy().sort("i"), "[plan]")
    _rejection(
        integers.lazy().group_by("i").agg(col("i").count().alias("n")), "[plan]"
    )
    _rejection(integers.lazy().select(col("i").rolling_sum(2)), "unsupported")
    var floats = DataFrame([Series("x", Column[Float32]([1, 2]))])
    var no_wide = RowCapabilities("limited", float64=False, wide_integer=False)
    _rejection(
        floats.lazy().select(col("x").sum()), "Float64 accumulation", no_wide
    )
    _rejection(
        integers.lazy().select(col("i").sum()),
        "exact wide accumulation",
        no_wide,
    )
    var double = DataFrame([Series("x", Column[Float64]([1, 2]))])
    _rejection(double.lazy().select(col("x")), "Float64 requires CPU", no_wide)
    # These contracts are capabilities, not a blanket ban on all reductions.
    var count = lower_rows(floats.lazy().select(col("x").count()), no_wide)
    assert_true(count.outputs[0].dtype == DataType.INT64)


def test_bounded_lowering_checks_source_slots_and_step_limits_at_the_boundary() raises:
    for count in [63, 64, 65]:
        var columns = List[Series]()
        for i in range(count):
            columns.append(Series("c" + String(i), Column[Int32]([1])))
        var source = DataFrame(columns^)
        if count <= 64:
            var actual = lower_rows(source.lazy())
            assert_equal(actual.source.width(), count)
            assert_equal(len(actual.outputs), count)
        else:
            _rejection(source.lazy(), "1 to 64 source columns")
    var source = DataFrame([Series("x", Column[Float32]([1, 2]))])
    for count in [62, 63, 64]:
        var query = source.lazy()
        for _ in range(count):
            query = query.select(col("x"))
        if count <= 63:
            var actual = lower_rows(query)
            assert_equal(len(actual.steps), count)
        else:
            _rejection(query, "at most 64 value slots")
    for count in [63, 64, 65]:
        var query = source.lazy()
        for _ in range(count):
            query = query.drop(List[String]())
        if count <= 64:
            var actual = lower_rows(query)
            assert_equal(len(actual.outputs), 1)
        else:
            _rejection(query, "at most 64 plan steps")


def test_memory_preflight_bitmap_geometry_exact_arithmetic_and_budget_fences() raises:
    for n in [0, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129]:
        for offset in range(8):
            var touched = List[Int]()
            for bit in range(offset, offset + n):
                var byte = bit // 8
                if byte not in touched:
                    touched.append(byte)
            assert_equal(bitmap_bytes(n, offset), len(touched))
    for n in [-1, 0, 1, 2, 7, Int.MAX // 2, Int.MAX - 1, Int.MAX]:
        for m in [-1, 0, 1, 2, 7, Int.MAX // 2, Int.MAX - 1, Int.MAX]:
            var sum = Int128(n) + Int128(m)
            var product = Int128(n) * Int128(m)
            if n >= 0 and m >= 0 and sum <= Int128(Int.MAX):
                assert_equal(checked_add(n, m), Int(sum))
            else:
                with assert_raises():
                    _ = checked_add(n, m)
            if n >= 0 and m >= 0 and product <= Int128(Int.MAX):
                assert_equal(checked_mul(n, m), Int(product))
            else:
                with assert_raises():
                    _ = checked_mul(n, m)
    for rows in [0, 1, 65, 257]:
        var source = DataFrame(
            [Series("x", Column[Float32](List[Float32](length=rows, fill=1)))]
        )
        var lowered = lower_rows(
            source.lazy().filter(col("x") > lit(Float32(0))).select(col("x"))
        )
        var memory = row_memory(lowered)
        var estimate = memory.summary.copy()
        assert_true(memory.upload_bytes >= rows * 4)
        assert_true(estimate.peak_bytes >= max(1, rows) * 4)
        assert_equal(
            estimate.peak_bytes,
            estimate.input_bytes
            + estimate.workspace_bytes
            + estimate.output_bytes,
        )
        estimate.require_budget(estimate.peak_bytes)
        estimate.require_budget(estimate.peak_bytes + 1)
        with assert_raises():
            estimate.require_budget(estimate.peak_bytes - 1)
        with assert_raises():
            estimate.require_budget(-1)
    with assert_raises():
        _ = bitmap_bytes(1, 8)


def _key() -> CalibrationKey:
    return CalibrationKey(
        "probe", "device", "runtime-v1", "cpu-v1", "host", 1, True, False, 7
    )


def _facts() -> PlanEstimate:
    return PlanEstimate(
        calibration_vector(), calibration_vector(), 64, "integer", "accel"
    )


def test_automatic_placement_uses_conservative_costs_memory_and_calibration_identity() raises:
    assert_equal(cost_feature_names(), calibration_features())
    var facts = _facts()
    var key = _key()
    for cpu in [149, 150, 151, 1000]:
        var decision = select_auto(
            facts,
            calibration_table(calibration_record(Float64(cpu), 100)),
            key,
            64,
            100,
        )
        assert_equal(decision.engine, "accel" if cpu > 150 else "cpu")
        assert_equal(decision.cpu_ns, Float64(cpu))
        assert_equal(decision.accelerator_ns, Float64(100))
    var valid = calibration_table(calibration_record())
    assert_equal(select_auto(facts, valid, key, 63, 100).engine, "cpu")
    assert_equal(select_auto(facts, valid, key, 64, 100).engine, "accel")
    assert_equal(select_auto(facts, valid, key, 65, 100).engine, "accel")
    assert_equal(
        select_auto(facts, calibration_table(String()), key, 64, 100).engine,
        "cpu",
    )
    assert_equal(
        select_auto(
            facts,
            calibration_table(calibration_record(expiry=100)),
            key,
            64,
            100,
        ).engine,
        "cpu",
    )
    assert_equal(
        select_auto(
            facts,
            calibration_table(calibration_record(validated=False)),
            key,
            64,
            100,
        ).engine,
        "cpu",
    )
    var mismatch = _key()
    mismatch.device = "other-device"
    assert_equal(select_auto(facts, valid, mismatch, 64, 100).engine, "cpu")
    mismatch = _key()
    mismatch.cpu_build_id = "development-source-no-calibration"
    assert_equal(select_auto(facts, valid, mismatch, 64, 100).engine, "cpu")
    facts.placement = "mixed"
    assert_equal(select_auto(facts, valid, key, 64, 100).engine, "cpu")
    facts.placement = "accel"
    facts.peak_bytes = -1
    assert_equal(select_auto(facts, valid, key, 64, 100).engine, "cpu")


def test_automatic_placement_isolates_emulated_float64_calibrations() raises:
    var facts = _facts()
    facts.family += ";metal-emulated-float64-v1"
    var key = _key()
    var old_native = calibration_table(calibration_record())
    var unmatched = select_auto(facts, old_native, key, 64, 100)
    assert_equal(unmatched.engine, "cpu")
    assert_true("no matching" in unmatched.reason)
    var expensive = calibration_table(
        calibration_record(
            1000,
            10000,
            family=facts.family,
        )
    )
    assert_equal(select_auto(facts, expensive, key, 64, 100).engine, "cpu")
    var faster = calibration_table(
        calibration_record(1000, 100, family=facts.family)
    )
    assert_equal(select_auto(facts, faster, key, 64, 100).engine, "accel")


def test_automatic_placement_rejects_ambiguous_nonfinite_or_unvalidated_estimates() raises:
    var facts = _facts()
    var key = _key()
    with assert_raises():
        _ = select_auto(
            facts,
            calibration_table(calibration_record(), duplicate=True),
            key,
            64,
            100,
        )
    for safety in [0.0, 1.4]:
        with assert_raises():
            _ = select_auto(
                facts,
                calibration_table(calibration_record(safety=safety)),
                key,
                64,
                100,
            )
    for residual in [1.01, 2.0]:
        with assert_raises():
            _ = select_auto(
                facts,
                calibration_table(calibration_record(cpu_residual=residual)),
                key,
                64,
                100,
            )
    with assert_raises():
        _ = select_auto(
            facts,
            calibration_table(calibration_record(device_residual=0.99)),
            key,
            64,
            100,
        )
    facts.cpu[0] = 1000001
    with assert_raises():
        _ = select_auto(
            facts, calibration_table(calibration_record()), key, 64, 100
        )
    facts = _facts()
    facts.accelerator[0] = Float64(0) / Float64(0)
    with assert_raises():
        _ = select_auto(
            facts, calibration_table(calibration_record()), key, 64, 100
        )
    facts = _facts()
    facts.cpu = [1.0]
    with assert_raises():
        _ = select_auto(
            facts, calibration_table(calibration_record()), key, 64, 100
        )
    with assert_raises():
        _ = select_auto(
            _facts(),
            calibration_table(calibration_record(cpu=-1)),
            key,
            64,
            100,
        )
