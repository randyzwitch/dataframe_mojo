"""The Metal cost model's placement decisions, before any device submission."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe_metal.accelerator.cost import (
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
