"""Shared CPU/accelerator decisions use terms, identity and capacity gates."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col
from dataframe.json_value import JsonValue, json_quote
from dataframe._build_identity import CPU_BUILD_ID
from dataframe.accelerator.cost import (
    COST_FEATURES,
    PlanEstimate,
    CalibrationKey,
    select_auto,
)
from dataframe.accelerator.cpu_work import cpu_features


def _model(fixed: Int, factor: String) -> String:
    var coefficients = String("{")
    var minimum = String("{")
    var maximum = String("{")
    comptime for i in range(len(COST_FEATURES)):
        if i:
            coefficients += ","
            minimum += ","
            maximum += ","
        var name = json_quote(COST_FEATURES[i])
        coefficients += name + ":" + String(fixed if i == 0 else 0)
        minimum += name + ":0"
        maximum += name + ":1000000"
    return (
        '{"coefficients":'
        + coefficients
        + '},"minimum":'
        + minimum
        + '},"maximum":'
        + maximum
        + '},"residual_factor":'
        + factor
        + "}"
    )


def _record(cpu: Int = 1000, device: Int = 100, failures: Int = 0) -> String:
    return (
        '{"provider":"test","device":"gpu","runtime_build_id":"runtime","cpu_build_id":"cpu","host":"host","threads":1,"optimize":true,"streaming":true,"batch_size":64,"family":"Float32","expires_at":10000,"validated":true,"training_queries":10,"validation_queries":5,"distribution_coverage":"conservative-null-selectivity-envelope","safety_factor":1.5,"cpu":'
        + _model(cpu, "0.9")
        + ',"accelerator":'
        + _model(device, "1.1")
        + ',"validation_failures":'
        + String(failures)
        + "}"
    )


def _table(record: String) raises -> JsonValue:
    return JsonValue(
        '{"format":"dataframe-analytical-cost-v1","records":[' + record + "]}"
    )


def _facts() -> PlanEstimate:
    var cpu = List[Float64](length=len(COST_FEATURES), fill=0)
    cpu[0] = 1
    return PlanEstimate(cpu.copy(), cpu^, 100, "Float32", "accel")


def _key() -> CalibrationKey:
    return CalibrationKey(
        "test", "gpu", "runtime", "cpu", "host", 1, True, True, 64
    )


def test_shared_comparison_explains_both_bounds_and_the_largest_term() raises:
    var facts = _facts()
    var decision = select_auto(facts, _table(_record()), _key(), 100, 10)
    assert_equal(decision.engine, "accel")
    assert_equal(decision.cpu_ns, Float64(900))
    assert_equal(decision.accelerator_ns, Float64(110))
    assert_true("cpu_lower_ns=" in decision.reason)
    assert_true("accelerator_upper_ns=" in decision.reason)
    assert_true("deciding_term=fixed" in decision.reason)
    assert_equal(
        select_auto(facts, _table(_record(100, 100)), _key(), 100, 10).engine,
        "cpu",
    )


def test_identity_memory_expiration_coverage_and_ambiguity_reject() raises:
    assert_equal(
        select_auto(
            _facts(), _table(_record(1000, 100, 1)), _key(), 100, 10
        ).engine,
        "cpu",
    )
    var facts = _facts()
    var table = _table(_record())
    var key = _key()
    key.cpu_build_id = "different-cpu"
    assert_equal(select_auto(facts, table, key, 100, 10).engine, "cpu")
    assert_equal(select_auto(facts, table, _key(), 99, 10).engine, "cpu")
    assert_equal(select_auto(facts, table, _key(), 100, 10000).engine, "cpu")
    facts.accelerator[1] = 1000001
    with assert_raises(contains="coverage"):
        _ = select_auto(facts, table, _key(), 100, 10)
    with assert_raises(contains="ambiguous"):
        _ = select_auto(
            _facts(), _table(_record() + "," + _record()), _key(), 100, 10
        )
    facts = _facts()
    facts.placement = "mixed"
    assert_equal(select_auto(facts, table, _key(), 100, 10).engine, "cpu")


def test_cpu_features_count_work_without_values_or_query_mutation() raises:
    var frame = DataFrame([Series("x", Column[Float32]([1, 2, 3]))])
    var query = frame.lazy().filter(col("x") > 1).select((col("x") + 1).sum())
    var before = query.explain(engine="cpu")
    var work = cpu_features(query)
    assert_equal(work[1], "Float32")
    assert_equal(work[0][1], Float64(3))
    assert_equal(work[0][2], Float64(12))
    assert_true(work[0][3] > 0)
    assert_true(work[0][4] > 0)
    assert_equal(work[0][5], Float64(3))
    assert_equal(query.explain(engine="cpu"), before)
    var profile = query.profile(engine="cpu")
    assert_equal(profile[1].item(0, "cpu_build_id").string(), CPU_BUILD_ID)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
