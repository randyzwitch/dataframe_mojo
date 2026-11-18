"""CPU profiles and pipeline JSON preserve observed work and result disposition."""
# contract-group: execution.reporting
from std.testing import assert_equal, assert_true
from std.memory import ArcPointer
from std.os import getenv
from dataframe import Column, DataFrame, Series, col
from dataframe.execution_observation import DeviceObservation
from dataframe.execution_report import pipeline_record_json
from dataframe.json.value import JsonValue
from tests.support.group_strategy_reference import ScopedEnvironment


def test_cpu_profiles_report_terminal_pipelines_without_device_work() raises:
    var query = (
        DataFrame([Series("v", Column[Int64]([1, 2, 3, 4]))])
        .lazy()
        .filter(col("v") > 1)
        .select(col("v").sum())
    )
    for streaming in [False, True]:
        var result = query.profile(
            engine="cpu", optimize=False, streaming=streaming, batch_size=2
        )
        assert_equal(result[0].item().int64(), Int64(9))
        ref report = result[1]
        var terminals = 0
        for row in range(report.height()):
            assert_equal(report.item(row, "executions").int64(), Int64(1))
            assert_equal(report.item(row, "result_device").string(), "cpu")
            assert_equal(report.item(row, "device_status").string(), "cpu")
            assert_equal(report.item(row, "gpu_launches").int64(), Int64(0))
            assert_equal(
                report.item(row, "gpu_completed_launches").int64(), Int64(0)
            )
            assert_equal(
                report.item(row, "phase_timing_kind").string(), "unavailable"
            )
            assert_equal(
                report.item(row, "device_work_wall_ms").float64(), Float64(-1)
            )
            var terminal = report.item(row, "pipeline").int64()
            var found = False
            for other in range(report.height()):
                found |= report.item(other, "node").int64() == terminal
            assert_true(found)
            terminals += Int(terminal == report.item(row, "node").int64())
        assert_true(terminals > 0)
        assert_true(not query._plan._report)


def test_auto_profiles_explain_missing_calibration_and_reset_query_observations() raises:
    var calibration = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", "")
    var query = (
        DataFrame([Series("v", Column[Int64]([1, 2, 3]))])
        .lazy()
        .select(col("v").sum())
    )
    var previous = Int64(-1)
    for _ in range(2):
        var result = query.profile(engine="auto")
        assert_true(result[0].equals(query.collect(engine="cpu")))
        ref report = result[1]
        var current = report.item(0, "query_id").int64()
        assert_true(current != previous)
        previous = current
        for row in range(report.height()):
            assert_equal(
                report.item(row, "device_status").string(), "cpu_fallback"
            )
            assert_equal(
                report.item(row, "fallback_reason").string(),
                "missing_calibration",
            )
            assert_equal(report.item(row, "gpu_launches").int64(), Int64(0))
            assert_equal(
                report.item(row, "execution_attempt").int64(), Int64(0)
            )
            assert_true(report.item(row, "result_used").bool())
    _ = calibration^


def test_pipeline_json_escapes_labels_and_discards_failed_query_results() raises:
    var name = String('value"\nwith tab\t')
    var query = (
        DataFrame([Series(name, Column[Int64]([1, 2, 3]))])
        .lazy()
        .select(col(name).sum())
    )
    var observed = query.profile(engine="cpu")
    ref report = observed[1]
    for row in range(report.height()):
        for status in ["completed", "failed", "cpu_adjudication_failed"]:
            var record = JsonValue(pipeline_record_json(report, row, status))
            assert_equal(
                record.get("format").string(),
                "dataframe-pipeline-observation-v1",
            )
            assert_equal(
                record.get("operator").string(),
                report.item(row, "operator").string(),
            )
            assert_equal(record.get("query_status").string(), status)
            assert_equal(record.get("gpu_launches").text, "0")
            assert_equal(
                record.get("result_used").text,
                "true" if status == "completed" else "false",
            )


def test_profiles_include_shared_producer_work_with_distinct_pipeline_ids() raises:
    var frame = DataFrame(
        [
            Series("k", Column[Int64]([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])),
            Series("v", Column[Int64]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])),
        ]
    )
    var shared = frame.lazy().group_by("k").agg([col("v").sum().alias("total")])
    var query = shared.join(shared, "k")
    var expected = DataFrame(
        [
            Series("k", Column[Int64]([0, 1, 2])),
            Series("total", Column[Int64]([18, 22, 26])),
            Series("total_right", Column[Int64]([18, 22, 26])),
        ]
    )
    var planned = query._optimized()
    var producers = planned._share_subplans(True, 2)
    assert_equal(len(producers), 1)
    for streaming in [False, True]:
        var observed = query.profile(
            engine="cpu", streaming=streaming, batch_size=2
        )
        assert_true(observed[0].sort("k").equals(expected))
        ref report = observed[1]
        var groups = 0
        var previous = Int64(-1)
        for row in range(report.height()):
            var id = report.item(row, "node").int64()
            assert_true(id > previous)
            previous = id
            if report.item(row, "operator").string().startswith("GROUP_BY"):
                groups += 1
                assert_equal(report.item(row, "executions").int64(), Int64(1))
                assert_equal(report.item(row, "input_rows").int64(), Int64(12))
                assert_equal(report.item(row, "output_rows").int64(), Int64(3))
                assert_equal(report.item(row, "result_device").string(), "cpu")
                assert_equal(report.item(row, "pipeline").int64(), id)
                assert_true(report.item(row, "plan_id").int64() > 0)
        assert_equal(groups, 1)


def test_observed_device_region_cpu_retry_does_not_create_another_query_trace() raises:
    var path = getenv("COVERAGE_TEST_TMP") + "/internal-cpu-retry.jsonl"
    assert_true(Bool(getenv("COVERAGE_TEST_TMP")))
    with open(path, "w") as target:
        target.write("")
    var trace = ScopedEnvironment("DATAFRAME_EXECUTION_TRACE", path)
    var region = (
        DataFrame([Series("v", Column[Int64]([1, 2, 3]))])
        .lazy()
        .select(col("v").sum())
    )
    region._plan._device_observation = ArcPointer(DeviceObservation())
    for printed in ["", "1"]:
        var legacy = ScopedEnvironment("DATAFRAME_EXECUTION_REPORT", printed)
        assert_equal(region.collect(engine="cpu").item().int64(), Int64(6))
        _ = legacy^
    with open(path, "r") as source:
        assert_equal(source.read(), "")
    _ = trace^
