"""Observed device launches, CPU fallbacks and host phase timings."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true
from std.memory import ArcPointer
from dataframe import Column, DataFrame, Expr, Series, col
from dataframe.execution_observation import DeviceObservation
from dataframe_metal.logical_reduce import (
    plan_packed_logical,
    execute_packed_logical,
)
from tests.support.metal_outcomes import require_metal


def _device_row(report: DataFrame, status: String) raises -> Int:
    var found = -1
    for row in range(report.height()):
        if report.item(row, "device_status").string() == status:
            assert_equal(found, -1)
            found = row
    assert_true(found >= 0, String(report))
    assert_equal(
        report.item(found, "pipeline").int64(),
        report.item(found, "node").int64(),
    )
    return found


# requires: metal
def test_metal_reports_completed_launches_and_independent_queries() raises:
    var runtime = require_metal()
    var query = (
        DataFrame([Series("v", Column[Int64]([1, 2, 3, 4]))])
        .lazy()
        .select(col("v").sum())
    )
    var previous_id = Int64(-1)
    var previous_launches = Int64(-1)
    for _ in range(2):
        var observed = query.profile(accelerator=runtime)
        assert_equal(observed[0].item().int64(), Int64(10))
        ref report = observed[1]
        var row = _device_row(report, "completed")
        var launches = report.item(row, "gpu_launches").int64()
        assert_true(launches > 0)
        assert_equal(
            report.item(row, "gpu_completed_launches").int64(), launches
        )
        assert_equal(report.item(row, "result_device").string(), "metal")
        assert_equal(
            report.item(row, "phase_timing_kind").string(), "host_wall"
        )
        assert_equal(report.item(row, "fallback_reason").string(), "")
        for phase in [
            "context_wall_ms",
            "input_wall_ms",
            "device_work_wall_ms",
            "finish_wall_ms",
            "synchronize_wall_ms",
        ]:
            assert_true(report.item(row, phase).float64() >= 0)
        var id = report.item(row, "query_id").int64()
        assert_true(id != previous_id)
        if previous_launches >= 0:
            assert_equal(launches, previous_launches)
        previous_id = id
        previous_launches = launches


# requires: metal
def test_metal_empty_copied_logical_inputs_report_no_launches() raises:
    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series("b", Column[Bool](List[Bool]())),
            Series("n", Column[Int32](List[Int32]())),
        ]
    )
    var outputs: List[Expr] = [
        col("b").any().alias("any"),
        col("b").all().alias("all"),
        col("n").count().alias("count"),
        col("n").len().alias("len"),
    ]
    var query = frame.lazy().select_exprs(outputs)
    var expected = DataFrame(
        [
            Series("any", Column[Bool]([False])),
            Series("all", Column[Bool]([True])),
            Series("count", Column[Int64]([0])),
            Series("len", Column[Int64]([0])),
        ]
    )
    for zero_copy in [False, True]:
        var observation = ArcPointer(DeviceObservation())
        query._plan._device_observation = observation
        var result = execute_packed_logical(
            query, plan_packed_logical(query).value(), zero_copy
        )
        assert_true(result[0].equals(expected))
        assert_equal(observation[].launches.load(), Int64(0))
        assert_equal(observation[].completed_launches.load(), Int64(0))
    var observed = query.profile(accelerator=runtime)
    assert_true(observed[0].equals(expected))
    var row = _device_row(observed[1], "no_device_work")
    assert_equal(observed[1].item(row, "gpu_launches").int64(), Int64(0))
    assert_equal(observed[1].item(row, "result_device").string(), "host")


# requires: metal
def test_metal_reports_unsupported_cpu_and_selected_cpu_producers() raises:
    var runtime = require_metal()
    var unsupported = (
        DataFrame([Series("v", Column[Float64]([1, 2, 3]))])
        .lazy()
        .select(col("v").sum())
    )
    var cpu = unsupported.profile(accelerator=runtime)
    var row = _device_row(cpu[1], "cpu_fallback")
    assert_equal(cpu[0].item().float64(), Float64(6))
    assert_equal(
        cpu[1].item(row, "fallback_reason").string(),
        "unsupported_operator_or_dtype",
    )
    assert_equal(cpu[1].item(row, "gpu_launches").int64(), Int64(0))
    assert_equal(cpu[1].item(row, "phase_timing_kind").string(), "unavailable")
    var frame = DataFrame(
        [
            Series("label", Column[String](["keep", "drop", "keep"])),
            Series("v", Column[Int64]([1, 2, 3])),
        ]
    )
    for keep in ["keep", "missing"]:
        var query = (
            frame.lazy().filter(col("label") == keep).select(col("v").sum())
        )
        var observed = query.profile(accelerator=runtime, optimize=False)
        assert_equal(
            observed[0].item().int64(), Int64(4 if keep == "keep" else 0)
        )
        ref report = observed[1]
        var selected = _device_row(
            report, "completed" if keep == "keep" else "no_device_work"
        )
        assert_equal(
            report.item(selected, "algorithm").string(), "handoff selections"
        )
        var producer = False
        for r in range(report.height()):
            if report.item(r, "operator").string().startswith("FILTER"):
                producer = True
                assert_equal(report.item(r, "result_device").string(), "cpu")
                assert_equal(report.item(r, "gpu_launches").int64(), Int64(0))
        assert_true(producer)


# requires: metal
def test_metal_adjudication_preserves_discarded_gpu_attempt() raises:
    var runtime = require_metal()
    var query = (
        DataFrame([Series("v", Column[Float32]([Float32.MAX_FINITE]))])
        .lazy()
        .select((col("v") * 2).sum())
    )
    var observed = query.profile(accelerator=runtime)
    assert_true(observed[0].equals(query.collect(engine="cpu")))
    ref report = observed[1]
    var discarded_gpu = False
    var cpu_result = False
    for row in range(report.height()):
        var launches = report.item(row, "gpu_launches").int64()
        if launches > 0:
            discarded_gpu = True
            assert_true(not report.item(row, "result_used").bool())
            assert_equal(
                report.item(row, "execution_attempt").int64(), Int64(0)
            )
            assert_equal(
                report.item(row, "gpu_completed_launches").int64(), launches
            )
        if report.item(row, "execution_attempt").int64() == 1:
            cpu_result = True
            assert_true(report.item(row, "result_used").bool())
            assert_equal(report.item(row, "result_device").string(), "cpu")
            assert_equal(launches, Int64(0))
            assert_equal(
                report.item(row, "fallback_reason").string(),
                "data_fault_adjudication",
            )
    assert_true(discarded_gpu)
    assert_true(cpu_result)
