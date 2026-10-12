"""Complete typed expectations and explicit CPU/native execution boundaries."""
from std.testing import assert_true, assert_equal
from dataframe import DataFrame
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from tests.support.coverage_compare import assert_value
from tests.execution_modes.test_backend_metal import assert_native_report
from tests.support.lifecycle_jobs import WorkerSetting


def assert_domain(actual: DataFrame, expected: DataFrame) raises:
    assert_equal(actual.height(), expected.height())
    assert_equal(actual.columns(), expected.columns())
    for name in expected.columns():
        assert_equal(actual.column(name).dtype(), expected.column(name).dtype())
        for row in range(expected.height()):
            try:
                assert_value(actual.item(row, name), expected.item(row, name))
            except error:
                raise Error(
                    "Metal domain expectation "
                    + name
                    + " row "
                    + String(row)
                    + ": "
                    + String(error)
                )


def check_domain(
    plan: LazyFrame,
    expected: DataFrame,
    runtime: MetalRuntime,
    native: Bool,
    submitted: Bool = True,
    resident: Bool = False,
    dictionary: Bool = False,
    mechanism: String = "",
    fused: Bool = False,
) raises:
    if native:
        var measured = plan.profile(accelerator=runtime, engine="accel")
        assert_domain(measured[0], expected)
        assert_native_report(measured[1], submitted)
        if mechanism:
            var found = False
            var expected_mechanism = String(
                mechanism[byte=1:]
            ) if mechanism.startswith("!") else mechanism.copy()
            for row in range(measured[1].height()):
                found = (
                    found
                    or measured[1].item(row, "algorithm").string()
                    == expected_mechanism
                )
            assert_equal(
                found,
                not mechanism.startswith("!"),
                "native mechanism " + mechanism + " found=" + String(found),
            )
        if resident:
            var input_bytes = Int64(0)
            var output_bytes = Int64(0)
            for row in range(measured[1].height()):
                input_bytes += (
                    measured[1].item(row, "resident_input_bytes").int64()
                )
                output_bytes += (
                    measured[1].item(row, "resident_output_bytes").int64()
                )
            assert_true(
                input_bytes > 0, "consumer must read resident native data"
            )
            assert_true(output_bytes > 0, "producer must retain native data")
        if fused:
            # One row describes an entire native region. Resident byte counters
            # measure edges between separate regions, not GPU-local operators
            # fused into this region; only its final result crosses to the CPU.
            assert_equal(measured[1].height(), 1)
            assert_equal(
                measured[1].item(0, "resident_input_bytes").int64(), Int64(0)
            )
            assert_equal(
                measured[1].item(0, "resident_output_bytes").int64(), Int64(0)
            )
            assert_true(measured[1].item(0, "upload_bytes").int64() > 0)
            assert_true(measured[1].item(0, "download_bytes").int64() > 0)
        if dictionary:
            var staging = Int64(0)
            for row in range(measured[1].height()):
                staging += (
                    measured[1].item(row, "dictionary_staging_bytes").int64()
                )
                var encode = (
                    measured[1].item(row, "dictionary_encode_ms").float64()
                )
                var decode = (
                    measured[1].item(row, "dictionary_decode_ms").float64()
                )
                assert_true(
                    measured[1].item(row, "query_wall_ms").float64()
                    >= encode + decode
                )
            assert_true(
                staging > 0,
                "text query must report real host dictionary staging",
            )
    # These CPU fixtures deliberately use one worker: public streamed global
    # positional reductions currently lose source order across worker states.
    # That scheduling-sensitive regression remains a separate audit obligation.
    var cpu_workers = WorkerSetting(1)
    for optimize in [False, True]:
        for streaming in [False, True]:
            for batch in [7, 65]:
                var actual = plan.collect(
                    engine="cpu",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=batch,
                )
                try:
                    assert_domain(actual, expected)
                except error:
                    print("Expected complete typed model:", expected)
                    print("Actual CPU result:", actual)
                    raise Error(
                        "CPU optimize="
                        + String(optimize)
                        + " streaming="
                        + String(streaming)
                        + " batch="
                        + String(batch)
                        + ": "
                        + String(error)
                    )
    _ = cpu_workers^
