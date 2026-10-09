"""Backend policy stays above CPU scheduling and rejects unavailable engines."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit, scan_csv


def test_cpu_and_auto_preserve_query_results() raises:
    var frame = DataFrame(
        [Series("x", Column[Float32]([-2, 1, 9, 3], [True, True, False, True]))]
    )
    var plan = (
        frame.lazy()
        .filter(col("x") > lit(Float32(0)))
        .select((col("x") * lit(Float32(1.25))).sum().alias("total"))
    )
    var expected = plan.collect()
    assert_equal(expected.item(0, "total").float32(), Float32(5.0))
    for engine in ["cpu", "auto"]:
        for streaming in [False, True]:
            for optimize in [False, True]:
                assert_true(
                    plan.collect(
                        engine=engine, streaming=streaming, optimize=optimize
                    ).equals(expected)
                )
        assert_true(plan.profile(engine=engine)[0].equals(expected))
        assert_true(plan.fetch(1, engine=engine).equals(expected))
    assert_equal(plan.explain(), plan.explain(engine="cpu"))
    assert_true("ENGINE cpu: auto uses CPU" in plan.explain(engine="auto"))


def test_unavailable_backend_rejected_before_reading_source() raises:
    var plan = scan_csv("/nonexistent/dataframe_backend_selection/input.csv")
    with assert_raises(contains="Accelerator execution is not implemented"):
        _ = plan.collect(engine="accel")
    with assert_raises(contains="Accelerator execution is not implemented"):
        _ = plan.profile(engine="accel")
    with assert_raises(contains="Accelerator execution is not implemented"):
        _ = plan.fetch(engine="accel")
    var description = plan.explain(engine="accel")
    assert_true("ENGINE accel:" in description)
    assert_true("Accelerator execution is not implemented" in description)


def test_invalid_engine_is_not_silently_ignored() raises:
    var plan = DataFrame([Series("x", Column[Int64]([1]))]).lazy()
    for engine in ["", "gpu", "nvidia", "CPU", "apple", "typo"]:
        with assert_raises(contains="Unknown engine"):
            _ = plan.collect(engine=engine)
        with assert_raises(contains="Unknown engine"):
            _ = plan.profile(engine=engine)
        with assert_raises(contains="Unknown engine"):
            _ = plan.fetch(engine=engine)
        with assert_raises(contains="Unknown engine"):
            _ = plan.explain(engine=engine)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
