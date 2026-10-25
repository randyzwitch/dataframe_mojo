"""The Metal runtime while its execution core is rebuilt: settings forwarding,
an explicit unsupported answer for every plan, and CPU placement."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col
from dataframe_metal.metal import MetalRuntime, UNSUPPORTED


def test_metal_runtime_forwards_query_execution_settings() raises:
    var frame = DataFrame([Series("x", Column[Int64]([1, 2, 3]))])
    var query = frame.lazy().select(col("x").sum())
    var copied = query._with_accelerator_settings(False, False, 2).copy()
    var original_runtime = MetalRuntime(device_id=999999)
    var configured_runtime = original_runtime._configured(copied)
    assert_false(configured_runtime._execution_settings.optimize)
    assert_false(configured_runtime._execution_settings.streaming)
    assert_equal(configured_runtime._execution_settings.batch_size, 2)
    assert_true(original_runtime._execution_settings.optimize)
    assert_equal(original_runtime._execution_settings.batch_size, 65536)


def test_metal_runtime_reports_every_plan_unsupported_and_places_on_cpu() raises:
    var frame = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 1])),
            Series("x", Column[Int32]([4, 5, 6])),
        ]
    )
    var plans: List[LazyFrame] = [
        frame.lazy().select(col("x").sum()),
        frame.lazy().filter(col("x") > 4),
        frame.lazy().group_by("k").agg([col("x").sum()]),
    ]
    var runtime = MetalRuntime(device_id=999999)
    for plan in plans:
        assert_true(UNSUPPORTED in runtime.describe(plan))
        assert_equal(runtime.select_auto(plan, True, True, 65536)[0], "cpu")
        with assert_raises(contains="unsupported"):
            _ = runtime.estimate_auto(plan)
        assert_true(
            plan.collect(engine="auto", accelerator=runtime).equals(
                plan.collect(engine="cpu")
            )
        )
