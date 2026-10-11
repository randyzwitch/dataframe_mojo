"""Provider-neutral package checks using deliberately synthetic cost fixtures."""
from std.sys import argv
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit
from dataframe._accel_provider import (
    calibration_key,
    calibration_runtime,
    estimate_auto,
    select_auto,
)
from dataframe.json_value import json_quote


def main() raises:
    var args = argv()
    var mode = String(args[1])
    var frame = DataFrame(
        [Series("x", Column[Float32]([2, 1, 2, -1], [True, True, False, True]))]
    )
    var plan = frame.lazy().select(
        ((col("x") + lit(Float32(1))) * lit(Float32(2))).alias("y")
    )
    if mode == "fixture":
        var key = calibration_key(True, True, 65536)
        var facts = estimate_auto(plan)
        print(
            '{"provider":'
            + json_quote(key.provider)
            + ',"device":'
            + json_quote(key.device)
            + ',"runtime_build_id":'
            + json_quote(key.runtime_build_id)
            + ',"cpu_build_id":'
            + json_quote(key.cpu_build_id)
            + ',"host":'
            + json_quote(key.host)
            + ',"threads":'
            + String(key.threads)
            + ',"optimize":true,"streaming":true,"batch_size":65536,"family":'
            + json_quote(facts.family)
            + "}"
        )
        return
    var expected = plan.collect(engine="cpu")
    var selected = select_auto(plan, True, True, 65536)
    var gpu = mode == "gpu"
    assert_equal(selected[0], "accel" if gpu else "cpu")
    assert_true(plan.collect(engine="auto").equals(expected))
    var measured = plan.profile(engine="auto")
    assert_true(measured[0].equals(expected))
    assert_true("selection_reason" in measured[1].columns())
    assert_true(
        ("ENGINE accel:" if gpu else "ENGINE cpu:")
        in plan.explain(engine="auto")
    )
    if gpu:
        var key = calibration_key(True, True, 65536)
        assert_equal(measured[1].item(0, "executor").string(), key.provider)
        var runtime = calibration_runtime()
        var reused = plan.profile(engine="auto", accelerator=runtime)
        assert_true(reused[0].equals(expected))
        assert_equal(reused[1].item(0, "executor").string(), key.provider)
        var bad = (
            DataFrame([Series("x", Column[Float32]([300]))])
            .lazy()
            .select(col("x").cast("int8"))
        )
        assert_equal(select_auto(bad, True, True, 65536)[0], "accel")
        with assert_raises(
            contains="strict cast failed" if key.provider
            == "metal" else "cast from float32"
        ):
            _ = bad.collect(engine="auto")
        assert_true(plan.collect(engine="auto").equals(expected))
    else:
        assert_true(measured[1].item(0, "executor").string() != "metal")
        assert_true(measured[1].item(0, "executor").string() != "nvidia")
    assert_true(
        frame.equals(
            DataFrame(
                [
                    Series(
                        "x",
                        Column[Float32](
                            [2, 1, 2, -1], [True, True, False, True]
                        ),
                    )
                ]
            )
        )
    )
    print("Shared packaged backend checks passed: " + mode)
