"""Read complete Apple calibration metadata without constructing GPU contexts."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe._build_identity import CPU_BUILD_ID
from dataframe.accelerator.gpu.identity import (
    available,
    hardware,
    key,
    runtime_build_id,
)


def main() raises:
    assert_true(available())
    var facts = hardware()
    assert_true(facts.total_bytes > 0)
    assert_true(facts.available_bytes > 0)
    assert_true(facts.available_bytes <= facts.total_bytes)
    assert_true(facts.device.startswith("mojo-metal;ordinal=0;"))
    assert_true("chip=Apple" in facts.device)
    assert_true("unified=1" in facts.device)
    assert_true("model=" in facts.host)
    assert_true("os=" in facts.host)
    assert_true("lowPower=" in facts.host and "thermal=" in facts.host)
    var first = key(0, True, True, 65_536)
    var second = key(0, False, False, 7)
    assert_equal(first.provider, "metal")
    assert_equal(first.device, second.device)
    assert_equal(first.host, second.host)
    assert_equal(first.runtime_build_id, runtime_build_id())
    assert_equal(first.cpu_build_id, CPU_BUILD_ID)
    assert_true(first.optimize and first.streaming)
    assert_equal(first.batch_size, 65_536)
    assert_true(not second.optimize and not second.streaming)
    assert_equal(second.batch_size, 7)
    for i in range(64):
        var repeated = key(0, i % 2 == 0, i % 3 == 0, i + 1)
        assert_equal(repeated.device, first.device)
        assert_equal(repeated.runtime_build_id, first.runtime_build_id)
        assert_equal(repeated.batch_size, i + 1)
        assert_true(
            "lowPower=" in repeated.host and "thermal=" in repeated.host
        )
    with assert_raises(contains="device 0"):
        _ = key(1, True, True, 1)
    with assert_raises(contains="batch_size"):
        _ = key(0, True, True, 0)
    print("DEVICE", facts.device)
    print("HOST", facts.host)
    print("MEMORY", facts.available_bytes, facts.total_bytes)
    print("BUILD", first.runtime_build_id)
    print(
        "PASS context-free Mojo Metal identity, 64 repeated keys, settings and host memory"
    )
