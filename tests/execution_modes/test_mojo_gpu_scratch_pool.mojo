"""The device service's scratch limits are metadata until a query runs."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_raises
from dataframe_metal.accelerator.gpu.query_scope import DeviceService


def test_mojo_gpu_scratch_limits_are_metadata_only() raises:
    var service = DeviceService(idle_scratch_bytes=24, idle_scratch_entries=1)
    assert_equal(service.cached_generations(), 0)
    assert_equal(service.scratch_info(), (0, 0, 0, 0))
    with assert_raises(contains="idle limit"):
        _ = DeviceService(idle_scratch_bytes=-1)
    with assert_raises(contains="idle limit"):
        _ = DeviceService(idle_scratch_entries=-1)
