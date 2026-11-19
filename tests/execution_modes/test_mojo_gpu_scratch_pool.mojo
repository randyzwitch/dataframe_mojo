"""The device service's scratch limits are metadata until a query runs."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_raises, assert_true
from dataframe_metal.accelerator.gpu.query_scope import DeviceService
from dataframe_metal.accelerator.gpu.scratch_pool import ScratchPool
from dataframe_metal.accelerator.gpu.target import METAL_TARGET, MetalContext
from tests.support.metal_outcomes import require_metal


def test_mojo_gpu_scratch_limits_are_metadata_only() raises:
    var service = DeviceService(idle_scratch_bytes=24, idle_scratch_entries=1)
    assert_equal(service.cached_generations(), 0)
    assert_equal(service.scratch_info(), (0, 0, 0, 0))
    with assert_raises(contains="idle limit"):
        _ = DeviceService(idle_scratch_bytes=-1)
    with assert_raises(contains="idle limit"):
        _ = DeviceService(idle_scratch_entries=-1)


# requires: metal
def test_mojo_gpu_scratch_reuse_checks_runtime_context_identity() raises:
    _ = require_metal()
    comptime if METAL_TARGET:
        var owner = MetalContext()
        var other = MetalContext()
        var pool = ScratchPool(1024, 4)
        comptime for i in range(2):
            comptime dtype = DType.int64 if i == 0 else DType.uint32
            var buffer = owner.enqueue_create_buffer[dtype](8)
            owner.synchronize()
            with assert_raises(contains="context mismatch"):
                pool.put[dtype](other, buffer.copy())
            assert_equal(pool.idle_entries, 0)
            pool.put[dtype](owner, buffer.copy())
            assert_equal(pool.idle_entries, 1)
            with assert_raises(contains="context mismatch"):
                _ = pool.take[dtype](other, 8)
            assert_equal(pool.idle_entries, 1)
            var reused = pool.take[dtype](owner, 8)
            assert_true(Bool(reused))
            assert_equal(len(reused.value()), 8)
            assert_true(
                reused.value().context().into_unknown_target()
                == owner.into_unknown_target()
            )
            assert_equal(pool.idle_entries, 0)
            assert_equal(pool.idle_bytes, 0)
        assert_equal(pool.hits, 2)
