"""Independent batch-boundary witnesses for the scalar-ANY helper."""
from max.gpu.host import DeviceContext
from std.testing import assert_equal, assert_raises
from dataframe.accelerator.gpu.iana_scalar_scopes import (
    resources,
    reduce_scalar_any,
)


def cpu_scalar_scope_resource_geometry() raises:
    var size = resources(8193, 3)
    assert_equal(size.batches, 2)
    assert_equal(size.scratch_bytes, 8193)
    assert_equal(size.cache_bytes, 6)
    assert_equal(size.workspace_bytes, 8199)
    assert_equal(resources(8192, 1).workspace_bytes, 8193)
    assert_equal(resources(0, 128).workspace_bytes, 0)
    assert_equal(resources(16385, 0).workspace_bytes, 0)
    assert_equal(resources(16385, 2).cache_bytes, 6)
    assert_equal(resources(16385, 2).workspace_bytes, 16391)
    assert_equal(resources(6, 2, 3).workspace_bytes, 10)
    with assert_raises(contains="geometry"):
        _ = resources(-1, 1)
    with assert_raises(contains="geometry"):
        _ = resources(Int(Int32.MAX) + 1, 1)
    with assert_raises(contains="geometry"):
        _ = resources(1, 129)
    with assert_raises(contains="geometry"):
        _ = resources(1, -1)
    with assert_raises(contains="geometry"):
        _ = resources(1, 1, 0)


def gpu_scalar_scope_any_batches_dense_slots_and_tail() raises:
    var n = 16385
    var first = List[UInt8](length=n, fill=0)
    var second = List[UInt8](length=n, fill=0)
    first[8191] = 1
    first[16384] = 1
    second[8192] = 1
    var actual = List[UInt8](length=6, fill=7)
    var ctx = DeviceContext()
    var active = ctx.enqueue_create_buffer[DType.uint8](n)
    var cache = ctx.enqueue_create_buffer[DType.uint8](6)
    # Every slot is overwritten before use; no cache initialization is
    # required, and a second queued source owner survives synchronization.
    ctx.enqueue_copy(active, first.unsafe_ptr())
    ctx.enqueue_function[reduce_scalar_any](
        active.unsafe_ptr(),
        cache.unsafe_ptr(),
        UInt32(n),
        UInt32(8192),
        UInt32(3),
        UInt32(0),
        grid_dim=1,
        block_dim=256,
    )
    ctx.enqueue_copy(active, second.unsafe_ptr())
    ctx.enqueue_function[reduce_scalar_any](
        active.unsafe_ptr(),
        cache.unsafe_ptr(),
        UInt32(n),
        UInt32(8192),
        UInt32(3),
        UInt32(1),
        grid_dim=1,
        block_dim=256,
    )
    ctx.enqueue_copy(actual.unsafe_ptr(), cache)
    ctx.synchronize()
    _ = first^
    _ = second^
    var expected: List[UInt8] = [1, 0, 1, 0, 1, 0]
    assert_equal(actual, expected)


def gpu_scalar_scope_all_false_overwrites_prior_true() raises:
    var first = List[UInt8](length=8193, fill=1)
    var second = List[UInt8](length=8193, fill=0)
    var actual = List[UInt8](length=2, fill=7)
    var ctx = DeviceContext()
    var active = ctx.enqueue_create_buffer[DType.uint8](8193)
    var cache = ctx.enqueue_create_buffer[DType.uint8](2)
    ctx.enqueue_copy(active, first.unsafe_ptr())
    ctx.enqueue_function[reduce_scalar_any](
        active.unsafe_ptr(),
        cache.unsafe_ptr(),
        UInt32(8193),
        UInt32(8192),
        UInt32(2),
        UInt32(0),
        grid_dim=1,
        block_dim=256,
    )
    ctx.enqueue_copy(active, second.unsafe_ptr())
    ctx.enqueue_function[reduce_scalar_any](
        active.unsafe_ptr(),
        cache.unsafe_ptr(),
        UInt32(8193),
        UInt32(8192),
        UInt32(2),
        UInt32(0),
        grid_dim=1,
        block_dim=256,
    )
    ctx.enqueue_copy(actual.unsafe_ptr(), cache)
    ctx.synchronize()
    _ = first^
    _ = second^
    var expected: List[UInt8] = [0, 0]
    assert_equal(actual, expected)
