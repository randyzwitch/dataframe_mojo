"""Resident row partition construction: sorted grouping versus equality hashing.

Warm primitive timings exclude allocation, upload and index readback. Every
route verifies complete row membership, uniqueness and key equality. These
are partition timings, not complete query or CPU/GPU speedups.
"""
from std.testing import assert_equal, assert_true
from std.time import perf_counter_ns
from dataframe_metal.accelerator.gpu.residency import Scope
from dataframe_metal.accelerator.gpu import hash_grouping as h
from dataframe_metal.accelerator.gpu.ordering import order_init, order_merge
from dataframe_metal.accelerator.gpu.grouping import (
    group_mark,
    group_offsets,
    group_starts,
)
from dataframe_metal.accelerator.rows import ROW_GROUP_HEADER, ROW_GROUP_KEY


def main() raises:
    for n in [257, 513, 1000003]:
        var dtype = Int64(1) if n == 513 else Int64(3)
        var groups = 17 if n == 513 else min(n, 100003)
        var scope = Scope(0, -1)
        var ctx = scope.context()
        var words = List[UInt64](length=n * 2, fill=0)
        var valid = List[UInt8](length=n, fill=1)
        var errors = List[UInt32](length=n, fill=0)
        for row in range(n):
            var key = UInt64((row * 7919) % groups)
            # Float32 equality ignores upper-word padding. Exercise distinct
            # storage words for equal finite keys on the actual device.
            words[row] = (UInt64(0x3F800000) + key) | (
                UInt64(row % 13) << 32
            ) if dtype == 1 else key
        var code: List[Int64] = [
            ROW_GROUP_HEADER,
            1,
            0,
            0,
            ROW_GROUP_KEY,
            0,
            0,
            0,
        ]
        var types: List[Int64] = [3, dtype]
        var dw = ctx.enqueue_create_buffer[DType.uint64](n * 2)
        var dv = ctx.enqueue_create_buffer[DType.uint8](n)
        var de = ctx.enqueue_create_buffer[DType.uint32](n)
        var dc = ctx.enqueue_create_buffer[DType.int64](8)
        var dt = ctx.enqueue_create_buffer[DType.int64](2)
        var dl = ctx.enqueue_create_buffer[DType.uint64](2)
        var index = ctx.enqueue_create_buffer[DType.uint32](n)
        var spare = ctx.enqueue_create_buffer[DType.uint32](n)
        var buckets = ctx.enqueue_create_buffer[DType.uint32](n)
        var starts = ctx.enqueue_create_buffer[DType.uint32](n)
        var metadata = ctx.enqueue_create_buffer[DType.uint32](4)
        var blocks = (n + 255) // 256
        var offsets = ctx.enqueue_create_buffer[DType.uint32](blocks * 2)
        var capacity = h.capacity(n)
        var reps = ctx.enqueue_create_buffer[DType.uint32](capacity)
        var counts = ctx.enqueue_create_buffer[DType.uint32](capacity)
        ctx.enqueue_copy(dw, words.unsafe_ptr())
        ctx.enqueue_copy(dv, valid.unsafe_ptr())
        ctx.enqueue_copy(de, errors.unsafe_ptr())
        ctx.enqueue_copy(dc, code.unsafe_ptr())
        ctx.enqueue_copy(dt, types.unsafe_ptr())
        ctx.enqueue_copy(dl, words.unsafe_ptr())
        scope.synchronize()
        for repeat in range(4):
            for hashed in [False, True]:
                var began = Int(perf_counter_ns())
                if hashed:
                    ctx.enqueue_function[h.initialize](
                        reps.unsafe_ptr(),
                        counts.unsafe_ptr(),
                        metadata.unsafe_ptr(),
                        UInt32(capacity),
                        grid_dim=(capacity + 255) // 256,
                        block_dim=256,
                    )
                    ctx.enqueue_function[h.build](
                        dw.unsafe_ptr(),
                        dv.unsafe_ptr(),
                        de.unsafe_ptr(),
                        dc.unsafe_ptr(),
                        dt.unsafe_ptr(),
                        reps.unsafe_ptr(),
                        counts.unsafe_ptr(),
                        buckets.unsafe_ptr(),
                        metadata.unsafe_ptr(),
                        UInt32(0),
                        UInt32(1),
                        UInt32(n),
                        UInt32(n),
                        UInt32(1),
                        UInt32(capacity),
                        grid_dim=blocks,
                        block_dim=256,
                    )
                    ctx.enqueue_function[h.offsets_local](
                        counts.unsafe_ptr(),
                        reps.unsafe_ptr(),
                        spare.unsafe_ptr(),
                        metadata.unsafe_ptr(),
                        UInt32(capacity),
                        grid_dim=(capacity + 255) // 256,
                        block_dim=256,
                    )
                    ctx.enqueue_function[h.offsets_blocks](
                        spare.unsafe_ptr(),
                        metadata.unsafe_ptr(),
                        UInt32((capacity + 255) // 256),
                        grid_dim=1,
                        block_dim=1,
                    )
                    ctx.enqueue_function[h.offsets_finish](
                        counts.unsafe_ptr(),
                        reps.unsafe_ptr(),
                        spare.unsafe_ptr(),
                        starts.unsafe_ptr(),
                        metadata.unsafe_ptr(),
                        UInt32(capacity),
                        grid_dim=(capacity + 255) // 256,
                        block_dim=256,
                    )
                    ctx.enqueue_function[h.scatter](
                        buckets.unsafe_ptr(),
                        counts.unsafe_ptr(),
                        index.unsafe_ptr(),
                        metadata.unsafe_ptr(),
                        UInt32(n),
                        grid_dim=blocks,
                        block_dim=256,
                    )
                else:
                    ctx.enqueue_function[order_init](
                        index.unsafe_ptr(),
                        UInt32(n),
                        grid_dim=blocks,
                        block_dim=256,
                    )
                    var width = 1
                    while width < n:
                        ctx.enqueue_function[order_merge](
                            dw.unsafe_ptr(),
                            dv.unsafe_ptr(),
                            index.unsafe_ptr(),
                            spare.unsafe_ptr(),
                            dc.unsafe_ptr(),
                            dt.unsafe_ptr(),
                            UInt32(1),
                            UInt32(1),
                            UInt32(n),
                            UInt32(n),
                            UInt32(width),
                            UInt32(1),
                            dl.unsafe_ptr(),
                            grid_dim=blocks,
                            block_dim=256,
                        )
                        var old = index^
                        index = spare^
                        spare = old^
                        width *= 2
                    ctx.enqueue_function[group_mark](
                        dw.unsafe_ptr(),
                        dv.unsafe_ptr(),
                        de.unsafe_ptr(),
                        index.unsafe_ptr(),
                        buckets.unsafe_ptr(),
                        offsets.unsafe_ptr(),
                        dc.unsafe_ptr(),
                        dt.unsafe_ptr(),
                        UInt32(0),
                        UInt32(1),
                        UInt32(n),
                        UInt32(n),
                        UInt32(1),
                        grid_dim=blocks,
                        block_dim=256,
                    )
                    ctx.enqueue_function[group_offsets](
                        offsets.unsafe_ptr(),
                        metadata.unsafe_ptr(),
                        UInt32(blocks),
                        grid_dim=1,
                        block_dim=1,
                    )
                    ctx.enqueue_function[group_starts](
                        dw.unsafe_ptr(),
                        dv.unsafe_ptr(),
                        index.unsafe_ptr(),
                        buckets.unsafe_ptr(),
                        offsets.unsafe_ptr(),
                        starts.unsafe_ptr(),
                        dc.unsafe_ptr(),
                        dt.unsafe_ptr(),
                        UInt32(0),
                        UInt32(1),
                        UInt32(n),
                        UInt32(n),
                        UInt32(1),
                        grid_dim=blocks,
                        block_dim=256,
                    )
                scope.synchronize()
                var elapsed = Int(perf_counter_ns()) - began
                var shape = scope.read_control(metadata)
                assert_equal(shape[0], UInt32(groups))
                assert_equal(shape[1], UInt32(0))
                assert_equal(shape[2], UInt32(0))
                var indices = scope.read_control(index)
                var beginnings = scope.read_control(starts)
                var seen = List[Bool](length=n, fill=False)
                for group in range(groups):
                    var begin = Int(beginnings[group])
                    var end = (
                        Int(beginnings[group + 1]) if group + 1 < groups else n
                    )
                    var word = words[Int(indices[begin])]
                    for at in range(begin, end):
                        var row = Int(indices[at])
                        assert_true(not seen[row])
                        seen[row] = True
                        assert_equal(
                            UInt64(UInt32(words[row])) if dtype
                            == 1 else words[row],
                            UInt64(UInt32(word)) if dtype == 1 else word,
                        )
                for row in range(n):
                    assert_true(seen[row])
                print(
                    "rows", n, "hashed", hashed, "round", repeat, "ns", elapsed
                )
