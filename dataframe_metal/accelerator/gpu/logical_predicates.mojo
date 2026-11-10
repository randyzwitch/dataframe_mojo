"""Selection/true/false ballots for native logical and count reductions."""
from std.memory import Pointer
from std.bit import pop_count
from max.gpu import thread_idx, block_idx
from max.gpu.primitives import warp, block
from .multi_reduce import Bytes
from .filter_sum import BLOCK_SIZE
from .compound_predicate import predicate_matches


def predicate_logical_partials(
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    v4: Bytes,
    v5: Bytes,
    v6: Bytes,
    v7: Bytes,
    v8: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    b4: Bytes,
    b5: Bytes,
    b6: Bytes,
    b7: Bytes,
    b8: Bytes,
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    rows: UInt32,
    stride: UInt32,
    partial_offset: UInt32,
    output: Pointer[UInt32, MutUntrackedOrigin],
):
    var fields = Int(metadata[unsafe_offset=0])
    var select = metadata.unsafe_offset(Int(metadata[unsafe_offset=1]))
    var lane = UInt32(thread_idx.x) & 31
    for field in range(fields):
        var positive = metadata.unsafe_offset(
            Int(metadata[unsafe_offset=2 + field * 2])
        )
        var negative = metadata.unsafe_offset(
            Int(metadata[unsafe_offset=3 + field * 2])
        )
        var kept_count = UInt32(0)
        var true_count = UInt32(0)
        var false_count = UInt32(0)
        # Every SIMD-group lane participates, including masked tail lanes.
        var base = UInt32(block_idx.x * BLOCK_SIZE + thread_idx.x) & ~UInt32(31)
        while base < rows:
            var row = base + lane
            var kept = False
            var t = False
            var f = False
            if row < rows:
                kept = predicate_matches(
                    select,
                    0,
                    Int(row),
                    v0,
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7,
                    v8,
                    b0,
                    b1,
                    b2,
                    b3,
                    b4,
                    b5,
                    b6,
                    b7,
                    b8,
                )
                if kept:
                    t = predicate_matches(
                        positive,
                        0,
                        Int(row),
                        v0,
                        v1,
                        v2,
                        v3,
                        v4,
                        v5,
                        v6,
                        v7,
                        v8,
                        b0,
                        b1,
                        b2,
                        b3,
                        b4,
                        b5,
                        b6,
                        b7,
                        b8,
                    )
                    f = predicate_matches(
                        negative,
                        0,
                        Int(row),
                        v0,
                        v1,
                        v2,
                        v3,
                        v4,
                        v5,
                        v6,
                        v7,
                        v8,
                        b0,
                        b1,
                        b2,
                        b3,
                        b4,
                        b5,
                        b6,
                        b7,
                        b8,
                    )
            var km = warp.vote[DType.uint32](kept)
            var tm = warp.vote[DType.uint32](t)
            var fm = warp.vote[DType.uint32](f)
            if lane == 0:
                kept_count += pop_count(km)
                true_count += pop_count(tm)
                false_count += pop_count(fm)
            base += stride
        var nk = block.sum[block_size=BLOCK_SIZE, broadcast=False](kept_count)
        var nt = block.sum[block_size=BLOCK_SIZE, broadcast=False](true_count)
        var nf = block.sum[block_size=BLOCK_SIZE, broadcast=False](false_count)
        if thread_idx.x == 0:
            var at = Int(partial_offset) + block_idx.x * (1 + fields * 2)
            if field == 0:
                output[unsafe_offset=at] = nk
            output[unsafe_offset=at + 1 + field * 2] = nt
            output[unsafe_offset=at + 2 + field * 2] = nf
