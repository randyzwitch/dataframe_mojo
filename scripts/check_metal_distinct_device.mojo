"""Actual Metal distinct equivalence checked against independent host equality."""
from std.memory import Pointer, bitcast
from std.math import isnan
from std.testing import assert_equal
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from dataframe_metal.accelerator.gpu.distinct_value import (
    distinct_same,
    distinct_hash,
)


def _distinct_probe(
    left: Pointer[UInt64, MutUntrackedOrigin],
    right: Pointer[UInt64, MutUntrackedOrigin],
    output: Pointer[UInt32, MutUntrackedOrigin],
    rows: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var a = left[unsafe_offset=row]
    var b = right[unsafe_offset=row]
    for tag in range(4):
        for flags in range(4):
            var av = Bool(flags & 1)
            var bv = Bool(flags & 2)
            var equal = distinct_same(a, av, b, bv, UInt32(tag))
            var consistent = not equal or distinct_hash(
                a, av, UInt32(tag), 127
            ) == distinct_hash(b, bv, UInt32(tag), 127)
            output[unsafe_offset=row * 32 + tag * 8 + flags * 2] = UInt32(equal)
            output[unsafe_offset=row * 32 + tag * 8 + flags * 2 + 1] = UInt32(
                consistent
            )


def main() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.accelerator.gpu.query_scope import Scope

    _ = require_metal()
    var edges: List[UInt64] = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x3F800000,
        0xBF800000,
        0x7F800000,
        0xFF800000,
        0x7F800001,
        0x7FC12345,
        0xFFC54321,
        0x8000000000000000,
        0x7FFFFFFFFFFFFFFF,
        UInt64(1 << 60),
        UInt64(1 << 60) + 1,
        UInt64.MAX,
    ]
    var left = List[UInt64]()
    var right = List[UInt64]()
    var expected = List[UInt32]()
    var state = UInt64(41378391)
    for i in range(len(edges) * len(edges) + 4096):
        state = state * 6364136223846793005 + 1442695040888963407
        var a = state
        state = state * 6364136223846793005 + 1442695040888963407
        var b = state
        if i < len(edges) * len(edges):
            a = edges[i // len(edges)]
            b = edges[i % len(edges)]
        left.append(a)
        right.append(b)
        for tag in range(4):
            for flags in range(4):
                var av = Bool(flags & 1)
                var bv = Bool(flags & 2)
                var equal = av == bv
                if av and bv:
                    if tag == 2:
                        var x = bitcast[DType.float32](UInt32(a))
                        var y = bitcast[DType.float32](UInt32(b))
                        equal = x == y or (isnan(x) and isnan(y))
                    elif tag == 3:
                        equal = Bool(a) == Bool(b)
                    elif tag == 0:
                        equal = UInt32(a) == UInt32(b)
                    else:
                        equal = a == b
                elif av or bv:
                    equal = False
                expected.append(UInt32(equal))
                expected.append(1)
    var n = len(left)
    var scope = Scope()
    var ctx = scope.context()
    var a = ctx.enqueue_create_buffer[DType.uint64](n)
    var b = ctx.enqueue_create_buffer[DType.uint64](n)
    var output = ctx.enqueue_create_buffer[DType.uint32](n * 32)
    ctx.enqueue_copy(a, left.unsafe_ptr())
    ctx.enqueue_copy(b, right.unsafe_ptr())
    var kernel = ctx.compile_function[_distinct_probe]()
    ctx.enqueue_function(
        kernel,
        a,
        b,
        output,
        UInt32(n),
        grid_dim=(n + 255) // 256,
        block_dim=256,
    )
    var actual = List[UInt32](length=n * 32, fill=0)
    ctx.enqueue_copy(actual.unsafe_ptr(), output)
    scope.synchronize()
    assert_equal(actual, expected)
    print("PASS", n * 32, "actual Metal distinct equality/hash checks")
