"""Actual Metal native ordering checked against independent host comparisons."""
from std.memory import Pointer, bitcast
from std.math import isnan
from std.testing import assert_equal
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from dataframe_metal.accelerator.gpu.ordered_value import ordered_better


def _ordered_probe(
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
        for minimum in range(2):
            for tie in range(2):
                var candidate_row = UInt32(1 if tie == 0 else 128)
                output[
                    unsafe_offset=row * 16 + tag * 4 + minimum * 2 + tie
                ] = UInt32(
                    ordered_better(
                        a, candidate_row, b, 64, UInt32(tag), Bool(minimum)
                    )
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
            for minimum in range(2):
                for tie in range(2):
                    var equal = False
                    var less = False
                    var greater = False
                    var forced = False
                    var answer = False
                    if tag == 2:
                        var x = bitcast[DType.float32](UInt32(a))
                        var y = bitcast[DType.float32](UInt32(b))
                        if isnan(x) or isnan(y):
                            forced = True
                            answer = tie == 0 if isnan(x) and isnan(
                                y
                            ) else not isnan(x)
                        else:
                            equal = x == y
                            less = x < y
                            greater = x > y
                    elif tag == 3:
                        equal = Bool(a) == Bool(b)
                        less = not Bool(a) and Bool(b)
                        greater = Bool(a) and not Bool(b)
                    else:
                        var x = bitcast[DType.int64](a)
                        var y = bitcast[DType.int64](b)
                        if tag == 0:
                            x = Int64(bitcast[DType.int32](UInt32(a)))
                            y = Int64(bitcast[DType.int32](UInt32(b)))
                        equal = x == y
                        less = x < y
                        greater = x > y
                    if not forced:
                        answer = (
                            tie == 0 if equal else less if minimum else greater
                        )
                    expected.append(UInt32(answer))
    var n = len(left)
    var scope = Scope()
    var ctx = scope.context()
    var a = ctx.enqueue_create_buffer[DType.uint64](n)
    var b = ctx.enqueue_create_buffer[DType.uint64](n)
    var output = ctx.enqueue_create_buffer[DType.uint32](n * 16)
    ctx.enqueue_copy(a, left.unsafe_ptr())
    ctx.enqueue_copy(b, right.unsafe_ptr())
    var kernel = ctx.compile_function[_ordered_probe]()
    ctx.enqueue_function(
        kernel,
        a,
        b,
        output,
        UInt32(n),
        grid_dim=(n + 255) // 256,
        block_dim=256,
    )
    var actual = List[UInt32](length=n * 16, fill=0)
    ctx.enqueue_copy(actual.unsafe_ptr(), output)
    scope.synchronize()
    assert_equal(actual, expected)
    print("PASS", n * 16, "actual Metal native ordering comparisons")
