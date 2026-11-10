"""Actual Metal division/remainder against exact host-generated expectations."""
from std.memory import Pointer, bitcast
from std.sys import argv
from std.testing import assert_equal
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from dataframe.expr import DIV, FLOORDIV, MOD
from dataframe_metal.accelerator.gpu.native_numeric import native_numeric_binary


def _probe(
    left: Pointer[UInt64, MutUntrackedOrigin],
    right: Pointer[UInt64, MutUntrackedOrigin],
    tags: Pointer[UInt32, MutUntrackedOrigin],
    operations: Pointer[Int32, MutUntrackedOrigin],
    output: Pointer[UInt64, MutUntrackedOrigin],
    rows: UInt32,
):
    var row = global_idx.x
    if row < Int(rows):
        var result = native_numeric_binary(
            left[unsafe_offset=row],
            right[unsafe_offset=row],
            tags[unsafe_offset=row],
            Int(operations[unsafe_offset=row]),
        )
        output[unsafe_offset=row * 3] = result[0]
        output[unsafe_offset=row * 3 + 1] = UInt64(result[1])
        output[unsafe_offset=row * 3 + 2] = UInt64(result[2])


def main() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.accelerator.gpu.query_scope import Scope

    _ = require_metal()
    var args = argv()
    var left = List[UInt64]()
    var right = List[UInt64]()
    var tags = List[UInt32]()
    var ops = List[Int32]()
    var expected = List[UInt64]()
    # The fixture's Float32 result is from exact Fraction distances, not this
    # implementation or host Float64 division.
    with open(args[1], "r") as source:
        for line in source.read().split("\n"):
            if not line:
                continue
            var parts = line.split(",")
            left.append(UInt64(Int(parts[0])))
            right.append(UInt64(Int(parts[1])))
            tags.append(2)
            ops.append(Int32(DIV))
            expected.append(UInt64(Int(parts[2])))
            expected.append(1)
            expected.append(0)
    var state = UInt64(70031)
    for tag in range(2):
        var smallest = Int64(Int32.MIN) if tag == 0 else Int64.MIN
        var largest = Int64(Int32.MAX) if tag == 0 else Int64.MAX
        for i in range(4096):
            state = state * 6364136223846793005 + 1442695040888963407
            var a = bitcast[DType.int64](state)
            state = state * 6364136223846793005 + 1442695040888963407
            var b = bitcast[DType.int64](state)
            if tag == 0:
                a = Int64(bitcast[DType.int32](UInt32(state >> 32)))
                b = Int64(bitcast[DType.int32](UInt32(state)))
            if i % 3 == 0:
                b = Int64(1) << Int64(i % (31 if tag == 0 else 63))
                if i % 2:
                    b = -b
            if i < 12:
                a = smallest if i % 2 == 0 else largest
                b = smallest if i < 4 else Int64(-1 if i < 8 else 0)
            for op in [FLOORDIV, MOD]:
                var present = b != 0
                var fault = a == smallest and b == -1 and op == FLOORDIV
                var value = Int64(0)
                if present and not fault:
                    value = a // b if op == FLOORDIV else (
                        Int64(0) if b == -1 else a % b
                    )
                left.append(bitcast[DType.uint64](a))
                right.append(bitcast[DType.uint64](b))
                tags.append(UInt32(tag))
                ops.append(Int32(op))
                expected.append(bitcast[DType.uint64](value))
                expected.append(UInt64(present))
                expected.append(UInt64(fault))
    var n = len(left)
    var scope = Scope()
    var ctx = scope.context()
    var a = ctx.enqueue_create_buffer[DType.uint64](n)
    var b = ctx.enqueue_create_buffer[DType.uint64](n)
    var types = ctx.enqueue_create_buffer[DType.uint32](n)
    var operations = ctx.enqueue_create_buffer[DType.int32](n)
    var output = ctx.enqueue_create_buffer[DType.uint64](n * 3)
    var actual = List[UInt64](length=n * 3, fill=UInt64.MAX)
    ctx.enqueue_copy(a, left.unsafe_ptr())
    ctx.enqueue_copy(b, right.unsafe_ptr())
    ctx.enqueue_copy(types, tags.unsafe_ptr())
    ctx.enqueue_copy(operations, ops.unsafe_ptr())
    ctx.enqueue_function[_probe](
        a,
        b,
        types,
        operations,
        output,
        UInt32(n),
        grid_dim=(n + 255) // 256,
        block_dim=256,
    )
    ctx.enqueue_copy(actual.unsafe_ptr(), output)
    scope.synchronize()
    for row in range(n):
        assert_equal(actual[row * 3 + 1], expected[row * 3 + 1], String(row))
        assert_equal(actual[row * 3 + 2], expected[row * 3 + 2], String(row))
        if expected[row * 3 + 1] and not expected[row * 3 + 2]:
            assert_equal(actual[row * 3], expected[row * 3], String(row))
    scope.mark_success()
    print(
        "PASS actual Metal exact Float32 and signed Int32/Int64 division/modulo:",
        n,
    )
