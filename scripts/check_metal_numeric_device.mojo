"""Actual Metal bit/validity/fault conformance for native numeric primitives.

Keep this device entrypoint in its own program: the pinned SDK cannot resolve
this test kernel's launch after importing the containing test function.
"""
from std.memory import Pointer, bitcast
from std.testing import assert_equal
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from dataframe.expr import (
    NEG,
    ABS,
    SIGN,
    FLOOR,
    CEIL,
    TRUNCATE,
    ROUND,
    CLIP_LOW,
    CLIP_HIGH,
    DIV,
    FLOORDIV,
    MOD,
)
from dataframe_metal.accelerator.gpu.native_numeric import (
    numeric_unary,
    float32_unary_bits,
    integer_unary_bits,
    native_numeric_binary,
)


def _numeric_bits_probe(
    left: Pointer[UInt64, MutUntrackedOrigin],
    right: Pointer[UInt64, MutUntrackedOrigin],
    operations: Pointer[Int32, MutUntrackedOrigin],
    tags: Pointer[UInt32, MutUntrackedOrigin],
    result: Pointer[UInt64, MutUntrackedOrigin],
    present: Pointer[UInt32, MutUntrackedOrigin],
    faults: Pointer[UInt32, MutUntrackedOrigin],
    rows: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var op = Int(operations[unsafe_offset=row])
    var tag = tags[unsafe_offset=row]
    var a = left[unsafe_offset=row]
    var b = right[unsafe_offset=row]
    var answer = (UInt64(0), True, UInt32(0))
    if numeric_unary(op):
        if tag == 2:
            answer = (
                UInt64(float32_unary_bits(UInt32(a), op)),
                True,
                UInt32(0),
            )
        else:
            var value = integer_unary_bits(a, tag, op)
            answer = (value[0], True, UInt32(value[1]))
    else:
        answer = native_numeric_binary(a, b, tag, op)
    result[unsafe_offset=row] = answer[0]
    present[unsafe_offset=row] = UInt32(answer[1])
    faults[unsafe_offset=row] = answer[2]


def main() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.accelerator.gpu.query_scope import Scope

    _ = require_metal()
    var left = List[UInt64]()
    var right = List[UInt64]()
    var ops = List[Int32]()
    var tags = List[UInt32]()
    var expected = List[UInt64]()
    var validity = List[UInt32]()
    var faults = List[UInt32]()
    var zero_divisors: List[UInt32] = [
        1,
        0x80000001,
        0x007FFFFF,
        0x00800000,
        0x3F800000,
        0xBF800000,
        0x40000000,
        0xC0000000,
        0x7F7FFFFF,
        0xFF7FFFFF,
    ]
    var state = UInt64(41378391)
    for op in [
        NEG,
        ABS,
        SIGN,
        FLOOR,
        CEIL,
        TRUNCATE,
        ROUND,
        CLIP_LOW,
        CLIP_HIGH,
        DIV,
        FLOORDIV,
        MOD,
    ]:
        for tag in range(3):
            if op == DIV and tag != 2:
                continue
            for i in range(512):
                state = state * 6364136223846793005 + 1442695040888963407
                var a = state
                state = state * 6364136223846793005 + 1442695040888963407
                var b = state
                if tag == 0:
                    a = bitcast[DType.uint64](
                        Int64(bitcast[DType.int32](UInt32(a)))
                    )
                    b = bitcast[DType.uint64](
                        Int64(bitcast[DType.int32](UInt32(b)))
                    )
                elif tag == 2:
                    a = UInt64(UInt32(a))
                    b = UInt64(UInt32(b))
                var answer = (UInt64(0), True, UInt32(0))
                if numeric_unary(op):
                    if tag == 2:
                        answer = (
                            UInt64(float32_unary_bits(UInt32(a), op)),
                            True,
                            UInt32(0),
                        )
                    else:
                        var value = integer_unary_bits(a, UInt32(tag), op)
                        answer = (value[0], True, UInt32(value[1]))
                else:
                    answer = native_numeric_binary(a, b, UInt32(tag), op)
                left.append(a)
                right.append(b)
                ops.append(Int32(op))
                tags.append(UInt32(tag))
                expected.append(answer[0])
                validity.append(UInt32(answer[1]))
                faults.append(answer[2])
    # Keep all random cases and add independently expected zero remainders.
    for zero in [UInt32(0), UInt32(0x80000000)]:
        for divisor in zero_divisors:
            left.append(UInt64(zero))
            right.append(UInt64(divisor))
            ops.append(Int32(MOD))
            tags.append(UInt32(2))
            expected.append(UInt64(0))
            validity.append(UInt32(1))
            faults.append(UInt32(0))
    var n = len(left)
    var scope = Scope()
    var ctx = scope.context()
    var a = ctx.enqueue_create_buffer[DType.uint64](n)
    var b = ctx.enqueue_create_buffer[DType.uint64](n)
    var operations = ctx.enqueue_create_buffer[DType.int32](n)
    var types = ctx.enqueue_create_buffer[DType.uint32](n)
    var output = ctx.enqueue_create_buffer[DType.uint64](n)
    var present = ctx.enqueue_create_buffer[DType.uint32](n)
    var flags = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(a, left.unsafe_ptr())
    ctx.enqueue_copy(b, right.unsafe_ptr())
    ctx.enqueue_copy(operations, ops.unsafe_ptr())
    ctx.enqueue_copy(types, tags.unsafe_ptr())
    var kernel = ctx.compile_function[_numeric_bits_probe]()
    ctx.enqueue_function(
        kernel,
        a,
        b,
        operations,
        types,
        output,
        present,
        flags,
        UInt32(n),
        grid_dim=(n + 255) // 256,
        block_dim=256,
    )
    var actual = List[UInt64](length=n, fill=0)
    var actual_validity = List[UInt32](length=n, fill=0)
    var actual_faults = List[UInt32](length=n, fill=0)
    ctx.enqueue_copy(actual.unsafe_ptr(), output)
    ctx.enqueue_copy(actual_validity.unsafe_ptr(), present)
    ctx.enqueue_copy(actual_faults.unsafe_ptr(), flags)
    scope.synchronize()
    assert_equal(actual, expected)
    assert_equal(actual_validity, validity)
    assert_equal(actual_faults, faults)
    print("PASS 17428 actual Metal native numeric bit/validity/fault cases")
