"""Actual Metal null/NaN primitives checked against host floating semantics."""
from std.memory import Pointer, bitcast
from std.math import isnan
from std.testing import assert_equal
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from dataframe.expr import FILL_NULL, FILL_NAN
from dataframe_metal.accelerator.gpu.null_value import (
    fill_value_bits,
    equal_missing_float32,
    equal_missing_integer,
)


def _null_probe(
    left: Pointer[UInt64, MutUntrackedOrigin],
    right: Pointer[UInt64, MutUntrackedOrigin],
    validity: Pointer[UInt32, MutUntrackedOrigin],
    output: Pointer[UInt64, MutUntrackedOrigin],
    rows: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var a = left[unsafe_offset=row]
    var b = right[unsafe_offset=row]
    var flags = validity[unsafe_offset=row]
    var av = Bool(flags & 1)
    var bv = Bool(flags & 2)
    var null = fill_value_bits(a, av, b, bv, FILL_NULL)
    var nan = fill_value_bits(a, av, b, bv, FILL_NAN)
    output[unsafe_offset=row * 5] = null[0]
    output[unsafe_offset=row * 5 + 1] = nan[0]
    output[unsafe_offset=row * 5 + 2] = UInt64(null[1]) | (UInt64(nan[1]) << 1)
    output[unsafe_offset=row * 5 + 3] = UInt64(
        equal_missing_float32(UInt32(a), av, UInt32(b), bv)
    )
    output[unsafe_offset=row * 5 + 4] = UInt64(
        equal_missing_integer(
            bitcast[DType.int64](a), av, bitcast[DType.int64](b), bv
        )
    )


def main() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.accelerator.gpu.query_scope import Scope

    _ = require_metal()
    var edges = List[UInt64](
        [
            UInt64(0),
            UInt64(0x80000000),
            UInt64(1),
            UInt64(0x80000001),
            UInt64(0x3F800000),
            UInt64(0xBF800000),
            UInt64(0x7F800000),
            UInt64(0xFF800000),
            UInt64(0x7F800001),
            UInt64(0x7FC12345),
            UInt64(0xFFC54321),
            UInt64(0x8000000000000000),
            UInt64.MAX,
        ]
    )
    var left = List[UInt64]()
    var right = List[UInt64]()
    var validity = List[UInt32]()
    var expected = List[UInt64]()
    var state = UInt64(41378391)
    for i in range(len(edges) * len(edges) + 4096):
        state = state * 6364136223846793005 + 1442695040888963407
        var a = state
        state = state * 6364136223846793005 + 1442695040888963407
        var b = state
        if i < len(edges) * len(edges):
            a = edges[i // len(edges)]
            b = edges[i % len(edges)]
        for flags in range(4):
            var av = Bool(flags & 1)
            var bv = Bool(flags & 2)
            var x = bitcast[DType.float32](UInt32(a))
            var y = bitcast[DType.float32](UInt32(b))
            var replace_nan = av and isnan(x)
            left.append(a)
            right.append(b)
            validity.append(UInt32(flags))
            expected.append(a if av else b)
            expected.append(b if replace_nan else a)
            expected.append(
                UInt64(av or bv) | (UInt64(bv if replace_nan else av) << 1)
            )
            expected.append(
                UInt64(
                    av == bv if not av
                    or not bv else x == y
                    or (isnan(x) and isnan(y))
                )
            )
            expected.append(UInt64(av == bv if not av or not bv else a == b))
    var n = len(left)
    var scope = Scope()
    var ctx = scope.context()
    var a = ctx.enqueue_create_buffer[DType.uint64](n)
    var b = ctx.enqueue_create_buffer[DType.uint64](n)
    var flags = ctx.enqueue_create_buffer[DType.uint32](n)
    var output = ctx.enqueue_create_buffer[DType.uint64](n * 5)
    ctx.enqueue_copy(a, left.unsafe_ptr())
    ctx.enqueue_copy(b, right.unsafe_ptr())
    ctx.enqueue_copy(flags, validity.unsafe_ptr())
    var kernel = ctx.compile_function[_null_probe]()
    ctx.enqueue_function(
        kernel,
        a,
        b,
        flags,
        output,
        UInt32(n),
        grid_dim=(n + 255) // 256,
        block_dim=256,
    )
    var actual = List[UInt64](length=n * 5, fill=0)
    ctx.enqueue_copy(actual.unsafe_ptr(), output)
    scope.synchronize()
    assert_equal(actual, expected)
    print("PASS", n, "actual Metal null/NaN/equality cases")
