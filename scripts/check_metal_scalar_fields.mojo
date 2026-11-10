"""Production composed scan: scalar and general fields share one reduction.

Independent integer fixtures verify every partial's sum, count and extrema.
"""
from std.math import abs
from std.sys import size_of
from std.testing import assert_equal
from max.gpu.host import DeviceContext, DeviceBuffer
from dataframe.expr import COL, LIT_INT, ABS, ADD, CLIP_LOW, CLIP_HIGH
from dataframe_metal.expression_format import (
    TREE_OP,
    TYPED_HEADER,
    TYPED_CHAIN,
    SCALAR_FIELD_SHIFT,
    VALUE_TYPE_SHIFT,
)
from dataframe_metal.accelerator.gpu.composed_reduce import mixed_partials


def node(mut words: List[UInt64], at: Int, op: Int, literal: UInt64 = 0):
    words[at] = UInt64(op)
    words[at + 2] = UInt64(1) << VALUE_TYPE_SHIFT
    words[at + 3] = literal


def run_check[D: DType](nullable: Bool, direct: Bool) raises:
    comptime ROWS = 65536
    comptime BLOCKS = 2
    var words = List[UInt64](length=100, fill=0)
    words[0] = ROWS
    words[1] = BLOCKS * 256
    words[3] = BLOCKS * 2
    words[14] = 3 | (UInt64(3) << 4) | (UInt64(3) << 8) | (UInt64(2) << 16)
    if direct:
        words[14] |= UInt64(1) << SCALAR_FIELD_SHIFT
    words[16] = UInt64(TREE_OP)
    words[22] = 48
    words[23] = 7
    words[24] = UInt64(TREE_OP)
    words[30] = 76
    words[31] = 6
    words[48] = UInt64(TYPED_HEADER)
    words[51] = UInt64(TYPED_CHAIN) | (UInt64(3) << 8) | (UInt64(1) << 16)
    node(words, 52, COL)
    node(words, 56, ABS)
    node(words, 60, LIT_INT)
    node(words, 64, CLIP_LOW)
    node(words, 68, LIT_INT, UInt64(12 if D == DType.int32 else 1 << 44))
    node(words, 72, CLIP_HIGH)
    words[76] = UInt64(TYPED_HEADER)
    words[77] = 576
    words[79] = 4 | (UInt64(2) << 8)
    node(words, 80, COL)
    node(words, 84, ABS)
    node(words, 88, COL)
    node(words, 92, ABS)
    node(words, 96, ADD)
    for at in [54, 58, 62, 66, 70, 74, 82, 86, 90, 94, 98]:
        words[at] = UInt64(1 if D == DType.int32 else 2) << VALUE_TYPE_SHIFT
    for at in [54, 82, 90]:
        words[at] |= UInt64(3) | (UInt64(nullable) << 32)
    var values = List[Scalar[D]](capacity=ROWS)
    var bitmap = List[UInt8](length=(ROWS + 10) // 8, fill=0)
    var sums = List[Int128](length=4, fill=0)
    var counts = List[UInt32](length=4, fill=0)
    var least = List[Int64](length=4, fill=Int64.MAX)
    var greatest = List[Int64](length=4, fill=Int64.MIN)
    for row in range(ROWS):
        var value = Int64(row % 32 - 16)
        comptime if D == DType.int64:
            value = value * (Int64(1) << 40) + Int64(row % 8)
        values.append(Scalar[D](value))
        var present = not nullable or row % 16 != 1
        if present:
            bitmap[(row + 3) // 8] |= UInt8(1) << UInt8((row + 3) % 8)
            var block = (row // 256) % BLOCKS
            for field in range(2):
                var result = Int64(
                    min(
                        abs(value), Int64(12 if D == DType.int32 else 1 << 44)
                    ) if field
                    == 0 else abs(value) * 2
                )
                var index = block * 2 + field
                sums[index] += Int128(result)
                counts[index] += 1
                least[index] = min(least[index], result)
                greatest[index] = max(greatest[index], result)
    var ctx = DeviceContext(0, api="metal")
    var dv = ctx.enqueue_create_buffer[D](ROWS)
    var db = ctx.enqueue_create_buffer[DType.uint8](len(bitmap))
    var dm = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var empty = ctx.enqueue_create_buffer[DType.uint8](1)
    var high = ctx.enqueue_create_buffer[DType.int64](20)
    var low = ctx.enqueue_create_buffer[DType.uint64](4)
    var fs = ctx.enqueue_create_buffer[DType.float32](4)
    var fc = ctx.enqueue_create_buffer[DType.float32](4)
    var nc = ctx.enqueue_create_buffer[DType.uint32](4)
    var bins = ctx.enqueue_create_buffer[DType.int64](1)
    var specials = ctx.enqueue_create_buffer[DType.uint32](1)
    var faults = ctx.enqueue_create_buffer[DType.uint32](BLOCKS)
    ctx.enqueue_copy(dv, values.unsafe_ptr())
    ctx.enqueue_copy(db, bitmap.unsafe_ptr())
    ctx.enqueue_copy(dm, words.unsafe_ptr())
    var bytes = dv.create_sub_buffer[DType.uint8](
        0, ROWS * size_of[Scalar[D]]()
    )
    ctx.enqueue_function[
        mixed_partials[
            False,
            False,
            2,
            0,
            2 if D == DType.int32 else 0,
            trees=True,
            cases=True,
        ]
    ](
        bytes,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        db,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        dm,
        UInt32(0),
        high,
        low,
        fs,
        fc,
        nc,
        bins,
        specials,
        faults,
        grid_dim=BLOCKS,
        block_dim=256,
    )
    var actual = List[Int64](length=20, fill=0)
    var actual_low = List[UInt64](length=4, fill=0)
    var actual_n = List[UInt32](length=4, fill=UInt32.MAX)
    var actual_f = List[UInt32](length=BLOCKS, fill=UInt32.MAX)
    ctx.enqueue_copy(actual.unsafe_ptr(), high)
    ctx.enqueue_copy(actual_low.unsafe_ptr(), low)
    ctx.enqueue_copy(actual_n.unsafe_ptr(), nc)
    ctx.enqueue_copy(actual_f.unsafe_ptr(), faults)
    ctx.synchronize()
    for i in range(4):
        var total = Int128(actual[i])
        comptime if D == DType.int64:
            total = (total << 32) + Int128(actual_low[i])
        assert_equal(total, sums[i])
        assert_equal(actual_n[i], counts[i])
        assert_equal(actual[4 + i * 4], least[i])
        assert_equal(actual[5 + i * 4], greatest[i])
        assert_equal(actual[6 + i * 4], Int64(counts[i]))
        assert_equal(actual[7 + i * 4], Int64(0))
    for fault in actual_f:
        assert_equal(fault, UInt32(0))
    _ = (
        dv^,
        db^,
        dm^,
        empty^,
        high^,
        low^,
        fs^,
        fc^,
        nc^,
        bins^,
        specials^,
        faults^,
        bytes^,
        actual^,
        actual_n^,
        actual_f^,
        values^,
        bitmap^,
        words^,
    )


def main() raises:
    for nullable in [False, True]:
        for direct in [False, True]:
            run_check[DType.int32](nullable, direct)
            run_check[DType.int64](nullable, direct)
    print(
        "PASS mixed scalar/general composed fields: sums/counts/extrema, Int32/Int64, both validity layouts and dispatch routes"
    )
