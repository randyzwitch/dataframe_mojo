"""Real-device stable sort against independent typed scalar reference order."""
from max.gpu.host import DeviceContext
from std.memory import bitcast, Pointer
from std.testing import assert_equal
from dataframe.accelerator.rows import ROW_SORT_KEY
from dataframe.accelerator.gpu.ordering import (
    order_init,
    order_merge,
    order_key_errors,
    order_gather,
    merge_grid,
)
from tests.support.key_kernel_reference import KeyCell, expected_order


def _value(kind: Int, row: Int) -> Tuple[UInt64, KeyCell]:
    var at = (row * 13 + row // 7) % 12
    if kind == 1 or kind == 11:
        var encodings: List[UInt64] = [
            0,
            0x8000000000000000,
            0x3FF0000000000000,
            0xBFF0000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000000001,
            0xFFF8000000000002,
            1,
            0x8000000000000001,
            0x0010000000000000,
            0x8010000000000000,
        ]
        if kind == 1:
            encodings = [
                0,
                0x80000000,
                0x3F800000,
                0xBF800000,
                0x7F800000,
                0xFF800000,
                0x7FC00001,
                0xFFC00002,
                1,
                0x80000001,
                0x00800000,
                0x80800000,
            ]
        var word = encodings[at]
        var value = Float64(
            bitcast[DType.float32](UInt32(word))
        ) if kind == 1 else bitcast[DType.float64](word)
        return (word, KeyCell(1, floating=value))
    if kind == 4:
        var word = UInt64(at % 2)
        return (word, KeyCell(0, integer=Int128(word)))
    var bits = 64
    if kind == 5 or kind == 7:
        bits = 8
    elif kind == 6 or kind == 8:
        bits = 16
    elif kind == 2 or kind == 9:
        bits = 32
    var unsigned = kind >= 7 and kind <= 10
    var low = Int128(0) if unsigned else -(Int128(1) << Int128(bits - 1))
    var high = (Int128(1) << Int128(bits if unsigned else bits - 1)) - 1
    var choices: List[Int128] = [
        low,
        high,
        0,
        1,
        high - 1,
        low + 1,
        2,
        3,
        0,
        high,
        low,
        1,
    ]
    var value = choices[at]
    return (value.cast[DType.uint64](), KeyCell(0, integer=value))


def _case(kind: Int, n: Int, descending: Bool, nulls_last: Bool) raises:
    var capacity = n + 5
    var words = List[UInt64](length=4 * capacity, fill=0xBAD0)
    var valid = List[UInt8](length=4 * capacity, fill=0)
    var errors = List[UInt32](length=4 * capacity, fill=0)
    var reference = List[List[KeyCell]]()
    for row in range(n):
        var pair = _value(kind, row)
        var first = pair[1].copy()
        first.valid = row % 7 != 3
        var second = KeyCell(0, row % 11 != 5, integer=Int128(row % 5 - 2))
        reference.append([first.copy(), second.copy()])
        words[2 * capacity + row] = pair[0]
        valid[2 * capacity + row] = UInt8(first.valid)
        words[capacity + row] = Int64(second.integer).cast[DType.uint64]()
        valid[capacity + row] = UInt8(second.valid)
        words[row] = UInt64(row)
        valid[row] = 1
        errors[row] = UInt32(row + 1000)
    errors[2 * capacity] = 16
    errors[capacity + n - 1] = 32
    var expected = expected_order(
        reference, [descending, not descending], [nulls_last, not nulls_last]
    )
    # Nonzero descriptor offset and reordered, noncontiguous gather slots.
    var code: List[Int64] = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        ROW_SORT_KEY,
        2,
        Int64(descending),
        Int64(nulls_last),
        ROW_SORT_KEY,
        1,
        Int64(not descending),
        Int64(not nulls_last),
    ]
    var types: List[Int64] = [0, 0, Int64(kind), 3]
    var gathers: List[Int64] = [2, 0, 1]
    var ctx = DeviceContext()
    var dw = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var dv = ctx.enqueue_create_buffer[DType.uint8](len(valid))
    var de = ctx.enqueue_create_buffer[DType.uint32](len(errors))
    var dc = ctx.enqueue_create_buffer[DType.int64](len(code))
    var dt = ctx.enqueue_create_buffer[DType.int64](len(types))
    var dg = ctx.enqueue_create_buffer[DType.int64](len(gathers))
    var a = ctx.enqueue_create_buffer[DType.uint32](n)
    var b = ctx.enqueue_create_buffer[DType.uint32](n)
    var ow = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var ov = ctx.enqueue_create_buffer[DType.uint8](len(valid))
    var oe = ctx.enqueue_create_buffer[DType.uint32](len(errors))
    var blocks = (n + 255) // 256
    var df = ctx.enqueue_create_buffer[DType.uint32](blocks)
    ctx.enqueue_copy(dw, words.unsafe_ptr())
    ctx.enqueue_copy(dv, valid.unsafe_ptr())
    ctx.enqueue_copy(de, errors.unsafe_ptr())
    ctx.enqueue_copy(dc, code.unsafe_ptr())
    ctx.enqueue_copy(dt, types.unsafe_ptr())
    ctx.enqueue_copy(dg, gathers.unsafe_ptr())
    ctx.enqueue_function[order_init](
        a.unsafe_ptr(), UInt32(n), grid_dim=blocks, block_dim=256
    )
    var width = 1
    while width < n:
        ctx.enqueue_function[order_merge](
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            a.unsafe_ptr(),
            b.unsafe_ptr(),
            dc.unsafe_ptr(),
            dt.unsafe_ptr(),
            UInt32(2),
            UInt32(2),
            UInt32(n),
            UInt32(capacity),
            UInt32(width),
            UInt32(0),
            Pointer[UInt64, ImmutAnyOrigin](unsafe_from_address=0),
            grid_dim=merge_grid(n, width),
            block_dim=256,
        )
        var previous = a^
        a = b^
        b = previous^
        width *= 2
    ctx.enqueue_function[order_key_errors](
        de.unsafe_ptr(),
        dc.unsafe_ptr(),
        df.unsafe_ptr(),
        UInt32(2),
        UInt32(2),
        UInt32(n),
        UInt32(capacity),
        grid_dim=blocks,
        block_dim=256,
    )
    ctx.enqueue_function[order_gather](
        dw.unsafe_ptr(),
        dv.unsafe_ptr(),
        de.unsafe_ptr(),
        ow.unsafe_ptr(),
        ov.unsafe_ptr(),
        oe.unsafe_ptr(),
        a.unsafe_ptr(),
        dg.unsafe_ptr(),
        UInt32(len(gathers)),
        UInt32(n),
        UInt32(capacity),
        grid_dim=blocks,
        block_dim=256,
    )
    var order = List[UInt32](length=n, fill=0)
    var result = List[UInt64](length=len(words), fill=0)
    var result_valid = List[UInt8](length=len(valid), fill=0)
    var result_errors = List[UInt32](length=len(errors), fill=0)
    var failures = List[UInt32](length=blocks, fill=0)
    ctx.enqueue_copy(order.unsafe_ptr(), a)
    ctx.enqueue_copy(result.unsafe_ptr(), ow)
    ctx.enqueue_copy(result_valid.unsafe_ptr(), ov)
    ctx.enqueue_copy(result_errors.unsafe_ptr(), oe)
    ctx.enqueue_copy(failures.unsafe_ptr(), df)
    ctx.synchronize()
    for row in range(n):
        assert_equal(Int(order[row]), expected[row])
        for slot in gathers:
            var base = Int(slot) * capacity
            var source = base + expected[row]
            assert_equal(result[base + row], words[source])
            assert_equal(result_valid[base + row], valid[source])
            assert_equal(result_errors[base + row], errors[source])
    for block in range(blocks):
        var expected_failures = UInt32(0)
        for row in range(block * 256, min(n, (block + 1) * 256)):
            expected_failures += UInt32(row == 0 or row == n - 1)
        assert_equal(failures[block], expected_failures)


def main() raises:
    for kind in range(1, 12):
        for descending in [False, True]:
            for nulls_last in [False, True]:
                _case(kind, 257, descending, nulls_last)
    for n in [1, 2, 3, 17, 255, 256, 1027]:
        _case(11, n, False, True)
        _case(10, n, True, False)
    print(
        "PASS Mojo GPU stable ordering: 58 mixed-key cases, every scalar kind,"
        " null placement, NaNs, signed zeros, subnormals, exact gathers and"
        " errors"
    )
