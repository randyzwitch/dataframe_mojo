"""Real-device partitioned ranks against independent typed scalar ordering."""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.math import isnan
from std.testing import assert_equal
from dataframe.accelerator.rows import ROW_RANK_KEY, ROW_SORT_KEY
from dataframe.accelerator.gpu.ordering import order_init
from dataframe.accelerator.gpu.ranking import (
    rank_order_merge,
    rank_prepare,
    rank_offsets,
    rank_assign,
)
from tests.tooling.mojo_gpu_order_probe import _value
from tests.support.key_kernel_reference import KeyCell, cell_equal, cell_less


def _partition(
    rows: List[List[KeyCell]], a: Int, b: Int, grouped: Bool
) -> Bool:
    if not grouped:
        return True
    return cell_equal(rows[a][1], rows[b][1]) and cell_equal(
        rows[a][2], rows[b][2]
    )


def _less(
    rows: List[List[KeyCell]], a: Int, b: Int, descending: Bool, grouped: Bool
) -> Bool:
    if grouped:
        for key in range(1, 3):
            if not cell_equal(rows[a][key], rows[b][key]):
                return cell_less(rows[a][key], rows[b][key], False, True)
    ref left = rows[a][0]
    ref right = rows[b][0]
    if not left.valid or not right.valid:
        return left.valid and not right.valid
    if left.kind == 1:
        var an = isnan(left.floating)
        var bn = isnan(right.floating)
        if an != bn:
            return an if descending else not an
    return cell_less(left, right, descending, True)


def _case(kind: Int, n: Int, descending: Bool, grouped: Bool) raises:
    var capacity = n + 7
    var words = List[UInt64](length=8 * capacity, fill=0)
    var valid = List[UInt8](length=8 * capacity, fill=0)
    var errors = List[UInt32](length=8 * capacity, fill=0)
    var reference = List[List[KeyCell]]()
    for row in range(n):
        var input = _value(kind, row)
        var value = input[1].copy()
        value.valid = row % 13 != 4
        var partition = _value(11, row // 7)
        var first = partition[1].copy()
        first.valid = row % 17 != 2
        var second = KeyCell(0, integer=Int128(row % 3))
        words[row] = input[0]
        valid[row] = UInt8(value.valid)
        words[capacity + row] = partition[0]
        valid[capacity + row] = UInt8(first.valid)
        words[2 * capacity + row] = UInt64(row % 3)
        valid[2 * capacity + row] = 1
        reference.append([value.copy(), first.copy(), second.copy()])
    var expected_order = List[Int]()
    for row in range(n):
        expected_order.append(row)
        var position = row
        while position and _less(
            reference,
            row,
            expected_order[position - 1],
            descending,
            grouped,
        ):
            expected_order[position] = expected_order[position - 1]
            position -= 1
        expected_order[position] = row
    var expected = List[UInt64](length=5 * n, fill=0)
    var expected_valid = List[UInt8](length=n, fill=0)
    var position = 0
    var base = 0
    var dense = 0
    while position < n:
        var own = expected_order[position]
        if position == 0 or not _partition(
            reference, expected_order[position - 1], own, grouped
        ):
            base = position
            dense = 0
        if not reference[own][0].valid:
            position += 1
            continue
        var end = position + 1
        while (
            end < n
            and _partition(reference, own, expected_order[end], grouped)
            and cell_equal(reference[own][0], reference[expected_order[end]][0])
        ):
            end += 1
        dense += 1
        for at in range(position, end):
            var row = expected_order[at]
            expected_valid[row] = 1
            var first = position - base + 1
            var last = end - base
            expected[row] = bitcast[DType.uint64](Float64(first + last) / 2)
            expected[n + row] = UInt64(first)
            expected[2 * n + row] = UInt64(last)
            expected[3 * n + row] = UInt64(dense)
            expected[4 * n + row] = UInt64(at - base + 1)
        position = end
    var nodes = 3 if grouped else 1
    var code = List[Int64]()
    var types = List[Int64]()
    for method in range(5):
        code.extend([Int64(ROW_RANK_KEY), 0, Int64(descending), Int64(method)])
        types.append(Int64(kind))
        if grouped:
            code.extend([Int64(ROW_SORT_KEY), 1, 0, 1])
            types.append(11)
            code.extend([Int64(ROW_SORT_KEY), 2, 0, 1])
            types.append(10)
    var ctx = DeviceContext()
    var dw = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var dv = ctx.enqueue_create_buffer[DType.uint8](len(valid))
    var de = ctx.enqueue_create_buffer[DType.uint32](len(errors))
    var dc = ctx.enqueue_create_buffer[DType.int64](len(code))
    var dt = ctx.enqueue_create_buffer[DType.int64](len(types))
    var a = ctx.enqueue_create_buffer[DType.uint32](n)
    var b = ctx.enqueue_create_buffer[DType.uint32](n)
    var blocks = (n + 255) // 256
    var totals = ctx.enqueue_create_buffer[DType.uint32](blocks)
    ctx.enqueue_copy(dw, words.unsafe_ptr())
    ctx.enqueue_copy(dv, valid.unsafe_ptr())
    ctx.enqueue_copy(de, errors.unsafe_ptr())
    ctx.enqueue_copy(dc, code.unsafe_ptr())
    ctx.enqueue_copy(dt, types.unsafe_ptr())
    ctx.enqueue_function[order_init](
        a.unsafe_ptr(), UInt32(n), grid_dim=blocks, block_dim=256
    )
    var width = 1
    while width < n:
        ctx.enqueue_function[rank_order_merge](
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            a.unsafe_ptr(),
            b.unsafe_ptr(),
            dc.unsafe_ptr(),
            dt.unsafe_ptr(),
            UInt32(0),
            UInt32(nodes),
            UInt32(n),
            UInt32(capacity),
            UInt32(width),
            grid_dim=blocks,
            block_dim=256,
        )
        var previous = a^
        a = b^
        b = previous^
        width *= 2
    ctx.enqueue_function[rank_prepare](
        dw.unsafe_ptr(),
        dv.unsafe_ptr(),
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        totals.unsafe_ptr(),
        dc.unsafe_ptr(),
        dt.unsafe_ptr(),
        UInt32(0),
        UInt32(nodes),
        UInt32(n),
        UInt32(capacity),
        grid_dim=blocks,
        block_dim=256,
    )
    ctx.enqueue_function[rank_offsets](
        totals.unsafe_ptr(), UInt32(blocks), grid_dim=1, block_dim=1
    )
    for method in range(5):
        ctx.enqueue_function[rank_assign](
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            de.unsafe_ptr(),
            a.unsafe_ptr(),
            b.unsafe_ptr(),
            totals.unsafe_ptr(),
            dc.unsafe_ptr(),
            dt.unsafe_ptr(),
            UInt32(method * nodes),
            UInt32(nodes),
            UInt32(method + 3),
            UInt32(n),
            UInt32(capacity),
            grid_dim=blocks,
            block_dim=256,
        )
    ctx.enqueue_copy(words.unsafe_ptr(), dw)
    ctx.enqueue_copy(valid.unsafe_ptr(), dv)
    ctx.enqueue_copy(errors.unsafe_ptr(), de)
    ctx.synchronize()
    for method in range(5):
        for row in range(n):
            var at = (method + 3) * capacity + row
            assert_equal(valid[at], expected_valid[row])
            assert_equal(words[at], expected[method * n + row])
            assert_equal(errors[at], UInt32(0))


def main() raises:
    for kind in range(1, 12):
        for descending in [False, True]:
            for grouped in [False, True]:
                _case(kind, 257, descending, grouped)
    for n in [1, 255, 256, 1_027, 4_099]:
        _case(11, n, True, True)
    print(
        "PASS Mojo GPU ranks: 49 layouts, five methods, every scalar kind,"
        " null partitions, NaN direction, zero ties and dense block boundaries"
    )
