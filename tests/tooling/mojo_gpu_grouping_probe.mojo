"""Direct grouping kernels: nullable IEEE keys, boundaries and reductions."""

from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe.accelerator.rows import (
    ROW_GROUP_HEADER,
    ROW_GROUP_KEY,
    ROW_GROUP_OUTPUT,
    ROW_QUANTILE_LINEAR,
)
from dataframe.aggregate import quantile_of
from dataframe.expr import SUM, COUNT, VAR, ANY, ARG_MIN
from dataframe.accelerator.gpu.ordering import (
    order_init,
    order_merge,
    merge_grid,
)
from dataframe.accelerator.gpu.grouping import (
    group_mark,
    group_offsets,
    group_starts,
    group_reduce,
)
from dataframe.accelerator.gpu.reductions import State as BasicState
from dataframe.accelerator.gpu.statistics import State as StatisticalState
from dataframe.accelerator.gpu import quantiles


def _indexed_expected(
    words: List[UInt64],
    masks: List[UInt8],
    index: List[UInt32],
    base: Int,
    begin: Int,
    end: Int,
) -> Tuple[UInt64, UInt8]:
    return quantiles.reduce[True](
        words.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
        masks.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
        index.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
        base,
        begin,
        end,
        bitcast[DType.uint64](Float64(0.37)),
        ROW_QUANTILE_LINEAR,
    )


def check(n: Int, fail_key: Bool, quantile: Bool = False) raises:
    var capacity = n + 5
    var cells = 13 * capacity
    var blocks = max(1, (n + 255) // 256)
    var code: List[Int64] = [
        ROW_GROUP_HEADER,
        2,
        12,
        5,
        ROW_GROUP_KEY,
        0,
        0,
        0,
        ROW_GROUP_KEY,
        1,
        0,
        0,
        ROW_GROUP_OUTPUT,
        2,
        6,
        SUM,
        ROW_GROUP_OUTPUT,
        2,
        7,
        COUNT,
        ROW_GROUP_OUTPUT,
        3,
        8,
        VAR,
        ROW_GROUP_OUTPUT,
        4,
        9,
        ANY,
        ROW_GROUP_OUTPUT,
        2,
        10,
        ARG_MIN,
    ]
    var literals: List[UInt64] = [0, 0, 0, 0, 0, 1 | (UInt64(4) << 47), 0, 0]
    var types: List[Int64] = [3, 11, 10, 3, 3, 11, 4, 9]
    var slot_types: List[Int64] = [11, 10, 3, 11, 4, 3, 3, 3, 11, 4, 9, 3, 3]
    if quantile:
        code[5 * 4 + 3] = ROW_QUANTILE_LINEAR
        literals[5] = bitcast[DType.uint64](Float64(0.37))
    var words = List[UInt64](length=cells, fill=0)
    var masks = List[UInt8](length=cells, fill=0)
    var errors = List[UInt32](length=cells, fill=0)
    var encodings: List[UInt64] = [
        0,
        0x8000000000000000,
        0x7FF8000000000001,
        0xFFF8000000000002,
        0x3FF0000000000000,
        0xBFF0000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
    ]
    for row in range(n):
        words[row] = encodings[row % len(encodings)]
        words[capacity + row] = UInt64.MAX - UInt64(row % 3)
        words[2 * capacity + row] = Int64(row % 11 - 5).cast[DType.uint64]()
        words[3 * capacity + row] = bitcast[DType.uint64](Float64(row % 11 - 5))
        words[4 * capacity + row] = UInt64(row % 2)
        masks[row] = UInt8(row % 13 != 0)
        masks[capacity + row] = UInt8(row % 17 != 0)
        for slot in range(2, 5):
            masks[slot * capacity + row] = UInt8(row % 7 != 0)
    if n and fail_key:
        errors[n - 1] = 32
    var result = List[UInt64](length=cells, fill=0)
    var result_masks = List[UInt8](length=cells, fill=0)
    var result_errors = List[UInt32](length=cells, fill=0)
    var starts = List[UInt32](length=max(1, n), fill=0)
    var sorted_rows = List[UInt32](length=max(1, n), fill=0)
    var shape = List[UInt32](length=4, fill=0)
    var ctx = DeviceContext()
    var dw = ctx.enqueue_create_buffer[DType.uint64](cells)
    var dv = ctx.enqueue_create_buffer[DType.uint8](cells)
    var de = ctx.enqueue_create_buffer[DType.uint32](cells)
    var dc = ctx.enqueue_create_buffer[DType.int64](len(code))
    var dt = ctx.enqueue_create_buffer[DType.int64](len(types))
    var dl = ctx.enqueue_create_buffer[DType.uint64](len(literals))
    var dst = ctx.enqueue_create_buffer[DType.int64](len(slot_types))
    var di = ctx.enqueue_create_buffer[DType.uint32](max(1, n))
    var dn = ctx.enqueue_create_buffer[DType.uint32](max(1, n))
    var dp = ctx.enqueue_create_buffer[DType.uint32](max(1, n))
    var db = ctx.enqueue_create_buffer[DType.uint32](blocks * 2)
    var ds = ctx.enqueue_create_buffer[DType.uint32](max(1, n))
    var dm = ctx.enqueue_create_buffer[DType.uint32](4)
    var dow = ctx.enqueue_create_buffer[DType.uint64](cells)
    var dov = ctx.enqueue_create_buffer[DType.uint8](cells)
    var doe = ctx.enqueue_create_buffer[DType.uint32](cells)
    ctx.enqueue_copy(dw, words.unsafe_ptr())
    ctx.enqueue_copy(dv, masks.unsafe_ptr())
    ctx.enqueue_copy(de, errors.unsafe_ptr())
    ctx.enqueue_copy(dc, code.unsafe_ptr())
    ctx.enqueue_copy(dt, types.unsafe_ptr())
    ctx.enqueue_copy(dl, literals.unsafe_ptr())
    ctx.enqueue_copy(dst, slot_types.unsafe_ptr())
    ctx.enqueue_copy(dow, result.unsafe_ptr())
    ctx.enqueue_copy(dov, result_masks.unsafe_ptr())
    ctx.enqueue_copy(doe, result_errors.unsafe_ptr())
    ctx.enqueue_function[order_init](
        di.unsafe_ptr(), UInt32(n), grid_dim=blocks, block_dim=256
    )
    var width = 1
    while width < n:
        ctx.enqueue_function[order_merge](
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            di.unsafe_ptr(),
            dn.unsafe_ptr(),
            dc.unsafe_ptr(),
            dt.unsafe_ptr(),
            UInt32(1),
            UInt32(2),
            UInt32(n),
            UInt32(capacity),
            UInt32(width),
            UInt32(0),
            dl.unsafe_ptr(),
            grid_dim=merge_grid(n, width),
            block_dim=256,
        )
        var previous = di^
        di = dn^
        dn = previous^
        width *= 2
    ctx.enqueue_function[group_mark](
        dw.unsafe_ptr(),
        dv.unsafe_ptr(),
        de.unsafe_ptr(),
        di.unsafe_ptr(),
        dp.unsafe_ptr(),
        db.unsafe_ptr(),
        dc.unsafe_ptr(),
        dt.unsafe_ptr(),
        UInt32(0),
        UInt32(2),
        UInt32(n),
        UInt32(capacity),
        UInt32(0),
        grid_dim=blocks,
        block_dim=256,
    )
    ctx.enqueue_function[group_offsets](
        db.unsafe_ptr(),
        dm.unsafe_ptr(),
        UInt32((n + 255) // 256),
        grid_dim=1,
        block_dim=1,
    )
    ctx.enqueue_copy(shape.unsafe_ptr(), dm)
    ctx.synchronize()
    assert_equal(shape[1] != 0, fail_key and n > 0)
    if shape[0]:
        ctx.enqueue_function[group_starts](
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            di.unsafe_ptr(),
            dp.unsafe_ptr(),
            db.unsafe_ptr(),
            ds.unsafe_ptr(),
            dc.unsafe_ptr(),
            dt.unsafe_ptr(),
            UInt32(0),
            UInt32(2),
            UInt32(n),
            UInt32(capacity),
            UInt32(0),
            grid_dim=blocks,
            block_dim=256,
        )
        ctx.enqueue_function[group_reduce](
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            de.unsafe_ptr(),
            dow.unsafe_ptr(),
            dov.unsafe_ptr(),
            doe.unsafe_ptr(),
            di.unsafe_ptr(),
            ds.unsafe_ptr(),
            dm.unsafe_ptr(),
            dc.unsafe_ptr(),
            dl.unsafe_ptr(),
            dst.unsafe_ptr(),
            UInt32(0),
            UInt32(8),
            shape[0],
            UInt32(n),
            UInt32(capacity),
            UInt32(0),
            grid_dim=(Int(shape[0]) * 8 + 255) // 256,
            block_dim=256,
        )
    ctx.enqueue_copy(result.unsafe_ptr(), dow)
    ctx.enqueue_copy(result_masks.unsafe_ptr(), dov)
    ctx.enqueue_copy(result_errors.unsafe_ptr(), doe)
    ctx.enqueue_copy(starts.unsafe_ptr(), ds)
    ctx.enqueue_copy(sorted_rows.unsafe_ptr(), di)
    ctx.synchronize()
    # Independent canonical bit keys determine every expected boundary.
    var groups = 0
    for at in range(n):
        var row = Int(sorted_rows[at])
        var previous = Int(sorted_rows[max(0, at - 1)])
        var aw = words[row] & 0x7FFFFFFFFFFFFFFF
        var bw = words[previous] & 0x7FFFFFFFFFFFFFFF
        var af = (
            UInt64.MAX if aw
            > 0x7FF0000000000000 else UInt64(0) if aw
            == 0 else words[row]
        )
        var bf = (
            UInt64.MAX if bw
            > 0x7FF0000000000000 else UInt64(0) if bw
            == 0 else words[previous]
        )
        var same = masks[row] == masks[previous] and (
            masks[row] == 0 or af == bf
        )
        same &= masks[capacity + row] == masks[capacity + previous] and (
            masks[capacity + row] == 0
            or words[capacity + row] == words[capacity + previous]
        )
        if at == 0 or not same:
            assert_equal(starts[groups], UInt32(at))
            groups += 1
        else:
            assert_true(row > previous)
    assert_equal(shape[0], UInt32(groups))
    for group in range(groups):
        var begin = Int(starts[group])
        var end = Int(starts[group + 1]) if group + 1 < groups else n
        var representative = Int(sorted_rows[begin])
        assert_equal(result[5 * capacity + group], UInt64(representative))
        for slot in range(2):
            assert_equal(
                result[slot * capacity + group],
                words[slot * capacity + representative],
            )
            assert_equal(
                result_masks[slot * capacity + group],
                masks[slot * capacity + representative],
            )
        for column in range(5):
            var node = column + 3
            var op = code[node * 4 + 3]
            var source_slot = Int(code[node * 4 + 1])
            var expected: Tuple[UInt64, UInt8, UInt32]
            if op == ROW_QUANTILE_LINEAR:
                var value = _indexed_expected(
                    words,
                    masks,
                    sorted_rows,
                    source_slot * capacity,
                    begin,
                    end,
                )
                expected = (value[0], value[1], 0)
                var samples = List[Float64]()
                for at in range(begin, end):
                    var source = source_slot * capacity + Int(sorted_rows[at])
                    if masks[source]:
                        samples.append(bitcast[DType.float64](words[source]))
                var cpu = quantile_of(samples^, 0.37, "linear")
                assert_equal(value[1] != 0, cpu.__bool__())
                if cpu:
                    assert_true(
                        abs(bitcast[DType.float64](value[0]) - cpu.value())
                        < 1e-12
                    )
            elif op == VAR:
                var state = StatisticalState()
                for at in range(begin, end):
                    var source = source_slot * capacity + Int(sorted_rows[at])
                    state.add(
                        words[source], 0, masks[source] != 0, errors[source], op
                    )
                expected = state.finish(op, Int64(1))
            else:
                var state = BasicState()
                for at in range(begin, end):
                    var row = Int(sorted_rows[at])
                    var source = source_slot * capacity + row
                    state.add(
                        words[source],
                        masks[source] != 0,
                        errors[source],
                        UInt64(at - begin if op == ARG_MIN else row),
                        op,
                        slot_types[source_slot],
                    )
                expected = state.finish(
                    op,
                    slot_types[source_slot],
                    slot_types[column + 6],
                    0,
                    UInt64(end - begin),
                )
            assert_equal(result[(column + 6) * capacity + group], expected[0])
            assert_equal(
                result_masks[(column + 6) * capacity + group], expected[1]
            )
            assert_equal(
                result_errors[(column + 6) * capacity + group], expected[2]
            )


def main() raises:
    check(0, False)
    check(1, False)
    check(257, False)
    check(1027, False)
    check(257, True)
    check(257, False, True)
    print(
        "PASS Mojo grouping: IEEE/mixed/null keys, stable starts, basic/stat"
        " reductions, local args, key errors"
    )
