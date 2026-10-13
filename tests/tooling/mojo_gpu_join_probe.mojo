"""Direct GPU joins checked against independent typed source-pair enumeration."""
from max.gpu.host import DeviceContext
from std.testing import assert_equal, assert_true
from dataframe.accelerator.gpu import joins as j
from tests.tooling.mojo_gpu_order_probe import _value
from tests.support.key_kernel_reference import KeyCell, cell_equal


def _matches(
    left: List[KeyCell], right: List[KeyCell], nulls_equal: Bool
) -> Bool:
    for key in range(len(left)):
        if not nulls_equal and (not left[key].valid or not right[key].valid):
            return False
        if not cell_equal(left[key], right[key]):
            return False
    return True


def _pairs(
    left: List[List[KeyCell]],
    right: List[List[KeyCell]],
    how: Int,
    nulls_equal: Bool,
) -> Tuple[List[Int], List[Int], Int]:
    var a = List[Int]()
    var b = List[Int]()
    var probe_total = 0
    var matched = List[Bool](length=len(right), fill=False)
    if how == 2:
        for r in range(len(right)):
            var found = False
            for l in range(len(left)):
                if _matches(left[l], right[r], nulls_equal):
                    a.append(l)
                    b.append(r)
                    found = True
            if not found:
                a.append(-1)
                b.append(r)
        probe_total = len(a)
        return (a^, b^, probe_total)
    for l in range(len(left)):
        var found = False
        for r in range(len(right)):
            if _matches(left[l], right[r], nulls_equal or how == 7):
                found = True
                matched[r] = True
                if how != 4 and how != 5:
                    a.append(l)
                    b.append(r)
        if (
            (how == 4 and found)
            or (how == 5 and not found)
            or (not found and (how == 1 or how == 3 or how == 7))
        ):
            a.append(l)
            b.append(-1)
    probe_total = len(a)
    if how == 3:
        for r in range(len(right)):
            if not matched[r]:
                a.append(-1)
                b.append(r)
    return (a^, b^, probe_total)


def _case(
    kind: Int,
    left_rows: Int,
    right_rows: Int,
    how: Int,
    nulls_equal: Bool,
    limit: Int = -1,
) raises:
    var left_capacity = left_rows + 5
    var right_capacity = right_rows + 7
    var cells = 3 * left_capacity + 3 * right_capacity
    var words = List[UInt64](length=cells, fill=0xBAD0)
    var valid = List[UInt8](length=cells, fill=0)
    var errors = List[UInt32](length=cells, fill=0)
    var columns = List[Int64]()
    var left = List[List[KeyCell]]()
    var right = List[List[KeyCell]]()
    for side in range(2):
        var rows = right_rows if side else left_rows
        var capacity = right_capacity if side else left_capacity
        var base = 3 * left_capacity + 3 if side else 2
        columns.extend([Int64(base), Int64(rows), Int64(kind)])
        columns.extend([Int64(base + capacity), Int64(rows), Int64(3)])
        columns.extend([Int64(base + 2 * capacity), Int64(rows), Int64(3)])
        for row in range(rows):
            var value = _value(kind, row + side * 3)
            var first = value[1].copy()
            first.valid = row % 11 != 3
            var second = KeyCell(0, row % 13 != 6, integer=Int128(row % 5))
            words[base + row] = value[0]
            valid[base + row] = UInt8(first.valid)
            words[base + capacity + row] = UInt64(row % 5)
            valid[base + capacity + row] = UInt8(second.valid)
            words[base + 2 * capacity + row] = UInt64(row + side * 100_000)
            valid[base + 2 * capacity + row] = UInt8(row % 17 != 4)
            errors[base + 2 * capacity + row] = UInt32(row % 19 == 2) * 16
            if side:
                right.append([first.copy(), second.copy()])
            else:
                left.append([first.copy(), second.copy()])
    var keys: List[Int64] = [0, 3, 1, 4]
    var outputs: List[Int64] = [
        0,
        3,
        Int64(how == 2),
        2,
        2,
        -1,
        0,
        0,
        -1,
        5,
        0,
        3,
        0,
        3,
        1,
        1,
    ]
    var expected_left = List[Int]()
    var expected_right = List[Int]()
    var full_total = 0
    var probe_total = 0
    if how == 6:
        full_total = left_rows * right_rows
        probe_total = full_total
        var rows = min(full_total, limit) if limit >= 0 else full_total
        for row in range(rows):
            expected_left.append(row // right_rows)
            expected_right.append(row % right_rows)
    else:
        var pairs = _pairs(left, right, how, nulls_equal)
        probe_total = pairs[2]
        full_total = len(pairs[0])
        var rows = min(full_total, limit) if limit >= 0 else full_total
        for row in range(rows):
            expected_left.append(pairs[0][row])
            expected_right.append(pairs[1][row])
    var expected_rows = len(expected_left)
    var probe_side = UInt32(how == 2)
    var probe = right_rows if how == 2 else left_rows
    var build = left_rows if how == 2 else right_rows
    var pb = max(1, (probe + 255) // 256)
    var bb = max(1, (build + 255) // 256)
    var ctx = DeviceContext()
    var dw = ctx.enqueue_create_buffer[DType.uint64](cells)
    var dv = ctx.enqueue_create_buffer[DType.uint8](cells)
    var de = ctx.enqueue_create_buffer[DType.uint32](cells)
    var dc = ctx.enqueue_create_buffer[DType.int64](len(columns))
    var dk = ctx.enqueue_create_buffer[DType.int64](len(keys))
    var dout = ctx.enqueue_create_buffer[DType.int64](len(outputs))
    var sorted_index = ctx.enqueue_create_buffer[DType.uint32](
        max(1, build if how != 6 else 0)
    )
    var scratch = ctx.enqueue_create_buffer[DType.uint32](
        max(1, build if how != 6 else 0)
    )
    var ranges = ctx.enqueue_create_buffer[DType.uint64](max(1, probe))
    var prefix = ctx.enqueue_create_buffer[DType.uint64](max(1, probe))
    var blocks = ctx.enqueue_create_buffer[DType.uint64](pb)
    var diff = ctx.enqueue_create_buffer[DType.int32](
        max(1, build + 1 if how == 3 else 0)
    )
    var bprefix = ctx.enqueue_create_buffer[DType.uint64](
        max(1, build if how == 3 else 0)
    )
    var bblocks = ctx.enqueue_create_buffer[DType.uint64](bb if how == 3 else 1)
    var flags = ctx.enqueue_create_buffer[DType.uint32](
        max(1, build if how == 3 else 0)
    )
    var unmatched = ctx.enqueue_create_buffer[DType.uint32](
        max(1, build if how == 3 else 0)
    )
    var metadata = ctx.enqueue_create_buffer[DType.uint64](j.METADATA_WORDS)
    ctx.enqueue_copy(dw, words.unsafe_ptr())
    ctx.enqueue_copy(dv, valid.unsafe_ptr())
    ctx.enqueue_copy(de, errors.unsafe_ptr())
    ctx.enqueue_copy(dc, columns.unsafe_ptr())
    ctx.enqueue_copy(dk, keys.unsafe_ptr())
    ctx.enqueue_copy(dout, outputs.unsafe_ptr())
    if how != 6:
        ctx.enqueue_function[j.join_init](
            sorted_index.unsafe_ptr(), UInt32(build), grid_dim=bb, block_dim=256
        )
        var width = 1
        while width < build:
            ctx.enqueue_function[j.join_merge](
                dw.unsafe_ptr(),
                dv.unsafe_ptr(),
                sorted_index.unsafe_ptr(),
                scratch.unsafe_ptr(),
                dc.unsafe_ptr(),
                dk.unsafe_ptr(),
                UInt32(2),
                probe_side,
                UInt32(build),
                UInt32(width),
                grid_dim=bb,
                block_dim=256,
            )
            var previous = sorted_index^
            sorted_index = scratch^
            scratch = previous^
            width *= 2
    if how == 3:
        ctx.enqueue_function[j.join_zero](
            diff.unsafe_ptr(),
            UInt32(build),
            grid_dim=(build + 256) // 256,
            block_dim=256,
        )
    ctx.enqueue_function[j.join_match](
        dw.unsafe_ptr(),
        dv.unsafe_ptr(),
        sorted_index.unsafe_ptr(),
        dc.unsafe_ptr(),
        dk.unsafe_ptr(),
        ranges.unsafe_ptr(),
        prefix.unsafe_ptr(),
        diff.unsafe_ptr(),
        UInt32(0 if how == 6 else 2),
        probe_side,
        UInt32(probe),
        UInt32(build),
        UInt32(how),
        UInt32(nulls_equal),
        grid_dim=pb,
        block_dim=256,
    )
    ctx.enqueue_function[j.join_scan](
        prefix.unsafe_ptr(),
        blocks.unsafe_ptr(),
        diff.unsafe_ptr(),
        flags.unsafe_ptr(),
        UInt32(probe),
        UInt32(0),
        grid_dim=pb,
        block_dim=256,
    )
    var bounded = UInt64(limit) if limit >= 0 else UInt64.MAX
    ctx.enqueue_function[j.join_offsets](
        blocks.unsafe_ptr(),
        metadata.unsafe_ptr(),
        UInt32(pb),
        UInt32(0),
        bounded,
        grid_dim=1,
        block_dim=1,
    )
    if how == 3:
        ctx.enqueue_function[j.join_scan](
            bprefix.unsafe_ptr(),
            bblocks.unsafe_ptr(),
            diff.unsafe_ptr(),
            flags.unsafe_ptr(),
            UInt32(build),
            UInt32(1),
            grid_dim=bb,
            block_dim=256,
        )
        ctx.enqueue_function[j.join_offsets](
            bblocks.unsafe_ptr(),
            metadata.unsafe_ptr(),
            UInt32(bb),
            UInt32(1),
            bounded,
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[j.join_coverage](
            bprefix.unsafe_ptr(),
            bblocks.unsafe_ptr(),
            diff.unsafe_ptr(),
            sorted_index.unsafe_ptr(),
            flags.unsafe_ptr(),
            UInt32(build),
            grid_dim=bb,
            block_dim=256,
        )
        ctx.enqueue_function[j.join_scan](
            bprefix.unsafe_ptr(),
            bblocks.unsafe_ptr(),
            diff.unsafe_ptr(),
            flags.unsafe_ptr(),
            UInt32(build),
            UInt32(2),
            grid_dim=bb,
            block_dim=256,
        )
        ctx.enqueue_function[j.join_offsets](
            bblocks.unsafe_ptr(),
            metadata.unsafe_ptr(),
            UInt32(bb),
            UInt32(2),
            bounded,
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[j.join_unmatched](
            bprefix.unsafe_ptr(),
            bblocks.unsafe_ptr(),
            flags.unsafe_ptr(),
            unmatched.unsafe_ptr(),
            UInt32(build),
            grid_dim=bb,
            block_dim=256,
        )
    var shape = List[UInt64](length=j.METADATA_WORDS, fill=0)
    ctx.enqueue_copy(shape.unsafe_ptr(), metadata)
    ctx.synchronize()
    assert_equal(shape[0], UInt64(expected_rows))
    assert_equal(shape[1], UInt64(probe_total))
    assert_equal(shape[2], UInt64(full_total - probe_total))
    assert_equal(shape[3], UInt64(full_total))
    assert_equal(shape[4], UInt64(0))
    var capacity = max(1, expected_rows)
    var ow = ctx.enqueue_create_buffer[DType.uint64](4 * capacity)
    var ov = ctx.enqueue_create_buffer[DType.uint8](4 * capacity)
    var oe = ctx.enqueue_create_buffer[DType.uint32](4 * capacity)
    var ol = ctx.enqueue_create_buffer[DType.uint32](capacity)
    var oright = ctx.enqueue_create_buffer[DType.uint32](capacity)
    if expected_rows:
        var ob = (expected_rows + 255) // 256
        ctx.enqueue_function[j.join_pairs](
            sorted_index.unsafe_ptr(),
            ranges.unsafe_ptr(),
            prefix.unsafe_ptr(),
            blocks.unsafe_ptr(),
            unmatched.unsafe_ptr(),
            metadata.unsafe_ptr(),
            ol.unsafe_ptr(),
            oright.unsafe_ptr(),
            UInt32(probe),
            probe_side,
            UInt32(how),
            grid_dim=ob,
            block_dim=256,
        )
        ctx.enqueue_function[j.join_gather](
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            de.unsafe_ptr(),
            dc.unsafe_ptr(),
            dout.unsafe_ptr(),
            sorted_index.unsafe_ptr(),
            ranges.unsafe_ptr(),
            prefix.unsafe_ptr(),
            blocks.unsafe_ptr(),
            unmatched.unsafe_ptr(),
            metadata.unsafe_ptr(),
            ow.unsafe_ptr(),
            ov.unsafe_ptr(),
            oe.unsafe_ptr(),
            UInt32(probe),
            probe_side,
            UInt32(how),
            UInt32(4),
            grid_dim=ob,
            block_dim=256,
        )
        var result = List[UInt64](length=4 * capacity, fill=0)
        var masks = List[UInt8](length=4 * capacity, fill=0)
        var failures = List[UInt32](length=4 * capacity, fill=0)
        var a = List[UInt32](length=capacity, fill=0)
        var b = List[UInt32](length=capacity, fill=0)
        ctx.enqueue_copy(result.unsafe_ptr(), ow)
        ctx.enqueue_copy(masks.unsafe_ptr(), ov)
        ctx.enqueue_copy(failures.unsafe_ptr(), oe)
        ctx.enqueue_copy(a.unsafe_ptr(), ol)
        ctx.enqueue_copy(b.unsafe_ptr(), oright)
        ctx.synchronize()
        for row in range(expected_rows):
            var l = expected_left[row]
            var r = expected_right[row]
            assert_equal(a[row], UInt32(l) if l >= 0 else j.MISSING)
            assert_equal(b[row], UInt32(r) if r >= 0 else j.MISSING)
            for column in range(4):
                var at = column * j.OUTPUT_WORDS
                var source = Int(outputs[at])
                var selected = l
                if (outputs[at + 2] != 0 and r >= 0) or (source < 0 or l < 0):
                    source = Int(outputs[at + 1])
                    selected = r
                var expected_word = UInt64(0)
                var expected_valid = UInt8(0)
                var expected_error = UInt32(0)
                if source >= 0 and selected >= 0:
                    var index = Int(columns[source * j.COLUMN_WORDS]) + selected
                    expected_valid = valid[index]
                    expected_error = errors[index]
                    if expected_valid:
                        expected_word = words[index]
                var target = Int(outputs[at + 3]) * capacity + row
                assert_equal(result[target], expected_word)
                assert_equal(masks[target], expected_valid)
                assert_equal(failures[target], expected_error)
    print(
        "JOIN CASE",
        kind,
        left_rows,
        right_rows,
        how,
        nulls_equal,
        limit,
        expected_rows,
    )


def main() raises:
    for kind in range(1, 12):
        for how in range(8):
            _case(kind, 17, 19, how, False)
            _case(kind, 257, 263, how, False, 513)
        _case(kind, 257, 263, 3, True)
        _case(kind, 257, 263, 2, True)
    for how in range(8):
        for shape in range(3):
            _case(
                11, 0 if shape != 2 else 17, 0 if shape != 1 else 17, how, False
            )
        _case(11, 17, 19, how, False, 0)
        _case(11, 17, 19, how, False, 1)
    _case(3, 50_000, 50_000, 6, False, 513)
    assert_true(j.workspace_bytes(50_000, 50_000, 6) < 1_000_000)
    assert_equal(j.result_bytes(513, 4), 26_676)
    print(
        "PASS Mojo GPU joins: all scalar kinds, seven styles, null/IEEE keys, empty sides, stable exact gathers/errors, and bounded 2.5-billion-row fanout"
    )
