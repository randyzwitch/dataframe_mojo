"""Real-device cumulative, shift and fill windows against scalar references."""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.math import isnan
from std.testing import assert_equal, assert_true
from dataframe.accelerator.rows import ROW_CUMULATIVE_KEY, ROW_SORT_KEY
from dataframe.accelerator.gpu.ordering import (
    order_init,
    order_merge,
    merge_grid,
)
from dataframe.accelerator.gpu.windows import window_assign, window_check_errors
from tests.tooling.mojo_gpu_order_probe import _value
from tests.support.key_kernel_reference import KeyCell, cell_equal


def _partition(
    rows: List[List[KeyCell]], a: Int, b: Int, grouped: Bool
) -> Bool:
    return not grouped or (
        cell_equal(rows[a][1], rows[b][1])
        and cell_equal(rows[a][2], rows[b][2])
    )


def _better(a: KeyCell, b: KeyCell, maximum: Bool) -> Bool:
    if a.kind == 1:
        var an = isnan(a.floating)
        var bn = isnan(b.floating)
        if an or bn:
            return an != bn and (an if maximum else bn)
        return a.floating > b.floating if maximum else a.floating < b.floating
    return a.integer > b.integer if maximum else a.integer < b.integer


def _case(kind: Int, n: Int, grouped: Bool) raises:
    var methods: List[Int64] = [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        6,
        6,
        6,
    ]
    var reverse: List[Bool] = [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    var parameters: List[Int64] = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -3,
        0,
        2,
        Int64.MIN,
        Int64.MAX,
        -1,
        0,
        2,
        -1,
        0,
        2,
    ]
    var count = len(methods)
    var capacity = n + 7
    var words = List[UInt64](length=(count + 3) * capacity, fill=0)
    var valid = List[UInt8](length=len(words), fill=0)
    var errors = List[UInt32](length=len(words), fill=0)
    var reference = List[List[KeyCell]]()
    var groups = List[List[Int]]()
    for row in range(n):
        var input = _value(kind, row)
        var value = input[1].copy()
        value.valid = row % 7 != 0 and row % 7 != 1
        var first = _value(11, row // 9)
        var partition = first[1].copy()
        partition.valid = row % 17 != 2
        reference.append(
            [
                value.copy(),
                partition.copy(),
                KeyCell(0, integer=Int128(row % 3)),
            ]
        )
        words[row] = input[0]
        valid[row] = UInt8(value.valid)
        words[capacity + row] = first[0]
        valid[capacity + row] = UInt8(partition.valid)
        words[2 * capacity + row] = UInt64(row % 3)
        valid[2 * capacity + row] = 1
        var found = -1
        for group in range(len(groups)):
            if _partition(reference, groups[group][0], row, grouped):
                found = group
                break
        if found < 0:
            found = len(groups)
            groups.append(List[Int]())
        groups[found].append(row)
    var expected = List[UInt64](length=count * n, fill=0)
    var expected_valid = List[UInt8](length=count * n, fill=0)
    var expected_errors = List[UInt32](length=count * n, fill=0)
    var output_types = List[UInt32]()
    for variant in range(count):
        var method = methods[variant]
        var output_type = kind
        if method == 3 or (
            method == 0 and (kind == 4 or (kind >= 5 and kind <= 8))
        ):
            output_type = 3
        output_types.append(UInt32(output_type))
        for group in groups:
            var state = Int128(0)
            var floating_state = Float64(0)
            var best = -1
            var distance = Int64(0)
            var failure = UInt32(0)
            var backwards = method == 6 if method >= 5 else reverse[variant]
            for offset in range(len(group)):
                var position = len(group) - 1 - offset if backwards else offset
                var row = group[position]
                var target = variant * n + row
                var present = reference[row][0].valid
                expected_errors[target] = failure
                if method == 4:
                    var parameter = parameters[variant]
                    if parameter >= 0:
                        if parameter > Int64(position):
                            continue
                    elif parameter < -Int64(len(group) - position - 1):
                        continue
                    var source = group[position - Int(parameter)]
                    expected[target] = words[source]
                    expected_valid[target] = valid[source]
                    continue
                if method >= 5:
                    if present:
                        best = row
                        distance = 0
                    else:
                        distance += 1
                    if best >= 0 and (
                        present
                        or parameters[variant] < 0
                        or distance <= parameters[variant]
                    ):
                        expected[target] = words[best]
                        expected_valid[target] = valid[best]
                    continue
                if method == 3:
                    state += Int128(present)
                    expected[target] = state.cast[DType.uint64]()
                    expected_valid[target] = 1
                    continue
                if not present:
                    continue
                expected_valid[target] = 1
                if method == 0:
                    if kind == 1 or kind == 11:
                        floating_state += reference[row][0].floating
                        expected[target] = UInt64(
                            bitcast[DType.uint32](Float32(floating_state))
                        ) if kind == 1 else bitcast[DType.uint64](
                            floating_state
                        )
                    else:
                        state += reference[row][0].integer
                        var low = Int128(Int64.MIN)
                        var high = Int128(Int64.MAX)
                        if output_type == 2:
                            low = Int128(Int32.MIN)
                            high = Int128(Int32.MAX)
                        elif output_type == 9:
                            low = 0
                            high = Int128(UInt32.MAX)
                        if (
                            state < low
                            or state > high
                            or (
                                kind == 10
                                and reference[row][0].integer
                                > Int128(Int64.MAX)
                            )
                        ):
                            failure |= 2
                        expected_errors[target] = failure
                        expected[target] = state.cast[DType.uint64]()
                else:
                    if best < 0 or _better(
                        reference[row][0], reference[best][0], method == 2
                    ):
                        best = row
                    expected[target] = words[best]
    var nodes = 3 if grouped else 1
    var code = List[Int64]()
    var types = List[Int64]()
    var literals = List[UInt64]()
    for variant in range(count):
        code.extend(
            [
                Int64(ROW_CUMULATIVE_KEY),
                0,
                Int64(reverse[variant]),
                methods[variant],
            ]
        )
        types.append(Int64(kind))
        literals.append(parameters[variant].cast[DType.uint64]())
        if grouped:
            code.extend(
                [Int64(ROW_SORT_KEY), 1, 0, 1, Int64(ROW_SORT_KEY), 2, 0, 1]
            )
            types.extend([11, 10])
            literals.extend([0, 0])
    var ctx = DeviceContext()
    var dw = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var dv = ctx.enqueue_create_buffer[DType.uint8](len(valid))
    var de = ctx.enqueue_create_buffer[DType.uint32](len(errors))
    var dc = ctx.enqueue_create_buffer[DType.int64](len(code))
    var dt = ctx.enqueue_create_buffer[DType.int64](len(types))
    var dl = ctx.enqueue_create_buffer[DType.uint64](len(literals))
    var a = ctx.enqueue_create_buffer[DType.uint32](n)
    var b = ctx.enqueue_create_buffer[DType.uint32](n)
    var blocks = (n + 255) // 256
    var df = ctx.enqueue_create_buffer[DType.uint32](blocks)
    ctx.enqueue_copy(dw, words.unsafe_ptr())
    ctx.enqueue_copy(dv, valid.unsafe_ptr())
    ctx.enqueue_copy(de, errors.unsafe_ptr())
    ctx.enqueue_copy(dc, code.unsafe_ptr())
    ctx.enqueue_copy(dt, types.unsafe_ptr())
    ctx.enqueue_copy(dl, literals.unsafe_ptr())
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
            UInt32(1),
            UInt32(nodes - 1),
            UInt32(n),
            UInt32(capacity),
            UInt32(width),
            UInt32(0),
            dl.unsafe_ptr(),
            grid_dim=merge_grid(n, width),
            block_dim=256,
        )
        var previous = a^
        a = b^
        b = previous^
        width *= 2
    for variant in range(count):
        ctx.enqueue_function[window_assign](
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            de.unsafe_ptr(),
            a.unsafe_ptr(),
            dc.unsafe_ptr(),
            dt.unsafe_ptr(),
            dl.unsafe_ptr(),
            UInt32(variant * nodes),
            UInt32(nodes),
            UInt32(variant + 3),
            output_types[variant],
            UInt32(n),
            UInt32(capacity),
            grid_dim=blocks,
            block_dim=256,
        )
    ctx.enqueue_function[window_check_errors](
        de.unsafe_ptr(),
        df.unsafe_ptr(),
        UInt32(3),
        UInt32(n),
        UInt32(capacity),
        grid_dim=blocks,
        block_dim=256,
    )
    var failures = List[UInt32](length=blocks, fill=0)
    ctx.enqueue_copy(words.unsafe_ptr(), dw)
    ctx.enqueue_copy(valid.unsafe_ptr(), dv)
    ctx.enqueue_copy(errors.unsafe_ptr(), de)
    ctx.enqueue_copy(failures.unsafe_ptr(), df)
    ctx.synchronize()
    for variant in range(count):
        for row in range(n):
            var actual = (variant + 3) * capacity + row
            var target = variant * n + row
            assert_equal(valid[actual], expected_valid[target])
            assert_equal(errors[actual], expected_errors[target])
            var expected_word = expected[target]
            var nan = (
                (expected_word & UInt64(0x7FFFFFFF))
                > UInt64(0x7F800000) if output_types[variant]
                == 1 else (expected_word & UInt64(0x7FFFFFFFFFFFFFFF))
                > UInt64(0x7FF0000000000000) if output_types[variant]
                == 11 else False
            )
            if nan and methods[variant] == 0:
                var word = words[actual]
                assert_true(
                    (word & UInt64(0x7FFFFFFF))
                    > UInt64(0x7F800000) if output_types[variant]
                    == 1 else (word & UInt64(0x7FFFFFFFFFFFFFFF))
                    > UInt64(0x7FF0000000000000)
                )
            else:
                assert_equal(words[actual], expected_word)
    for block in range(blocks):
        var expected_failures = UInt32(0)
        for row in range(block * 256, min(n, (block + 1) * 256)):
            expected_failures += UInt32(expected_errors[row] != 0)
        assert_equal(failures[block], expected_failures)


def main() raises:
    for kind in range(1, 12):
        for grouped in [False, True]:
            _case(kind, 257, grouped)
    for n in [1, 255, 256, 1_027]:
        _case(11, n, True)
    print(
        "PASS Mojo GPU windows: 26 layouts x19 variants, all scalar kinds,"
        " cumulative directions, checked prefixes, extreme shifts and bounded"
        " fills"
    )
