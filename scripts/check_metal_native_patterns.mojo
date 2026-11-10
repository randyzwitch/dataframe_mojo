"""Leaf fill validity/payload contracts on the actual Metal device."""
from std.memory import Pointer, bitcast
from std.testing import assert_equal
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from dataframe.expr import FILL_NULL, FILL_NAN
from dataframe_metal.accelerator.gpu.native_fill import native_fill_bits


def _probe(
    left: Pointer[UInt64, MutUntrackedOrigin],
    right: Pointer[UInt64, MutUntrackedOrigin],
    flags: Pointer[UInt32, MutUntrackedOrigin],
    output: Pointer[UInt64, MutUntrackedOrigin],
    rows: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var bits = flags.unsafe_offset(row).unsafe_bitcast[UInt8]()
    var a = left.unsafe_offset(row).unsafe_bitcast[UInt8]()
    var b = right.unsafe_offset(row).unsafe_bitcast[UInt8]()
    var at = row * 80
    for tag in range(3):
        for op in range(2 if tag == 2 else 1):
            for choice in range(5):
                var side = choice if choice < 3 else choice + 2
                for requested in range(2):
                    var want = Bool(requested)
                    var literal = (
                        left[unsafe_offset=row] if side & 3
                        == 1 else right[unsafe_offset=row]
                    )
                    if tag == 0:
                        literal = bitcast[DType.uint64](
                            Int64(bitcast[DType.int32](UInt32(literal)))
                        )
                    elif tag == 2:
                        literal = UInt64(UInt32(literal))
                    var raw = UInt32(left[unsafe_offset=row])
                    var reused = tag == 2 and side & 3 != 1
                    var result = native_fill_bits(
                        a,
                        b,
                        bits,
                        bits,
                        0,
                        1,
                        0,
                        UInt32(tag),
                        UInt32(FILL_NAN if op else FILL_NULL),
                        3,
                        UInt32(side),
                        literal,
                        want,
                        reused,
                        raw,
                        Bool(flags[unsafe_offset=row] & 1),
                    )
                    output[unsafe_offset=at] = result[0]
                    output[unsafe_offset=at + 1] = UInt64(result[1])
                    at += 2


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
    var flags = List[UInt32]()
    for a in edges:
        for b in edges:
            for mask in range(4):
                left.append(a)
                right.append(b)
                flags.append(UInt32(mask))
    var n = len(left)
    var scope = Scope()
    var ctx = scope.context()
    var a = ctx.enqueue_create_buffer[DType.uint64](n)
    var b = ctx.enqueue_create_buffer[DType.uint64](n)
    var masks = ctx.enqueue_create_buffer[DType.uint32](n)
    var result = ctx.enqueue_create_buffer[DType.uint64](n * 80)
    var actual = List[UInt64](length=n * 80, fill=UInt64.MAX)
    ctx.enqueue_copy(a, left.unsafe_ptr())
    ctx.enqueue_copy(b, right.unsafe_ptr())
    ctx.enqueue_copy(masks, flags.unsafe_ptr())
    ctx.enqueue_function[_probe](
        a, b, masks, result, UInt32(n), grid_dim=(n + 255) // 256, block_dim=256
    )
    ctx.enqueue_copy(actual.unsafe_ptr(), result)
    scope.synchronize()
    var checked = 0
    for row in range(n):
        var at = row * 80
        for tag in range(3):
            for op in range(2 if tag == 2 else 1):
                for side in [0, 1, 2, 5, 6]:
                    for want in [False, True]:
                        var av = not Bool(side & 4) if side & 3 == 1 else Bool(
                            flags[row] & 1
                        )
                        var bv = not Bool(side & 4) if side & 3 == 2 else Bool(
                            flags[row] & 2
                        )
                        var x = left[row]
                        var y = right[row]
                        if tag == 0:
                            x = bitcast[DType.uint64](
                                Int64(bitcast[DType.int32](UInt32(x)))
                            )
                            y = bitcast[DType.uint64](
                                Int64(bitcast[DType.int32](UInt32(y)))
                            )
                        elif tag == 2:
                            x = UInt64(UInt32(x))
                            y = UInt64(UInt32(y))
                        var replace = (
                            av
                            and (UInt32(x) & 0x7FFFFFFF)
                            > 0x7F800000 if op else not av
                        )
                        var valid = bv if replace else av
                        var expected = y if replace else x
                        assert_equal(actual[at + 1], UInt64(valid))
                        if valid and want:
                            assert_equal(actual[at], expected)
                        at += 2
                        checked += 1
    scope.mark_success()
    print(
        "PASS native leaf fill masks, literals/nulls, shared payloads and COUNT:",
        checked,
    )
