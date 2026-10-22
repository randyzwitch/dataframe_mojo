"""Execute integer-only floating primitives on the GPU against host IEEE bits."""
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import Pointer
from std.sys import argv
from std.testing import assert_equal, assert_true
from dataframe_metal.accelerator.gpu import floating as f

comptime OPERATIONS = 41
comptime RECORD = OPERATIONS + 2


def evaluate(
    inputs: Pointer[UInt64, ImmutAnyOrigin],
    outputs: Pointer[UInt64, MutAnyOrigin],
    count: UInt32,
):
    var row = global_idx.x
    if row >= Int(count):
        return
    var a = inputs[unsafe_offset=row * RECORD]
    var b = inputs[unsafe_offset=row * RECORD + 1]
    var p = f.from_float32(UInt32(a))
    var q = f.from_float32(UInt32(b))
    var at = row * OPERATIONS
    outputs[unsafe_offset=at] = f.add(a, b)
    outputs[unsafe_offset=at + 1] = f.sub(a, b)
    outputs[unsafe_offset=at + 2] = f.mul(a, b)
    outputs[unsafe_offset=at + 3] = f.div(a, b)
    outputs[unsafe_offset=at + 4] = UInt64(f.equal(a, b))
    outputs[unsafe_offset=at + 5] = UInt64(f.less(a, b))
    outputs[unsafe_offset=at + 6] = f.from_unsigned(a)
    outputs[unsafe_offset=at + 7] = f.from_signed(a.cast[DType.int64]())
    outputs[unsafe_offset=at + 8] = p
    outputs[unsafe_offset=at + 9] = UInt64(f.to_float32(a))
    outputs[unsafe_offset=at + 10] = f.integral(a, True)
    outputs[unsafe_offset=at + 11] = f.integral(a, False)
    outputs[unsafe_offset=at + 12] = f.sqrt(a)
    outputs[unsafe_offset=at + 13] = f.fma(a, b, (a >> 31) | (a << 33))
    outputs[unsafe_offset=at + 14] = f.fma(a, b, f.mul(a, b) ^ f.SIGN)
    outputs[unsafe_offset=at + 15] = UInt64(f.to_float32(f.add(p, q)))
    outputs[unsafe_offset=at + 16] = UInt64(f.to_float32(f.sub(p, q)))
    outputs[unsafe_offset=at + 17] = UInt64(f.to_float32(f.mul(p, q)))
    outputs[unsafe_offset=at + 18] = UInt64(f.to_float32(f.div(p, q)))
    outputs[unsafe_offset=at + 19] = f.fma(
        p, q, f.from_float32(UInt32((a >> 13) | (a << 19))), True
    )
    outputs[unsafe_offset=at + 20] = f.fma(
        p, q, f.from_float32(0x17800000), True
    )
    outputs[unsafe_offset=at + 21] = f.fma(
        p, q, f.from_float32(0x97800000), True
    )
    outputs[unsafe_offset=at + 22] = f.from_wide(a, b, True)
    outputs[unsafe_offset=at + 23] = f.from_wide(a, b, False)

    for pair in range(4):
        var converted = f.to_integer(
            a, UInt32(64 if pair < 2 else 8), pair % 2 != 0
        )
        outputs[unsafe_offset=at + 24 + 2 * pair] = converted[0] if converted[
            1
        ] else UInt64(0)
        outputs[unsafe_offset=at + 25 + 2 * pair] = UInt64(converted[1])
    outputs[unsafe_offset=at + 32] = f.round_integer(a)
    outputs[unsafe_offset=at + 33] = f.decimal_round(a, b, False)
    outputs[unsafe_offset=at + 34] = f.decimal_round(a, b, True)
    outputs[unsafe_offset=at + 35] = f.integral(f.div(a, b), True)
    outputs[unsafe_offset=at + 36] = f.modulo(a, b)
    outputs[unsafe_offset=at + 37] = UInt64(f.to_float32(f.round_integer(p)))
    outputs[unsafe_offset=at + 38] = UInt64(f.to_float32(f.sqrt(p)))
    var narrow_quotient = f.from_float32(f.to_float32(f.div(p, q)))
    outputs[unsafe_offset=at + 39] = UInt64(
        f.to_float32(f.integral(narrow_quotient, True))
    )
    outputs[unsafe_offset=at + 40] = f.modulo(p, q, True)


def main() raises:
    var args = argv()
    assert_equal(len(args), 2)
    var bytes = List[UInt8]()
    with open(args[1], "r") as file:
        bytes = file.read_bytes()
    assert_true(len(bytes) % (8 * RECORD) == 0)
    var count = len(bytes) // (8 * RECORD)
    assert_true(count > 100_000)
    var inputs = List[UInt64](length=count * RECORD, fill=0)
    # Decode explicitly so alignment of a byte allocation is irrelevant.
    for i in range(len(inputs)):
        for byte in range(8):
            inputs[i] |= UInt64(bytes[i * 8 + byte]) << UInt64(8 * byte)
    var result = List[UInt64](length=count * OPERATIONS, fill=0)
    var ctx = DeviceContext()
    var device_inputs = ctx.enqueue_create_buffer[DType.uint64](len(inputs))
    var device_outputs = ctx.enqueue_create_buffer[DType.uint64](len(result))
    ctx.enqueue_copy(device_inputs, inputs.unsafe_ptr())
    ctx.enqueue_function[evaluate](
        device_inputs.unsafe_ptr(),
        device_outputs.unsafe_ptr(),
        UInt32(count),
        grid_dim=(count + 255) // 256,
        block_dim=256,
    )
    ctx.enqueue_copy(result.unsafe_ptr(), device_outputs)
    ctx.synchronize()
    for row in range(count):
        for op in range(OPERATIONS):
            var expected = inputs[row * RECORD + 2 + op]
            var actual = result[row * OPERATIONS + op]
            var narrow = op == 9 or (op >= 15 and op <= 21) or op >= 37
            var integer = op >= 24 and op <= 31
            var nan = False if integer else (
                expected & 0x7FFFFFFF
            ) > 0x7F800000 if narrow else f.is_nan(expected)
            var actual_nan = (
                actual & 0x7FFFFFFF
            ) > 0x7F800000 if narrow else f.is_nan(actual)
            var same = actual_nan if nan else actual == expected
            if not same:
                print(
                    "IEEE mismatch row/op/a/b/expected/actual",
                    row,
                    op,
                    inputs[row * RECORD],
                    inputs[row * RECORD + 1],
                    expected,
                    actual,
                )
            assert_true(same)
    print("PASS Mojo GPU IEEE host differential cases:", count * OPERATIONS)
