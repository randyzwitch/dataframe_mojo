"""Compare direct Mojo GPU decimal helpers with independent integer vectors.

Generate the binary input using mojo_gpu_decimal_probe.py. No native dataframe
library or Metal source is loaded by this driver.
"""
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import Pointer
from std.sys import argv
from std.testing import assert_equal, assert_true
from dataframe.accelerator.gpu import decimal as d

comptime INPUTS = 10
comptime OUTPUTS = 4
comptime RECORD = INPUTS + OUTPUTS


def evaluate(
    inputs: Pointer[UInt64, ImmutAnyOrigin],
    outputs: Pointer[UInt64, MutAnyOrigin],
    count: UInt32,
):
    var row = global_idx.x
    if row >= Int(count):
        return
    var at = row * RECORD
    var op = inputs[unsafe_offset=at]
    var a = d.Wide(inputs[unsafe_offset=at + 1], inputs[unsafe_offset=at + 2])
    var b = d.Wide(inputs[unsafe_offset=at + 3], inputs[unsafe_offset=at + 4])
    var left_scale = UInt32(inputs[unsafe_offset=at + 5])
    var right_scale = UInt32(inputs[unsafe_offset=at + 6])
    var option = UInt32(inputs[unsafe_offset=at + 7])
    var bits = UInt32(inputs[unsafe_offset=at + 8])
    var flags = inputs[unsafe_offset=at + 9]
    var value = d.Wide(0)
    var extra = d.Wide(0)
    if op == 0:
        value = d.wide_add(a, b)
    elif op == 1:
        value = d.wide_subtract(a, b)
    elif op == 2:
        value = d.wide_multiply(a, b)
    elif op == 3 or op == 4:
        var parts = d.wide_divide(a, b) if op == 3 else d.signed_divide(a, b)
        value = parts.value
        extra = parts.remainder
    elif op == 5:
        value = d.half_even(a, left_scale)
    elif op == 6:
        var result = d.scale_checked(a, left_scale)
        value = result.value
        extra.low = UInt64(result.valid)
    elif op == 7 or op == 8:
        var result = d.add_subtract(a, b, left_scale, right_scale, op == 8)
        value = result.value
        extra.low = UInt64(result.error)
    elif op == 9:
        var result = d.multiply(
            a, b, left_scale, right_scale, option if option else UInt32(38)
        )
        value = result.value
        extra.low = UInt64(result.error)
    elif op == 10:
        var result = d.divide_rows(a, b, left_scale, right_scale)
        value = result.value
        extra.low = UInt64(result.error)
    elif op == 11:
        var result = d.unary(a, bits, flags != 0)
        value = result.value
        extra.low = UInt64(result.error)
    elif op == 12:
        var result = d.compare(a, b, left_scale, right_scale, option)
        value.low = UInt64(result.value)
        extra.low = UInt64(result.valid)
    elif op == 13:
        var result = d.cast(
            a,
            left_scale,
            right_scale,
            option,
            bits,
            (flags & 1) != 0,
            (flags & 2) != 0,
        )
        value = result.value
        extra.low = UInt64(result.valid)
    elif op == 14:
        value.low = d.mean(a.low, a.high, b.low, left_scale)
    elif op == 15:
        value.low = d.mean128(a.low, a.high, b.low, left_scale)
    elif op == 16:
        value = d.wide_pow10(left_scale)
    elif op == 17:
        value.low = UInt64(d.wide_ge(a, b))
    elif op == 18:
        value.low = UInt64(d.signed_less(a, b))
    elif op == 19:
        value = d.negate(a)
    elif op == 20:
        value = d.increment(a)
    elif op == 21:
        var parts = d.divide(a.low, a.high, b.low)
        value.low = parts.value
        extra.low = parts.remainder
    outputs[unsafe_offset=row * OUTPUTS] = value.low
    outputs[unsafe_offset=row * OUTPUTS + 1] = value.high
    outputs[unsafe_offset=row * OUTPUTS + 2] = extra.low
    outputs[unsafe_offset=row * OUTPUTS + 3] = extra.high


def main() raises:
    var args = argv()
    assert_equal(len(args), 2)
    var bytes = List[UInt8]()
    with open(args[1], "r") as file:
        bytes = file.read_bytes()
    assert_true(len(bytes) % (8 * RECORD) == 0)
    var count = len(bytes) // (8 * RECORD)
    assert_true(count > 1000)
    var inputs = List[UInt64](length=count * RECORD, fill=0)
    for i in range(len(inputs)):
        for byte in range(8):
            inputs[i] |= UInt64(bytes[i * 8 + byte]) << UInt64(8 * byte)
    var results = List[UInt64](length=count * OUTPUTS, fill=0)
    var ctx = DeviceContext()
    var device_inputs = ctx.enqueue_create_buffer[DType.uint64](len(inputs))
    var device_outputs = ctx.enqueue_create_buffer[DType.uint64](len(results))
    ctx.enqueue_copy(device_inputs, inputs.unsafe_ptr())
    ctx.enqueue_function[evaluate](
        device_inputs.unsafe_ptr(),
        device_outputs.unsafe_ptr(),
        UInt32(count),
        grid_dim=(count + 127) // 128,
        block_dim=128,
    )
    ctx.enqueue_copy(results.unsafe_ptr(), device_outputs)
    ctx.synchronize()
    for row in range(count):
        for word in range(OUTPUTS):
            var expected = inputs[row * RECORD + INPUTS + word]
            var actual = results[row * OUTPUTS + word]
            if expected != actual:
                print(
                    "Decimal mismatch row/op/word/expected/actual",
                    row,
                    inputs[row * RECORD],
                    word,
                    expected,
                    actual,
                )
            assert_equal(actual, expected)
    print("PASS Mojo GPU decimal independent integer cases:", count)
