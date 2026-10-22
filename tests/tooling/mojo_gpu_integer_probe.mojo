"""Device checks against independent Python integer arithmetic vectors."""
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import Pointer
from std.sys import argv
from std.testing import assert_equal, assert_true
from dataframe_metal.accelerator.gpu import integers

comptime INPUTS = 5
comptime OUTPUTS = 3
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
    var result = integers.evaluate(
        Int64(inputs[unsafe_offset=at]),
        inputs[unsafe_offset=at + 1],
        inputs[unsafe_offset=at + 2],
        UInt32(inputs[unsafe_offset=at + 3]),
        inputs[unsafe_offset=at + 4] != 0,
    )
    outputs[unsafe_offset=row * OUTPUTS] = result[0]
    outputs[unsafe_offset=row * OUTPUTS + 1] = UInt64(result[1])
    outputs[unsafe_offset=row * OUTPUTS + 2] = UInt64(result[2])


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
                    "Integer mismatch row/op/word/expected/actual",
                    row,
                    inputs[row * RECORD],
                    word,
                    expected,
                    actual,
                )
            assert_equal(actual, expected)
    print("PASS Mojo GPU checked integer independent cases:", count)
