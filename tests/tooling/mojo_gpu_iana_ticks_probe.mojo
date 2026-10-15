"""Execute zone-aware tick/calendar cases on device against CPU reference."""
from max.gpu import global_idx
from std.memory import Pointer
from std.testing import assert_equal
from dataframe.accelerator.gpu.residency import Scope
from tests.tooling.iana_ticks_cases import build_cases, evaluate


def _evaluate(
    tables: Pointer[UInt64, ImmutAnyOrigin],
    inputs: Pointer[UInt64, ImmutAnyOrigin],
    output: Pointer[UInt64, MutAnyOrigin],
    count: UInt32,
):
    var row = global_idx.x
    if row >= Int(count):
        return
    var result = evaluate(tables, inputs, row)
    output[unsafe_offset=3 * row] = result[0] if result[1] and not result[
        2
    ] else UInt64(0)
    output[unsafe_offset=3 * row + 1] = UInt64(result[1])
    output[unsafe_offset=3 * row + 2] = UInt64(result[2])


def main() raises:
    var fixture = build_cases()
    var count = len(fixture.inputs) // 11
    var scope = Scope()
    var context = scope.context()
    var device_tables = context.enqueue_create_buffer[DType.uint64](
        len(fixture.tables)
    )
    var device_inputs = context.enqueue_create_buffer[DType.uint64](
        len(fixture.inputs)
    )
    var device_output = context.enqueue_create_buffer[DType.uint64](
        len(fixture.expected)
    )
    var actual = List[UInt64](length=len(fixture.expected), fill=0xBAD)
    try:
        context.enqueue_copy(device_tables, fixture.tables.unsafe_ptr())
        context.enqueue_copy(device_inputs, fixture.inputs.unsafe_ptr())
        context.enqueue_function[_evaluate](
            device_tables.unsafe_ptr(),
            device_inputs.unsafe_ptr(),
            device_output.unsafe_ptr(),
            UInt32(count),
            grid_dim=(count + 127) // 128,
            block_dim=128,
        )
        scope.read_control_into(device_output, actual)
    except error:
        scope.synchronize()
        _ = device_tables^
        _ = device_inputs^
        _ = device_output^
        _ = fixture^
        _ = actual^
        raise error^
    _ = device_tables^
    _ = device_inputs^
    _ = device_output^
    for index in range(len(actual)):
        assert_equal(actual[index], fixture.expected[index])
    _ = fixture^
    print(
        "PASS actual GPU IANA ticks/localize/relocalize/calendar cases:",
        count,
        flush=True,
    )
