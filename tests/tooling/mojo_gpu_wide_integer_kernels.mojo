from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import Pointer, bitcast
from std.testing import assert_equal
from dataframe.accelerator.gpu import wide_integer_sum


def initialize(values: Pointer[UInt64, MutAnyOrigin], a: UInt64, b: UInt64):
    if global_idx.x == 0:
        values[unsafe_offset=0] = a
        values[unsafe_offset=1] = b


def check[
    unsigned: Bool, multiply: Bool
](a: UInt64, b: UInt64, factor: UInt64, expected: UInt64, failed: Int64) raises:
    var ctx = DeviceContext(0, api="metal")
    var values = ctx.enqueue_create_buffer[DType.uint64](2)
    var bits = values.create_sub_buffer[DType.uint8](0, 1)
    var lows = ctx.enqueue_create_buffer[DType.int64](1)
    var highs = ctx.enqueue_create_buffer[DType.int64](1)
    var counts = ctx.enqueue_create_buffer[DType.uint32](1)
    var faults = ctx.enqueue_create_buffer[DType.uint32](1)
    var result = ctx.enqueue_create_buffer[DType.int64](3)
    ctx.enqueue_function[initialize](values, a, b, grid_dim=1, block_dim=1)
    comptime if unsigned:
        ctx.enqueue_function[
            wide_integer_sum.partials[True, False, False, multiply]
        ](
            values,
            bits,
            UInt32(0),
            UInt32(2),
            UInt32(256),
            UInt64(0),
            factor,
            lows,
            highs,
            counts,
            faults,
            grid_dim=1,
            block_dim=256,
        )
    else:
        var signed_values = values.create_sub_buffer[DType.int64](0, 2)
        ctx.enqueue_function[
            wide_integer_sum.partials[False, False, False, multiply]
        ](
            signed_values,
            bits,
            UInt32(0),
            UInt32(2),
            UInt32(256),
            UInt64(0),
            factor,
            lows,
            highs,
            counts,
            faults,
            grid_dim=1,
            block_dim=256,
        )
    ctx.enqueue_function[wide_integer_sum.finish[unsigned]](
        lows,
        highs,
        counts,
        faults,
        UInt32(1),
        UInt32(0),
        Int64(0),
        result,
        grid_dim=1,
        block_dim=256,
    )
    ctx.synchronize()
    with result.map_to_host() as record:
        assert_equal(record[2], failed)
        if not failed:
            assert_equal(bitcast[DType.uint64](record[0]), expected)
            assert_equal(record[1], Int64(2))


def main() raises:
    check[True, True](UInt64.MAX, 0, 2, 0, 1)
    check[True, True](UInt64.MAX, 0, 1, UInt64.MAX, 0)
    check[True, True](UInt64.MAX, 0, 0, 0, 0)
    check[True, True](UInt64(1) << 32, 0, UInt64(1) << 32, 0, 1)
    check[False, True](UInt64(1) << 63, 0, UInt64.MAX, 0, 1)
    check[False, True](UInt64(1) << 63, 0, 1, UInt64(1) << 63, 0)
    check[False, True](UInt64(Int64.MAX), 0, 1, UInt64(Int64.MAX), 0)
    check[False, True](UInt64.MAX - 2, 0, 2, UInt64.MAX - 5, 0)
    check[True, False](UInt64.MAX, 1, 1, 0, 1)
    check[False, False](UInt64(Int64.MAX), 1, 1, 0, 1)
    check[False, False](UInt64(1) << 63, UInt64.MAX, 1, 0, 1)
    print(
        "PASS 11 exact wide integer kernel multiplication and final-overflow"
        " probes"
    )
