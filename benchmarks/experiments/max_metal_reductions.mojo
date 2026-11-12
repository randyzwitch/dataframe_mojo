"""Focused hardware check of existing MAX-backed Metal exact Float32 state.

Run on Apple Silicon: pixi run mojo run -I .
    benchmarks/experiments/max_metal_reductions.mojo
This bypasses query planning so it does not instantiate every query kernel.
"""
from max.gpu import thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives import block
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe_metal.accelerator.gpu.exact_float_state import ExactFloatState
from dataframe_metal.exact_float import finish_float_bins


def _reduce(
    words: DeviceBuffer[DType.uint32].device_type,
    valid: DeviceBuffer[DType.uint8].device_type,
    bins: DeviceBuffer[DType.int64].device_type,
    specials: DeviceBuffer[DType.uint32].device_type,
    count: DeviceBuffer[DType.int64].device_type,
):
    var row = Int(thread_idx.x)
    var state = ExactFloatState[True, 1]()
    if valid[unsafe_offset=row]:
        state.add[0](bitcast[DType.float32](words[unsafe_offset=row]))
    state.finish[0](
        0,
        bins.unsafe_origin_cast[MutUntrackedOrigin](),
        specials.unsafe_origin_cast[MutUntrackedOrigin](),
    )
    var total = block.sum[block_size=256, broadcast=False](
        Int64(valid[unsafe_offset=row] != 0)
    )
    var wide = block.sum[block_size=256, broadcast=False](
        (Int64(1) << 54) + Int64(row)
    )
    if row == 0:
        count[unsafe_offset=0] = total
        count[unsafe_offset=1] = wide


def main() raises:
    var ctx = DeviceContext()
    for pattern in range(4):
        var words = List[UInt32](length=256, fill=0)
        var valid = List[UInt8](length=256, fill=0)
        for row in range(256):
            valid[row] = UInt8(pattern != 0)
            if pattern <= 1:
                # Cancellation of huge values leaves 128 tiny subnormals.
                words[row] = UInt32(0x7F7FFFFF) if row % 4 == 0 else (
                    UInt32(0xFF7FFFFF) if row % 4 == 2 else UInt32(1)
                )
            elif pattern == 2:
                words[row] = 0x3F800000  # 1.0
            else:
                words[row] = UInt32(0x7FC00005) if row % 4 == 0 else (
                    UInt32(0x7F800000) if row % 4 == 1 else UInt32(0xFF800000)
                )
                valid[row] = UInt8(row % 4 != 3)
        var device_words = ctx.enqueue_create_buffer[DType.uint32](256)
        var device_valid = ctx.enqueue_create_buffer[DType.uint8](256)
        var bins = ctx.enqueue_create_buffer[DType.int64](9)
        var specials = ctx.enqueue_create_buffer[DType.uint32](6)
        var count = ctx.enqueue_create_buffer[DType.int64](2)
        ctx.enqueue_copy(device_words, words.unsafe_ptr())
        ctx.enqueue_copy(device_valid, valid.unsafe_ptr())
        ctx.enqueue_function[_reduce](
            device_words,
            device_valid,
            bins,
            specials,
            count,
            grid_dim=1,
            block_dim=256,
        )
        ctx.synchronize()
        var totals = List[Int128]()
        with bins.map_to_host() as host:
            for index in range(9):
                totals.append(Int128(host[index]))
        with count.map_to_host() as host:
            assert_equal(
                host[0],
                Int64(0 if pattern == 0 else (192 if pattern == 3 else 256)),
            )
            assert_equal(host[1], (Int64(1) << 62) + 32640)
        with specials.map_to_host() as host:
            for index in range(3):
                assert_equal(host[index], UInt32(64 if pattern == 3 else 0))
            assert_equal(host[5], UInt32(0))
            var result = finish_float_bins(
                totals, host[0] != 0, host[1] != 0, host[2] != 0
            )
            if pattern == 3:
                assert_true(isnan(result))
            else:
                var expected = Float64(0 if pattern == 0 else 256)
                if pattern == 1:
                    expected = bitcast[DType.float64](UInt64(1023 - 142) << 52)
                assert_equal(
                    bitcast[DType.uint64](result),
                    bitcast[DType.uint64](expected),
                )
    print(
        "Metal exact bins, subnormals, null counts, nonfinite flags, and Int64 MAX sums: PASS"
    )
