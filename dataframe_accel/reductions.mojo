"""Shared deterministic two-stage float/count reduction primitives."""
from max.gpu import thread_idx
from max.gpu.host import DeviceBuffer
from max.gpu.sync import barrier
from std.memory import AddressSpace, stack_allocation
from dataframe.accel_memory import ACCEL_THREADS as THREADS


@always_inline
def reduce_pair(total: Float64, count: Int64) -> Tuple[Float64, Int64]:
    # This pinned SDK does not support Float64 block.sum warp shuffles.
    var sums = stack_allocation[
        THREADS, DType.float64, address_space=AddressSpace.SHARED
    ]()
    var counts = stack_allocation[
        THREADS, DType.int64, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    sums[unsafe_offset=tid] = total
    counts[unsafe_offset=tid] = count
    barrier()
    comptime for step in range(8):
        comptime stride = THREADS >> (step + 1)
        if tid < stride:
            sums[unsafe_offset=tid] += sums[unsafe_offset=tid + stride]
            counts[unsafe_offset=tid] += counts[unsafe_offset=tid + stride]
        barrier()
    return (sums[unsafe_offset=0], counts[unsafe_offset=0])


def finish(
    sums: DeviceBuffer[DType.float64].device_type,
    counts: DeviceBuffer[DType.int64].device_type,
    output: DeviceBuffer[DType.float64].device_type,
    output_count: DeviceBuffer[DType.int64].device_type,
    blocks: Int64,
):
    var total = Float64(0)
    var count = Int64(0)
    var i = Int(thread_idx.x)
    while i < Int(blocks):
        total += sums[unsafe_offset=i]
        count += counts[unsafe_offset=i]
        i += THREADS
    var reduced = reduce_pair(total, count)
    if thread_idx.x == 0:
        output[unsafe_offset=0] = reduced[0]
        output_count[unsafe_offset=0] = reduced[1]
