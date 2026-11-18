"""Reduce retained source buffers through CPU-owned row selections."""
from std.memory import Pointer, bitcast
from max.gpu import block_idx, thread_idx
from max.gpu.primitives import block
from .multi_reduce import Bytes
from .filter_sum import BLOCK_SIZE, _valid
from .float_predicate import _ordered
from .exact_float_state import ExactFloatState

# Per segment: first block, block count, row count, source offset, selection
# offset (MAX: a range, MAX-1: a constant), and the first output partial.
comptime SELECTION_WORDS = 6
# Per partial: split integer sum, valid count, extrema, numeric count, flags.
comptime SELECTION_STATE_WORDS = 7


def selection_partials[
    dtype: DType, sum_values: Bool, extrema: Bool
](
    values: Bytes,
    validity: Bytes,
    validity_offset: UInt32,
    nullable: UInt32,
    selections: Pointer[UInt32, UntrackedOrigin[mut=False]],
    descriptors: Pointer[UInt64, UntrackedOrigin[mut=False]],
    first_segment: UInt32,
    segments: UInt32,
    states: Pointer[Int64, MutUntrackedOrigin],
    bins: Pointer[Int64, MutUntrackedOrigin],
    specials: Pointer[UInt32, MutUntrackedOrigin],
):
    comptime assert (
        dtype == DType.int32 or dtype == DType.int64 or dtype == DType.float32
    )
    var first = 0
    var end = Int(segments)
    while first + 1 < end:
        var middle = (first + end) >> 1
        var at = (Int(first_segment) + middle) * SELECTION_WORDS
        if Int(descriptors[unsafe_offset=at]) <= block_idx.x:
            first = middle
        else:
            end = middle
    var meta = descriptors.unsafe_offset(
        (Int(first_segment) + first) * SELECTION_WORDS
    )
    var local = block_idx.x - Int(meta[unsafe_offset=0])
    var stride = Int(meta[unsafe_offset=1]) * BLOCK_SIZE
    var count = Int(meta[unsafe_offset=2])
    var base = Int(meta[unsafe_offset=3])
    var selection = meta[unsafe_offset=4]
    var output = Int(meta[unsafe_offset=5]) + local
    var row = local * BLOCK_SIZE + thread_idx.x
    var high = Int64(0)
    var low = UInt64(0)
    var valid_count = UInt32(0)
    var numeric_count = UInt32(0)
    var minimum = Int64.MAX
    var maximum = Int64.MIN
    var flags = UInt32(0)
    var exact = ExactFloatState[True, 1]()
    while row < count:
        var source = base + row
        if selection == UInt64.MAX - 1:
            source = 0
        elif selection != UInt64.MAX:
            source = base + Int(selections[unsafe_offset=Int(selection) + row])
        var valid = True
        if nullable:
            valid = _valid(validity, validity_offset, source)
        if valid:
            valid_count += 1
            comptime if dtype == DType.float32:
                comptime if sum_values or extrema:
                    var raw = values.unsafe_bitcast[UInt32]()[
                        unsafe_offset=source
                    ]
                    comptime if sum_values:
                        exact.add[0](bitcast[DType.float32](raw))
                    comptime if extrema:
                        var magnitude = raw & 0x7FFF_FFFF
                        if magnitude > 0x7F80_0000:
                            flags |= 1
                        else:
                            if magnitude == 0:
                                flags |= UInt32(1) << (
                                    UInt32(20) if raw >> 31 else UInt32(10)
                                )
                            var value = Int64(_ordered(raw))
                            minimum = min(minimum, value)
                            maximum = max(maximum, value)
                            numeric_count += 1
            else:
                comptime if sum_values or extrema:
                    var value = Int64(
                        values.unsafe_bitcast[Scalar[dtype]]()[
                            unsafe_offset=source
                        ]
                    )
                    comptime if sum_values:
                        comptime if dtype == DType.int64:
                            high += value >> 32
                            low += UInt64(value & 0xFFFF_FFFF)
                        else:
                            high += value
                    comptime if extrema:
                        minimum = min(minimum, value)
                        maximum = max(maximum, value)
                        numeric_count += 1
        row += stride
    var h = block.sum[block_size=BLOCK_SIZE, broadcast=False](high)
    var l = block.sum[block_size=BLOCK_SIZE, broadcast=False](low)
    var n = block.sum[block_size=BLOCK_SIZE, broadcast=False](valid_count)
    var nn = block.sum[block_size=BLOCK_SIZE, broadcast=False](numeric_count)
    var lo = block.min[block_size=BLOCK_SIZE, broadcast=False](minimum)
    var hi = block.max[block_size=BLOCK_SIZE, broadcast=False](maximum)
    var special = block.sum[block_size=BLOCK_SIZE, broadcast=False](flags)
    comptime if dtype == DType.float32 and sum_values:
        exact.finish[0](output, bins, specials)
    if thread_idx.x == 0:
        var at = output * SELECTION_STATE_WORDS
        states[unsafe_offset=at] = h
        states[unsafe_offset=at + 1] = Int64(l)
        states[unsafe_offset=at + 2] = Int64(n)
        states[unsafe_offset=at + 3] = lo
        states[unsafe_offset=at + 4] = hi
        states[unsafe_offset=at + 5] = Int64(nn)
        states[unsafe_offset=at + 6] = Int64(
            UInt32((special & 1023) != 0)
            | (UInt32(((special >> 10) & 1023) != 0) << 1)
            | (UInt32((special >> 20) != 0) << 2)
        )
