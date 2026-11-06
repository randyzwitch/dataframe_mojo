"""One native Float32 value, with shared aggregate and extrema state.

The binding ABI matches the general composition scan. Only v0/b0 are value
inputs: native single-field plans have no operand slots, literals or binary
operators to select. Float32 values and extrema use the same raw-bit load.
"""
from std.memory import Pointer, bitcast
from max.gpu import block_idx, thread_idx
from max.gpu.primitives import block
from .multi_reduce import Bytes, read_integer
from .composed_reduce import static_segment_info, HEADER_WORDS
from .float_predicate import float_in_range, _ordered
from .filter_sum import BLOCK_SIZE, _valid
from .exact_float_state import ExactFloatState


def native_float_mixed_partials[
    precise: Bool, wide: Bool, batched: Bool = False
](
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    v4: Bytes,
    v5: Bytes,
    v6: Bytes,
    v7: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    b4: Bytes,
    b5: Bytes,
    b6: Bytes,
    b7: Bytes,
    predicate: Bytes,
    predicate_bits: Bytes,
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    metadata_offset: UInt32,
    high: Pointer[Int64, MutUntrackedOrigin],
    low: Pointer[UInt64, MutUntrackedOrigin],
    sums: Pointer[Float32, MutUntrackedOrigin],
    compensations: Pointer[Float32, MutUntrackedOrigin],
    counts: Pointer[UInt32, MutUntrackedOrigin],
    bins: Pointer[Int64, MutUntrackedOrigin],
    specials: Pointer[UInt32, MutUntrackedOrigin],
    faults: Pointer[UInt32, MutUntrackedOrigin],
):
    var geometry = static_segment_info[batched](
        metadata, metadata_offset, Int(block_idx.x)
    )
    var meta = geometry[0]
    var row_base = geometry[3]
    var rows = Int(meta[unsafe_offset=0])
    var stride = Int(meta[unsafe_offset=1])
    var values = v0.unsafe_offset(row_base * 4).unsafe_bitcast[UInt32]()
    var value_offset = UInt32(meta[unsafe_offset=HEADER_WORDS + 4]) + UInt32(
        row_base
    )
    var nullable = Bool(meta[unsafe_offset=HEADER_WORDS + 3] & 1)
    var filtered = Bool(meta[unsafe_offset=7])
    var shared = filtered and meta[unsafe_offset=15] == 1
    var tag = UInt32(meta[unsafe_offset=4])
    var pv = predicate.unsafe_offset(row_base * (8 if tag == 1 else 4))
    var predicate_offset = UInt32(meta[unsafe_offset=5]) + UInt32(row_base)
    var predicate_nullable = Bool(meta[unsafe_offset=6])
    var inclusive = UInt32(meta[unsafe_offset=8])
    var low_bound = bitcast[DType.int64](meta[unsafe_offset=9])
    var high_bound = bitcast[DType.int64](meta[unsafe_offset=10])
    var low_float = bitcast[DType.float32](UInt32(meta[unsafe_offset=11]))
    var high_float = bitcast[DType.float32](UInt32(meta[unsafe_offset=12]))
    var needs_sum = Bool(meta[unsafe_offset=14] & 1)
    var exact = ExactFloatState[wide, 1]()
    var total = Float32(0)
    var correction = Float32(0)
    var count = UInt32(0)
    var minimum = UInt32.MAX
    var maximum = UInt32(0)
    var numeric = UInt32(0)
    var flags = UInt32(0)
    var fault = UInt32(0)
    var row = geometry[1] * BLOCK_SIZE + Int(thread_idx.x)
    while row < rows:
        var raw = UInt32(0)
        if shared:
            raw = values[unsafe_offset=row]
        var kept = True
        if filtered:
            if predicate_nullable:
                kept = _valid(predicate_bits, predicate_offset, row)
            if tag == 2:
                var value = bitcast[DType.float32](
                    raw
                ) if shared else pv.unsafe_bitcast[Float32]()[unsafe_offset=row]
                kept = kept & float_in_range(
                    value, low_float, high_float, inclusive
                )
            else:
                var value = read_integer(pv, tag, row)
                kept = kept & (value >= low_bound) & (value <= high_bound)
        if nullable:
            kept = kept & _valid(b0, value_offset, row)
        if kept:
            if not shared:
                raw = values[unsafe_offset=row]
            count += 1
            var magnitude = raw & 0x7FFF_FFFF
            if magnitude > 0x7F80_0000:
                flags |= 1
            else:
                numeric += 1
                var key = _ordered(raw)
                minimum = min(minimum, key)
                maximum = max(maximum, key)
                if magnitude == 0:
                    flags |= UInt32(1) << (
                        UInt32(20) if raw >> 31 else UInt32(10)
                    )
            if needs_sum:
                var value = bitcast[DType.float32](raw)
                comptime if precise:
                    exact.add[0](value)
                else:
                    fault |= (
                        UInt32(magnitude != 0 and magnitude < 0x0C80_0000) << 1
                    )
                    var adjusted = value - correction
                    var next = total + adjusted
                    correction = (next - total) - adjusted
                    total = next
        row += stride
    var output = geometry[2]
    var n = block.sum[block_size=BLOCK_SIZE, broadcast=False](count)
    var lo = block.min[block_size=BLOCK_SIZE, broadcast=False](minimum)
    var hi = block.max[block_size=BLOCK_SIZE, broadcast=False](maximum)
    var numeric_count = block.sum[block_size=BLOCK_SIZE, broadcast=False](
        numeric
    )
    var packed_flags = block.sum[block_size=BLOCK_SIZE, broadcast=False](flags)
    var bad = block.max[block_size=BLOCK_SIZE, broadcast=False](fault)
    var s = Float32(0)
    var c = Float32(0)
    comptime if precise:
        if needs_sum:
            exact.finish[0](output, bins, specials)
    else:
        if needs_sum:
            s = block.sum[block_size=BLOCK_SIZE, broadcast=False](total)
            c = block.sum[block_size=BLOCK_SIZE, broadcast=False](correction)
    if thread_idx.x == 0:
        high[unsafe_offset=output] = 0
        low[unsafe_offset=output] = 0
        sums[unsafe_offset=output] = s
        compensations[unsafe_offset=output] = c
        counts[unsafe_offset=output] = n
        faults[unsafe_offset=output] = bad
        var at = Int(meta[unsafe_offset=3]) + output * 4
        high[unsafe_offset=at] = Int64(lo)
        high[unsafe_offset=at + 1] = Int64(hi)
        high[unsafe_offset=at + 2] = Int64(numeric_count)
        high[unsafe_offset=at + 3] = Int64(
            UInt32((packed_flags & 1023) != 0)
            | (UInt32(((packed_flags >> 10) & 1023) != 0) << 1)
            | (UInt32((packed_flags >> 20) != 0) << 2)
        )
