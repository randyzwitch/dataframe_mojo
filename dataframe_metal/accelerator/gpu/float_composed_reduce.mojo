"""Compact Float32 composition: one binary value and shared column states."""
from std.memory import Pointer, bitcast
from std.utils.static_tuple import StaticTuple
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from .filter_sum import BLOCK_SIZE, _valid
from .multi_reduce import Bytes, read_integer
from .composed_reduce import (
    input_pointer,
    float_value,
    HEADER_WORDS,
    FIELD_WORDS,
    segment_info,
)
from .float_predicate import float_in_range
from .exact_float_state import ExactFloatState


def float_composed_partials[
    wide: Bool, arity: Int, exact_fields: Int, native_predicate: Int = -1
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
    comptime assert 1 <= arity and arity <= 4
    comptime assert 0 <= exact_fields and exact_fields <= arity
    comptime assert -1 <= native_predicate and native_predicate <= 2
    var geometry = segment_info(metadata, metadata_offset, Int(block_idx.x))
    var meta = geometry[0]
    var row_base = geometry[3]
    var rows = Int(meta[unsafe_offset=0])
    var stride = Int(meta[unsafe_offset=1])
    var output_offset = Int(meta[unsafe_offset=2])
    var predicate_tag = UInt32(
        native_predicate
    ) if native_predicate >= 0 else UInt32(meta[unsafe_offset=4])
    var predicate_offset = UInt32(meta[unsafe_offset=5]) + UInt32(row_base)
    var predicate_nullable = UInt32(meta[unsafe_offset=6])
    var filtered = UInt32(meta[unsafe_offset=7])
    var inclusive = UInt32(meta[unsafe_offset=8])
    var low_bound = bitcast[DType.int64](meta[unsafe_offset=9])
    var high_bound = bitcast[DType.int64](meta[unsafe_offset=10])
    var low_float = bitcast[DType.float32](UInt32(meta[unsafe_offset=11]))
    var high_float = bitcast[DType.float32](UInt32(meta[unsafe_offset=12]))
    var low_int32 = Int32(0)
    var high_int32 = Int32(0)
    comptime if native_predicate == 0:
        # An absent bound retains an Int64 sentinel in the matcher. Clamp
        # once before narrowing, without retaining Int64 state in the scan.
        low_int32 = max(low_bound, Int64(Int32.MIN)).cast[DType.int32]()
        high_int32 = min(high_bound, Int64(Int32.MAX)).cast[DType.int32]()
    var sum_flags = UInt32(meta[unsafe_offset=14])
    var predicate_slot = UInt32(meta[unsafe_offset=15])
    var predicate_values = predicate.unsafe_offset(
        row_base * (8 if predicate_tag == 1 else 4)
    )
    var values = StaticTuple[Bytes, arity]()
    var bits = StaticTuple[Bytes, arity]()
    var offsets = SIMD[DType.uint32, 4](0)
    var nullable = UInt32(0)
    var reuse = UInt32(0)
    comptime for field in range(arity):
        comptime at = HEADER_WORDS + field * FIELD_WORDS
        var slot = UInt32(meta[unsafe_offset=at + 1])
        values[field] = input_pointer(
            Int(slot) - 1, v0, v1, v2, v3, v4, v5, v6, v7
        )
        if slot != 0:
            values[field] = values[field].unsafe_offset(row_base * 4)
        bits[field] = input_pointer(
            Int(slot) - 1, b0, b1, b2, b3, b4, b5, b6, b7
        )
        offsets[field] = UInt32(meta[unsafe_offset=at + 4]) + UInt32(row_base)
        nullable |= (UInt32(meta[unsafe_offset=at + 3]) & 1) << UInt32(field)
        if slot != 0 and slot == predicate_slot and filtered:
            reuse |= UInt32(1) << UInt32(field)
    var right_slot = UInt32(meta[unsafe_offset=HEADER_WORDS + 2])
    var right = input_pointer(
        Int(right_slot) - 1, v0, v1, v2, v3, v4, v5, v6, v7
    )
    if right_slot != 0:
        right = right.unsafe_offset(row_base * 4)
    var right_bits = input_pointer(
        Int(right_slot) - 1, b0, b1, b2, b3, b4, b5, b6, b7
    )
    var right_nullable = UInt32(meta[unsafe_offset=HEADER_WORDS + 3]) & 2
    var right_offset = UInt32(meta[unsafe_offset=HEADER_WORDS + 5]) + UInt32(
        row_base
    )
    var side = UInt32(meta[unsafe_offset=HEADER_WORDS + 6])
    var literal = bitcast[DType.float32](
        UInt32(meta[unsafe_offset=HEADER_WORDS + 7])
    )
    var op = UInt32(meta[unsafe_offset=HEADER_WORDS])
    var reuse_right = (
        right_slot != 0 and right_slot == predicate_slot and filtered != 0
    )
    var total = SIMD[DType.float32, 4](0)
    var comp = SIMD[DType.float32, 4](0)
    var count = SIMD[DType.uint32, 4](0)
    var exact = ExactFloatState[wide, max(1, exact_fields)]()
    var fault = UInt32(0)
    var row = geometry[1] * BLOCK_SIZE + Int(thread_idx.x)
    while row < rows:
        var kept = True
        var pf = Float32(0)
        if filtered:
            if predicate_nullable:
                kept = _valid(predicate_bits, predicate_offset, row)
            comptime if native_predicate == 0:
                var pi = predicate_values.unsafe_bitcast[Int32]()[unsafe_offset=row]
                kept = kept & (pi >= low_int32) & (pi <= high_int32)
            else:
                if predicate_tag == 2:
                    pf = predicate_values.unsafe_bitcast[Float32]()[unsafe_offset=row]
                    kept = kept & float_in_range(
                        pf, low_float, high_float, inclusive
                    )
                else:
                    var pi = read_integer(predicate_values, predicate_tag, row)
                    kept = kept & (pi >= low_bound) & (pi <= high_bound)
        comptime for field in range(arity):
            var valid = kept
            if nullable & (UInt32(1) << UInt32(field)):
                valid = valid & _valid(bits[field], offsets[field], row)
            comptime if field == 0:
                if right_nullable:
                    valid = valid & _valid(right_bits, right_offset, row)
            if valid:
                count[field] += 1
                if sum_flags & (UInt32(1) << UInt32(field)):
                    var value = Float32(0)
                    comptime if field == 0:
                        var computed = float_value(
                            values[0],
                            right,
                            op,
                            side,
                            literal,
                            pf,
                            Bool(reuse & 1),
                            reuse_right,
                            row,
                        )
                        fault |= UInt32(computed[1]) << 1
                        value = computed[0]
                    else:
                        value = (
                            pf if reuse
                            & (UInt32(1) << UInt32(field)) else values[
                                field
                            ].unsafe_bitcast[Float32]()[unsafe_offset=row]
                        )
                    comptime if field < exact_fields:
                        exact.add[field](value)
                    else:
                        # Kahan still performs Float32 arithmetic. Tiny
                        # contributions or cancellation can require gradual
                        # underflow even when the value itself was loaded.
                        var magnitude = (
                            bitcast[DType.uint32](value) & 0x7FFF_FFFF
                        )
                        fault |= (
                            UInt32(magnitude != 0 and magnitude < 0x0C80_0000)
                            << 1
                        )
                        var adjusted = value - comp[field]
                        var next = total[field] + adjusted
                        comp[field] = (next - total[field]) - adjusted
                        total[field] = next
        row += stride
    comptime for field in range(arity):
        var n = block.sum[block_size=BLOCK_SIZE, broadcast=False](count[field])
        var s = Float32(0)
        var c = Float32(0)
        comptime if field >= exact_fields:
            s = block.sum[block_size=BLOCK_SIZE, broadcast=False](total[field])
            c = block.sum[block_size=BLOCK_SIZE, broadcast=False](comp[field])
        var output = geometry[2] * arity + field
        if thread_idx.x == 0:
            high[unsafe_offset=output] = 0
            low[unsafe_offset=output] = 0
            sums[unsafe_offset=output] = s
            compensations[unsafe_offset=output] = c
            counts[unsafe_offset=output] = n
        comptime if field < exact_fields:
            exact.finish[field](output, bins, specials)
    var bad = block.max[block_size=BLOCK_SIZE, broadcast=False](fault)
    if thread_idx.x == 0:
        faults[unsafe_offset=geometry[2]] = bad
