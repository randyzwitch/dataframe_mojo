"""Compact Float32 composition: one binary value and shared column states."""
from std.memory import Pointer, bitcast
from dataframe.expr import ADD, SUB, MUL
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


@always_inline
def _pair(values: Bytes, row: Int, rows: Int) -> SIMD[DType.float32, 2]:
    var pointer = values.unsafe_bitcast[Float32]()
    if row + 1 < rows:
        # Native windows retain scalar alignment even at an odd element.
        # Promise only that alignment, and guard the second element.
        return pointer.unsafe_load[width=2](row)
    return SIMD[DType.float32, 2](pointer[unsafe_offset=row], Float32(0))


@always_inline
def _binary(a: Float32, b: Float32, op: UInt32) -> Tuple[Float32, Bool]:
    var ar = bitcast[DType.uint32](a) & 0x7FFF_FFFF
    var br = bitcast[DType.uint32](b) & 0x7FFF_FFFF
    var ae = ar >> 23
    var be = br >> 23
    var tiny = (ar != 0 and ae == 0) or (br != 0 and be == 0)
    var underflow = (ar != 0 and br != 0 and ae + be <= 127) if op == UInt32(
        MUL
    ) else ((ar != 0 and ae <= 24) or (br != 0 and be <= 24))
    var result = a + b if op == UInt32(ADD) else (
        a - b if op == UInt32(SUB) else a * b
    )
    return (result, tiny or underflow)


def float_composed_partials[
    wide: Bool, arity: Int, exact_fields: Int
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
    var geometry = segment_info(metadata, metadata_offset, Int(block_idx.x))
    var meta = geometry[0]
    var row_base = geometry[3]
    var rows = Int(meta[unsafe_offset=0])
    var stride = Int(meta[unsafe_offset=1])
    var output_offset = Int(meta[unsafe_offset=2])
    var predicate_tag = UInt32(meta[unsafe_offset=4])
    var predicate_offset = UInt32(meta[unsafe_offset=5]) + UInt32(row_base)
    var predicate_nullable = UInt32(meta[unsafe_offset=6])
    var filtered = UInt32(meta[unsafe_offset=7])
    var inclusive = UInt32(meta[unsafe_offset=8])
    var low_bound = bitcast[DType.int64](meta[unsafe_offset=9])
    var high_bound = bitcast[DType.int64](meta[unsafe_offset=10])
    var low_float = bitcast[DType.float32](UInt32(meta[unsafe_offset=11]))
    var high_float = bitcast[DType.float32](UInt32(meta[unsafe_offset=12]))
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
    # Independent compensated streams expose memory/ALU work within a thread.
    # Exact means retain their original single-stream traversal.
    comptime LANES = 2 if exact_fields == 0 else 1
    var total = SIMD[DType.float32, 4](0)
    var comp = SIMD[DType.float32, 4](0)
    var second_total = SIMD[DType.float32, 4](0)
    var second_comp = SIMD[DType.float32, 4](0)
    var count = SIMD[DType.uint32, 4](0)
    var exact = ExactFloatState[wide, max(1, exact_fields)]()
    var fault = UInt32(0)
    var row = (geometry[1] * BLOCK_SIZE + Int(thread_idx.x)) * LANES
    while row < rows:
        var pairs = StaticTuple[SIMD[DType.float32, 2], arity]()
        var right_pair = SIMD[DType.float32, 2](0)
        var predicate_pair = SIMD[DType.float32, 2](0)
        comptime if LANES == 2:
            if filtered:
                if predicate_tag == 2:
                    predicate_pair = _pair(predicate_values, row, rows)
            comptime for field in range(arity):
                pairs[field] = SIMD[DType.float32, 2](0)
                if sum_flags & (UInt32(1) << UInt32(field)):
                    var load_value = True
                    comptime if field == 0:
                        load_value = side != 1
                    if load_value:
                        pairs[field] = predicate_pair if reuse & (
                            UInt32(1) << UInt32(field)
                        ) else _pair(values[field], row, rows)
            if side != 2 and sum_flags & 1:
                right_pair = predicate_pair if reuse_right else _pair(
                    right, row, rows
                )
        comptime for lane in range(LANES):
            var position = row + lane
            var active = True
            comptime if lane != 0:
                active = position < rows
            if active:
                var kept = True
                var pf = Float32(0)
                if filtered:
                    if predicate_nullable:
                        kept = _valid(
                            predicate_bits, predicate_offset, position
                        )
                    if predicate_tag == 2:
                        comptime if LANES == 2:
                            pf = predicate_pair[lane]
                        else:
                            pf = predicate_values.unsafe_bitcast[Float32]()[
                                unsafe_offset=position
                            ]
                        kept = kept & float_in_range(
                            pf, low_float, high_float, inclusive
                        )
                    else:
                        var pi = read_integer(
                            predicate_values, predicate_tag, position
                        )
                        kept = kept & (pi >= low_bound) & (pi <= high_bound)
                comptime for field in range(arity):
                    var valid = kept
                    if nullable & (UInt32(1) << UInt32(field)):
                        valid = valid & _valid(
                            bits[field], offsets[field], position
                        )
                    comptime if field == 0:
                        if right_nullable:
                            valid = valid & _valid(
                                right_bits, right_offset, position
                            )
                    if valid:
                        count[field] += 1
                        if sum_flags & (UInt32(1) << UInt32(field)):
                            var value = Float32(0)
                            comptime if field == 0:
                                var computed: Tuple[Float32, Bool]
                                comptime if LANES == 2:
                                    var a = (
                                        literal if side == 1 else pairs[0][lane]
                                    )
                                    var b = (
                                        literal if side
                                        == 2 else right_pair[lane]
                                    )
                                    computed = _binary(a, b, op)
                                else:
                                    computed = float_value(
                                        values[0],
                                        right,
                                        op,
                                        side,
                                        literal,
                                        pf,
                                        Bool(reuse & 1),
                                        reuse_right,
                                        position,
                                    )
                                fault |= UInt32(computed[1]) << 1
                                value = computed[0]
                            else:
                                comptime if LANES == 2:
                                    value = pairs[field][lane]
                                else:
                                    value = (
                                        pf if reuse
                                        & (
                                            UInt32(1) << UInt32(field)
                                        ) else values[field].unsafe_bitcast[
                                            Float32
                                        ]()[
                                            unsafe_offset=position
                                        ]
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
                                    UInt32(
                                        magnitude != 0
                                        and magnitude < 0x0C80_0000
                                    )
                                    << 1
                                )
                                comptime if lane == 0:
                                    var adjusted = value - comp[field]
                                    var next = total[field] + adjusted
                                    comp[field] = (
                                        next - total[field]
                                    ) - adjusted
                                    total[field] = next
                                else:
                                    var adjusted = value - second_comp[field]
                                    var next = second_total[field] + adjusted
                                    second_comp[field] = (
                                        next - second_total[field]
                                    ) - adjusted
                                    second_total[field] = next
        row += stride * LANES
    comptime for field in range(arity):
        var n = block.sum[block_size=BLOCK_SIZE, broadcast=False](count[field])
        var s = Float32(0)
        var c = Float32(0)
        comptime if field >= exact_fields:
            # The flag is uniform across the whole block. COUNT-only fields
            # need no compensated-sum collectives.
            if sum_flags & (UInt32(1) << UInt32(field)):
                var local_total = total[field]
                var local_comp = comp[field]
                comptime for lane in range(1, LANES):
                    comptime for component in range(2):
                        var contribution = (
                            second_total[field] if component
                            == 0 else -second_comp[field]
                        )
                        var adjusted = contribution - local_comp
                        var next = local_total + adjusted
                        local_comp = (next - local_total) - adjusted
                        local_total = next
                s = block.sum[block_size=BLOCK_SIZE, broadcast=False](
                    local_total
                )
                c = block.sum[block_size=BLOCK_SIZE, broadcast=False](
                    local_comp
                )
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
