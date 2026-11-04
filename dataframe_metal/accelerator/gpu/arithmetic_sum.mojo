"""Fused native binary arithmetic and reduction, without a temporary column."""
from std.memory import Pointer
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from dataframe.expr import ADD, SUB, MUL
from .filter_sum import BLOCK_SIZE, _valid
from .multi_reduce import Bytes, read_integer


@always_inline
def multiply_overflows(a: Int64, b: Int64) -> Bool:
    # Multiply unsigned magnitudes into high/low words using 32-bit limbs.
    # This also handles Int64.MIN without signed negation or GPU Int128.
    var x = UInt64(0) - UInt64(a) if a < 0 else UInt64(a)
    var y = UInt64(0) - UInt64(b) if b < 0 else UInt64(b)
    var x0 = x & 0xFFFF_FFFF
    var x1 = x >> 32
    var y0 = y & 0xFFFF_FFFF
    var y1 = y >> 32
    var w0 = x0 * y0
    var t = x1 * y0 + (w0 >> 32)
    var mid = (t & 0xFFFF_FFFF) + x0 * y1
    var high = x1 * y1 + (t >> 32) + (mid >> 32)
    var low = (mid << 32) | (w0 & 0xFFFF_FFFF)
    var limit = UInt64(1) << 63 if (a < 0) != (b < 0) else UInt64(Int64.MAX)
    return high != 0 or low > limit


@always_inline
def integer_binary(
    a: Int64, b: Int64, op: UInt32, tag: UInt32
) -> Tuple[Int64, Bool]:
    var result = Int64(0)
    var fault = False
    if op == UInt32(ADD):
        result = Int64(UInt64(a) + UInt64(b))
        fault = ((a ^ result) & (b ^ result)) < 0
    elif op == UInt32(SUB):
        result = Int64(UInt64(a) - UInt64(b))
        fault = ((a ^ b) & (a ^ result)) < 0
    else:
        result = Int64(UInt64(a) * UInt64(b))
        fault = multiply_overflows(a, b) if tag == 1 else False
    if tag == 0:
        fault = result < Int64(Int32.MIN) or result > Int64(Int32.MAX)
    return (result, fault)


def arithmetic_partials(
    left: Bytes,
    right: Bytes,
    predicate: Bytes,
    left_bits: Bytes,
    right_bits: Bytes,
    predicate_bits: Bytes,
    metadata: UInt32,
    left_offset: UInt32,
    right_offset: UInt32,
    predicate_offset: UInt32,
    rows: UInt32,
    stride: UInt32,
    output_offset: UInt32,
    operation: UInt32,
    low_bound: Int64,
    high_bound: Int64,
    low_float: Float32,
    high_float: Float32,
    literal_integer: Int64,
    literal_float: Float32,
    high: Pointer[Int64, MutUntrackedOrigin],
    low: Pointer[UInt64, MutUntrackedOrigin],
    sums: Pointer[Float32, MutUntrackedOrigin],
    compensation: Pointer[Float32, MutUntrackedOrigin],
    counts: Pointer[UInt32, MutUntrackedOrigin],
    faults: Pointer[UInt32, MutUntrackedOrigin],
):
    var tag = metadata & 3
    var predicate_tag = (metadata >> 2) & 3
    var flags = (metadata >> 4) & 7
    var filtered = (metadata >> 7) & 1
    var inclusive = (operation >> 8) & 3
    var op = operation & 15
    var side = (operation >> 4) & 3
    var h = Int64(0)
    var l = UInt64(0)
    var s = Float32(0)
    var c = Float32(0)
    var n = UInt32(0)
    var fault = UInt32(0)
    var row = global_idx.x
    while row < Int(rows):
        var kept = True
        if filtered:
            if flags & 4:
                kept = _valid(predicate_bits, predicate_offset, row)
            if predicate_tag == 2:
                var p = predicate.unsafe_bitcast[Float32]()[unsafe_offset=row]
                var above = p >= low_float if inclusive & 1 else p > low_float
                var below = p <= high_float if inclusive & 2 else p < high_float
                kept = kept & above & below
            else:
                var p = read_integer(predicate, predicate_tag, row)
                kept = kept & (p >= low_bound) & (p <= high_bound)
        if side != 1 and flags & 1:
            kept = kept & _valid(left_bits, left_offset, row)
        if side != 2 and flags & 2:
            kept = kept & _valid(right_bits, right_offset, row)
        if kept:
            n += 1
            if tag == 2:
                var a = (
                    literal_float if side
                    == 1 else left.unsafe_bitcast[Float32]()[unsafe_offset=row]
                )
                var b = (
                    literal_float if side
                    == 2 else right.unsafe_bitcast[Float32]()[unsafe_offset=row]
                )
                var value = a + b if op == UInt32(ADD) else (
                    a - b if op == UInt32(SUB) else a * b
                )
                var adjusted = value - c
                var next = s + adjusted
                c = (next - s) - adjusted
                s = next
            else:
                var a = literal_integer if side == 1 else read_integer(
                    left, tag, row
                )
                var b = literal_integer if side == 2 else read_integer(
                    right, tag, row
                )
                var computed = integer_binary(a, b, op, tag)
                fault |= UInt32(computed[1])
                # Invalid intermediates are adjudicated on CPU. Keep their
                # wrapped payloads out of otherwise bounded accumulators.
                var value = computed[0] if not computed[1] else Int64(0)
                if tag == 1:
                    h += value >> 32
                    l += UInt64(value & 0xFFFF_FFFF)
                else:
                    h += value
        row += Int(stride)
    h = block.sum[block_size=BLOCK_SIZE, broadcast=False](h)
    l = block.sum[block_size=BLOCK_SIZE, broadcast=False](l)
    s = block.sum[block_size=BLOCK_SIZE, broadcast=False](s)
    c = block.sum[block_size=BLOCK_SIZE, broadcast=False](c)
    n = block.sum[block_size=BLOCK_SIZE, broadcast=False](n)
    fault = block.max[block_size=BLOCK_SIZE, broadcast=False](fault)
    if thread_idx.x == 0:
        var output = Int(output_offset) + block_idx.x
        high[unsafe_offset=output] = h
        low[unsafe_offset=output] = l
        sums[unsafe_offset=output] = s
        compensation[unsafe_offset=output] = c
        counts[unsafe_offset=output] = n
        faults[unsafe_offset=output] = fault
