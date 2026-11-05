"""Four fixed native expression slots sharing one predicate and reduction.

Metadata contains only scalar values and input-slot indices. Device pointers
are real buffer arguments selected once before scanning, never CPU address
integers or a per-row expression-tree interpreter.
"""
from .float_predicate import float_in_range
from std.memory import Pointer, bitcast
from std.utils.static_tuple import StaticTuple
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from dataframe.expr import COL, ADD, SUB, MUL
from .filter_sum import BLOCK_SIZE, _valid
from .multi_reduce import Bytes, read_integer
from .arithmetic_sum import integer_binary
from .exact_float_state import ExactFloatState

comptime HEADER_WORDS = 16
comptime FIELD_WORDS = 8
comptime META_WORDS = HEADER_WORDS + 4 * FIELD_WORDS


@always_inline
def segment_info(
    root: Pointer[UInt64, UntrackedOrigin[mut=False]],
    offset: UInt32,
    group: Int,
) -> Tuple[Pointer[UInt64, UntrackedOrigin[mut=False]], Int, Int, Int]:
    """Metadata, local block, output block and row base for one workgroup.

    A batch keeps the original segment grids and reduction order. Only the
    submission is combined; no CPU pointer addresses enter this table.
    """
    if offset != UInt32.MAX:
        var meta = root.unsafe_offset(Int(offset))
        return (meta, group, Int(meta[unsafe_offset=2]) + group, 0)
    var first = 0
    var last = Int(root[unsafe_offset=0])
    while first + 1 < last:
        var middle = (first + last) >> 1
        if Int(root[unsafe_offset=2 + middle * META_WORDS + 2]) <= group:
            first = middle
        else:
            last = middle
    var meta = root.unsafe_offset(2 + first * META_WORDS)
    return (
        meta,
        group - Int(meta[unsafe_offset=2]),
        group,
        Int(meta[unsafe_offset=13]),
    )


@always_inline
def static_segment_info[
    batched: Bool
](
    root: Pointer[UInt64, UntrackedOrigin[mut=False]],
    offset: UInt32,
    group: Int,
) -> Tuple[Pointer[UInt64, UntrackedOrigin[mut=False]], Int, Int, Int]:
    # Separate entry points let the compiler erase the batch table and all
    # row-base pointer arithmetic for ordinary contiguous/copy inputs.
    comptime if batched:
        return segment_info(root, UInt32.MAX, group)
    else:
        var meta = root.unsafe_offset(Int(offset))
        return (meta, group, Int(meta[unsafe_offset=2]) + group, 0)


@always_inline
def input_pointer(
    slot: Int,
    a: Bytes,
    b: Bytes,
    c: Bytes,
    d: Bytes,
    e: Bytes,
    f: Bytes,
    g: Bytes,
    h: Bytes,
) -> Bytes:
    if slot == 0:
        return a
    if slot == 1:
        return b
    if slot == 2:
        return c
    if slot == 3:
        return d
    if slot == 4:
        return e
    if slot == 5:
        return f
    if slot == 6:
        return g
    return h


@always_inline
def float_value(
    left: Bytes,
    right: Bytes,
    op: UInt32,
    side: UInt32,
    literal: Float32,
    pf: Float32,
    reuse_left: Bool,
    reuse_right: Bool,
    row: Int,
) -> Tuple[Float32, Bool]:
    var a = literal if side == 1 else (
        pf if reuse_left else left.unsafe_bitcast[Float32]()[unsafe_offset=row]
    )
    if op == UInt32(COL):
        return (a, False)
    var b = literal if side == 2 else (
        pf if reuse_right else right.unsafe_bitcast[Float32]()[
            unsafe_offset=row
        ]
    )
    # Metal may flush subnormal operands/results during arithmetic. Reading
    # raw column bits into exact bins is safe, but arithmetic needs CPU
    # adjudication when the device cannot preserve gradual underflow.
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


@always_inline
def int_value(
    left: Bytes,
    right: Bytes,
    op: UInt32,
    side: UInt32,
    literal: Int64,
    pi: Int64,
    reuse_left: Bool,
    reuse_right: Bool,
    row: Int,
    tag: UInt32,
) -> Tuple[Int64, Bool]:
    var a = literal if side == 1 else (
        pi if reuse_left else read_integer(left, tag, row)
    )
    if op == UInt32(COL):
        return (a, False)
    var b = literal if side == 2 else (
        pi if reuse_right else read_integer(right, tag, row)
    )
    return integer_binary(a, b, op, tag)


def composed_partials[
    precise: Bool,
    wide: Bool,
    arity: Int,
    float_fields: Int,
    batched: Bool = False,
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
    var geometry = static_segment_info[batched](
        metadata, metadata_offset, Int(block_idx.x)
    )
    var meta = geometry[0]
    var row_base = geometry[3]
    var rows = UInt32(meta[unsafe_offset=0])
    var stride = UInt32(meta[unsafe_offset=1])
    var partial_offset = UInt32(meta[unsafe_offset=2])
    var tags = UInt32(meta[unsafe_offset=3])
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
    var lefts = StaticTuple[Bytes, arity]()
    var rights = StaticTuple[Bytes, arity]()
    var left_bits = StaticTuple[Bytes, arity]()
    var right_bits = StaticTuple[Bytes, arity]()
    var ops = SIMD[DType.uint32, 4](0)
    var sides = SIMD[DType.uint32, 4](0)
    var flags = SIMD[DType.uint32, 4](0)
    var loff = SIMD[DType.uint32, 4](0)
    var roff = SIMD[DType.uint32, 4](0)
    var reuse_l = SIMD[DType.uint32, 4](0)
    var reuse_r = SIMD[DType.uint32, 4](0)
    var ilit = SIMD[DType.int64, 4](0)
    var flit = SIMD[DType.float32, 4](0)
    comptime for field in range(arity):
        comptime at = HEADER_WORDS + field * FIELD_WORDS
        var li = UInt32(meta[unsafe_offset=at + 1])
        var ri = UInt32(meta[unsafe_offset=at + 2])
        lefts[field] = input_pointer(
            Int(li) - 1, v0, v1, v2, v3, v4, v5, v6, v7
        )
        rights[field] = input_pointer(
            Int(ri) - 1, v0, v1, v2, v3, v4, v5, v6, v7
        )
        var width = 8 if ((tags >> UInt32(2 * field)) & 3) == 1 else 4
        if li != 0:
            lefts[field] = lefts[field].unsafe_offset(row_base * width)
        if ri != 0:
            rights[field] = rights[field].unsafe_offset(row_base * width)
        left_bits[field] = input_pointer(
            Int(li) - 1, b0, b1, b2, b3, b4, b5, b6, b7
        )
        right_bits[field] = input_pointer(
            Int(ri) - 1, b0, b1, b2, b3, b4, b5, b6, b7
        )
        ops[field] = UInt32(meta[unsafe_offset=at])
        flags[field] = UInt32(meta[unsafe_offset=at + 3])
        loff[field] = UInt32(meta[unsafe_offset=at + 4]) + UInt32(row_base)
        roff[field] = UInt32(meta[unsafe_offset=at + 5]) + UInt32(row_base)
        sides[field] = UInt32(meta[unsafe_offset=at + 6])
        ilit[field] = bitcast[DType.int64](meta[unsafe_offset=at + 7])
        flit[field] = bitcast[DType.float32](UInt32(meta[unsafe_offset=at + 7]))
        reuse_l[field] = UInt32(
            li != 0 and li == predicate_slot and filtered != 0
        )
        reuse_r[field] = UInt32(
            ri != 0 and ri == predicate_slot and filtered != 0
        )
    var hi = SIMD[DType.int64, 4](0)
    var lo = SIMD[DType.uint64, 4](0)
    var fs = SIMD[DType.float32, 4](0)
    var fc = SIMD[DType.float32, 4](0)
    var nc = SIMD[DType.uint32, 4](0)
    var exact = ExactFloatState[wide, max(1, float_fields)]()
    var fault = UInt32(0)
    var row = geometry[1] * BLOCK_SIZE + Int(thread_idx.x)
    while row < Int(rows):
        var kept = True
        var pi = Int64(0)
        var pf = Float32(0)
        if filtered:
            if predicate_nullable:
                kept = _valid(predicate_bits, predicate_offset, row)
            if predicate_tag == 2:
                pf = predicate_values.unsafe_bitcast[Float32]()[
                    unsafe_offset=row
                ]
                kept = kept & float_in_range(
                    pf, low_float, high_float, inclusive
                )
            else:
                pi = read_integer(predicate_values, predicate_tag, row)
                kept = kept & (pi >= low_bound) & (pi <= high_bound)
        comptime for field in range(arity):
            var valid = kept
            if flags[field] & 1:
                valid = valid & _valid(left_bits[field], loff[field], row)
            if flags[field] & 2:
                valid = valid & _valid(right_bits[field], roff[field], row)
            if valid:
                nc[field] += 1
                var tag = (tags >> UInt32(2 * field)) & 3
                comptime if precise:
                    comptime if field < float_fields:
                        if sum_flags & (UInt32(1) << UInt32(field)):
                            var value = float_value(
                                lefts[field],
                                rights[field],
                                ops[field],
                                sides[field],
                                flit[field],
                                pf,
                                Bool(reuse_l[field]),
                                Bool(reuse_r[field]),
                                row,
                            )
                            fault |= UInt32(value[1]) << 1
                            exact.add[field](value[0])
                    else:
                        # COUNT of arithmetic must still check integer errors.
                        if (sum_flags & (UInt32(1) << UInt32(field))) or ops[
                            field
                        ] != UInt32(COL):
                            var result = int_value(
                                lefts[field],
                                rights[field],
                                ops[field],
                                sides[field],
                                ilit[field],
                                pi,
                                Bool(reuse_l[field]),
                                Bool(reuse_r[field]),
                                row,
                                tag,
                            )
                            fault |= UInt32(result[1])
                            var value = result[0] if not result[1] else Int64(0)
                            if sum_flags & (UInt32(1) << UInt32(field)):
                                if tag == 1:
                                    hi[field] += value >> 32
                                    lo[field] += UInt64(value & 0xFFFF_FFFF)
                                else:
                                    hi[field] += value
                else:
                    if tag == 2:
                        if sum_flags & (UInt32(1) << UInt32(field)):
                            var value = float_value(
                                lefts[field],
                                rights[field],
                                ops[field],
                                sides[field],
                                flit[field],
                                pf,
                                Bool(reuse_l[field]),
                                Bool(reuse_r[field]),
                                row,
                            )
                            fault |= UInt32(value[1]) << 1
                            var adjusted = value[0] - fc[field]
                            var next = fs[field] + adjusted
                            fc[field] = (next - fs[field]) - adjusted
                            fs[field] = next
                    elif (sum_flags & (UInt32(1) << UInt32(field))) or ops[
                        field
                    ] != UInt32(COL):
                        var result = int_value(
                            lefts[field],
                            rights[field],
                            ops[field],
                            sides[field],
                            ilit[field],
                            pi,
                            Bool(reuse_l[field]),
                            Bool(reuse_r[field]),
                            row,
                            tag,
                        )
                        fault |= UInt32(result[1])
                        var value = result[0] if not result[1] else Int64(0)
                        if sum_flags & (UInt32(1) << UInt32(field)):
                            if tag == 1:
                                hi[field] += value >> 32
                                lo[field] += UInt64(value & 0xFFFF_FFFF)
                            else:
                                hi[field] += value
        row += Int(stride)
    comptime for field in range(arity):
        var h = Int64(0)
        var l = UInt64(0)
        var s = Float32(0)
        var c = Float32(0)
        comptime if precise:
            comptime if field >= float_fields:
                h = block.sum[block_size=BLOCK_SIZE, broadcast=False](hi[field])
                l = block.sum[block_size=BLOCK_SIZE, broadcast=False](lo[field])
        else:
            h = block.sum[block_size=BLOCK_SIZE, broadcast=False](hi[field])
            l = block.sum[block_size=BLOCK_SIZE, broadcast=False](lo[field])
            s = block.sum[block_size=BLOCK_SIZE, broadcast=False](fs[field])
            c = block.sum[block_size=BLOCK_SIZE, broadcast=False](fc[field])
        var n = block.sum[block_size=BLOCK_SIZE, broadcast=False](nc[field])
        var output = geometry[2] * arity + field
        if thread_idx.x == 0:
            high[unsafe_offset=output] = h
            low[unsafe_offset=output] = l
            sums[unsafe_offset=output] = s
            compensations[unsafe_offset=output] = c
            counts[unsafe_offset=output] = n
        comptime if precise and field < float_fields:
            exact.finish[field](output, bins, specials)
    var bad = block.max[block_size=BLOCK_SIZE, broadcast=False](fault)
    if thread_idx.x == 0:
        faults[unsafe_offset=geometry[2]] = bad
