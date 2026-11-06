"""Four fixed native expression slots sharing one predicate and reduction.

Metadata contains only scalar values and input-slot indices. Device pointers
are real buffer arguments selected once before scanning, never CPU address
integers or a per-row expression-tree interpreter.
"""
from .float_predicate import float_in_range, _ordered
from std.memory import Pointer, bitcast
from std.utils.static_tuple import StaticTuple
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from dataframe.expr import COL, ADD, SUB, MUL
from .filter_sum import BLOCK_SIZE, _valid
from .multi_reduce import Bytes, read_integer
from .arithmetic_sum import integer_binary
from .exact_float_state import ExactFloatState
from .compound_predicate import predicate_matches
from .value_ops import checked_float_value
from .expression_value import expression_value
from ...expression_format import TREE_OP

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
    return checked_float_value(a, b, op)


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


@always_inline
def _composed_scan[
    precise: Bool,
    wide: Bool,
    arity: Int,
    float_fields: Int,
    int32_fields: Int,
    batched: Bool,
    with_extrema: Bool,
    compound: Bool = False,
    trees: Bool = False,
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
    extrema: Pointer[Int64, MutUntrackedOrigin],
):
    comptime assert 1 <= arity and arity <= 4
    comptime assert float_fields >= 0 and int32_fields >= 0
    comptime assert float_fields + int32_fields <= arity
    var geometry = static_segment_info[batched](
        metadata, metadata_offset, Int(block_idx.x)
    )
    var meta = geometry[0]
    var row_base = geometry[3]
    var rows = UInt32(meta[unsafe_offset=0])
    var stride = UInt32(meta[unsafe_offset=1])
    var partial_offset = UInt32(meta[unsafe_offset=2])
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
        comptime width = 8 if field >= float_fields + int32_fields else 4
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
    var ext_lo = SIMD[DType.int64, 4](Int64.MAX)
    var ext_hi = SIMD[DType.int64, 4](Int64.MIN)
    var ext_count = SIMD[DType.uint32, 4](0)
    var ext_special = SIMD[DType.uint32, 4](0)
    var extent_flags = (sum_flags >> 4) | (sum_flags >> 8)
    var fault = UInt32(0)
    var row = geometry[1] * BLOCK_SIZE + Int(thread_idx.x)
    while row < Int(rows):
        var kept = True
        comptime if compound:
            kept = predicate_matches(
                metadata.unsafe_offset(Int(meta[unsafe_offset=3] >> 32)),
                row_base,
                row,
                v0,
                v1,
                v2,
                v3,
                v4,
                v5,
                v6,
                v7,
                predicate,
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
                predicate_bits,
            )
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
            var tree_bits = UInt64(0)
            comptime if trees:
                if kept and ops[field] == UInt32(TREE_OP):
                    var computed = expression_value[field < float_fields](
                        metadata.unsafe_offset(
                            Int(
                                meta[
                                    unsafe_offset=HEADER_WORDS
                                    + field * FIELD_WORDS
                                    + 6
                                ]
                            )
                        ),
                        Int(
                            meta[
                                unsafe_offset=HEADER_WORDS
                                + field * FIELD_WORDS
                                + 7
                            ]
                        ),
                        row_base,
                        row,
                        UInt32(0 if field < float_fields + int32_fields else 1),
                        Bool(sum_flags & (UInt32(1) << UInt32(field)))
                        or (
                            with_extrema
                            and Bool(
                                extent_flags & (UInt32(1) << UInt32(field))
                            )
                        ),
                        v0,
                        v1,
                        v2,
                        v3,
                        v4,
                        v5,
                        v6,
                        v7,
                        b0,
                        b1,
                        b2,
                        b3,
                        b4,
                        b5,
                        b6,
                        b7,
                        predicate,
                        predicate_bits,
                    )
                    tree_bits = computed[0]
                    valid = computed[1]
                    fault |= computed[2]
            if valid:
                nc[field] += 1
                comptime if field < float_fields:
                    if (sum_flags & (UInt32(1) << UInt32(field))) or (
                        with_extrema
                        and (extent_flags & (UInt32(1) << UInt32(field)))
                    ):
                        var raw = UInt32(0)
                        var result = (Float32(0), False)
                        if trees and ops[field] == UInt32(TREE_OP):
                            raw = UInt32(tree_bits)
                            result = (bitcast[DType.float32](raw), False)
                        else:
                            comptime if with_extrema:
                                if ops[field] == UInt32(COL):
                                    raw = lefts[field].unsafe_bitcast[UInt32]()[
                                        unsafe_offset=row
                                    ]
                                    result = (
                                        bitcast[DType.float32](raw),
                                        False,
                                    )
                                else:
                                    result = float_value(
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
                                    raw = bitcast[DType.uint32](result[0])
                            else:
                                result = float_value(
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
                        # Extrema-only arithmetic also needs gradual-underflow
                        # adjudication; its SUM bit is deliberately absent.
                        fault |= UInt32(result[1]) << 1
                        if sum_flags & (UInt32(1) << UInt32(field)):
                            comptime if precise:
                                exact.add[field](result[0])
                            else:
                                var magnitude = (
                                    bitcast[DType.uint32](result[0])
                                    & 0x7FFF_FFFF
                                )
                                fault |= (
                                    UInt32(
                                        magnitude != 0
                                        and magnitude < 0x0C80_0000
                                    )
                                    << 1
                                )
                                var adjusted = result[0] - fc[field]
                                var next = fs[field] + adjusted
                                fc[field] = (next - fs[field]) - adjusted
                                fs[field] = next
                        comptime if with_extrema:
                            if extent_flags & (UInt32(1) << UInt32(field)):
                                var magnitude = raw & 0x7FFF_FFFF
                                if magnitude > 0x7F80_0000:
                                    ext_special[field] |= 1
                                else:
                                    if magnitude == 0:
                                        ext_special[field] |= UInt32(1) << (
                                            UInt32(20) if raw
                                            >> 31 else UInt32(10)
                                        )
                                    var key = Int64(_ordered(raw))
                                    ext_lo[field] = min(ext_lo[field], key)
                                    ext_hi[field] = max(ext_hi[field], key)
                                    ext_count[field] += 1
                else:
                    comptime TAG = 0 if field < float_fields + int32_fields else 1
                    # Arithmetic COUNT still evaluates checked intermediates.
                    if (
                        (sum_flags & (UInt32(1) << UInt32(field)))
                        or ops[field] != UInt32(COL)
                        or (
                            with_extrema
                            and (extent_flags & (UInt32(1) << UInt32(field)))
                        )
                    ):
                        var result = (Int64(0), False)
                        if trees and ops[field] == UInt32(TREE_OP):
                            result = (bitcast[DType.int64](tree_bits), False)
                        else:
                            result = int_value(
                                lefts[field],
                                rights[field],
                                ops[field],
                                sides[field],
                                ilit[field],
                                pi,
                                Bool(reuse_l[field]),
                                Bool(reuse_r[field]),
                                row,
                                UInt32(TAG),
                            )
                        fault |= UInt32(result[1])
                        var value = result[0] if not result[1] else Int64(0)
                        comptime if with_extrema:
                            if extent_flags & (UInt32(1) << UInt32(field)):
                                ext_lo[field] = min(ext_lo[field], value)
                                ext_hi[field] = max(ext_hi[field], value)
                                ext_count[field] += 1
                        if sum_flags & (UInt32(1) << UInt32(field)):
                            comptime if TAG == 1:
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
        # Every branch is uniform within the block, and each field owns
        # only its physical type's reduction work.
        if sum_flags & (UInt32(1) << UInt32(field)):
            comptime if field >= float_fields:
                h = block.sum[block_size=BLOCK_SIZE, broadcast=False](hi[field])
                comptime if field >= float_fields + int32_fields:
                    l = block.sum[block_size=BLOCK_SIZE, broadcast=False](
                        lo[field]
                    )
            elif not precise:
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
            if sum_flags & (UInt32(1) << UInt32(field)):
                exact.finish[field](output, bins, specials)
        comptime if with_extrema:
            if extent_flags & (UInt32(1) << UInt32(field)):
                var minimum = block.min[block_size=BLOCK_SIZE](ext_lo[field])
                var maximum = block.max[block_size=BLOCK_SIZE](ext_hi[field])
                var numeric = block.sum[block_size=BLOCK_SIZE](ext_count[field])
                var flags_out = block.sum[block_size=BLOCK_SIZE](
                    ext_special[field]
                )
                if thread_idx.x == 0:
                    extrema[unsafe_offset=output * 4] = minimum
                    extrema[unsafe_offset=output * 4 + 1] = maximum
                    extrema[unsafe_offset=output * 4 + 2] = Int64(numeric)
                    extrema[unsafe_offset=output * 4 + 3] = Int64(
                        UInt32((flags_out & 1023) != 0)
                        | (UInt32(((flags_out >> 10) & 1023) != 0) << 1)
                        | (UInt32((flags_out >> 20) != 0) << 2)
                    )
    var bad = block.max[block_size=BLOCK_SIZE, broadcast=False](fault)
    if thread_idx.x == 0:
        faults[unsafe_offset=geometry[2]] = bad


def composed_partials[
    precise: Bool,
    wide: Bool,
    arity: Int,
    float_fields: Int,
    int32_fields: Int,
    batched: Bool = False,
    compound: Bool = False,
    trees: Bool = False,
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
    _composed_scan[
        precise,
        wide,
        arity,
        float_fields,
        int32_fields,
        batched,
        False,
        compound,
        trees,
    ](
        v0,
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
        predicate,
        predicate_bits,
        metadata,
        metadata_offset,
        high,
        low,
        sums,
        compensations,
        counts,
        bins,
        specials,
        faults,
        high,
    )


def mixed_partials[
    precise: Bool,
    wide: Bool,
    arity: Int,
    float_fields: Int,
    int32_fields: Int,
    batched: Bool = False,
    compound: Bool = False,
    trees: Bool = False,
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
    var tail_offset = geometry[0][unsafe_offset=3]
    comptime if compound:
        tail_offset &= 0xFFFF_FFFF
    var extrema_tail = high.unsafe_offset(Int(tail_offset))
    _composed_scan[
        precise,
        wide,
        arity,
        float_fields,
        int32_fields,
        batched,
        True,
        compound,
        trees,
    ](
        v0,
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
        predicate,
        predicate_bits,
        metadata,
        metadata_offset,
        high,
        low,
        sums,
        compensations,
        counts,
        bins,
        specials,
        faults,
        extrema_tail,
    )
