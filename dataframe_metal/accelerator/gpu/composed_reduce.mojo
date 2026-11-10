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
from dataframe.expr import COL, ADD, SUB, MUL, FILL_NULL, FILL_NAN
from .native_numeric import (
    numeric_unary,
    float32_unary_bits,
    integer_unary_bits,
)
from .native_fill import native_fill_bits
from .filter_sum import BLOCK_SIZE, _valid
from .multi_reduce import Bytes, read_integer
from .arithmetic_sum import integer_binary
from .exact_float_state import ExactFloatState
from .compound_predicate import predicate_matches
from .value_ops import checked_float_value
from .expression_value import expression_value
from ...expression_format import TREE_OP, SCALAR_FIELD_SHIFT
from .chain_expression_value import chain_expression_value
from ...moment_format import MOMENT_PARTIAL_WORDS, MOMENT_FLAG_SHIFT
from .moment_squares import SquareBins, IntegerMoments

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
    if numeric_unary(Int(op)):
        return (
            bitcast[DType.float32](
                float32_unary_bits(bitcast[DType.uint32](a), Int(op))
            ),
            False,
        )
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
    if numeric_unary(Int(op)):
        var result = integer_unary_bits(bitcast[DType.uint64](a), tag, Int(op))
        return (bitcast[DType.int64](result[0]), result[1])
    var b = literal if side == 2 else (
        pi if reuse_right else read_integer(right, tag, row)
    )
    return integer_binary(a, b, op, tag)


@always_inline
def conditional_tree_fields[
    floating: Bool, simple_cases: Bool
](
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    descriptors: Pointer[UInt64, UntrackedOrigin[mut=False]],
    first: Int,
    end: Int,
    row_base: Int,
    row: Int,
    tag: UInt32,
    requested: UInt32,
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
) -> Tuple[SIMD[DType.uint64, 4], SIMD[DType.uint32, 4], UInt32]:
    """Share one checked evaluator across a native dtype's CASE fields.

    Runtime metadata bounds keep the machine-code compiler from duplicating
    the entire conditional evaluator for every field. Reduction accumulators
    remain statically typed in the caller; no row values are materialized.
    """
    var values = SIMD[DType.uint64, 4](0)
    var valid = SIMD[DType.uint32, 4](0)
    var fault = UInt32(0)
    for field in range(first, end):
        var at = HEADER_WORDS + field * FIELD_WORDS
        if UInt32(descriptors[unsafe_offset=at]) != UInt32(TREE_OP):
            continue
        if descriptors[unsafe_offset=14] & (
            UInt64(1) << UInt64(SCALAR_FIELD_SHIFT + field)
        ):
            continue
        var value = expression_value[floating, True, simple_cases](
            metadata.unsafe_offset(Int(descriptors[unsafe_offset=at + 6])),
            Int(descriptors[unsafe_offset=at + 7]),
            row_base,
            row,
            tag,
            Bool(requested & (UInt32(1) << UInt32(field))),
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
        values[field] = value[0]
        valid[field] = UInt32(value[1])
        fault |= value[2]
    return (values, valid, fault)


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
    cases: Bool = False,
    simple_cases: Bool = False,
    moments: Bool = False,
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
    comptime assert not moments or with_extrema
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
    var scalar_fields = (sum_flags >> UInt32(SCALAR_FIELD_SHIFT)) & 15
    var generic_fields = UInt32(0)
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
        if ops[field] == UInt32(TREE_OP) and not (
            scalar_fields & (UInt32(1) << UInt32(field))
        ):
            generic_fields |= UInt32(1) << UInt32(field)
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
    var squares = SquareBins[max(1, float_fields), wide]()
    var integer_moments = IntegerMoments[max(1, arity - float_fields)]()
    var moment_flags = sum_flags >> UInt32(MOMENT_FLAG_SHIFT)
    var ext_lo = SIMD[DType.int64, 4](Int64.MAX)
    var ext_hi = SIMD[DType.int64, 4](Int64.MIN)
    var ext_count = SIMD[DType.uint32, 4](0)
    var ext_special = SIMD[DType.uint32, 4](0)
    var extent_flags = (sum_flags >> 4) | (sum_flags >> 8)
    var shared_left = Bool(sum_flags & (UInt32(1) << 12))
    var shared_want = Bool((sum_flags | extent_flags) & 15)
    comptime for field in range(arity):
        shared_want = shared_want or ops[field] == UInt32(FILL_NAN)
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
        var case_values = SIMD[DType.uint64, 4](0)
        var case_valid = SIMD[DType.uint32, 4](0)
        comptime if cases:
            if kept and generic_fields:
                # Bits 16..18 carry the field count; zero accepts older probe
                # metadata while retaining a runtime loop bound.
                var bound = Int((meta[unsafe_offset=14] >> 16) & 7)
                if bound == 0:
                    bound = arity
                var requested = sum_flags
                comptime if with_extrema:
                    requested |= extent_flags
                comptime for kind in range(3):
                    comptime FIRST = 0 if kind == 0 else (
                        float_fields if kind
                        == 1 else float_fields + int32_fields
                    )
                    comptime END = float_fields if kind == 0 else (
                        float_fields + int32_fields if kind == 1 else arity
                    )
                    comptime if END > FIRST:
                        var evaluated = conditional_tree_fields[
                            kind == 0, simple_cases
                        ](
                            metadata,
                            meta,
                            FIRST,
                            min(bound, END),
                            row_base,
                            row,
                            UInt32(2 if kind == 0 else (0 if kind == 1 else 1)),
                            requested,
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
                        case_values |= evaluated[0]
                        case_valid |= evaluated[1]
                        fault |= evaluated[2]
        var shared_valid = kept
        var shared_raw = UInt32(0)
        if shared_left:
            if flags[0] & 1:
                shared_valid = shared_valid and _valid(
                    left_bits[0], loff[0], row
                )
            if shared_valid and shared_want:
                shared_raw = lefts[0].unsafe_bitcast[UInt32]()[
                    unsafe_offset=row
                ]
        comptime for field in range(arity):
            var valid = shared_valid if shared_left else kept
            if not shared_left and flags[field] & 1:
                valid = valid & _valid(left_bits[field], loff[field], row)
            if flags[field] & 2:
                valid = valid & _valid(right_bits[field], roff[field], row)
            var tree_bits = UInt64(0)
            var fill = ops[field] == UInt32(FILL_NULL) or ops[field] == UInt32(
                FILL_NAN
            )
            if kept and fill:
                comptime TAG = 2 if field < float_fields else (
                    0 if field < float_fields + int32_fields else 1
                )
                var filled = native_fill_bits(
                    lefts[field],
                    rights[field],
                    left_bits[field],
                    right_bits[field],
                    loff[field],
                    roff[field],
                    row,
                    UInt32(TAG),
                    ops[field],
                    flags[field],
                    sides[field],
                    UInt64(bitcast[DType.uint32](flit[field])) if TAG
                    == 2 else bitcast[DType.uint64](ilit[field]),
                    Bool(
                        (sum_flags | extent_flags)
                        & (UInt32(1) << UInt32(field))
                    ),
                    shared_left,
                    shared_raw,
                    shared_valid,
                )
                tree_bits = filled[0]
                valid = filled[1]
            comptime if trees:
                if kept and ops[field] == UInt32(TREE_OP):
                    comptime if cases:
                        comptime if field < float_fields:
                            # Shared Float32 field evaluation keeps exact-bin
                            # scans compact; direct inlining regressed timings.
                            tree_bits = case_values[field]
                            valid = Bool(case_valid[field])
                        else:
                            if scalar_fields & (UInt32(1) << UInt32(field)):
                                comptime at = HEADER_WORDS + field * FIELD_WORDS
                                var computed = chain_expression_value(
                                    metadata.unsafe_offset(
                                        Int(meta[unsafe_offset=at + 6])
                                    ),
                                    Int(meta[unsafe_offset=at + 7]),
                                    row_base,
                                    row,
                                    Bool(
                                        sum_flags & (UInt32(1) << UInt32(field))
                                    )
                                    or (
                                        with_extrema
                                        and Bool(
                                            extent_flags
                                            & (UInt32(1) << UInt32(field))
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
                                )
                                tree_bits = computed[0]
                                valid = computed[1]
                                fault |= computed[2]
                            else:
                                tree_bits = case_values[field]
                                valid = Bool(case_valid[field])
                    else:
                        var computed = expression_value[
                            field < float_fields, cases, simple_cases
                        ](
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
                            UInt32(
                                0 if field < float_fields + int32_fields else 1
                            ),
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
                        if fill or (trees and ops[field] == UInt32(TREE_OP)):
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
                                        bitcast[DType.float32](
                                            shared_raw
                                        ) if shared_left else pf,
                                        shared_left or Bool(reuse_l[field]),
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
                                    bitcast[DType.float32](
                                        shared_raw
                                    ) if shared_left else pf,
                                    shared_left or Bool(reuse_l[field]),
                                    Bool(reuse_r[field]),
                                    row,
                                )
                        # Extrema-only arithmetic also needs gradual-underflow
                        # adjudication; its SUM bit is deliberately absent.
                        fault |= UInt32(result[1]) << 1
                        comptime if moments:
                            if moment_flags & (UInt32(1) << UInt32(field)):
                                squares.add[field](
                                    bitcast[DType.uint32](result[0])
                                )
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
                        if fill or (trees and ops[field] == UInt32(TREE_OP)):
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
                        comptime if moments:
                            if moment_flags & (UInt32(1) << UInt32(field)):
                                integer_moments.add[field - float_fields](value)
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
        comptime if moments:
            # The normal mixed scan owns five words per partial value. The
            # moment tail shares that existing buffer, adding no Metal binding.
            var total_values = Int(meta[unsafe_offset=3] & UInt64(0xFFFF_FFFF))
            var tail = high.unsafe_offset(
                5 * total_values + output * MOMENT_PARTIAL_WORDS
            )
            if thread_idx.x == 0:
                comptime for word in range(MOMENT_PARTIAL_WORDS):
                    tail[unsafe_offset=word] = 0
            if moment_flags & (UInt32(1) << UInt32(field)):
                comptime if field < float_fields:
                    comptime for bin in range(
                        SquareBins[max(1, float_fields), wide].BIN_COUNT
                    ):
                        var square = block.sum[
                            block_size=BLOCK_SIZE, broadcast=False
                        ](
                            squares.words[
                                field
                                * SquareBins[max(1, float_fields), wide].STRIDE
                                + bin
                            ]
                        )
                        if thread_idx.x == 0:
                            tail[
                                unsafe_offset=bin
                                + SquareBins[
                                    max(1, float_fields), wide
                                ].FIRST_BIN
                            ] = square
                    var largest = block.max[
                        block_size=BLOCK_SIZE, broadcast=False
                    ](squares.largest[field])
                    if thread_idx.x == 0:
                        tail[unsafe_offset=18] = Int64(largest)
                else:
                    integer_moments.finish[field - float_fields](0, tail)
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
    cases: Bool = False,
    simple_cases: Bool = False,
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
        cases,
        simple_cases,
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
    cases: Bool = False,
    simple_cases: Bool = False,
    moments: Bool = False,
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
        cases,
        simple_cases,
        moments,
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
