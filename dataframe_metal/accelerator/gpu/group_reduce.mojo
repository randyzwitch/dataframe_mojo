"""One bounded hash aggregation scan over native numeric inputs."""
from std.atomic import Atomic
from std.memory import Pointer, bitcast
from max.gpu import global_idx, block_idx, thread_idx
from dataframe.expr import COL
from ...expression_format import TREE_OP
from .multi_reduce import Bytes, read_integer
from .filter_sum import _valid
from .compound_predicate import _pointer, predicate_matches
from .arithmetic_sum import integer_binary
from .value_ops import checked_float_value
from .expression_value import expression_value
from .group_hash import claim_group, record_group, NO_GROUP, GROUP_SLOTS
from .group_state import (
    GROUP_STATE_WORDS,
    GROUP_COUNT,
    GROUP_MIN,
    GROUP_MAX,
    atomic_or,
    add_integer_sum,
    add_float_sum,
    add_ordered_extrema,
)


@always_inline
def group_value[
    floating: Bool
](
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    field: Int,
    row: Int,
    tag: UInt32,
    want: Bool,
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
) -> Tuple[UInt64, Bool, UInt32]:
    var descriptor = metadata.unsafe_offset(16 + field * 8)
    var op = UInt32(descriptor[unsafe_offset=0])
    if op == UInt32(TREE_OP):
        return expression_value[floating](
            metadata.unsafe_offset(Int(descriptor[unsafe_offset=6])),
            Int(descriptor[unsafe_offset=7]),
            0,
            row,
            tag,
            want,
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
    var li = Int(descriptor[unsafe_offset=1]) - 1
    var ri = Int(descriptor[unsafe_offset=2]) - 1
    var flags = UInt32(descriptor[unsafe_offset=3])
    var valid = True
    if flags & 1:
        valid = _valid(
            _pointer(li, b0, b1, b2, b3, b4, b5, b6, b7, b7),
            UInt32(descriptor[unsafe_offset=4]),
            row,
        )
    if flags & 2:
        valid = valid & _valid(
            _pointer(ri, b0, b1, b2, b3, b4, b5, b6, b7, b7),
            UInt32(descriptor[unsafe_offset=5]),
            row,
        )
    if not valid:
        return (UInt64(0), False, UInt32(0))
    comptime if floating:
        if not want:
            return (UInt64(0), True, UInt32(0))
    var side = UInt32(descriptor[unsafe_offset=6])
    var literal = descriptor[unsafe_offset=7]
    var left = _pointer(li, v0, v1, v2, v3, v4, v5, v6, v7, v7)
    var right = _pointer(ri, v0, v1, v2, v3, v4, v5, v6, v7, v7)
    comptime if floating:
        var a = (
            UInt32(literal) if side
            == 1 else left.unsafe_bitcast[UInt32]()[unsafe_offset=row]
        )
        if op == UInt32(COL):
            return (UInt64(a), True, UInt32(0))
        var b = (
            UInt32(literal) if side
            == 2 else right.unsafe_bitcast[UInt32]()[unsafe_offset=row]
        )
        var result = checked_float_value(
            bitcast[DType.float32](a), bitcast[DType.float32](b), op
        )
        return (
            UInt64(bitcast[DType.uint32](result[0])),
            True,
            UInt32(result[1]) << 1,
        )
    else:
        var a = bitcast[DType.int64](literal) if side == 1 else read_integer(
            left, tag, row
        )
        if op == UInt32(COL):
            return (bitcast[DType.uint64](a), True, UInt32(0))
        var b = bitcast[DType.int64](literal) if side == 2 else read_integer(
            right, tag, row
        )
        var result = integer_binary(a, b, op, tag)
        return (bitcast[DType.uint64](result[0]), True, UInt32(result[1]))


@always_inline
def update_int64_ticket[
    minimum: Bool
](
    ticket: Pointer[UInt32, MutUntrackedOrigin],
    candidate: Int64,
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    field: Int,
    row: Int,
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
):
    var expected = Atomic[UInt32].load(ticket)
    while True:
        if expected != UInt32.MAX:
            var previous = group_value[False](
                metadata,
                field,
                Int(expected),
                UInt32(1),
                True,
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
            var old = bitcast[DType.int64](previous[0])
            if candidate >= old if minimum else candidate <= old:
                return
        if Atomic[UInt32].compare_exchange[weak=True](
            ticket, expected, UInt32(row)
        ):
            return


def grouped_partials[
    arity: Int, float_fields: Int, int32_fields: Int
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
    keys: Bytes,
    key_bits: Bytes,
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    table: Pointer[UInt32, MutUntrackedOrigin],
    control: Pointer[UInt32, MutUntrackedOrigin],
    counts: Pointer[UInt32, MutUntrackedOrigin],
    first: Pointer[UInt32, MutUntrackedOrigin],
    states: Pointer[UInt32, MutUntrackedOrigin],
):
    var rows = Int(metadata[unsafe_offset=0])
    var stride = Int(metadata[unsafe_offset=1])
    var blocks = UInt32(metadata[unsafe_offset=2])
    var flags = UInt32(metadata[unsafe_offset=9])
    var row = global_idx.x
    while row < rows:
        if Atomic[UInt32].load(control.unsafe_offset(1)):
            return
        if predicate_matches(
            metadata.unsafe_offset(Int(metadata[unsafe_offset=10])),
            0,
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
        ):
            var slot = claim_group(
                keys,
                key_bits,
                UInt32(metadata[unsafe_offset=3]),
                UInt32(metadata[unsafe_offset=4]),
                UInt32(metadata[unsafe_offset=5]),
                row,
                table,
                control,
                control.unsafe_offset(1),
            )
            if slot == NO_GROUP:
                return
            record_group(slot, row, blocks, counts, first)
            comptime for field in range(arity):
                comptime floating = field < float_fields
                comptime narrow = field < float_fields + int32_fields
                var sum_requested = Bool(flags & (UInt32(1) << UInt32(field)))
                var extrema_requested = Bool(
                    (flags >> 4 | flags >> 8) & (UInt32(1) << UInt32(field))
                )
                var value = group_value[floating](
                    metadata,
                    field,
                    row,
                    UInt32(0 if narrow else 1),
                    sum_requested or extrema_requested,
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
                if value[2]:
                    atomic_or(control.unsafe_offset(1), value[2])
                    return
                if value[1]:
                    var state = states.unsafe_offset(
                        (
                            (Int(slot) * Int(blocks) + Int(block_idx.x)) * arity
                            + field
                        )
                        * GROUP_STATE_WORDS
                    )
                    _ = Atomic[UInt32].fetch_add(
                        state.unsafe_offset(GROUP_COUNT), UInt32(1)
                    )
                    if sum_requested:
                        comptime if floating:
                            add_float_sum(state, UInt32(value[0]))
                        else:
                            add_integer_sum(
                                state, bitcast[DType.int64](value[0])
                            )
                    if extrema_requested:
                        comptime if narrow:
                            add_ordered_extrema(
                                state, UInt32(value[0]), floating
                            )
                        else:
                            var candidate = bitcast[DType.int64](value[0])
                            update_int64_ticket[True](
                                state.unsafe_offset(GROUP_MIN),
                                candidate,
                                metadata,
                                field,
                                row,
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
                            update_int64_ticket[False](
                                state.unsafe_offset(GROUP_MAX),
                                candidate,
                                metadata,
                                field,
                                row,
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
        row += stride
    if thread_idx.x == 0:
        _ = Atomic[UInt32].fetch_add(control.unsafe_offset(2), UInt32(1))


def resolve_int64_extrema[
    arity: Int, float_fields: Int, int32_fields: Int
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
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    states: Pointer[UInt32, UntrackedOrigin[mut=False]],
    resolved: Pointer[Int64, MutUntrackedOrigin],
    control: Pointer[UInt32, MutUntrackedOrigin],
):
    var index = global_idx.x
    var fields = GROUP_SLOTS * Int(metadata[unsafe_offset=2]) * arity
    if index < fields:
        var field = index % arity
        if field >= float_fields + int32_fields:
            for side in range(2):
                var ticket = states[
                    unsafe_offset=index * GROUP_STATE_WORDS + GROUP_MIN + side
                ]
                if ticket != UInt32.MAX:
                    var value = group_value[False](
                        metadata,
                        field,
                        Int(ticket),
                        UInt32(1),
                        True,
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
                    resolved[unsafe_offset=index * 2 + side] = bitcast[
                        DType.int64
                    ](value[0])
    if thread_idx.x == 0:
        _ = Atomic[UInt32].fetch_add(control.unsafe_offset(3), UInt32(1))
