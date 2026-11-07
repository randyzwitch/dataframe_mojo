"""Packed scratch keeps native tuple buffers within a bounded kernel ABI."""
from std.memory import Pointer
from max.gpu import global_idx
from ...group_layout import GroupLayout, GROUP_VALUE_WORDS, GROUP_MOMENT_WORDS
from .group_state import initial_group_word

from std.atomic import Atomic
from max.gpu import block_idx, thread_idx
from .multi_reduce import Bytes
from .compound_predicate import predicate_matches
from .group_tuple_hash import claim_tuple
from .group_hash import NO_GROUP
from .group_reduce import _update_group_field, group_value
from std.memory import bitcast


def initialize_general_groups[
    global_state: Bool, moments: Bool = False
](
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    scratch: Pointer[UInt32, MutUntrackedOrigin],
):
    var layout = GroupLayout(
        Int(metadata[unsafe_offset=3]),
        Int(metadata[unsafe_offset=6]),
        Int(metadata[unsafe_offset=2]),
        global_state,
        GROUP_MOMENT_WORDS if moments else GROUP_VALUE_WORDS,
    )
    var index = global_idx.x
    if index < layout.words():
        if index >= layout.states_offset():
            scratch[unsafe_offset=index] = initial_group_word(
                (index - layout.states_offset()) % layout.value_words
            )
        elif index >= layout.first_offset() and index < layout.rows_offset():
            scratch[unsafe_offset=index] = UInt32.MAX
        else:
            scratch[unsafe_offset=index] = 0


# Only scratch/results are packed. Every native value/key/bitmap stays in its
# original mapped buffer; no per-row tuples, group IDs or values are produced.


def general_grouped_partials[
    cases: Bool = False,
    simple_cases: Bool = False,
    global_state: Bool = True,
    value_types: Int = 7,
    moments: Bool = False,
    with_extrema: Bool = True,
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
    k0: Bytes,
    k1: Bytes,
    k2: Bytes,
    k3: Bytes,
    kb0: Bytes,
    kb1: Bytes,
    kb2: Bytes,
    kb3: Bytes,
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    scratch: Pointer[UInt32, MutUntrackedOrigin],
):
    comptime assert 1 <= value_types and value_types <= 7
    var layout = GroupLayout(
        Int(metadata[unsafe_offset=3]),
        Int(metadata[unsafe_offset=6]),
        Int(metadata[unsafe_offset=2]),
        global_state,
        GROUP_MOMENT_WORDS if moments else GROUP_VALUE_WORDS,
    )
    var table = scratch.unsafe_offset(layout.table_offset())
    var first = scratch.unsafe_offset(layout.first_offset())
    var counts = scratch.unsafe_offset(layout.rows_offset())
    var states = scratch.unsafe_offset(layout.states_offset())
    var flags = UInt32(metadata[unsafe_offset=9])
    var row = global_idx.x
    while row < Int(metadata[unsafe_offset=0]):
        if Atomic[UInt32].load(scratch.unsafe_offset(1)):
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
            var slot = claim_tuple(
                metadata,
                row,
                k0,
                k1,
                k2,
                k3,
                kb0,
                kb1,
                kb2,
                kb3,
                table,
                scratch,
            )
            if slot == NO_GROUP:
                return
            var row_index = Int(slot)
            comptime if not global_state:
                row_index = row_index * layout.blocks + Int(block_idx.x)
            _ = Atomic[UInt32].fetch_add(
                counts.unsafe_offset(row_index), UInt32(1)
            )
            _ = Atomic[UInt32].min(first.unsafe_offset(Int(slot)), UInt32(row))
            comptime for kind in range(3):
                comptime if (value_types & (1 << kind)) != 0:
                    var begin = 0
                    var end = Int(metadata[unsafe_offset=7])
                    comptime if kind == 1:
                        begin = end
                        end += Int(metadata[unsafe_offset=8])
                    elif kind == 2:
                        begin = end + Int(metadata[unsafe_offset=8])
                        end = layout.fields
                    for field in range(begin, end):
                        if _update_group_field[
                            kind == 0,
                            kind < 2,
                            cases,
                            simple_cases,
                            global_state,
                            moments,
                            with_extrema,
                        ](
                            metadata,
                            slot,
                            UInt32(layout.blocks),
                            flags,
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
                            predicate,
                            predicate_bits,
                            states,
                            scratch,
                            layout.fields,
                        ):
                            return
        row += Int(metadata[unsafe_offset=1])
    if thread_idx.x == 0:
        _ = Atomic[UInt32].fetch_add(scratch.unsafe_offset(2), UInt32(1))


def resolve_general_int64[
    cases: Bool = False,
    simple_cases: Bool = False,
    global_state: Bool = True,
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
    scratch: Pointer[UInt32, UntrackedOrigin[mut=False]],
    resolved: Pointer[Int64, MutUntrackedOrigin],
    control: Pointer[UInt32, MutUntrackedOrigin],
):
    var layout = GroupLayout(
        Int(metadata[unsafe_offset=3]),
        Int(metadata[unsafe_offset=6]),
        Int(metadata[unsafe_offset=2]),
        global_state,
        GROUP_MOMENT_WORDS if moments else GROUP_VALUE_WORDS,
    )
    var index = global_idx.x
    if index < layout.entries():
        var field = index % layout.fields
        if field >= Int(metadata[unsafe_offset=7] + metadata[unsafe_offset=8]):
            var states = scratch.unsafe_offset(layout.states_offset())
            for side in range(2):
                var ticket = states[
                    unsafe_offset=index * layout.value_words + 4 + side
                ]
                if ticket != UInt32.MAX:
                    var value = group_value[False, cases, simple_cases](
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
                        predicate,
                        predicate_bits,
                    )
                    resolved[unsafe_offset=index * 2 + side] = bitcast[
                        DType.int64
                    ](value[0])
    if thread_idx.x == 0:
        _ = Atomic[UInt32].fetch_add(control.unsafe_offset(3), UInt32(1))
