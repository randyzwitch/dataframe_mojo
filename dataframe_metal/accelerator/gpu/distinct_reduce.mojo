"""Collision-checked exact native distinct sets with original-row tickets."""
from std.atomic import Atomic
from std.memory import Pointer
from max.gpu import global_idx, thread_idx
from ...distinct_layout import (
    DistinctLayout,
    DISTINCT_CAPACITY_FAULT,
    MAX_DISTINCT_PROBES,
)
from .multi_reduce import Bytes
from .compound_predicate import _pointer, predicate_matches
from .filter_sum import _valid
from .group_tuple_hash import claim_tuple, tuple_hash, tuple_equal
from .group_hash import NO_GROUP
from .group_state import atomic_or
from .ordered_value import ordered_bits
from .distinct_value import distinct_hash, distinct_same


@always_inline
def distinct_layout(
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]]
) -> DistinctLayout:
    return DistinctLayout(
        Int(metadata[unsafe_offset=3]),
        Int(metadata[unsafe_offset=2]),
        Int(metadata[unsafe_offset=6]),
    )


@always_inline
def distinct_valid(bits: Bytes, descriptor: UInt64, row: Int) -> Bool:
    return (
        _valid(bits, UInt32(descriptor), row) if (descriptor >> 32)
        & 1 else True
    )


def initialize_distinct(
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    scratch: Pointer[UInt32, MutUntrackedOrigin],
):
    var layout = distinct_layout(metadata)
    var index = global_idx.x
    if index < layout.words():
        scratch[
            unsafe_offset=index
        ] = UInt32.MAX if index >= layout.first_offset() and index < layout.counts_offset() else UInt32(
            0
        )


@always_inline
def insert_distinct(
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    scratch: Pointer[UInt32, MutUntrackedOrigin],
    field: Int,
    group: UInt32,
    group_hash: UInt32,
    row: Int,
    values: Bytes,
    bits: Bytes,
    descriptor: UInt64,
    k0: Bytes,
    k1: Bytes,
    k2: Bytes,
    k3: Bytes,
    kb0: Bytes,
    kb1: Bytes,
    kb2: Bytes,
    kb3: Bytes,
):
    var layout = distinct_layout(metadata)
    var tag = UInt32(descriptor >> 33)
    var valid = distinct_valid(bits, descriptor, row)
    var raw = ordered_bits(values, tag, row) if valid else UInt64(0)
    if not valid:
        atomic_or(
            scratch.unsafe_offset(
                layout.nulls_offset() + Int(group) * layout.fields + field
            ),
            UInt32(1),
        )
    var slot = distinct_hash(raw, valid, tag, group_hash) & UInt32(
        layout.unique_capacity - 1
    )
    var probes = 0
    while probes < min(layout.unique_capacity, MAX_DISTINCT_PROBES):
        if Atomic[UInt32].load(scratch.unsafe_offset(1)):
            return
        var address = scratch.unsafe_offset(
            layout.unique_offset() + field * layout.unique_capacity + Int(slot)
        )
        var owner = Atomic[UInt32].load(address)
        if owner == 0:
            var expected = UInt32(0)
            if Atomic[UInt32].compare_exchange[weak=True](
                address, expected, UInt32(row) + 1
            ):
                var count = Atomic[UInt32].fetch_add(
                    scratch.unsafe_offset(4 + field), UInt32(1)
                )
                if count >= UInt32(layout.unique_capacity // 2):
                    atomic_or(scratch.unsafe_offset(1), DISTINCT_CAPACITY_FAULT)
                    return
                _ = Atomic[UInt32].fetch_add(
                    scratch.unsafe_offset(
                        layout.counts_offset()
                        + Int(group) * layout.fields
                        + field
                    ),
                    UInt32(1),
                )
                return
            if expected == 0:
                continue
            owner = expected
        var previous = Int(owner - 1)
        var old_valid = distinct_valid(bits, descriptor, previous)
        var old = ordered_bits(values, tag, previous) if old_valid else UInt64(
            0
        )
        if distinct_same(raw, valid, old, old_valid, tag) and tuple_equal(
            metadata, row, previous, k0, k1, k2, k3, kb0, kb1, kb2, kb3
        ):
            return
        slot = (slot + 1) & UInt32(layout.unique_capacity - 1)
        probes += 1
    atomic_or(scratch.unsafe_offset(1), DISTINCT_CAPACITY_FAULT)


def distinct_partials(
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
    var layout = distinct_layout(metadata)
    var row = global_idx.x
    while row < Int(metadata[unsafe_offset=0]):
        if Atomic[UInt32].load(scratch.unsafe_offset(1)):
            break
        if predicate_matches(
            metadata.unsafe_offset(Int(metadata[unsafe_offset=5])),
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
            var group = UInt32(0)
            var hash = UInt32(0x9E3779B9)
            if metadata[unsafe_offset=11]:
                group = claim_tuple(
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
                    scratch.unsafe_offset(layout.table_offset()),
                    scratch,
                )
                hash = tuple_hash(
                    metadata, row, k0, k1, k2, k3, kb0, kb1, kb2, kb3
                )
            if group != NO_GROUP:
                if metadata[unsafe_offset=11]:
                    _ = Atomic[UInt32].min(
                        scratch.unsafe_offset(
                            layout.first_offset() + Int(group)
                        ),
                        UInt32(row),
                    )
                for field in range(layout.fields):
                    var at = 16 + field * 2
                    var input = Int(metadata[unsafe_offset=at])
                    insert_distinct(
                        metadata,
                        scratch,
                        field,
                        group,
                        hash,
                        row,
                        _pointer(
                            input, v0, v1, v2, v3, v4, v5, v6, v7, predicate
                        ),
                        _pointer(
                            input,
                            b0,
                            b1,
                            b2,
                            b3,
                            b4,
                            b5,
                            b6,
                            b7,
                            predicate_bits,
                        ),
                        metadata[unsafe_offset=at + 1],
                        k0,
                        k1,
                        k2,
                        k3,
                        kb0,
                        kb1,
                        kb2,
                        kb3,
                    )
        row += Int(metadata[unsafe_offset=1])
    if thread_idx.x == 0:
        _ = Atomic[UInt32].fetch_add(scratch.unsafe_offset(2), UInt32(1))
