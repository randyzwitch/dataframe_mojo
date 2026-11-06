"""Native range leaves and a truth-table lookup inside the reduction scan."""
from std.memory import Pointer, bitcast
from .multi_reduce import Bytes, read_integer
from .filter_sum import _valid
from .float_predicate import float_in_range

comptime PREDICATE_HEADER_WORDS = 9
comptime PREDICATE_FIELD_WORDS = 6


@always_inline
def _pointer(
    slot: Int,
    a: Bytes,
    b: Bytes,
    c: Bytes,
    d: Bytes,
    e: Bytes,
    f: Bytes,
    g: Bytes,
    h: Bytes,
    ninth: Bytes,
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
    if slot == 7:
        return h
    return ninth


@always_inline
def predicate_matches(
    program: Pointer[UInt64, UntrackedOrigin[mut=False]],
    row_base: Int,
    row: Int,
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    v4: Bytes,
    v5: Bytes,
    v6: Bytes,
    v7: Bytes,
    predicate: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    b4: Bytes,
    b5: Bytes,
    b6: Bytes,
    b7: Bytes,
    predicate_bits: Bytes,
) -> Bool:
    var mask = UInt32(0)
    for i in range(Int(program[unsafe_offset=0])):
        var clause = program.unsafe_offset(
            PREDICATE_HEADER_WORDS + i * PREDICATE_FIELD_WORDS
        )
        var slot = Int(clause[unsafe_offset=0])
        var tag = UInt32(clause[unsafe_offset=1] >> 33)
        var nullable = (clause[unsafe_offset=1] >> 32) & 1
        var offset = UInt32(clause[unsafe_offset=1]) + UInt32(row_base)
        var values = _pointer(
            slot, v0, v1, v2, v3, v4, v5, v6, v7, predicate
        ).unsafe_offset(row_base * (8 if tag == 1 else 4))
        var bits = _pointer(
            slot, b0, b1, b2, b3, b4, b5, b6, b7, predicate_bits
        )
        var valid = True
        if nullable:
            valid = _valid(bits, offset, row)
        var flags = UInt32(clause[unsafe_offset=5])
        var kind = (flags >> 2) & 3
        var inverted = (flags & 16) != 0
        var matched = False
        var kept = False
        if kind == 1:
            matched = not valid
            kept = not matched if inverted else matched
        elif tag == 2:
            var bounds = clause[unsafe_offset=4]
            matched = float_in_range(
                values.unsafe_bitcast[Float32]()[unsafe_offset=row],
                bitcast[DType.float32](UInt32(bounds)),
                bitcast[DType.float32](UInt32(bounds >> 32)),
                flags & 3,
            )
        else:
            var value = read_integer(values, tag, row)
            matched = (
                value >= bitcast[DType.int64](clause[unsafe_offset=2])
            ) & (value <= bitcast[DType.int64](clause[unsafe_offset=3]))
        if kind != 1:
            kept = valid & (not matched if inverted else matched)
        mask |= UInt32(kept) << UInt32(i)
    return Bool(
        UInt32(program[unsafe_offset=1 + Int(mask >> 5)])
        & (UInt32(1) << (mask & 31))
    )
