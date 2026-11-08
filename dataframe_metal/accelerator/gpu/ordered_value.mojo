"""Stable native comparisons and original-row tickets for ordered reductions."""
from std.memory import bitcast, Pointer
from dataframe.expr import FIRST, LAST, ARG_MIN, ARG_MAX
from .float_predicate import _ordered
from .multi_reduce import Bytes


@always_inline
def ordered_bits(values: Bytes, tag: UInt32, row: Int) -> UInt64:
    if tag == 1:
        return values.unsafe_bitcast[UInt64]()[unsafe_offset=row]
    if tag == 3:
        return UInt64(values[unsafe_offset=row] != 0)
    return UInt64(values.unsafe_bitcast[UInt32]()[unsafe_offset=row])


@always_inline
def ordered_better(
    candidate: UInt64,
    candidate_row: UInt32,
    previous: UInt64,
    previous_row: UInt32,
    tag: UInt32,
    minimum: Bool,
) -> Bool:
    """NaNs lose to numbers; numeric ties, all NaNs and signed zero use first row."""
    if tag == 2:
        var a = UInt32(candidate)
        var b = UInt32(previous)
        var am = a & 0x7FFF_FFFF
        var bm = b & 0x7FFF_FFFF
        var an = am > 0x7F80_0000
        var bn = bm > 0x7F80_0000
        if an or bn:
            if an != bn:
                return not an
            return candidate_row < previous_row
        if am == 0:
            a = 0
        if bm == 0:
            b = 0
        var ak = _ordered(a)
        var bk = _ordered(b)
        return (
            candidate_row
            < previous_row if ak
            == bk else (ak < bk if minimum else ak > bk)
        )
    var a = bitcast[DType.int64](candidate)
    var b = bitcast[DType.int64](previous)
    if tag == 0:
        a = Int64(bitcast[DType.int32](UInt32(candidate)))
        b = Int64(bitcast[DType.int32](UInt32(previous)))
    elif tag == 3:
        a = Int64(candidate != 0)
        b = Int64(previous != 0)
    return (
        candidate_row
        < previous_row if a
        == b else (a < b if minimum else a > b)
    )


@always_inline
def ordered_empty(op: Int) -> UInt32:
    return UInt32.MAX if op == FIRST else UInt32(0)


@always_inline
def ordered_ticket_row(ticket: UInt32, op: Int) -> UInt32:
    return UInt32.MAX if ticket == ordered_empty(op) else ticket - 1


@always_inline
def update_ordered_arg(
    ticket: Pointer[UInt32, MutUntrackedOrigin],
    candidate: UInt64,
    values: Bytes,
    tag: UInt32,
    row: Int,
    minimum: Bool,
):
    from std.atomic import Atomic

    var expected = Atomic[UInt32].load(ticket)
    while True:
        if expected:
            var previous_row = expected - 1
            if not ordered_better(
                candidate,
                UInt32(row),
                ordered_bits(values, tag, Int(previous_row)),
                previous_row,
                tag,
                minimum,
            ):
                return
        if Atomic[UInt32].compare_exchange[weak=True](
            ticket, expected, UInt32(row) + 1
        ):
            return
