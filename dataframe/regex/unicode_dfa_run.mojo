"""Consume valid UTF8 packets whose cached DFA edges keep one state."""
from .alphabet_batch import alphabet_batch, alphabet_interval
from .class_batch import _contains
from .program import Program
from .unicode_run import _decode_block, _structure


@always_inline
def _scan_width[
    width: Int
](
    program: Program,
    transitions: List[Int32],
    offset: Int,
    text: Span[UInt8, ImmutAnyOrigin],
    var at: Int,
    end: Int,
    expected: Int32,
) -> Int:
    comptime lanes = 8 if width == 2 else 4
    comptime consumed = width * lanes
    var p = text.unsafe_ptr()
    var table = transitions.unsafe_ptr()
    var interval: UInt64 = 0
    while end - at >= 16:
        var block = p.unsafe_load[width=16](at)
        if not _structure[width, False](block):
            return at
        var cp, valid = _decode_block[width, False](block)
        if not valid.reduce_and():
            return at
        if interval == 0:
            var cls: Int32
            interval, cls = alphabet_interval(program, cp[0])
            if table[unsafe_offset=offset + Int(cls)] != expected:
                return at
        if not _contains(interval, cp).reduce_and():
            comptime if width == 2:
                # Two-byte code points use the bounded direct map.
                var classes = alphabet_batch[lanes](program, cp)
                var entries = table.unsafe_gather[width=lanes](
                    classes.cast[DType.int]() + offset
                )
                if not entries.eq(expected).reduce_and():
                    return at
            else:
                # Wider mixed pages retain the scalar directory search;
                # decoding is still shared by all four characters.
                for lane in range(lanes):
                    var cls = program.class_of(cp[lane])
                    if table[unsafe_offset=offset + cls] != expected:
                        return at
        at += consumed
    return at


@no_inline
def scan_unicode_self_loop(
    program: Program,
    transitions: List[Int32],
    offset: Int,
    text: Span[UInt8, ImmutAnyOrigin],
    at: Int,
    end: Int,
    expected: Int32,
) -> Int:
    """Skip complete packets, leaving unproved input for ordinary dispatch.
    Every decoded character must be valid and every cached edge must equal
    the caller's self-loop entry. No transition is computed or changed."""
    if end - at < 16:
        return at
    var first = text.unsafe_ptr()[unsafe_offset=at]
    if first >= 0xC2 and first < 0xE0:
        return _scan_width[2](
            program, transitions, offset, text, at, end, expected
        )
    if first >= 0xE0 and first < 0xF0:
        return _scan_width[3](
            program, transitions, offset, text, at, end, expected
        )
    if first >= 0xF0 and first <= 0xF4:
        return _scan_width[4](
            program, transitions, offset, text, at, end, expected
        )
    return at
