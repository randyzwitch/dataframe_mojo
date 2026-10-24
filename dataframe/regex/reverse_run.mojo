"""Scalar reverse runs with consuming-predicate dispatch outside the loop."""
from .pikevm import decode, previous_start
from .program import OP_CHAR, OP_CLASS, OP_ANY, Program


@always_inline
def _decode_previous(
    text: Span[UInt8, ImmutAnyOrigin], at: Int
) -> Tuple[Int, UInt32]:
    """Decode the previous character without rereading its UTF8 bytes.
    A proved header/continuation layout has the scalar decoder's exact
    result, including overlong and out-of-range values. Other layouts use
    the original full-span boundary search and permissive decoder."""
    var p = text.unsafe_ptr()
    var a = UInt32(p[unsafe_offset=at - 1])
    if a < 128:
        return at - 1, a
    if (a & 0xC0) == 0x80 and at >= 2:
        var b = UInt32(p[unsafe_offset=at - 2])
        if (b & 0xE0) == 0xC0:
            return at - 2, ((b & 31) << 6) | (a & 63)
        if (b & 0xC0) == 0x80 and at >= 3:
            var c = UInt32(p[unsafe_offset=at - 3])
            if (c & 0xF0) == 0xE0:
                return at - 3, ((c & 15) << 12) | ((b & 63) << 6) | (a & 63)
            if (c & 0xC0) == 0x80 and at >= 4:
                var d = UInt32(p[unsafe_offset=at - 4])
                if d >= 0xF0:
                    return at - 4, (
                        ((d & 7) << 18)
                        | ((c & 63) << 12)
                        | ((b & 63) << 6)
                        | (a & 63)
                    )
    var prev = previous_start(text, at)
    return prev, UInt32(decode(text, prev) >> 3)


@always_inline
def _scan_class(
    program: Program,
    id: Int,
    text: Span[UInt8, ImmutAnyOrigin],
    var at: Int,
    floor: Int,
) -> Int:
    while at > floor:
        var prev, cp = _decode_previous(text, at)
        if prev < floor:
            break
        if not program.class_has(id, cp):
            break
        at = prev
    return at


@always_inline
def _scan_other[
    literal: Bool, exclude_newline: Bool
](
    codepoint: UInt32,
    text: Span[UInt8, ImmutAnyOrigin],
    var at: Int,
    floor: Int,
) -> Int:
    while at > floor:
        var prev, cp = _decode_previous(text, at)
        if prev < floor:
            break
        comptime if literal:
            if cp != codepoint:
                break
        elif exclude_newline:
            if cp == 10:
                break
        at = prev
    return at


@no_inline
def scan_reverse_run(
    program: Program,
    pc: Int,
    text: Span[UInt8, ImmutAnyOrigin],
    at: Int,
    floor: Int,
) -> Int:
    """Consume a capture-stable predicate, retaining scalar UTF8 semantics.
    The one-pass compiler already proves that this run has no changing
    capture actions, assertions or match priority. Fused decoding preserves
    the original full-span boundary and code-point results."""
    var op = program.op.unsafe_ptr()[unsafe_offset=pc]
    var arg = Int(program.arg.unsafe_ptr()[unsafe_offset=pc])
    if op == OP_CLASS:
        return _scan_class(program, arg, text, at, floor)
    if op == OP_CHAR:
        return _scan_other[True, False](UInt32(arg), text, at, floor)
    if op == OP_ANY:
        return _scan_other[False, False](0, text, at, floor)
    return _scan_other[False, True](0, text, at, floor)
