"""Scalar reverse runs with consuming-predicate dispatch outside the loop."""
from .pikevm import decode, previous_start
from .program import OP_CHAR, OP_CLASS, OP_ANY, Program


@always_inline
def _scan_class(
    program: Program,
    id: Int,
    text: Span[UInt8, ImmutAnyOrigin],
    var at: Int,
    floor: Int,
) -> Int:
    while at > floor:
        var prev = previous_start(text, at)
        if prev < floor:
            break
        var cp = UInt32(decode(text, prev) >> 3)
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
        var prev = previous_start(text, at)
        if prev < floor:
            break
        var cp = UInt32(decode(text, prev) >> 3)
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
    capture actions, assertions or match priority. Only dispatch moves;
    previous_start/decode still observe the complete input span."""
    var op = program.op.unsafe_ptr()[unsafe_offset=pc]
    var arg = Int(program.arg.unsafe_ptr()[unsafe_offset=pc])
    if op == OP_CLASS:
        return _scan_class(program, arg, text, at, floor)
    if op == OP_CHAR:
        return _scan_other[True, False](UInt32(arg), text, at, floor)
    if op == OP_ANY:
        return _scan_other[False, False](0, text, at, floor)
    return _scan_other[False, True](0, text, at, floor)
