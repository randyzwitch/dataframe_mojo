"""Scan capture-stable ASCII runs with the program's existing class bits."""
from std.bit import count_leading_zeros, count_trailing_zeros
from std.memory import bitcast, pack_bits
from .byte_lookup import lookup16
from .program import OP_CHAR, OP_CLASS, OP_ANY, Program


@always_inline
def _bits(program: Program, pc: Int) -> SIMD[DType.uint8, 16]:
    var words = SIMD[DType.uint64, 2]()
    var op = program.op.unsafe_ptr()[unsafe_offset=pc]
    var arg = Int(program.arg.unsafe_ptr()[unsafe_offset=pc])
    if op == OP_CHAR:
        if arg < 128:
            words[arg >> 6] = UInt64(1) << UInt64(arg & 63)
    elif op == OP_CLASS:
        words = program.class_ascii.unsafe_ptr().unsafe_load[width=2](2 * arg)
    else:
        words = SIMD[DType.uint64, 2](UInt64.MAX)
        if op != OP_ANY:
            words[0] &= ~(UInt64(1) << 10)
    return bitcast[DType.uint8, 16](words)


@always_inline
def scan_ascii_run[
    reverse: Bool
](
    program: Program,
    pc: Int,
    text: Span[UInt8, ImmutAnyOrigin],
    var at: Int,
    bound: Int,
) -> Int:
    """Consume complete 16-byte blocks, stopping before a rejected byte.
    The caller handles the remaining short tail. `pc` must consume input;
    the range is [bound, at) in reverse and [at, bound) forward."""
    var table = _bits(program, pc)
    var powers = SIMD[DType.uint8, 16](
        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128
    )
    var p = text.unsafe_ptr()
    while True:
        var begin = at
        comptime if reverse:
            if at - bound < 16:
                return at
            begin -= 16
        else:
            if bound - at < 16:
                return at
        var block = p.unsafe_load[width=16](begin)
        var bytes = lookup16(table, (block >> 3) & 15)
        var bits = lookup16(powers, block & 7)
        var rejected = (bytes & bits).eq(0) | block.ge(128)
        var mask = UInt64(pack_bits[DType.uint16](rejected))
        if mask:
            comptime if reverse:
                return begin + 64 - Int(count_leading_zeros(mask))
            else:
                return at + Int(count_trailing_zeros(mask))
        comptime if reverse:
            at -= 16
        else:
            at += 16
