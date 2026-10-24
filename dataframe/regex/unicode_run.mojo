"""Scan fixed-width UTF8 batches proved by Unicode class membership."""
from std.bit import count_leading_zeros, count_trailing_zeros
from std.memory import pack_bits
from .charclass import MAX_CODEPOINT
from .pikevm import previous_start
from .program import OP_CHAR, OP_CLASS, Program
from .class_batch import class_batch


def has_unicode_run(program: Program, pc: Int) -> Bool:
    """Whether a consuming predicate has a Unicode batch predicate."""
    var op = program.op[pc]
    if op == OP_CHAR:
        return program.arg[pc] >= 128
    if op == OP_CLASS:
        ref ranges = program.class_ranges[Int(program.arg[pc])]
        return len(ranges) > 0 and ranges[len(ranges) - 1] >= 128
    return True


@always_inline
def _part[
    width: Int, byte: Int, reverse: Bool
](block: SIMD[DType.uint8, 16]) -> SIMD[DType.uint32, 8 if width == 2 else 4]:
    comptime if width == 2:
        return (
            block.shuffle[
                byte,
                byte + 2,
                byte + 4,
                byte + 6,
                byte + 8,
                byte + 10,
                byte + 12,
                byte + 14,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]()
            .slice[8 if width == 2 else 4]()
            .cast[DType.uint32]()
        )
    else:
        comptime offset = 4 if reverse and width == 3 else 0
        return (
            block.shuffle[
                offset + byte,
                offset + byte + width,
                offset + byte + 2 * width,
                offset + byte + 3 * width,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]()
            .slice[8 if width == 2 else 4]()
            .cast[DType.uint32]()
        )


@always_inline
def _structure[width: Int, reverse: Bool](block: SIMD[DType.uint8, 16]) -> Bool:
    comptime consumed = 16 if width == 2 else 4 * width
    comptime offset = 4 if reverse and width == 3 else 0
    var structure_mask = SIMD[DType.uint8, 16]()
    var headers = SIMD[DType.uint8, 16]()
    comptime for i in range(16):
        comptime if i >= offset and i < offset + consumed:
            comptime if (i - offset) % width == 0:
                structure_mask[i] = UInt8(0x100 - (1 << (7 - width)))
                headers[i] = UInt8(0x100 - (1 << (8 - width)))
            else:
                structure_mask[i] = 0xC0
                headers[i] = 0x80
    return (block & structure_mask).eq(headers).reduce_and()


@always_inline
def _decode_block[
    width: Int, reverse: Bool
](block: SIMD[DType.uint8, 16]) -> Tuple[
    SIMD[DType.uint32, 8 if width == 2 else 4],
    SIMD[DType.bool, 8 if width == 2 else 4],
]:
    comptime lanes = 8 if width == 2 else 4
    var a = _part[width, 0, reverse](block)
    var b = _part[width, 1, reverse](block)
    var cp: SIMD[DType.uint32, lanes]
    var valid: SIMD[DType.bool, lanes]
    comptime if width == 2:
        cp = ((a & 31) << 6) | (b & 63)
        valid = cp.ge(0x80)
    else:
        var c = _part[width, 2, reverse](block)
        comptime if width == 3:
            cp = ((a & 15) << 12) | ((b & 63) << 6) | (c & 63)
            valid = cp.ge(0x800) & (cp.lt(0xD800) | cp.gt(0xDFFF))
        else:
            var d = _part[width, 3, reverse](block)
            cp = ((a & 7) << 18) | ((b & 63) << 12) | ((c & 63) << 6) | (d & 63)
            valid = cp.ge(0x10000) & cp.le(MAX_CODEPOINT)
    return cp, valid


@always_inline
def _mask[width: Int, lanes: Int](rejected: SIMD[DType.bool, lanes]) -> UInt64:
    var mask: UInt64
    comptime if width == 2:
        mask = UInt64(pack_bits[DType.uint8](rejected))
    else:
        mask = UInt64(
            pack_bits[DType.uint8](rejected.join(SIMD[DType.bool, lanes]()))
        )
    return mask


@always_inline
def _scan_width[
    width: Int, reverse: Bool
](
    text: Span[UInt8, ImmutAnyOrigin],
    var at: Int,
    bound: Int,
    low: UInt32,
    high: UInt32,
) -> Int:
    comptime lanes = 8 if width == 2 else 4
    comptime consumed = width * lanes
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
        # Validate the selected byte layout before shuffling or decoding.
        # Three-byte batches ignore the unused four bytes in the load.
        if not _structure[width, reverse](block):
            return at
        var cp, valid = _decode_block[width, reverse](block)
        var rejected = ~valid | cp.lt(low) | cp.gt(high)
        var mask = _mask[width, lanes](rejected)
        if mask:
            comptime if reverse:
                return (
                    at
                    - consumed
                    + (64 - Int(count_leading_zeros(mask))) * width
                )
            else:
                return at + Int(count_trailing_zeros(mask)) * width
        comptime if reverse:
            at -= consumed
        else:
            at += consumed


@always_inline
def _scan_class_width[
    width: Int, reverse: Bool
](
    program: Program,
    id: Int,
    text: Span[UInt8, ImmutAnyOrigin],
    var at: Int,
    bound: Int,
) -> Int:
    comptime lanes = 8 if width == 2 else 4
    comptime consumed = width * lanes
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
        if not _structure[width, reverse](block):
            return at
        var cp, valid = _decode_block[width, reverse](block)
        # Invalid lanes use a safe directory index and cannot be consumed.
        var safe_cp = valid.select(cp, SIMD[DType.uint32, lanes](128))
        var accepted = class_batch[width, lanes](program, id, safe_cp) & valid
        var mask = _mask[width, lanes](~accepted)
        if mask:
            comptime if reverse:
                return (
                    at
                    - consumed
                    + (64 - Int(count_leading_zeros(mask))) * width
                )
            else:
                return at + Int(count_trailing_zeros(mask)) * width
        comptime if reverse:
            at -= consumed
        else:
            at += consumed


@no_inline
def scan_unicode_run[
    reverse: Bool
](
    program: Program,
    pc: Int,
    text: Span[UInt8, ImmutAnyOrigin],
    at: Int,
    bound: Int,
) -> Int:
    """Consume whole UTF8 characters inside the requested search range.
    Unproved membership, incomplete sequences and mixed widths end the
    batch scan; the caller resumes its scalar engine at the returned byte."""
    comptime if reverse:
        if at - bound < 16:
            return at
    else:
        if bound - at < 16:
            return at
    var low: UInt32 = 128
    var high: UInt32 = MAX_CODEPOINT
    var use_class = False
    var op = program.op.unsafe_ptr()[unsafe_offset=pc]
    var arg = Int(program.arg.unsafe_ptr()[unsafe_offset=pc])
    if op == OP_CHAR:
        low = UInt32(arg)
        high = low
    elif op == OP_CLASS:
        ref ranges = program.class_ranges[arg]
        if len(ranges) == 2:
            low = ranges[0]
            high = ranges[1]
        else:
            use_class = True
    var width = 0
    comptime if reverse:
        width = at - previous_start(text, at)
    else:
        var first = text.unsafe_ptr()[unsafe_offset=at]
        if first >= 0xC2 and first < 0xE0:
            width = 2
        elif first >= 0xE0 and first < 0xF0:
            width = 3
        elif first >= 0xF0 and first <= 0xF4:
            width = 4
    if use_class:
        if width == 2:
            return _scan_class_width[2, reverse](program, arg, text, at, bound)
        if width == 3:
            return _scan_class_width[3, reverse](program, arg, text, at, bound)
        if width == 4:
            return _scan_class_width[4, reverse](program, arg, text, at, bound)
        return at
    if width == 2:
        return _scan_width[2, reverse](text, at, bound, low, high)
    if width == 3:
        return _scan_width[3, reverse](text, at, bound, low, high)
    if width == 4:
        return _scan_width[4, reverse](text, at, bound, low, high)
    return at
