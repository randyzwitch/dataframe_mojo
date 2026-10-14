"""Fixed-pass binary64 order statistics expressed entirely in Mojo.

Eight radix refinements select one rank without allocating a sorted row copy.
Indexed inputs let grouped kernels share the same selection and interpolation.
"""
from max.gpu import global_idx
from std.memory import Pointer, unsafe_stack_allocation
from dataframe.accelerator.rows import (
    ROW_QUANTILE_LINEAR,
    ROW_QUANTILE_LOWER,
    ROW_QUANTILE_HIGHER,
    ROW_QUANTILE_NEAREST,
    ROW_QUANTILE_MIDPOINT,
)
from . import floating as f

comptime STATE_WORDS = 8
comptime DESCRIPTOR_WORDS = 5


def supported(op: Int64) -> Bool:
    return op >= ROW_QUANTILE_LINEAR and op <= ROW_QUANTILE_MIDPOINT


def key(word: UInt64) -> UInt64:
    if f.is_nan(word):
        return UInt64.MAX
    if f.absolute(word) == 0:
        return f.SIGN
    return ~word if word >> 63 else word ^ f.SIGN


def select[
    Indexed: Bool
](
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    base: Int,
    begin: Int,
    end: Int,
    rank: UInt32,
) -> UInt64:
    var prefix = UInt64(0)
    var mask = UInt64(0)
    var wanted = rank
    var bins = unsafe_stack_allocation[256, DType.uint32]()
    for pass_number in range(8):
        for bin in range(256):
            bins[unsafe_offset=bin] = 0
        var shift = UInt64(56 - 8 * pass_number)
        for at in range(begin, end):
            var row = at
            comptime if Indexed:
                row = Int(index[unsafe_offset=at])
            var source = base + row
            if validity[unsafe_offset=source] == 0:
                continue
            var value = key(words[unsafe_offset=source])
            if value & mask == prefix:
                bins[unsafe_offset=Int((value >> shift) & 255)] += 1
        var chosen = 0
        while chosen < 255 and wanted >= bins[unsafe_offset=chosen]:
            wanted -= bins[unsafe_offset=chosen]
            chosen += 1
        prefix |= UInt64(chosen) << shift
        mask |= UInt64(255) << shift
    if prefix == UInt64.MAX:
        return 0x7FF8000000000000
    return prefix ^ f.SIGN if prefix >> 63 else ~prefix


def reduce[
    Indexed: Bool
](
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    base: Int,
    begin: Int,
    end: Int,
    probability: UInt64,
    op: Int64,
) -> Tuple[UInt64, UInt8]:
    var count = UInt32(0)
    for at in range(begin, end):
        var row = at
        comptime if Indexed:
            row = Int(index[unsafe_offset=at])
        count += UInt32(validity[unsafe_offset=base + row] != 0)
    if count == 0:
        return (0, 0)
    var position = f.mul(probability, f.from_unsigned(UInt64(count - 1)))
    var lower = UInt32(f.to_integer(position, 32, True)[0])
    var fraction = f.sub(position, f.from_unsigned(UInt64(lower)))
    var upper = min(lower + UInt32(f.absolute(fraction) != 0), count - 1)
    var wanted = lower
    if op == ROW_QUANTILE_HIGHER:
        wanted = upper
    elif op == ROW_QUANTILE_NEAREST:
        wanted = min(
            UInt32(
                f.to_integer(f.add(position, 0x3FE0000000000000), 32, True)[0]
            ),
            count - 1,
        )
    var lo = select[Indexed](words, validity, index, base, begin, end, wanted)
    if (
        op == ROW_QUANTILE_LOWER
        or op == ROW_QUANTILE_HIGHER
        or op == ROW_QUANTILE_NEAREST
        or lower == upper
        or f.absolute(fraction) == 0
    ):
        return (lo, 1)
    var hi = select[Indexed](words, validity, index, base, begin, end, upper)
    if op == ROW_QUANTILE_MIDPOINT:
        return (f.div(f.add(lo, hi), 0x4000000000000000), 1)
    return (f.fma(f.sub(hi, lo), fraction, lo), 1)


def empty_result(op: Int64, minimum: Int64) -> Tuple[UInt64, UInt8, UInt32]:
    """Every order statistic of an input with no valid values is null."""
    return (0, 0, 0)


def reduce_partials(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    descriptors: Pointer[Int64, ImmutAnyOrigin],
    partials: Pointer[UInt64, MutAnyOrigin],
    outputs: UInt32,
    groups: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    var work = global_idx.x
    if work >= Int(outputs) * Int(groups):
        return
    var column = work // Int(groups)
    if work % Int(groups) != 0:
        return
    var descriptor = column * DESCRIPTOR_WORDS
    var op = descriptors[unsafe_offset=descriptor + 3]
    if not supported(op):
        return
    var base = Int(descriptors[unsafe_offset=descriptor]) * Int(stride)
    var probability = descriptors[unsafe_offset=descriptor + 4].cast[
        DType.uint64
    ]()
    # The unindexed specialization does not read its index argument.
    var result = reduce[False](
        words,
        validity,
        words.unsafe_bitcast[UInt32](),
        base,
        0,
        Int(rows),
        probability,
        op,
    )
    var error = UInt32(0)
    for row in range(Int(rows)):
        error |= errors[unsafe_offset=base + row]
    var at = work * STATE_WORDS
    for field in range(STATE_WORDS):
        partials[unsafe_offset=at + field] = 0
    partials[unsafe_offset=at] = result[0]
    partials[unsafe_offset=at + 1] = UInt64(result[1])
    partials[unsafe_offset=at + 6] = UInt64(error)


def reduce_finish(
    partials: Pointer[UInt64, ImmutAnyOrigin],
    descriptors: Pointer[Int64, ImmutAnyOrigin],
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    outputs: UInt32,
    groups: UInt32,
    rows: UInt32,
):
    var column = global_idx.x
    if column >= Int(outputs):
        return
    if not supported(descriptors[unsafe_offset=column * DESCRIPTOR_WORDS + 3]):
        return
    var at = column * Int(groups) * STATE_WORDS
    words[unsafe_offset=column] = partials[unsafe_offset=at]
    validity[unsafe_offset=column] = UInt8(partials[unsafe_offset=at + 1])
    errors[unsafe_offset=column] = UInt32(partials[unsafe_offset=at + 6])
