"""Global Boolean and validity counts over packed bitmap words."""
from std.bit import pop_count
from std.memory import Pointer
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from .multi_reduce import Bytes
from .filter_sum import BLOCK_SIZE


@always_inline
def bitmap_word(
    data: Bytes, offset: UInt32, row: UInt32, rows: UInt32
) -> UInt32:
    """Up to 32 logical bits, without reading past the last bitmap byte."""
    var first = offset + row
    var byte = Int(first >> 3)
    var shift = first & 7
    var end = Int((offset + rows + 7) >> 3)
    var word = UInt32(0)
    comptime for k in range(4):
        if byte + k < end:
            word |= UInt32(data[unsafe_offset=byte + k]) << UInt32(k * 8)
    word >>= shift
    if shift and byte + 4 < end:
        word |= UInt32(data[unsafe_offset=byte + 4]) << (32 - shift)
    var mask = UInt32.MAX >> (32 - min(UInt32(32), rows - row))
    return word & mask


def logical_word_partials(
    values: Bytes,
    validity: Bytes,
    value_offset: UInt32,
    validity_offset: UInt32,
    rows: UInt32,
    flags: UInt32,
    stride: UInt32,
    result: Pointer[UInt32, MutUntrackedOrigin],
    partial_offset: UInt32,
):
    var valid = UInt32(0)
    var true = UInt32(0)
    var need_values = Bool(flags & 1)
    var nullable = Bool(flags & 2)
    if need_values or nullable:
        var row = UInt32(global_idx.x) * 32
        while row < rows:
            var mask = UInt32.MAX >> (32 - min(UInt32(32), rows - row))
            var present = bitmap_word(
                validity, validity_offset, row, rows
            ) if nullable else mask
            valid += pop_count(present)
            if need_values:
                true += pop_count(
                    bitmap_word(values, value_offset, row, rows) & present
                )
            row += stride * 32
    elif global_idx.x == 0:
        # A plain count over an all-valid column needs no input scan.
        valid = rows
    var valid_total = block.sum[block_size=BLOCK_SIZE, broadcast=False](valid)
    var true_total = block.sum[block_size=BLOCK_SIZE, broadcast=False](true)
    if thread_idx.x == 0:
        var at = Int(partial_offset) + block_idx.x * 2
        result[unsafe_offset=at] = valid_total
        result[unsafe_offset=at + 1] = true_total
