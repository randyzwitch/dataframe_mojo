"""Bulk fixed-width row gathering over typed storage."""
from dataframe.storage.column import Column, _validity_bit
from std.memory import unsafe_memcpy


def gather_scalars[
    D: DType
](
    column: Column[Scalar[D]],
    indices: List[Int],
    allow_missing: Bool,
    first: Int = 0,
    last: Int = -1,
    base: Int = 0,
) raises -> Column[Scalar[D]]:
    """Rows of a fixed-width column, in bulk (#328).

    One pass checks bounds; values are then copied with one memcpy per run
    of consecutive source rows, which filters produce in long stretches.
    Validity is gathered only when the source has nulls or a row is missing
    (-1 with allow_missing, as outer joins pass); missing rows hold zero.
    """
    # indices[first:last], each minus `base`: a chunk of a larger column
    # reads its share of the shared filter indices without a copy.
    var n = (len(indices) if last < 0 else last) - first
    var rows = indices.unsafe_ptr().unsafe_offset(first)
    var limit = len(column)
    var missing = False
    # Every position is written below: copied rows by the run copies, and
    # missing rows (-1) with an explicit zero, so nothing is filled first.
    var values = List[Scalar[D]](unsafe_uninit_length=n)
    var src = column._ptr()
    var dst = values.unsafe_ptr()
    var k = 0
    # One pass: bounds checks ride along with run detection.
    while k < n:
        var row = rows.unsafe_offset(k)[]
        if row != -1:
            row -= base
        if row < 0 or row >= limit:
            if row == -1 and allow_missing:
                missing = True
                dst.unsafe_offset(k)[] = 0
                k += 1
                continue
            raise Error("Column index out of bounds")
        var end = k + 1
        var next = row + 1
        while end < n and rows.unsafe_offset(end)[] - base == next:
            end += 1
            next += 1
        if next > limit:
            raise Error("Column index out of bounds")
        unsafe_memcpy(
            dest=dst.unsafe_offset(k), src=src.unsafe_offset(row), count=end - k
        )
        k = end
    ref source_bits = column._bits[]
    var bits = List[UInt8]()
    if missing or len(source_bits) > 0:
        bits = List[UInt8](length=(n + 7) // 8, fill=0)
        for at in range(n):
            var row = rows.unsafe_offset(at)[]
            if row == -1:
                continue
            row -= base
            if _validity_bit(source_bits, column._offset + row):
                bits[at >> 3] |= UInt8(1) << UInt8(at & 7)
    return Column[Scalar[D]](values=values^, bits=bits^)
