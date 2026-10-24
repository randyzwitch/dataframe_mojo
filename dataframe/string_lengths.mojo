"""String lengths from storage metadata and UTF-8 leading bytes."""
from std.memory import Pointer
from .column import Column, _copy_validity
from .series import Series
from .string_bytes import _Bytes
from .string_column import StringColumn


@always_inline
def _count_chars(row: Span[UInt8, ImmutAnyOrigin]) -> Int64:
    # Every UTF-8 byte except a continuation starts one code point. This
    # counts combining marks separately, just as String.codepoints does.
    var p = row.unsafe_ptr()
    var count = Int64(0)
    var i = 0
    while i + 32 <= len(row):
        var bytes = p.unsafe_offset(i).unsafe_load[width=32]()
        count += Int64(
            (bytes & SIMD[DType.uint8, 32](192))
            .ne(SIMD[DType.uint8, 32](128))
            .cast[DType.uint8]()
            .reduce_add()
        )
        i += 32
    while i < len(row):
        count += Int64((p.unsafe_offset(i)[] & 192) != 128)
        i += 1
    return count


def string_lengths[characters: Bool](column: StringColumn) -> Series:
    var n = len(column)
    var rows = _Bytes(column)
    var values = List[Int64](length=n, fill=0)
    var out = values.unsafe_ptr()
    var nullable = len(column._bits[]) != 0
    comptime if characters:
        for i in range(n):
            if not nullable or column._valid(i):
                out.unsafe_offset(i)[] = _count_chars(rows.get(i))
    else:
        if rows.is_views():
            var views = Pointer[UInt32, ImmutAnyOrigin](
                unsafe_from_address=rows.views
            )
            for i in range(n):
                out.unsafe_offset(i)[] = Int64(views.unsafe_offset(4 * i)[])
        else:
            var offsets = Pointer[Int64, ImmutAnyOrigin](
                unsafe_from_address=rows.offsets
            )
            var i = 0
            while i + 4 <= n:
                out.unsafe_offset(i).unsafe_store(
                    offsets.unsafe_offset(i + 1).unsafe_load[width=4]()
                    - offsets.unsafe_offset(i).unsafe_load[width=4]()
                )
                i += 4
            while i < n:
                out.unsafe_offset(i)[] = (
                    offsets.unsafe_offset(i + 1)[] - offsets.unsafe_offset(i)[]
                )
                i += 1
    return Series(
        "",
        Column[Int64](
            values=values^,
            bits=_copy_validity(column._bits[], column._offset, n),
        ),
    )
