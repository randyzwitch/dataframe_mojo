"""Raw row bytes of string columns, for kernels that compare many rows.

`StringColumn._get` re-checks the storage kind and builds a `StringSlice`
for every row, and `Span`'s `==` checks the bounds of every byte; in a loop
over every row those checks cost more than the comparison itself. `_Bytes`
reads offsets and bytes (or view storage) directly, and `_same_bytes` and
`_order_bytes` compare eight bytes at a time.
"""
from std.bit import byte_swap

from .string_column import StringColumn
from .string_view import StringViewStorage


@always_inline
def _same_bytes(
    a: Span[UInt8, ImmutAnyOrigin], b: Span[UInt8, ImmutAnyOrigin]
) -> Bool:
    """Byte equality eight bytes at a time; Span's == checks the bounds of
    each byte, which was most of a string count."""
    var n = len(a)
    if n != len(b):
        return False
    var x = a.unsafe_ptr()
    var y = b.unsafe_ptr()
    var i = 0
    while i + 8 <= n:
        if (
            x.unsafe_offset(i).unsafe_bitcast[UInt64]().unsafe_load()
            != y.unsafe_offset(i).unsafe_bitcast[UInt64]().unsafe_load()
        ):
            return False
        i += 8
    while i < n:
        if x.unsafe_offset(i)[] != y.unsafe_offset(i)[]:
            return False
        i += 1
    return True


struct _Bytes(Copyable, Movable):
    """Row bytes of one string chunk, read from its buffers directly: the
    column's own accessor re-checks its storage kind and copies the view
    storage handle, whose reference count every worker shares, per row."""

    var column: StringColumn
    var storage: List[StringViewStorage]
    var offsets: Int
    var data: Int

    def __init__(out self, column: StringColumn):
        self.column = column.copy()
        self.storage = List[StringViewStorage]()
        self.offsets = 0
        self.data = 0
        if column._is_view_storage():
            self.storage.append(column._view_storage_unchecked())
        else:
            self.offsets = Int(column.unsafe_offsets()) + 8 * column._offset
            self.data = Int(column.unsafe_bytes())

    @always_inline
    def get(self, i: Int) -> Span[UInt8, ImmutAnyOrigin]:
        if len(self.storage) > 0:
            return (
                self.storage[0]
                ._get_unchecked(self.column._offset + i)
                .as_bytes()
            )
        var offsets = Pointer[Int64, ImmutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        var start = Int(offsets.unsafe_offset(i)[])
        var end = Int(offsets.unsafe_offset(i + 1)[])
        return Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=self.data + start
            ),
            length=end - start,
        )


@always_inline
def _order_bytes(
    a: Span[UInt8, ImmutAnyOrigin], b: Span[UInt8, ImmutAnyOrigin]
) -> Int:
    """-1, 0 or 1 as `a` sorts before, equal to or after `b`, byte by byte
    (which for UTF-8 is code point order), the shorter first on a tie."""
    var n = min(len(a), len(b))
    var x = a.unsafe_ptr()
    var y = b.unsafe_ptr()
    var i = 0
    while i + 8 <= n:
        var u = x.unsafe_offset(i).unsafe_bitcast[UInt64]().unsafe_load()
        var v = y.unsafe_offset(i).unsafe_bitcast[UInt64]().unsafe_load()
        if u != v:
            # Loaded little-endian; swapped, the first byte is the most
            # significant, so integer order is byte order.
            return -1 if byte_swap(u) < byte_swap(v) else 1
        i += 8
    while i < n:
        var u = x.unsafe_offset(i)[]
        var v = y.unsafe_offset(i)[]
        if u != v:
            return -1 if u < v else 1
        i += 1
    if len(a) == len(b):
        return 0
    return -1 if len(a) < len(b) else 1
