"""Raw row bytes of string columns, for kernels that compare many rows.

`StringColumn._get` re-checks the storage kind and builds a `StringSlice`
for every row, and `Span`'s `==` checks the bounds of every byte; in a loop
over every row those checks cost more than the comparison itself. `_Bytes`
reads offsets and bytes (or view storage) directly, and `_same_bytes` and
`_order_bytes` compare eight bytes at a time.
"""
from dataframe.storage.buffer import _Buffer
from std.bit import byte_swap

from std.memory import ArcPointer, Pointer

from dataframe.regex.dfa import find_literal_bytes
from dataframe.storage.string import StringColumn
from dataframe.storage.string_view import StringView


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
    """Row bytes of one string chunk, read through pointers taken once, for
    either storage: offsets and bytes, or Arrow views (16-byte descriptors:
    length and a four-byte prefix, then eight more inline bytes up to
    twelve or a buffer and an offset). The column's own accessor re-checks
    its storage kind and copies the view storage handle, whose reference
    count every worker shares, per row; that made every kernel slower on
    views than on offsets."""

    var column: StringColumn
    var views: Int
    var buffers: Int
    var offsets: Int
    var data: Int

    def __init__(out self, column: StringColumn):
        # Views are read one descriptor a row: a column selecting another's
        # descriptors is copied out here, and held.
        self.column = column._flat_views()
        self.views = 0
        self.buffers = 0
        self.offsets = 0
        self.data = 0
        if self.column._is_view_storage():
            ref storage = self.column._view_storage.value()
            self.views = Int(
                storage._views[].unsafe_ptr().unsafe_offset(column._offset)
            )
            self.buffers = Int(storage._buffers[].unsafe_ptr())
        else:
            self.offsets = Int(column.unsafe_offsets()) + 8 * column._offset
            self.data = Int(column.unsafe_bytes())

    @always_inline
    def is_views(self) -> Bool:
        return self.views != 0

    @always_inline
    def get(self, i: Int) -> Span[UInt8, ImmutAnyOrigin]:
        if self.views != 0:
            var view = Pointer[StringView, ImmutAnyOrigin](
                unsafe_from_address=self.views
            ).unsafe_offset(i)
            var length = Int(view[].length)
            var at: Int
            if length <= 12:
                at = Int(view) + 4
            else:
                at = Int(
                    Pointer[_Buffer[UInt8], ImmutAnyOrigin](
                        unsafe_from_address=self.buffers
                    )[unsafe_offset=Int(view[].buffer_index)][].unsafe_ptr()
                ) + Int(view[].offset)
            return Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                    unsafe_from_address=at
                ),
                length=length,
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
    def head(self, i: Int) -> UInt64:
        """Row i's length and first four bytes, as a view's first word (zero
        past the end of a shorter value): equal strings have equal heads,
        so unequal heads settle most unequal pairs without their bytes."""
        if self.views != 0:
            var word = Pointer[UInt64, ImmutAnyOrigin](
                unsafe_from_address=self.views
            )[unsafe_offset=2 * i]
            var length = Int(word & 0xFFFFFFFF)
            if length >= 4:
                return word
            # Bytes past a short value are not promised to be zero.
            var keep = (UInt64(1) << UInt64(8 * length)) - 1
            return (word & 0xFFFFFFFF) | (((word >> 32) & keep) << 32)
        var bytes = self.get(i)
        var head = UInt64(len(bytes))
        for b in range(min(len(bytes), 4)):
            head |= UInt64(bytes.unsafe_ptr()[unsafe_offset=b]) << UInt64(
                32 + 8 * b
            )
        return head

    @always_inline
    def equal(self, i: Int, other: _Bytes, j: Int) -> Bool:
        """Whether row i equals `other`'s row j (validity aside): heads
        first, then the bytes."""
        if self.views != 0 and other.views != 0:
            if self.head(i) != other.head(j):
                return False
            if len(self.get(i)) <= 4:
                return True
        return _same_bytes(self.get(i), other.get(j))


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


@always_inline
def _find_bytes(
    hay: Span[UInt8, ImmutAnyOrigin],
    needle: Span[UInt8, ImmutAnyOrigin],
    start: Int,
) -> Int:
    """The first position at or after `start` where `needle` occurs in
    `hay`, or -1: the regex engine's SIMD memmem (`find_literal_bytes`),
    which extracts candidate positions from a lane mask and scans 64 bytes a
    step, as Polars' `memmem` and DuckDB's `ContainsFun` find a literal."""
    return find_literal_bytes(hay, needle.unsafe_ptr(), len(needle), start)


@always_inline
def _utf8_width(byte: UInt8) -> Int:
    if byte < 0x80:
        return 1
    if byte < 0xE0:
        return 2
    if byte < 0xF0:
        return 3
    return 4


def _match_at(
    value: Span[UInt8, ImmutAnyOrigin],
    at: Int,
    segment: Span[UInt8, ImmutAnyOrigin],
) -> Int:
    """Where `segment` (a LIKE pattern piece without '%', in which '_'
    matches one code point) ends when matched at `at`, or -1."""
    var i = at
    var v = value.unsafe_ptr()
    for k in range(len(segment)):
        var b = segment.unsafe_ptr().unsafe_offset(k)[]
        if i >= len(value):
            return -1
        if b == 95:
            i += _utf8_width(v.unsafe_offset(i)[])
        elif v.unsafe_offset(i)[] == b:
            i += 1
        else:
            return -1
    return i if i <= len(value) else -1


def _search(
    value: Span[UInt8, ImmutAnyOrigin],
    at: Int,
    segment: Span[UInt8, ImmutAnyOrigin],
    wildcards: Bool,
) -> Tuple[Int, Int]:
    """The first match of `segment` at or after `at`: (start, end), or
    (-1, -1)."""
    if not wildcards:
        var found = _find_bytes(value, segment, at)
        return (found, found + len(segment)) if found >= 0 else (-1, -1)
    for p in range(at, len(value) + 1):
        var end = _match_at(value, p, segment)
        if end >= 0:
            return (p, end)
    return (-1, -1)
