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


@always_inline
def _find_bytes(
    hay: Span[UInt8, ImmutAnyOrigin],
    needle: Span[UInt8, ImmutAnyOrigin],
    start: Int,
) -> Int:
    """The first position at or after `start` where `needle` occurs in
    `hay`, or -1. Candidates are filtered on the needle's first and last
    bytes before the rest is compared (#374)."""
    var m = len(needle)
    var n = len(hay)
    if m == 0:
        return start if start <= n else -1
    if m > n - start:
        return -1
    var h = hay.unsafe_ptr()
    var p = needle.unsafe_ptr()
    var first = p[]
    var last = p.unsafe_offset(m - 1)[]
    # 32 candidate positions at a time: compare them with the needle's
    # first and last bytes, and verify only where both match (#374).
    comptime lanes = 32
    var firsts = SIMD[DType.uint8, lanes](first)
    var lasts = SIMD[DType.uint8, lanes](last)
    var i = start
    while i + m - 1 + lanes <= n:
        var hits = h.unsafe_load[width=lanes](i).eq(firsts) & h.unsafe_load[
            width=lanes
        ](i + m - 1).eq(lasts)
        if hits.reduce_or():
            for k in range(lanes):
                if hits[k]:
                    var same = True
                    for j in range(1, m - 1):
                        if h.unsafe_offset(i + k + j)[] != p.unsafe_offset(j)[]:
                            same = False
                            break
                    if same:
                        return i + k
        i += lanes
    for i in range(i, n - m + 1):
        if (
            h.unsafe_offset(i)[] == first
            and h.unsafe_offset(i + m - 1)[] == last
        ):
            var same = True
            for k in range(1, m - 1):
                if h.unsafe_offset(i + k)[] != p.unsafe_offset(k)[]:
                    same = False
                    break
            if same:
                return i
    return -1


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
