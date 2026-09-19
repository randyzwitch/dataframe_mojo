"""Typed columns over shared, immutable, Arrow-compatible buffers.

A column is a window (offset, length) onto a reference-counted value buffer
and an LSB-first validity bitmap, as in an Arrow array. Copying a column,
slicing it, and every operation that only selects columns (select, rename,
drop, head, GroupBy snapshots) share buffers in O(1) instead of copying
values. Buffers are never mutated while shared: the only mutating operation,
batch reassembly in `_append_column`, first takes a private copy unless the
column already owns its buffers outright.
"""
from std.bit import pop_count
from std.memory import ArcPointer, Pointer
from std.sys.info import is_little_endian


struct Column[T: Copyable & Deinitable](Copyable, Sized):
    """A window onto a shared payload buffer and validity bitmap.

    Underscored storage is internal. Public operations return new columns;
    no mutable buffers or implicit negative indexing are exposed.
    """

    var _data: ArcPointer[List[Self.T]]
    var _bits: ArcPointer[List[UInt8]]
    var _offset: Int
    var _length: Int

    def __init__(out self, var values: List[Self.T]):
        # Bitmaps are shared as little-endian bytes (Arrow layout).
        comptime assert is_little_endian(), "dataframe requires little-endian"
        self._length = len(values)
        self._offset = 0
        self._bits = ArcPointer(
            List[UInt8](length=(len(values) + 7) // 8, fill=255)
        )
        self._data = ArcPointer(values^)

    def __init__(out self, var values: List[Self.T], valid: List[Bool]) raises:
        if len(values) != len(valid):
            raise Error("Column values and validity must have equal lengths")
        var bits = _pack_bits(valid)
        self._length = len(values)
        self._offset = 0
        self._bits = ArcPointer(bits^)
        self._data = ArcPointer(values^)

    def __len__(self) -> Int:
        return self._length

    def _get(self, i: Int) -> ref[ImmutAnyOrigin] Self.T:
        """Unchecked read of row i; valid while this column is alive."""
        return self._data[][self._offset + i]

    def _ptr(self) -> Pointer[Self.T, MutAnyOrigin]:
        """Pointer to row 0 of this window, for SIMD loads (read-only use)."""
        return (
            self._data[]
            .unsafe_ptr()
            .unsafe_offset(self._offset)
            .unsafe_origin_cast[MutAnyOrigin]()
        )

    def _to_list(self) -> List[Self.T]:
        """An owned copy of this window's payloads (including null slots)."""
        var values = List[Self.T](capacity=self._length)
        for i in range(self._length):
            values.append(self._get(i).copy())
        return values^

    def _shares_buffers_with(self, other: Self) -> Bool:
        return (
            self._data.ptr() == other._data.ptr()
            and self._bits.ptr() == other._bits.ptr()
        )

    def _check_index(self, index: Int) raises:
        if index < 0 or index >= self._length:
            raise Error("Column index out of bounds")

    def _valid(self, i: Int) -> Bool:
        """Internal unchecked validity read after bounds validation."""
        return _bit(self._bits[], self._offset + i)

    def is_null(self, index: Int) raises -> Bool:
        self._check_index(index)
        return not self._valid(index)

    def value(self, index: Int) raises -> Self.T:
        if self.is_null(index):
            raise Error("Cannot read a null value")
        return self._get(index).copy()

    def null_count(self) -> Int:
        return self._length - _count_set(
            self._bits[], self._offset, self._length
        )

    def take(self, indices: List[Int]) raises -> Self:
        for i in indices:
            self._check_index(i)
        ref data = self._data[]
        ref bits = self._bits[]
        var base = self._offset
        var values = List[Self.T](capacity=len(indices))
        var out_bits = List[UInt8](length=(len(indices) + 7) // 8, fill=0)
        for k in range(len(indices)):
            var row = base + indices[k]
            values.append(data[row].copy())
            if (bits[row // 8] >> UInt8(row % 8)) & 1 == 1:
                out_bits[k // 8] |= UInt8(1) << UInt8(k % 8)
        var result = Self(values^)
        result._bits = ArcPointer(out_bits^)
        return result^

    def take_or_null(self, indices: List[Int], fill: Self.T) raises -> Self:
        """Gather rows, treating only -1 as a missing row (for outer joins)."""
        for i in indices:
            if i != -1:
                self._check_index(i)
        ref data = self._data[]
        ref bits = self._bits[]
        var base = self._offset
        var values = List[Self.T](capacity=len(indices))
        var out_bits = List[UInt8](length=(len(indices) + 7) // 8, fill=0)
        for k in range(len(indices)):
            var i = indices[k]
            if i == -1:
                values.append(fill.copy())
                continue
            var row = base + i
            values.append(data[row].copy())
            if (bits[row // 8] >> UInt8(row % 8)) & 1 == 1:
                out_bits[k // 8] |= UInt8(1) << UInt8(k % 8)
        var result = Self(values^)
        result._bits = ArcPointer(out_bits^)
        return result^

    def slice(self, offset: Int, length: Int) raises -> Self:
        """A zero-copy window sharing this column's buffers."""
        if (
            offset < 0
            or length < 0
            or offset > self._length
            or length > self._length - offset
        ):
            raise Error("Invalid column slice")
        var result = self.copy()
        result._offset = self._offset + offset
        result._length = length
        return result^

    def _owned(self) -> Bool:
        """Whether the buffers are unshared and exactly this window."""
        return (
            self._data.count() == 1
            and self._bits.count() == 1
            and self._offset == 0
            and len(self._data[]) == self._length
        )

    def _compact(self) -> Self:
        """A private copy of this window with offset 0."""
        var result = Self(self._to_list())
        result._bits = ArcPointer(
            _copy_bits(self._bits[], self._offset, self._length)
        )
        return result^

    def _append_column(mut self, other: Self):
        """Append payloads and validity bytewise, shifting when unaligned.

        Copies first unless this column owns its buffers, so values shared
        with other columns are never modified.
        """
        if not self._owned():
            self = self._compact()
        var start = self._length
        var count = other._length
        _append_bits(self._bits[], start, other._bits[], other._offset, count)
        ref values = self._data[]
        # Grow geometrically: batch reassembly appends many small chunks, and
        # an exact reservation would copy the whole column on every append.
        if values.capacity() < start + count:
            values.reserve(max(start + count, 2 * values.capacity()))
        for i in range(count):
            values.append(other._get(i).copy())
        self._length = start + count

    @staticmethod
    def _nulls(length: Int, fill: Self.T) -> Self:
        var result = Self(List[Self.T](length=length, fill=fill.copy()))
        result._bits = ArcPointer(List[UInt8](length=(length + 7) // 8, fill=0))
        return result^

    def _broadcast(self, length: Int) raises -> Self:
        if self._length != 1 or length < 0:
            raise Error("Only a scalar result can broadcast")
        var values = List[Self.T](length=length, fill=self._get(0).copy())
        var valid = List[Bool](length=length, fill=self._valid(0))
        return Self(values^, valid)


# Validity bitmaps: LSB-first bytes, 1 = valid, as in Arrow. Bits past a
# column's length are unspecified; readers never look at them.


def _bit(bits: List[UInt8], i: Int) -> Bool:
    return (bits[i // 8] >> UInt8(i % 8)) & 1 == 1


def _count_set(bits: List[UInt8], offset: Int, length: Int) -> Int:
    """Set bits in [offset, offset + length): popcount over whole bytes."""
    var count = 0
    var i = 0
    while i < length and (offset + i) % 8 != 0:
        count += Int(_bit(bits, offset + i))
        i += 1
    var byte = (offset + i) // 8
    while length - i >= 8:
        count += Int(pop_count(bits[byte]))
        byte += 1
        i += 8
    while i < length:
        count += Int(_bit(bits, offset + i))
        i += 1
    return count


def _pack_bits(valid: List[Bool]) -> List[UInt8]:
    var bits = List[UInt8](length=(len(valid) + 7) // 8, fill=0)
    for i in range(len(valid)):
        if valid[i]:
            bits[i // 8] |= UInt8(1) << UInt8(i % 8)
    return bits^


def _copy_bits(bits: List[UInt8], offset: Int, length: Int) -> List[UInt8]:
    """Bits [offset, offset + length) rebased to bit 0."""
    var out = List[UInt8](capacity=(length + 7) // 8)
    _append_bits(out, 0, bits, offset, length)
    return out^


def _append_bits(
    mut bits: List[UInt8],
    length: Int,
    incoming: List[UInt8],
    offset: Int,
    count: Int,
):
    """Append bits [offset, offset + count) of incoming after bit `length`.

    Byte-aligned sources merge bytewise (shifted when the destination is not
    byte-aligned); unaligned sources fall back to per-bit copies.
    """
    if count == 0:
        return
    var needed = (length + count + 7) // 8
    if offset % 8 != 0:
        # Clear stale bits above the current length, then set bit by bit.
        if length % 8 != 0:
            bits[len(bits) - 1] &= (UInt8(1) << UInt8(length % 8)) - 1
        while len(bits) < needed:
            bits.append(0)
        for i in range(count):
            if _bit(incoming, offset + i):
                var bit = length + i
                bits[bit // 8] |= UInt8(1) << UInt8(bit % 8)
        return
    var first_byte = offset // 8
    var shift = length % 8
    var full_bytes = count // 8
    var tail_bits = count % 8
    if shift == 0:
        for b in range(full_bytes):
            bits.append(incoming[first_byte + b])
        if tail_bits > 0:
            var mask = (UInt8(1) << UInt8(tail_bits)) - 1
            bits.append(incoming[first_byte + full_bytes] & mask)
        return
    var last = len(bits) - 1
    bits[last] &= (UInt8(1) << UInt8(shift)) - 1
    var source_bytes = (count + 7) // 8
    for b in range(source_bytes):
        var byte = incoming[first_byte + b]
        if b == source_bytes - 1 and tail_bits > 0:
            byte &= (UInt8(1) << UInt8(tail_bits)) - 1
        bits[len(bits) - 1] |= byte << UInt8(shift)
        bits.append(byte >> UInt8(8 - shift))
    while len(bits) > needed:
        _ = bits.pop()
