"""Boolean columns with bit-packed values (Arrow layout).

Values are an LSB-first bitmap, like validity, so a Boolean column costs one
bit per row instead of one byte and exports to Arrow without converting.
Like `Column`, a column is a window (offset, length) onto reference-counted
buffers that are never mutated while shared.

Both buffers use the bitmap helpers in `column.mojo`, so windows at any
offset append, slice, and compact correctly.
"""
from std.bit import pop_count
from std.memory import ArcPointer
from .column import (
    Column,
    _append_bits,
    _bit,
    _copy_bits,
    _count_set,
    _pack_bits,
)


struct BoolColumn(Copyable, Sized):
    """A window onto shared, bit-packed values and validity."""

    var _data: ArcPointer[List[UInt8]]
    var _bits: ArcPointer[List[UInt8]]
    var _offset: Int
    var _length: Int

    def __init__(out self, values: List[Bool]):
        self._length = len(values)
        self._offset = 0
        self._data = ArcPointer(_pack_bits(values))
        self._bits = ArcPointer(
            List[UInt8](length=(len(values) + 7) // 8, fill=255)
        )

    def __init__(out self, values: List[Bool], valid: List[Bool]) raises:
        if len(values) != len(valid):
            raise Error("Column values and validity must have equal lengths")
        self = Self(values)
        self._bits = ArcPointer(_pack_bits(valid))

    def __init__(out self, column: Column[Bool]) raises:
        """Pack a byte-per-value Boolean column."""
        var values = List[Bool](capacity=len(column))
        var valid = List[Bool](capacity=len(column))
        for i in range(len(column)):
            values.append(column._get(i))
            valid.append(column._valid(i))
        self = Self(values, valid)

    def __init__(
        out self,
        *,
        var values: List[UInt8],
        var bits: List[UInt8],
        length: Int,
    ):
        """Adopt finished value and validity bitmaps."""
        self._data = ArcPointer(values^)
        self._bits = ArcPointer(bits^)
        self._offset = 0
        self._length = length

    def __len__(self) -> Int:
        return self._length

    def _get(self, i: Int) -> Bool:
        """Unchecked read of row i."""
        return _bit(self._data[], self._offset + i)

    def _valid(self, i: Int) -> Bool:
        return _bit(self._bits[], self._offset + i)

    def _to_list(self) -> List[Bool]:
        var values = List[Bool](capacity=self._length)
        for i in range(self._length):
            values.append(self._get(i))
        return values^

    def _shares_buffers_with(self, other: Self) -> Bool:
        return (
            self._data.ptr() == other._data.ptr()
            and self._bits.ptr() == other._bits.ptr()
        )

    def _check_index(self, index: Int) raises:
        if index < 0 or index >= self._length:
            raise Error("Column index out of bounds")

    def is_null(self, index: Int) raises -> Bool:
        self._check_index(index)
        return not self._valid(index)

    def value(self, index: Int) raises -> Bool:
        if self.is_null(index):
            raise Error("Cannot read a null value")
        return self._get(index)

    def null_count(self) -> Int:
        return self._length - _count_set(
            self._bits[], self._offset, self._length
        )

    def true_count(self) -> Int:
        """Rows whose value bit is set (including null rows)."""
        return _count_set(self._data[], self._offset, self._length)

    def take(self, indices: List[Int]) raises -> Self:
        for i in indices:
            self._check_index(i)
        ref data = self._data[]
        ref bits = self._bits[]
        var base = self._offset
        var values = List[UInt8](length=(len(indices) + 7) // 8, fill=0)
        var out_bits = List[UInt8](length=(len(indices) + 7) // 8, fill=0)
        for k in range(len(indices)):
            var row = base + indices[k]
            var mask = UInt8(1) << UInt8(k % 8)
            if _bit(data, row):
                values[k // 8] |= mask
            if _bit(bits, row):
                out_bits[k // 8] |= mask
        return Self(values=values^, bits=out_bits^, length=len(indices))

    def take_or_null(self, indices: List[Int], fill: Bool) raises -> Self:
        """Gather rows, treating only -1 as a missing row (for outer joins)."""
        for i in indices:
            if i != -1:
                self._check_index(i)
        ref data = self._data[]
        ref bits = self._bits[]
        var base = self._offset
        var values = List[UInt8](length=(len(indices) + 7) // 8, fill=0)
        var out_bits = List[UInt8](length=(len(indices) + 7) // 8, fill=0)
        for k in range(len(indices)):
            var i = indices[k]
            var mask = UInt8(1) << UInt8(k % 8)
            if i == -1:
                if fill:
                    values[k // 8] |= mask
                continue
            var row = base + i
            if _bit(data, row):
                values[k // 8] |= mask
            if _bit(bits, row):
                out_bits[k // 8] |= mask
        return Self(values=values^, bits=out_bits^, length=len(indices))

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
        return (
            self._data.count() == 1
            and self._bits.count() == 1
            and self._offset == 0
            and len(self._data[]) == (self._length + 7) // 8
        )

    def _compact(self) -> Self:
        return Self(
            values=_copy_bits(self._data[], self._offset, self._length),
            bits=_copy_bits(self._bits[], self._offset, self._length),
            length=self._length,
        )

    def _append_column(mut self, other: Self):
        """Append values and validity bitwise, copying first unless this
        column owns its buffers."""
        if not self._owned():
            self = self._compact()
        var count = other._length
        if count == 0:
            return
        _append_bits(
            self._data[], self._length, other._data[], other._offset, count
        )
        _append_bits(
            self._bits[], self._length, other._bits[], other._offset, count
        )
        self._length += count

    @staticmethod
    def _nulls(length: Int, fill: Bool = False) -> Self:
        return Self(
            values=List[UInt8](
                length=(length + 7) // 8, fill=UInt8(255) if fill else UInt8(0)
            ),
            bits=List[UInt8](length=(length + 7) // 8, fill=0),
            length=length,
        )

    def _broadcast(self, length: Int) raises -> Self:
        if self._length != 1 or length < 0:
            raise Error("Only a scalar result can broadcast")
        var bytes = (length + 7) // 8
        return Self(
            values=List[UInt8](
                length=bytes, fill=UInt8(255) if self._get(0) else UInt8(0)
            ),
            bits=List[UInt8](
                length=bytes, fill=UInt8(255) if self._valid(0) else UInt8(0)
            ),
            length=length,
        )
