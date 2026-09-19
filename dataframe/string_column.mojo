"""String columns in the Arrow `large_utf8` layout.

Row i of the underlying buffers is the bytes `[offsets[i], offsets[i + 1])` of
one contiguous UTF-8 buffer; `offsets` has one more entry than there are
buffer rows and starts at 0. Offsets are Int64 so a column can exceed 2 GiB of
text, matching Arrow `large_utf8`. Validity is the same LSB-first bitmap that
`Column` uses. A column is a window (offset, length) onto shared buffers, so
copies, slices, and projections are O(1); like `Column`, buffers are never
mutated while shared.

Internal kernels read rows as borrowed `StringSlice`s via `_get`; public
accessors return owned `String`s.
"""
from std.memory import ArcPointer, Pointer
from .column import (
    Column,
    _append_bits,
    _bit,
    _copy_bits,
    _count_set,
    _pack_bits,
)


struct StringColumn(Copyable, Sized):
    """A window onto shared UTF-8 bytes, Int64 offsets, and validity."""

    var _bytes: ArcPointer[List[UInt8]]
    var _offsets: ArcPointer[List[Int64]]
    var _bits: ArcPointer[List[UInt8]]
    var _offset: Int
    var _length: Int

    def __init__(out self, values: List[String]):
        var builder = StringBuilder(len(values))
        for value in values:
            builder.append(value)
        self = builder^.finish()

    def __init__(out self, values: List[String], valid: List[Bool]) raises:
        if len(values) != len(valid):
            raise Error("Column values and validity must have equal lengths")
        self = Self(values)
        self._bits = ArcPointer(_pack_bits(valid))

    def __init__(out self, column: Column[String]):
        """Convert a list-backed string column into the UTF-8 layout."""
        var builder = StringBuilder(len(column))
        for i in range(len(column)):
            if column._valid(i):
                builder.append(column._get(i))
            else:
                builder.append_null()
        self = builder^.finish()

    def __init__(
        out self,
        *,
        var bytes: List[UInt8],
        var offsets: List[Int64],
        var bits: List[UInt8],
        length: Int,
    ):
        """Adopt finished buffers; offsets must have length + 1 entries."""
        self._bytes = ArcPointer(bytes^)
        self._offsets = ArcPointer(offsets^)
        self._bits = ArcPointer(bits^)
        self._offset = 0
        self._length = length

    def __len__(self) -> Int:
        return self._length

    def _start(self, i: Int) -> Int:
        return Int(self._offsets[][self._offset + i])

    def _end(self, i: Int) -> Int:
        return Int(self._offsets[][self._offset + i + 1])

    def _byte_length(self, i: Int) -> Int:
        return self._end(i) - self._start(i)

    def _get(self, i: Int) -> StringSlice[ImmutAnyOrigin]:
        """Unchecked borrowed read of row i; valid while this column lives.

        Null rows read as the empty string.
        """
        var start = self._start(i)
        return StringSlice[ImmutAnyOrigin](
            unsafe_from_utf8=Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=self._base().unsafe_offset(start),
                length=self._end(i) - start,
            )
        )

    def _base(self) -> Pointer[UInt8, ImmutAnyOrigin]:
        return (
            self._bytes[]
            .unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )

    def _to_list(self) -> List[String]:
        """Owned copies of this window's rows (nulls read as "")."""
        var values = List[String](capacity=self._length)
        for i in range(self._length):
            values.append(String(self._get(i)))
        return values^

    def _shares_buffers_with(self, other: Self) -> Bool:
        return (
            self._bytes.unsafe_ptr() == other._bytes.unsafe_ptr()
            and self._offsets.unsafe_ptr() == other._offsets.unsafe_ptr()
            and self._bits.unsafe_ptr() == other._bits.unsafe_ptr()
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

    def value(self, index: Int) raises -> String:
        if self.is_null(index):
            raise Error("Cannot read a null value")
        return String(self._get(index))

    def null_count(self) -> Int:
        return self._length - _count_set(
            self._bits[], self._offset, self._length
        )

    def _value_bytes(self) -> Int:
        """Bytes of text in this window."""
        if self._length == 0:
            return 0
        return self._end(self._length - 1) - self._start(0)

    def take(self, indices: List[Int]) raises -> Self:
        for i in indices:
            self._check_index(i)
        var total = 0
        for i in indices:
            total += self._byte_length(i)
        var builder = StringBuilder(len(indices), total)
        for i in indices:
            builder._append_row(self, i)
        return builder^.finish()

    def take_or_null(self, indices: List[Int], fill: String) raises -> Self:
        """Gather rows, treating only -1 as a missing row (for outer joins)."""
        var total = 0
        for i in indices:
            if i != -1:
                self._check_index(i)
                total += self._byte_length(i)
        var builder = StringBuilder(len(indices), total)
        for i in indices:
            if i == -1:
                builder.append_null()
            else:
                builder._append_row(self, i)
        return builder^.finish()

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
            self._bytes.count() == 1
            and self._offsets.count() == 1
            and self._bits.count() == 1
            and self._offset == 0
            and len(self._offsets[]) == self._length + 1
            and Int(self._offsets[][self._length]) == len(self._bytes[])
        )

    def _compact(self) -> Self:
        """A private copy of this window with offsets rebased to 0."""
        # offsets[_offset] exists even for an empty window at the end.
        var first = self._start(0)
        var total = self._value_bytes()
        var bytes = List[UInt8](capacity=total)
        bytes.extend(
            Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=self._base().unsafe_offset(first), length=total
            )
        )
        var offsets = List[Int64](capacity=self._length + 1)
        for i in range(self._length + 1):
            offsets.append(self._offsets[][self._offset + i] - Int64(first))
        return Self(
            bytes=bytes^,
            offsets=offsets^,
            bits=_copy_bits(self._bits[], self._offset, self._length),
            length=self._length,
        )

    def _append_column(mut self, other: Self):
        """Append rows in bulk: bytes and validity copied, offsets rebased.

        Copies first unless this column owns its buffers, so values shared
        with other columns are never modified.
        """
        if not self._owned():
            self = self._compact()
        var count = other._length
        if count == 0:
            return
        _append_bits(
            self._bits[], self._length, other._bits[], other._offset, count
        )
        var first = other._start(0)
        var total = other._value_bytes()
        ref bytes = self._bytes[]
        if bytes.capacity() < len(bytes) + total:
            bytes.reserve(max(len(bytes) + total, 2 * bytes.capacity()))
        var shift = Int64(len(bytes) - first)
        bytes.extend(
            Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=other._base().unsafe_offset(first), length=total
            )
        )
        ref offsets = self._offsets[]
        if offsets.capacity() < len(offsets) + count:
            offsets.reserve(max(len(offsets) + count, 2 * offsets.capacity()))
        ref incoming = other._offsets[]
        for i in range(1, count + 1):
            offsets.append(incoming[other._offset + i] + shift)
        self._length += count

    @staticmethod
    def _nulls(length: Int, fill: String = "") -> Self:
        """All-null rows; payloads are empty regardless of fill."""
        return Self(
            bytes=List[UInt8](),
            offsets=List[Int64](length=length + 1, fill=0),
            bits=List[UInt8](length=(length + 7) // 8, fill=0),
            length=length,
        )

    def _broadcast(self, length: Int) raises -> Self:
        if self._length != 1 or length < 0:
            raise Error("Only a scalar result can broadcast")
        if not self._valid(0):
            return Self._nulls(length)
        var builder = StringBuilder(length, length * self._byte_length(0))
        for _ in range(length):
            builder._append_row(self, 0)
        return builder^.finish()


struct StringBuilder(Copyable):
    """Appends rows into fresh UTF-8, offset, and validity buffers.

    Kernels build string results here instead of collecting `String`s.
    """

    var _bytes: List[UInt8]
    var _offsets: List[Int64]
    var _bits: List[UInt8]
    var _length: Int

    def __init__(out self, rows: Int = 0, bytes: Int = 0):
        """Reserve for about `rows` rows and `bytes` bytes of text."""
        self._bytes = List[UInt8](capacity=bytes)
        self._offsets = List[Int64](capacity=rows + 1)
        self._offsets.append(0)
        self._bits = List[UInt8](capacity=(rows + 7) // 8)
        self._length = 0

    def __len__(self) -> Int:
        return self._length

    def _push_bit(mut self, valid: Bool):
        if self._length % 8 == 0:
            self._bits.append(0)
        if valid:
            self._bits[len(self._bits) - 1] |= UInt8(1) << UInt8(
                self._length % 8
            )
        self._length += 1

    def append(mut self, text: StringSlice):
        self._bytes.extend(text.as_bytes())
        self._offsets.append(Int64(len(self._bytes)))
        self._push_bit(True)

    def append(mut self, text: String):
        self.append(StringSlice(text))

    def append_null(mut self):
        self._offsets.append(Int64(len(self._bytes)))
        self._push_bit(False)

    def _pop(mut self):
        """Drop the last row (the CSV reader discards partial records)."""
        _ = self._offsets.pop()
        self._bytes.resize(Int(self._offsets[len(self._offsets) - 1]), 0)
        self._length -= 1
        if self._length % 8 == 0:
            _ = self._bits.pop()
        else:
            self._bits[len(self._bits) - 1] &= ~(
                UInt8(1) << UInt8(self._length % 8)
            )

    def _append_row(mut self, column: StringColumn, i: Int):
        """Copy row i of column, including its validity."""
        if column._valid(i):
            self.append(column._get(i))
        else:
            self.append_null()

    def finish(deinit self) -> StringColumn:
        return StringColumn(
            bytes=self._bytes^,
            offsets=self._offsets^,
            bits=self._bits^,
            length=self._length,
        )
