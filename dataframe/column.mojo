"""Owned typed columns with bit-packed validity, independent of payload values."""
from std.sys.info import is_little_endian


struct Column[T: Copyable & Deinitable](Copyable, Sized):
    """A contiguous payload and an LSB-first validity bitmap.

    Underscored storage is internal. Public operations return owned copies;
    no borrowed mutable buffers or implicit negative indexing are exposed.
    """

    var _values: List[Self.T]
    var _validity: List[UInt8]

    def __init__(out self, var values: List[Self.T]):
        # Bitmaps use in-byte shifts, but planned Arrow interchange shares
        # buffers as little-endian bytes; fail the build elsewhere.
        comptime assert is_little_endian(), "dataframe requires little-endian"
        self._values = values^
        self._validity = List[UInt8](
            length=(len(self._values) + 7) // 8, fill=255
        )

    def __init__(out self, var values: List[Self.T], valid: List[Bool]) raises:
        if len(values) != len(valid):
            raise Error("Column values and validity must have equal lengths")
        self._values = values^
        self._validity = List[UInt8](length=(len(valid) + 7) // 8, fill=0)
        for i in range(len(valid)):
            if valid[i]:
                self._validity[i // 8] |= UInt8(1) << UInt8(i % 8)

    def __len__(self) -> Int:
        return len(self._values)

    def _check_index(self, index: Int) raises:
        if index < 0 or index >= len(self):
            raise Error("Column index out of bounds")

    def is_null(self, index: Int) raises -> Bool:
        self._check_index(index)
        return (
            self._validity[index // 8] & (UInt8(1) << UInt8(index % 8))
        ) == 0

    def value(self, index: Int) raises -> Self.T:
        if self.is_null(index):
            raise Error("Cannot read a null value")
        return self._values[index].copy()

    def null_count(self) -> Int:
        var count = 0
        for i in range(len(self)):
            count += Int(
                (self._validity[i // 8] & (UInt8(1) << UInt8(i % 8))) == 0
            )
        return count

    def take(self, indices: List[Int]) raises -> Self:
        var values = List[Self.T](capacity=len(indices))
        var valid = List[Bool](capacity=len(indices))
        for i in indices:
            self._check_index(i)
            values.append(self._values[i].copy())
            valid.append(not self.is_null(i))
        return Self(values^, valid)

    def take_or_null(self, indices: List[Int], fill: Self.T) raises -> Self:
        """Gather rows, treating only -1 as a missing row (for outer joins)."""
        var values = List[Self.T](capacity=len(indices))
        var valid = List[Bool](capacity=len(indices))
        for i in indices:
            if i == -1:
                values.append(fill.copy())
                valid.append(False)
            else:
                self._check_index(i)
                values.append(self._values[i].copy())
                valid.append(not self.is_null(i))
        return Self(values^, valid)

    def _valid(self, i: Int) -> Bool:
        """Internal unchecked access after batch bounds validation."""
        return (self._validity[i // 8] & (UInt8(1) << UInt8(i % 8))) != 0

    def slice(self, offset: Int, length: Int) raises -> Self:
        if (
            offset < 0
            or length < 0
            or offset > len(self)
            or length > len(self) - offset
        ):
            raise Error("Invalid column slice")
        var values = List[Self.T](capacity=length)
        var valid = List[Bool](capacity=length)
        for i in range(offset, offset + length):
            values.append(self._values[i].copy())
            valid.append(self._valid(i))
        return Self(values^, valid)

    def _append_column(mut self, other: Self):
        """Append payloads and validity bytewise, shifting when unaligned."""
        var start = len(self)
        var shift = start % 8
        var count = len(other)
        var full_bytes = count // 8
        var tail_bits = count % 8
        if shift == 0:
            for b in range(full_bytes):
                self._validity.append(other._validity[b])
            if tail_bits > 0:
                var mask = (UInt8(1) << UInt8(tail_bits)) - 1
                self._validity.append(other._validity[full_bytes] & mask)
        elif count > 0:
            # Clear stale bits above the current length before merging.
            var last = len(self._validity) - 1
            self._validity[last] &= (UInt8(1) << UInt8(shift)) - 1
            var source_bytes = (count + 7) // 8
            for b in range(source_bytes):
                var byte = other._validity[b]
                if b == source_bytes - 1 and tail_bits > 0:
                    byte &= (UInt8(1) << UInt8(tail_bits)) - 1
                self._validity[len(self._validity) - 1] |= byte << UInt8(shift)
                self._validity.append(byte >> UInt8(8 - shift))
            var needed = (start + count + 7) // 8
            while len(self._validity) > needed:
                _ = self._validity.pop()
        # Grow geometrically: batch reassembly appends many small chunks, and
        # an exact reservation would copy the whole column on every append.
        if self._values.capacity() < start + count:
            self._values.reserve(max(start + count, 2 * self._values.capacity()))
        for i in range(count):
            self._values.append(other._values[i].copy())

    @staticmethod
    def _nulls(length: Int, fill: Self.T) -> Self:
        var result = Self(List[Self.T](length=length, fill=fill.copy()))
        for i in range(len(result._validity)):
            result._validity[i] = 0
        return result^

    def _broadcast(self, length: Int) raises -> Self:
        if len(self) != 1 or length < 0:
            raise Error("Only a scalar result can broadcast")
        var values = List[Self.T](length=length, fill=self._values[0].copy())
        var valid = List[Bool](length=length, fill=self._valid(0))
        return Self(values^, valid)
