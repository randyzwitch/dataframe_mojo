"""Owned typed columns with bit-packed validity, independent of payload values."""


struct Column[T: Copyable & Deinitable](Copyable, Sized):
    """A contiguous payload and an LSB-first validity bitmap.

    Underscored storage is internal. Public operations return owned copies;
    no borrowed mutable buffers or implicit negative indexing are exposed.
    """

    var _values: List[Self.T]
    var _validity: List[UInt8]

    def __init__(out self, var values: List[Self.T]):
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
