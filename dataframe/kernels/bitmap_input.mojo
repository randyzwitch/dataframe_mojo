"""Owned physical bitmap windows without materializing source payloads."""
from dataframe.series import Series, Storage, FixedElements
from dataframe.storage.buffer import _Buffer
from dataframe.storage.column import Column
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.string import StringColumn
from dataframe.storage.nested import ListColumn, StructColumn


@always_inline
def _bitmap_byte(
    bits: Span[UInt8, _], offset: Int, count: Int, scalar: Bool, empty: UInt8
) -> UInt8:
    """At most eight logical bits, with whole-operand scalar broadcasting."""
    assert count >= 0 and count <= 8
    if count == 0:
        return 0
    var mask = UInt8((UInt16(1) << UInt16(count)) - 1)
    if len(bits) == 0:
        return empty & mask
    var byte = offset // 8
    var shift = UInt8(offset % 8)
    var value = bits[byte] >> shift
    if scalar:
        return mask if value & 1 != 0 else UInt8(0)
    if shift != 0 and byte + 1 < len(bits):
        value |= bits[byte + 1] << (UInt8(8) - shift)
    return value & mask


@always_inline
def _write_bitmap_byte(
    mut output: List[UInt8], row: Int, value: UInt8, count: Int
):
    """Append nonoverlapping bits to a zeroed output at its global offset."""
    if count == 0:
        return
    var mask = UInt8((UInt16(1) << UInt16(count)) - 1)
    var byte = row // 8
    var shift = UInt8(row % 8)
    var bits = value & mask
    output[byte] |= bits << shift
    if Int(shift) + count > 8:
        output[byte + 1] |= bits >> (UInt8(8) - shift)


def _validity_window(storage: Storage) -> Tuple[_Buffer[UInt8], Int, Int]:
    comptime for i in range(len(FixedElements.Ts)):
        comptime E: Copyable & Deinitable = FixedElements.Ts[i]
        if storage.isa[Column[E]]():
            ref column = storage[Column[E]]
            return (column._bits.copy(), column._offset, len(column))
    if storage.isa[BoolColumn]():
        ref column = storage[BoolColumn]
        return (column._bits.copy(), column._offset, len(column))
    if storage.isa[StringColumn]():
        ref column = storage[StringColumn]
        return (column._bits.copy(), column._offset, len(column))
    if storage.isa[ListColumn]():
        ref column = storage[ListColumn]
        return (column._bits.copy(), column._offset, len(column))
    ref column = storage[StructColumn]
    return (column._bits.copy(), column._offset, len(column))


struct _ValidityInput(Movable):
    """Only a physical parent's bitmap; nested children stay untouched."""

    var owner: Series
    var bits: _Buffer[UInt8]
    var offset: Int
    var length: Int
    var chunk: Int
    var row: Int
    var scalar: Bool

    def __init__(out self, input: Series):
        self.owner = input.copy()
        self.chunk = 0
        self.row = 0
        self.scalar = len(input) == 1
        var window: Tuple[_Buffer[UInt8], Int, Int]
        if input.is_chunked():
            window = _validity_window(input._chunked.value()[].arrays[0])
        else:
            window = _validity_window(input._data)
        self.bits = window[0].copy()
        self.offset = window[1]
        self.length = window[2]
        self._skip_empty()

    def _skip_empty(mut self):
        if not self.owner.is_chunked():
            return
        ref source = self.owner._chunked.value()[]
        while self.row == self.length and self.chunk + 1 < len(source.arrays):
            self.chunk += 1
            var window = _validity_window(source.arrays[self.chunk])
            self.bits = window[0].copy()
            self.offset = window[1]
            self.length = window[2]
            self.row = 0

    def available(self, remaining: Int) -> Int:
        return remaining if self.scalar else min(
            remaining, self.length - self.row
        )

    def advance(mut self, count: Int):
        if not self.scalar:
            assert count >= 0 and count <= self.length - self.row
            self.row += count
            self._skip_empty()

    @always_inline
    def valid(self, count: Int, relative: Int = 0) -> UInt8:
        return _bitmap_byte(
            self.bits[].span(),
            self.offset + self.row + (0 if self.scalar else relative),
            count,
            self.scalar,
            255,
        )


struct _BoolInput(Movable):
    """Packed Boolean payload and validity at the same physical window."""

    var owner: Series
    var column: BoolColumn
    var chunk: Int
    var row: Int
    var scalar: Bool

    def __init__(out self, input: Series):
        self.owner = input.copy()
        self.chunk = 0
        self.row = 0
        self.scalar = len(input) == 1
        self.column = (
            input._chunked.value()[]
            .arrays[0][BoolColumn]
            .copy() if input.is_chunked() else input._data[BoolColumn]
            .copy()
        )
        self._skip_empty()

    def _skip_empty(mut self):
        if not self.owner.is_chunked():
            return
        ref source = self.owner._chunked.value()[]
        while self.row == len(self.column) and self.chunk + 1 < len(
            source.arrays
        ):
            self.chunk += 1
            self.column = source.arrays[self.chunk][BoolColumn].copy()
            self.row = 0

    def available(self, remaining: Int) -> Int:
        return remaining if self.scalar else min(
            remaining, len(self.column) - self.row
        )

    def advance(mut self, count: Int):
        if not self.scalar:
            assert count >= 0 and count <= len(self.column) - self.row
            self.row += count
            self._skip_empty()

    @always_inline
    def values(self, count: Int) -> UInt8:
        return _bitmap_byte(
            self.column._data[].span(),
            self.column._offset + self.row,
            count,
            self.scalar,
            0,
        )

    @always_inline
    def valid(self, count: Int) -> UInt8:
        return _bitmap_byte(
            self.column._bits[].span(),
            self.column._offset + self.row,
            count,
            self.scalar,
            255,
        )
