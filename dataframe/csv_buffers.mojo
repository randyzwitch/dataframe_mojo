"""Typed CSV buffers.

Numeric values are built in their output dtype and validity is packed during
append. Variant payloads are swapped out at finish, transferring ownership.
Strings use 16-byte views with inline short values and retained blocks.
"""
from std.memory import ArcPointer
from std.utils import Variant
from .column import Column, _append_validity_bit
from .bool_column import BoolColumn
from .string_column import StringColumn
from .string_view import StringViewBuilder
from .series import Series
from .dtype import DataType, NUMERIC_DTYPES
from .csv_types import CsvField
from .csv_integer import parse_csv_integer
from .csv_numeric import parse_csv_float32, parse_csv_float64
from .temporal import parse as parse_temporal


def _push_bit(mut bits: List[UInt8], index: Int, value: Bool):
    if index % 8 == 0:
        bits.append(0)
    if value:
        bits[len(bits) - 1] |= UInt8(1) << UInt8(index % 8)


struct _NumericBuffer[D: DType](Copyable):
    var values: List[Scalar[Self.D]]
    var bits: List[UInt8]

    def __init__(out self, capacity: Int):
        self.values = List[Scalar[Self.D]](capacity=capacity)
        self.bits = List[UInt8]()

    def append(mut self, value: Scalar[Self.D], valid: Bool):
        _append_validity_bit(
            self.bits, len(self.values), valid, self.values.capacity()
        )
        self.values.append(value)

    def finish(mut self, name: String) -> Series:
        var values = List[Scalar[Self.D]]()
        var bits = List[UInt8]()
        swap(values, self.values)
        swap(bits, self.bits)
        return Series(name, Column[Scalar[Self.D]](values=values^, bits=bits^))


struct _BoolBuffer(Copyable):
    var values: List[UInt8]
    var bits: List[UInt8]
    var length: Int

    def __init__(out self, capacity: Int):
        self.values = List[UInt8](capacity=(capacity + 7) // 8)
        self.bits = List[UInt8]()
        self.length = 0

    def append(mut self, value: Bool, valid: Bool):
        _push_bit(self.values, self.length, value)
        _append_validity_bit(
            self.bits, self.length, valid, self.values.capacity() * 8
        )
        self.length += 1

    def finish(mut self, name: String) -> Series:
        var values = List[UInt8]()
        var bits = List[UInt8]()
        swap(values, self.values)
        swap(bits, self.bits)
        return Series(
            name, BoolColumn(values=values^, bits=bits^, length=self.length)
        )


struct _Utf8Buffer(Movable):
    """builder.rs Utf8Field: encoding, quote byte, and scratch owned by builder."""

    var builder: StringViewBuilder
    var scratch: List[UInt8]
    var quote_char: UInt8
    var lossy: Bool

    def __init__(out self, capacity: Int, quote_char: UInt8, lossy: Bool):
        self.builder = StringViewBuilder(capacity)
        self.scratch = List[UInt8]()
        self.quote_char = quote_char
        self.lossy = lossy

    def add(
        mut self,
        raw: Span[UInt8, ImmutAnyOrigin],
        needs_escaping: Bool,
        ignore_errors: Bool,
    ) raises:
        if len(raw) == 0:
            self.builder.append_null()
            return
        var bytes = raw
        if needs_escaping:
            var quote = self.quote_char
            if len(raw) < 2 or raw[len(raw) - 1] != quote:
                raise Error("CSV string field is not properly escaped")
            self.scratch.clear()
            self.scratch.reserve(len(raw))
            # escape_field writes into reserved spare capacity; avoid a List
            # append/capacity branch for every output byte. UInt8 is trivial,
            # so the borrowed span can track length without publishing elements.
            var scratch_ptr = self.scratch.unsafe_ptr()
            var written = 0
            var previous_quote = False
            for byte in raw[1 : len(raw) - 1]:
                if byte == quote:
                    if previous_quote:
                        previous_quote = False
                        scratch_ptr.unsafe_offset(written)[] = byte
                        written += 1
                    else:
                        previous_quote = True
                else:
                    previous_quote = False
                    scratch_ptr.unsafe_offset(written)[] = byte
                    written += 1
            bytes = Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=self.scratch.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                length=written,
            )
        if self.lossy or ignore_errors:
            try:
                _ = StringSlice(from_utf8=bytes)
            except:
                if self.lossy:
                    var lossy = String(from_utf8_lossy=bytes)
                    self.builder.append(StringSlice(lossy))
                else:
                    self.builder.append_null()
                return
        self.builder.append(StringSlice(unsafe_from_utf8=bytes))

    def finish(deinit self) -> StringColumn:
        return StringColumn(self.builder^.finish())


comptime _Buffers = Variant[
    _NumericBuffer[DType.int64],
    _NumericBuffer[DType.float64],
    _NumericBuffer[DType.int8],
    _NumericBuffer[DType.int16],
    _NumericBuffer[DType.int32],
    _NumericBuffer[DType.uint8],
    _NumericBuffer[DType.uint16],
    _NumericBuffer[DType.uint32],
    _NumericBuffer[DType.uint64],
    _NumericBuffer[DType.float32],
    _BoolBuffer,
    _Utf8Buffer,
]


def _ascii_equal(
    bytes: Span[UInt8, ImmutAnyOrigin], expected: StringSlice
) -> Bool:
    var target = expected.as_bytes()
    if len(bytes) != len(target):
        return False
    for i in range(len(bytes)):
        if (bytes[i] | UInt8(32)) != target[i]:
            return False
    return True


@fieldwise_init
struct CsvCell(Copyable):
    var start: Int
    var length: Int
    var needs_escaping: Bool
    var record: Int


struct CsvBuffer(Movable):
    var field: CsvField
    var storage: _Buffers

    def __init__(
        out self,
        field: CsvField,
        capacity: Int,
        quote_char: UInt8 = 34,
        lossy: Bool = False,
    ):
        self.field = field.copy()
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if field.dtype.physical() == DataType.of(D):
                self.storage = _Buffers(_NumericBuffer[D](capacity))
                return
        if field.dtype == DataType.BOOL:
            self.storage = _Buffers(_BoolBuffer(capacity))
        else:
            self.storage = _Buffers(_Utf8Buffer(capacity, quote_char, lossy))

    def add_null(mut self) raises:
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if self.storage.isa[_NumericBuffer[D]]():
                self.storage[_NumericBuffer[D]].append(0, False)
                return
        if self.storage.isa[_BoolBuffer]():
            self.storage[_BoolBuffer].append(False, False)
        else:
            self.storage[_Utf8Buffer].builder.append_null()

    def add(
        mut self,
        raw: Span[UInt8, ImmutAnyOrigin],
        needs_escaping: Bool,
        ignore_errors: Bool,
    ) raises:
        if self.storage.isa[_Utf8Buffer]():
            self.storage[_Utf8Buffer].add(raw, needs_escaping, ignore_errors)
            return
        var value = raw
        if needs_escaping and len(raw) >= 2:
            value = raw[1 : len(raw) - 1]
        if self.storage.isa[_BoolBuffer]():
            if _ascii_equal(value, "false"):
                self.storage[_BoolBuffer].append(False, True)
            elif _ascii_equal(value, "true"):
                self.storage[_BoolBuffer].append(True, True)
            elif len(value) == 0 or ignore_errors:
                self.add_null()
            else:
                raise Error("invalid Boolean CSV value")
            return
        # PrimitiveChunkedBuilder strips leading SP/TAB; temporal builders
        # consume their quoted contents without whitespace trimming.
        var start = 0
        while (
            not self.field.dtype.is_temporal()
            and start < len(value)
            and (value[start] == 32 or value[start] == 9)
        ):
            start += 1
        value = value[start:]
        if len(value) == 0:
            self.add_null()
            return
        try:
            var text = StringSlice(unsafe_from_utf8=value)
            if self.field.dtype.is_temporal():
                var parsed = parse_temporal(
                    String(text), self.field.dtype, self.field.format
                )
                self.storage[_NumericBuffer[DType.int64]].append(parsed, True)
                return
            comptime for i in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[i]
                if self.storage.isa[_NumericBuffer[D]]():
                    comptime if D == DType.float32:
                        self.storage[_NumericBuffer[D]].append(
                            parse_csv_float32(text).cast[D](), True
                        )
                    elif D == DType.float64:
                        self.storage[_NumericBuffer[D]].append(
                            parse_csv_float64(text).cast[D](), True
                        )
                    else:
                        self.storage[_NumericBuffer[D]].append(
                            parse_csv_integer[D](text), True
                        )
                    return
        except e:
            if ignore_errors:
                self.add_null()
            else:
                raise e^

    def add_many(
        mut self,
        bytes: Span[UInt8, ImmutAnyOrigin],
        cells: List[CsvCell],
        count: Int,
        ignore_errors: Bool,
    ) raises -> Tuple[Int, String]:
        var first_record = -1
        var first_message = String()
        # Select the numeric representation once for the whole chunk column.
        # The row scanner records byte ranges; this loop parses one dtype.
        if not self.field.dtype.is_temporal():
            comptime for i in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[i]
                if self.storage.isa[_NumericBuffer[D]]():
                    for index in range(count):
                        var cell = (
                            cells.unsafe_ptr().unsafe_offset(index)[].copy()
                        )
                        if cell.start < 0:
                            self.storage[_NumericBuffer[D]].append(0, False)
                            continue
                        var value = Span[UInt8, ImmutAnyOrigin](
                            unsafe_ptr=bytes.unsafe_ptr().unsafe_offset(
                                cell.start
                            ),
                            length=cell.length,
                        )
                        if cell.needs_escaping and len(value) >= 2:
                            value = value[1 : len(value) - 1]
                        var start = 0
                        while start < len(value) and (
                            value[start] == 32 or value[start] == 9
                        ):
                            start += 1
                        value = value[start:]
                        if len(value) == 0:
                            self.storage[_NumericBuffer[D]].append(0, False)
                            continue
                        try:
                            var text = StringSlice(unsafe_from_utf8=value)
                            comptime if D == DType.float32:
                                self.storage[_NumericBuffer[D]].append(
                                    parse_csv_float32(text).cast[D](), True
                                )
                            elif D == DType.float64:
                                self.storage[_NumericBuffer[D]].append(
                                    parse_csv_float64(text).cast[D](), True
                                )
                            else:
                                self.storage[_NumericBuffer[D]].append(
                                    parse_csv_integer[D](text), True
                                )
                        except error:
                            if ignore_errors:
                                self.storage[_NumericBuffer[D]].append(0, False)
                            else:
                                self.storage[_NumericBuffer[D]].append(0, False)
                                if first_record < 0:
                                    first_record = cell.record
                                    first_message = String(error)
                    return (first_record, first_message)
        for index in range(count):
            var cell = cells.unsafe_ptr().unsafe_offset(index)[].copy()
            if cell.start < 0:
                self.add_null()
                continue
            var value = Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=bytes.unsafe_ptr().unsafe_offset(cell.start),
                length=cell.length,
            )
            try:
                self.add(value, cell.needs_escaping, ignore_errors)
            except error:
                self.add_null()
                if first_record < 0:
                    first_record = cell.record
                    first_message = String(error)
        return (first_record, first_message)

    def finish(mut self) raises -> Series:
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if self.storage.isa[_NumericBuffer[D]]():
                return (
                    self.storage[_NumericBuffer[D]]
                    .finish(self.field.name)
                    .with_dtype(self.field.dtype)
                )
        if self.storage.isa[_BoolBuffer]():
            return self.storage[_BoolBuffer].finish(self.field.name)
        var builder = _Utf8Buffer(0, 34, False)
        swap(builder, self.storage[_Utf8Buffer])
        return Series(self.field.name, builder^.finish())
