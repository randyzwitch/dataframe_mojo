"""Typed CSV buffers mapped to Polars 1.44.2 csv/read/builder.rs.

Numeric values are built in their output dtype and validity is packed during
append. Variant payloads are swapped out at finish, transferring ownership.
Strings currently adapt to this library's Arrow large_utf8 layout; Polars uses
MutableBinaryViewArray. That representation difference remains explicit.
"""
from std.memory import ArcPointer
from std.utils import Variant
from .column import Column, _append_validity_bit
from .bool_column import BoolColumn
from .string_column import StringBuilder
from .series import Series
from .dtype import DataType, NUMERIC_DTYPES
from .csv import CsvField, CsvOptions
from .parse import parse_integer
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
    StringBuilder,
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


struct CsvBuffer(Movable):
    var field: CsvField
    var storage: _Buffers
    var scratch: List[UInt8]

    def __init__(out self, field: CsvField, capacity: Int):
        self.field = field.copy()
        self.scratch = List[UInt8]()
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if field.dtype.physical() == DataType.of(D):
                self.storage = _Buffers(_NumericBuffer[D](capacity))
                return
        if field.dtype == DataType.BOOL:
            self.storage = _Buffers(_BoolBuffer(capacity))
        else:
            self.storage = _Buffers(StringBuilder(capacity))

    def add_null(mut self) raises:
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if self.storage.isa[_NumericBuffer[D]]():
                self.storage[_NumericBuffer[D]].append(0, False)
                return
        if self.storage.isa[_BoolBuffer]():
            self.storage[_BoolBuffer].append(False, False)
        else:
            self.storage[StringBuilder].append_null()

    def add(
        mut self,
        raw: Span[UInt8, ImmutAnyOrigin],
        needs_escaping: Bool,
        options: CsvOptions,
    ) raises:
        # parse_lines compares configured null tokens after removing wrappers,
        # including quoted null tokens (unlike the legacy reader's policy).
        var value = raw
        if needs_escaping and len(raw) >= 2:
            value = raw[1 : len(raw) - 1]
        for marker in options.null_values:
            if value == marker.as_bytes():
                self.add_null()
                return
        if self.storage.isa[StringBuilder]():
            self._add_string(raw, needs_escaping, options)
            return
        if self.storage.isa[_BoolBuffer]():
            if _ascii_equal(value, "false"):
                self.storage[_BoolBuffer].append(False, True)
            elif _ascii_equal(value, "true"):
                self.storage[_BoolBuffer].append(True, True)
            elif len(value) == 0 or options.ignore_errors:
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
                            parse_integer[D](text), True
                        )
                    return
        except e:
            if options.ignore_errors:
                self.add_null()
            else:
                raise e^

    def _add_string(
        mut self,
        raw: Span[UInt8, ImmutAnyOrigin],
        needs_escaping: Bool,
        options: CsvOptions,
    ) raises:
        if len(raw) == 0:
            self.add_null()
            return
        var bytes = raw
        if needs_escaping:
            var quote = options.quote_char.as_bytes()[0]
            if len(raw) < 2 or raw[len(raw) - 1] != quote:
                raise Error("CSV string field is not properly escaped")
            self.scratch.clear()
            self.scratch.reserve(len(raw))
            var previous_quote = False
            for byte in raw[1 : len(raw) - 1]:
                if byte == quote:
                    if previous_quote:
                        previous_quote = False
                        self.scratch.append(byte)
                    else:
                        previous_quote = True
                else:
                    previous_quote = False
                    self.scratch.append(byte)
            bytes = Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=self.scratch.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                length=len(self.scratch),
            )
        if options.encoding == "utf8-lossy" or options.ignore_errors:
            try:
                _ = StringSlice(from_utf8=bytes)
            except:
                if options.encoding == "utf8-lossy":
                    var lossy = String(from_utf8_lossy=bytes)
                    self.storage[StringBuilder].append(lossy)
                else:
                    self.add_null()
                return
        self.storage[StringBuilder].append(StringSlice(unsafe_from_utf8=bytes))

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
        var builder = StringBuilder()
        swap(builder, self.storage[StringBuilder])
        return Series(self.field.name, builder^.finish())
