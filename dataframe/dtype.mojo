"""Structured logical data types.

`DataType` identifies a column's logical type. Values are compile-time
constants (`DataType.INT64`) and compare with `==`. `String(dtype)` gives the
canonical name used in schemas, casts, and error messages, and
`DataType.parse(name)` is its inverse. Parameterized types (such as datetime
units) keep their parameter in `_unit`, and a datetime's time zone is kept
as text behind the same shared pointer nested types use; nested types (`DataType.list(inner)`
and `DataType.struct(names, dtypes)`) keep their child types as an encoded
spec behind a shared pointer and decode them on demand. A DataType cannot
hold a `List[DataType]`, even indirectly: Mojo rejects that cycle in an
imported module.
"""
from std.collections import Optional
from std.memory import ArcPointer
from std.sys import size_of

from .timezone import canonical_zone

# Every numeric type shares one code; its DType tells them apart.
comptime _NUMERIC = 0
comptime _BOOL = 2
comptime _STRING = 3
comptime _DATE = 4
comptime _DATETIME = 5
comptime _DURATION = 6
comptime _TIME = 7
comptime _DECIMAL = 8
# Arbitrary bytes in the string layout (Arrow large_binary); no UTF-8.
comptime _BINARY = 9
# Dictionary-encoded strings (Arrow dictionary<uint32, large_utf8>): UInt32
# codes into a dictionary of distinct values carried by the DataType.
comptime _CATEGORICAL = 10
comptime _LIST = 16
# Marks a type without numeric storage (bool, string, nested). Bool columns
# are bit-packed, so DType.bool never names a numeric storage type.
comptime _NO_STORAGE = DType.bool
comptime _STRUCT = 17
# Binder-only types of untyped numeric literals (`col("x") > 0`) before they
# adopt the dtype of the operand they meet. Never stored in a column.
comptime _UNTYPED_INT = 100
comptime _UNTYPED_FLOAT = 101

# Numeric storage types. Dispatch loops over this list at compile time, so
# each iteration sees a concrete Scalar[D] with arithmetic, ordering, and
# hashing; adding a numeric type means extending this list and DataType.
comptime NUMERIC_DTYPES: Array[DType, 10] = [
    DType.int64,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float32,
]

# Time units for datetime and duration, kept in DataType._unit.
comptime NS = 1
comptime US = 2
comptime MS = 3


struct CategoricalDictionary(Movable, Sized):
    """The distinct values of a categorical column, in code order, as UTF-8
    bytes and Int64 offsets (the large_utf8 layout). Kept to plain lists so
    a DataType can hold one without importing StringColumn."""

    var bytes: List[UInt8]
    var offsets: List[Int64]

    def __init__(out self):
        self.bytes = List[UInt8]()
        self.offsets = [0]

    def __init__(out self, var bytes: List[UInt8], var offsets: List[Int64]):
        self.bytes = bytes^
        self.offsets = offsets^

    def __len__(self) -> Int:
        return len(self.offsets) - 1

    def get(self, code: Int) -> StringSlice[ImmutAnyOrigin]:
        """The value with this code (unchecked)."""
        var start = Int(self.offsets[code])
        return StringSlice[ImmutAnyOrigin](
            unsafe_from_utf8=Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=self.bytes.unsafe_ptr()
                .unsafe_offset(start)
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                length=Int(self.offsets[code + 1]) - start,
            )
        )

    def append(mut self, value: StringSlice):
        self.bytes.extend(value.as_bytes())
        self.offsets.append(Int64(len(self.bytes)))

    def copy_values(self) -> Self:
        return Self(self.bytes.copy(), self.offsets.copy())

    def same_values(self, other: Self) -> Bool:
        if len(self) != len(other):
            return False
        for i in range(len(self.offsets)):
            if self.offsets[i] != other.offsets[i]:
                return False
        for i in range(len(self.bytes)):
            if self.bytes[i] != other.bytes[i]:
                return False
        return True


struct _NestedSpec(Copyable, Movable):
    """The encoded child types of a nested DataType (see _encode)."""

    var text: String

    def __init__(out self, var text: String):
        self.text = text^


def _decimal_storage(unit: Int) -> DType:
    """The integer a decimal's `unit` (width code, precision, scale) is
    stored in."""
    var code = unit // 10000
    return DType.int128 if code == 0 else (
        DType.int64 if code == 1 else DType.int32
    )


struct DataType(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """A logical column type.

    Numeric: INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64,
    FLOAT32, FLOAT64. Also BOOL, STRING, DATE (days since 1970-01-01), TIME
    (nanoseconds since midnight), and datetime(unit, time_zone) /
    duration(unit) with unit "ns", "us", or "ms". A datetime with a time zone
    holds UTC ticks, like an Arrow timestamp whose zone is set. Temporal types are stored as Int64. Nested:
    list(inner) holds a variable number of `inner` values per row, and
    struct(names, dtypes) holds one value of each named field per row.
    """

    var _code: Int
    var _unit: Int
    # The fixed-width storage type: the numeric DType itself, int64 for
    # temporal types, _NO_STORAGE for bool, string and nested types.
    var _storage: DType
    var _nested: Optional[ArcPointer[_NestedSpec]]
    # A categorical's dictionary; None for other types, and for the bare
    # CATEGORICAL used as a cast target or schema entry.
    var _dictionary: Optional[ArcPointer[CategoricalDictionary]]

    def __init__(out self, code: Int, unit: Int):
        self._code = code
        self._unit = unit
        self._storage = (
            _decimal_storage(unit) if code
            == _DECIMAL else DType.int64 if code
            >= _DATE
            and code
            <= _TIME else DType.uint32 if code
            == _CATEGORICAL else _NO_STORAGE
        )
        self._nested = None
        self._dictionary = None

    def __init__(out self, storage: DType):
        """A numeric type stored as Scalar[storage]."""
        self._code = _NUMERIC
        self._unit = 0
        self._storage = storage
        self._nested = None
        self._dictionary = None

    def __init__(out self, code: Int, var spec: String):
        self._code = code
        self._unit = 0
        self._storage = _NO_STORAGE
        self._nested = ArcPointer(_NestedSpec(spec^))
        self._dictionary = None

    def __init__(out self, code: Int, unit: Int, var zone: String):
        """A datetime in a time zone; the zone rides in `_nested`."""
        self = DataType(code, unit)
        if zone.byte_length() > 0:
            self._nested = ArcPointer(_NestedSpec(zone^))

    @staticmethod
    def list(inner: DataType) -> DataType:
        """A list column whose elements have dtype `inner`."""
        return DataType(_LIST, "L" + inner._encode())

    @staticmethod
    def struct(names: List[String], dtypes: List[DataType]) raises -> DataType:
        """A struct column with one field per name, in order."""
        if len(names) != len(dtypes):
            raise Error("struct needs one dtype per field name")
        if len(names) == 0:
            raise Error("struct needs at least one field")
        for i in range(len(names)):
            for j in range(i):
                if names[i] == names[j]:
                    raise Error(
                        "struct field names must be unique: " + names[i]
                    )
        var spec = String("S") + String(len(names)) + ":"
        for i in range(len(names)):
            spec += String(names[i].byte_length()) + ":" + names[i]
            spec += dtypes[i]._encode()
        return DataType(_STRUCT, spec^)

    def _encode(self) -> String:
        """A self-delimiting spec: P<len>:<name> for a flat type,
        L<inner> for a list, S<n>:(<len>:<name><dtype>)* for a struct."""
        if self.is_nested():
            return self._nested.value()[].text
        var name = self.name()
        return "P" + String(name.byte_length()) + ":" + name

    def is_list(self) -> Bool:
        return self._code == _LIST

    def is_struct(self) -> Bool:
        return self._code == _STRUCT

    def is_nested(self) -> Bool:
        return self._code == _LIST or self._code == _STRUCT

    def inner(self) raises -> DataType:
        """The element type of a list."""
        if not self.is_list():
            raise Error("inner() needs a list dtype, found " + self.name())
        var cursor = 1
        return _decode(self._nested.value()[].text, cursor)

    def field_count(self) -> Int:
        """A struct's number of fields (0 for other types)."""
        if not self.is_struct():
            return 0
        try:
            return len(self.field_names())
        except:
            return 0

    def field_names(self) raises -> List[String]:
        """A struct's field names, in order (empty for other types)."""
        var names = List[String]()
        if not self.is_struct():
            return names^
        ref spec = self._nested.value()[].text
        var cursor = 1
        var count = _read_count(spec, cursor)
        for _ in range(count):
            names.append(_read_text(spec, cursor))
            _ = _decode(spec, cursor)
        return names^

    def field_dtypes(self) raises -> List[DataType]:
        """A struct's field types, in order (empty for other types)."""
        var dtypes = List[DataType]()
        if not self.is_struct():
            return dtypes^
        ref spec = self._nested.value()[].text
        var cursor = 1
        var count = _read_count(spec, cursor)
        for _ in range(count):
            _ = _read_text(spec, cursor)
            dtypes.append(_decode(spec, cursor))
        return dtypes^

    def field_index(self, name: String) raises -> Int:
        if not self.is_struct():
            raise Error(
                "field_index needs a struct dtype, found " + self.name()
            )
        var names = self.field_names()
        for i in range(len(names)):
            if names[i] == name:
                return i
        raise Error("struct has no field named " + name)

    def field_dtype(self, index: Int) raises -> DataType:
        var dtypes = self.field_dtypes()
        if index < 0 or index >= len(dtypes):
            raise Error("struct field index out of range")
        return dtypes[index]

    comptime INT64 = DataType(DType.int64)
    comptime FLOAT64 = DataType(DType.float64)
    comptime BOOL = DataType(_BOOL, 0)
    comptime STRING = DataType(_STRING, 0)
    comptime BINARY = DataType(_BINARY, 0)
    # Categorical with no dictionary yet: a cast target or schema entry.
    comptime CATEGORICAL = DataType(_CATEGORICAL, 0)

    @staticmethod
    def categorical(var dictionary: CategoricalDictionary) -> DataType:
        """A categorical whose codes index `dictionary`."""
        var dtype = DataType(_CATEGORICAL, 0)
        dtype._dictionary = ArcPointer(dictionary^)
        return dtype^

    def is_categorical(self) -> Bool:
        return self._code == _CATEGORICAL

    def has_dictionary(self) -> Bool:
        return Bool(self._dictionary)

    def dictionary(self) -> ArcPointer[CategoricalDictionary]:
        """The dictionary of a categorical that has one (check first)."""
        return self._dictionary.value()

    comptime DATE = DataType(_DATE, 0)
    comptime TIME = DataType(_TIME, 0)
    comptime INT8 = DataType(DType.int8)
    comptime INT16 = DataType(DType.int16)
    comptime INT32 = DataType(DType.int32)
    comptime UINT8 = DataType(DType.uint8)
    comptime UINT16 = DataType(DType.uint16)
    comptime UINT32 = DataType(DType.uint32)
    comptime UINT64 = DataType(DType.uint64)
    comptime FLOAT32 = DataType(DType.float32)
    comptime UNTYPED_INT = DataType(_UNTYPED_INT, 0)
    comptime UNTYPED_FLOAT = DataType(_UNTYPED_FLOAT, 0)

    def is_untyped(self) -> Bool:
        """Whether this is an untyped literal awaiting a dtype (binder only)."""
        return self._code == _UNTYPED_INT or self._code == _UNTYPED_FLOAT

    def default(self) -> DataType:
        """The dtype an untyped literal takes on its own: Int64 or Float64."""
        if self._code == _UNTYPED_INT:
            return DataType.INT64
        if self._code == _UNTYPED_FLOAT:
            return DataType.FLOAT64
        return self

    @staticmethod
    def of(dtype: DType) -> DataType:
        """The DataType stored as Scalar[dtype] (numeric types only)."""
        return DataType(dtype)

    def storage(self) -> Optional[DType]:
        """The numeric storage DType (int64 for temporal types); None for
        bool, string and nested types."""
        if self._storage == _NO_STORAGE:
            return None
        return self._storage

    @staticmethod
    def datetime(
        unit: String = "us", time_zone: String = ""
    ) raises -> DataType:
        """An instant counted in unit since the epoch. Without a time zone
        it is naive (a wall-clock reading); with one it is UTC, shown and
        broken into fields in that zone. The zone is an IANA name, "UTC",
        or a fixed offset "+HH:MM", as in Arrow (see timezone.mojo)."""
        var code = _unit_code(unit)
        if time_zone.byte_length() == 0:
            return DataType(_DATETIME, code)
        return DataType(_DATETIME, code, canonical_zone(time_zone))

    def time_zone(self) -> String:
        """A datetime's time zone; empty when naive or not a datetime."""
        if self._code == _DATETIME and self._nested:
            return self._nested.value()[].text
        return ""

    def with_time_zone(self, time_zone: String) raises -> DataType:
        """This datetime's unit in `time_zone` (empty for naive)."""
        return DataType.datetime(self.unit(), time_zone)

    @staticmethod
    def duration(unit: String = "us") raises -> DataType:
        """A signed length of time counted in unit."""
        return DataType(_DURATION, _unit_code(unit))

    @staticmethod
    def decimal(
        precision: Int, scale: Int, width: Int = 128
    ) raises -> DataType:
        """An Arrow decimal: a scaled integer of `width` bits (Arrow's
        decimal32, decimal64 or decimal128), kept at the width its source
        declares. Results that could exceed it widen to 128 bits."""
        if width != 128 and width != 64 and width != 32:
            raise Error("decimal width must be 32, 64 or 128")
        var most = 38 if width == 128 else (18 if width == 64 else 9)
        if precision < 1 or precision > most:
            raise Error(
                "decimal"
                + ("" if width == 128 else String(width))
                + " precision must be between 1 and "
                + String(most)
            )
        if scale < 0 or scale > precision:
            raise Error("decimal scale must be between 0 and precision")
        var code = 0 if width == 128 else (1 if width == 64 else 2)
        return DataType(_DECIMAL, code * 10000 + precision * 100 + scale)

    def is_decimal(self) -> Bool:
        return self._code == _DECIMAL

    def precision(self) -> Int:
        return (self._unit % 10000) // 100 if self.is_decimal() else 0

    def scale(self) -> Int:
        return self._unit % 100 if self.is_decimal() else 0

    def decimal_width(self) -> Int:
        """Bits per value of a decimal (32, 64 or 128); 0 otherwise."""
        if not self.is_decimal():
            return 0
        var code = self._unit // 10000
        return 128 if code == 0 else (64 if code == 1 else 32)

    @staticmethod
    def parse(name: String) raises -> DataType:
        """The type with this canonical name; raises for unknown names."""
        if name == "int64":
            return DataType.INT64
        if name == "float64":
            return DataType.FLOAT64
        if name == "bool":
            return DataType.BOOL
        if name == "string":
            return DataType.STRING
        if name == "binary":
            return DataType.BINARY
        if name == "categorical" or name == "cat":
            return DataType.CATEGORICAL
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if name == String(D):
                return DataType.of(D)
        if name == "date":
            return DataType.DATE
        if name == "time":
            return DataType.TIME
        if name == "datetime":
            return DataType.datetime()
        if name == "duration":
            return DataType.duration()
        var width = 128
        var open = 8
        if name.startswith("decimal64["):
            width = 64
            open = 10
        elif name.startswith("decimal32["):
            width = 32
            open = 10
        if (
            name.startswith("decimal[")
            or name.startswith("decimal64[")
            or name.startswith("decimal32[")
        ) and name.endswith("]"):
            var body = String(name[byte = open : name.byte_length() - 1])
            var comma = body.find(",")
            if comma < 0:
                raise Error("Unknown dtype: " + name)
            try:
                return DataType.decimal(
                    Int(String(body[byte=0:comma])),
                    Int(String(body[byte = comma + 1 : body.byte_length()])),
                    width,
                )
            except:
                raise Error("Unknown dtype: " + name)
        for unit in ["ns", "us", "ms"]:
            if name == "datetime[" + unit + "]":
                return DataType.datetime(unit)
            var prefix = "datetime[" + unit + ","
            if name.startswith(prefix) and name.endswith("]"):
                var zone = String(
                    name[byte = prefix.byte_length() : name.byte_length() - 1]
                ).strip()
                return DataType.datetime(unit, String(zone))
            if name == "duration[" + unit + "]":
                return DataType.duration(unit)
        if name.startswith("list[") and name.endswith("]"):
            return DataType.list(
                DataType.parse(String(name[byte = 5 : name.byte_length() - 1]))
            )
        raise Error("Unknown dtype: " + name)

    @staticmethod
    def is_known(name: String) -> Bool:
        try:
            _ = DataType.parse(name)
            return True
        except:
            return False

    def __eq__(self, other: Self) -> Bool:
        if (
            self._code != other._code
            or self._unit != other._unit
            or self._storage != other._storage
        ):
            return False
        if self._code == _CATEGORICAL:
            # A categorical without a dictionary (a cast target or schema
            # entry) matches any; two dictionaries must hold the same
            # values in the same order, so that codes mean the same.
            if not self._dictionary or not other._dictionary:
                return True
            if (
                self._dictionary.value().ptr()
                == other._dictionary.value().ptr()
            ):
                return True
            return self._dictionary.value()[].same_values(
                other._dictionary.value()[]
            )
        if not self._nested and not other._nested:
            return True
        if not self._nested or not other._nested:
            return False
        return self._nested.value()[].text == other._nested.value()[].text

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def name(self) -> String:
        """The canonical name, as accepted by parse."""
        if self._code == _NUMERIC:
            return String(self._storage)
        if self._code == _BOOL:
            return "bool"
        if self._code == _BINARY:
            return "binary"
        if self._code == _CATEGORICAL:
            return "categorical"
        if self._code == _DATE:
            return "date"
        if self._code == _TIME:
            return "time"
        if self._code == _DATETIME:
            var zone = self.time_zone()
            if zone.byte_length() > 0:
                return "datetime[" + self.unit() + ", " + zone + "]"
            return "datetime[" + self.unit() + "]"
        if self._code == _DURATION:
            return "duration[" + self.unit() + "]"
        if self._code == _DECIMAL:
            var width = self.decimal_width()
            return (
                (
                    "decimal[" if width
                    == 128 else "decimal" + String(width) + "["
                )
                + String(self.precision())
                + ","
                + String(self.scale())
                + "]"
            )
        if self._code == _UNTYPED_INT:
            return "integer literal"
        if self._code == _UNTYPED_FLOAT:
            return "float literal"
        if self._code == _LIST:
            try:
                return "list[" + self.inner().name() + "]"
            except:
                return "list[?]"
        if self._code == _STRUCT:
            var out = String("struct[")
            try:
                var names = self.field_names()
                var dtypes = self.field_dtypes()
                for i in range(len(names)):
                    if i > 0:
                        out += ", "
                    out += names[i] + ": " + dtypes[i].name()
            except:
                out += "?"
            return out + "]"
        return "string"

    def short_name(self) -> String:
        """The compact name used in table headers (i64, f64, bool, str)."""
        if self._code == _NUMERIC:
            var letter = "f" if self.is_float() else (
                "u" if self.is_unsigned() else "i"
            )
            return letter + String(self.bit_width())
        if self._code == _BOOL:
            return "bool"
        if self._code == _STRING:
            return "str"
        if self._code == _CATEGORICAL:
            return "cat"
        if self._code == _LIST:
            try:
                return "list[" + self.inner().short_name() + "]"
            except:
                return "list[?]"
        if self._code == _STRUCT:
            return "struct[" + String(self.field_count()) + "]"
        return self.name()

    def unit(self) -> String:
        """The time unit of a datetime or duration ("" otherwise)."""
        if self._unit == NS:
            return "ns"
        if self._unit == US:
            return "us"
        if self._unit == MS:
            return "ms"
        return ""

    def per_second(self) -> Int64:
        """Ticks per second: 1e9 for ns, 1e6 for us, 1e3 for ms, and 1e9
        for TIME; 0 for other types."""
        if self._code == _TIME or self._unit == NS:
            return 1000000000
        if self._unit == US:
            return 1000000
        if self._unit == MS:
            return 1000
        return 0

    def is_temporal(self) -> Bool:
        return self._code >= _DATE and self._code <= _TIME

    def is_date(self) -> Bool:
        return self._code == _DATE

    def is_datetime(self) -> Bool:
        return self._code == _DATETIME

    def is_duration(self) -> Bool:
        return self._code == _DURATION

    def is_time(self) -> Bool:
        return self._code == _TIME

    def is_binary(self) -> Bool:
        return self._code == _BINARY

    def physical(self) -> DataType:
        """The storage type: INT64 for temporal types, STRING for binary
        (the same offsets-and-bytes layout), otherwise self."""
        if self.is_temporal():
            return DataType.INT64
        if self._code == _BINARY:
            return DataType.STRING
        if self._code == _CATEGORICAL:
            return DataType.of(DType.uint32)
        if self.is_decimal():
            return DataType.of(_decimal_storage(self._unit))
        return self

    def is_numeric(self) -> Bool:
        return self.is_integer() or self.is_float() or self.is_decimal()

    def is_integer(self) -> Bool:
        return self._code == _NUMERIC and self._storage.is_integral()

    def is_float(self) -> Bool:
        return self._code == _NUMERIC and self._storage.is_floating_point()

    def is_unsigned(self) -> Bool:
        return self._code == _NUMERIC and self._storage.is_unsigned()

    def is_signed(self) -> Bool:
        """Whether values can be negative (numeric types only)."""
        return self.is_numeric() and not self.is_unsigned()

    def bit_width(self) -> Int:
        """Bits per value for fixed-width types; 0 for variable-width."""
        if self._code == _BOOL:
            return 1
        if self.is_decimal():
            return self.decimal_width()
        # A runtime DType cannot report its width in Mojo 1.2; match it
        # against the comptime list, whose members can.
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if self._storage == D:
                return size_of[Scalar[D]]() * 8
        return 0

    def sum_type(self) -> DataType:
        """The result type of sum and cumulative sums (as in Polars): 8-
        and 16-bit integers widen to INT64; other types keep their type."""
        if self.is_integer() and self.bit_width() <= 16:
            return DataType.INT64
        if self.is_decimal() and self.decimal_width() != 128:
            # A decimal32 or decimal64 sum can outgrow its width: widen it.
            try:
                return DataType.decimal(38, self.scale())
            except:
                return self
        return self

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.name())


def _unit_code(unit: String) raises -> Int:
    if unit == "ns":
        return NS
    if unit == "us":
        return US
    if unit == "ms":
        return MS
    raise Error("time unit must be 'ns', 'us', or 'ms', found '" + unit + "'")


def _read_count(spec: String, mut cursor: Int) raises -> Int:
    """Read digits up to ':' at cursor; leave cursor after the colon."""
    var value = 0
    var digits = 0
    while cursor < spec.byte_length():
        var byte = spec.as_bytes()[cursor]
        cursor += 1
        if byte == 58:  # ':'
            if digits == 0:
                raise Error("Malformed nested dtype spec")
            return value
        if byte < 48 or byte > 57:
            raise Error("Malformed nested dtype spec")
        value = value * 10 + Int(byte - 48)
        digits += 1
    raise Error("Malformed nested dtype spec")


def _read_text(spec: String, mut cursor: Int) raises -> String:
    var length = _read_count(spec, cursor)
    if cursor + length > spec.byte_length():
        raise Error("Malformed nested dtype spec")
    var text = String(spec[byte = cursor : cursor + length])
    cursor += length
    return text^


def _decode(spec: String, mut cursor: Int) raises -> DataType:
    """Decode one dtype at cursor (see DataType._encode)."""
    if cursor >= spec.byte_length():
        raise Error("Malformed nested dtype spec")
    var kind = spec.as_bytes()[cursor]
    cursor += 1
    if kind == 80:  # 'P'
        return DataType.parse(_read_text(spec, cursor))
    if kind == 76:  # 'L'
        return DataType(_LIST, "L" + _decode(spec, cursor)._encode())
    if kind == 83:  # 'S'
        var start = cursor - 1
        var count = _read_count(spec, cursor)
        for _ in range(count):
            _ = _read_text(spec, cursor)
            _ = _decode(spec, cursor)
        return DataType(_STRUCT, String(spec[byte=start:cursor]))
    raise Error("Malformed nested dtype spec")
