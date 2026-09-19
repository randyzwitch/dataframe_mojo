"""Structured logical data types.

`DataType` identifies a column's logical type. Values are compile-time
constants (`DataType.INT64`) and compare with `==`. `String(dtype)` gives the
canonical name used in schemas, casts, and error messages, and
`DataType.parse(name)` is its inverse. Parameterized types (such as datetime
units) keep their parameter in `_unit`.
"""
from std.sys import size_of

comptime _INT64 = 0
comptime _FLOAT64 = 1
comptime _BOOL = 2
comptime _STRING = 3
comptime _DATE = 4
comptime _DATETIME = 5
comptime _DURATION = 6
comptime _TIME = 7
comptime _INT8 = 8
comptime _INT16 = 9
comptime _INT32 = 10
comptime _UINT8 = 11
comptime _UINT16 = 12
comptime _UINT32 = 13
comptime _UINT64 = 14
comptime _FLOAT32 = 15
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


@fieldwise_init
struct DataType(Copyable, Equatable, ImplicitlyCopyable, Writable):
    """A logical column type.

    Numeric: INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64,
    FLOAT32, FLOAT64. Also BOOL, STRING, DATE (days since 1970-01-01), TIME
    (nanoseconds since midnight), and datetime(unit) / duration(unit) with
    unit "ns", "us", or "ms". Temporal types are stored as Int64.
    """

    var _code: Int
    var _unit: Int

    comptime INT64 = DataType(_INT64, 0)
    comptime FLOAT64 = DataType(_FLOAT64, 0)
    comptime BOOL = DataType(_BOOL, 0)
    comptime STRING = DataType(_STRING, 0)
    comptime DATE = DataType(_DATE, 0)
    comptime TIME = DataType(_TIME, 0)
    comptime INT8 = DataType(_INT8, 0)
    comptime INT16 = DataType(_INT16, 0)
    comptime INT32 = DataType(_INT32, 0)
    comptime UINT8 = DataType(_UINT8, 0)
    comptime UINT16 = DataType(_UINT16, 0)
    comptime UINT32 = DataType(_UINT32, 0)
    comptime UINT64 = DataType(_UINT64, 0)
    comptime FLOAT32 = DataType(_FLOAT32, 0)
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
        if dtype == DType.int64:
            return DataType.INT64
        if dtype == DType.float64:
            return DataType.FLOAT64
        if dtype == DType.int8:
            return DataType.INT8
        if dtype == DType.int16:
            return DataType.INT16
        if dtype == DType.int32:
            return DataType.INT32
        if dtype == DType.uint8:
            return DataType.UINT8
        if dtype == DType.uint16:
            return DataType.UINT16
        if dtype == DType.uint32:
            return DataType.UINT32
        if dtype == DType.uint64:
            return DataType.UINT64
        return DataType.FLOAT32

    def storage(self) -> Optional[DType]:
        """The numeric storage DType (int64 for temporal types); None for
        bool and string."""
        var physical = self.physical()
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if physical == DataType.of(D):
                return D
        return None

    @staticmethod
    def datetime(unit: String = "us") raises -> DataType:
        """A time-zone-naive instant counted in unit since the epoch."""
        return DataType(_DATETIME, _unit_code(unit))

    @staticmethod
    def duration(unit: String = "us") raises -> DataType:
        """A signed length of time counted in unit."""
        return DataType(_DURATION, _unit_code(unit))

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
        for unit in ["ns", "us", "ms"]:
            if name == "datetime[" + unit + "]":
                return DataType.datetime(unit)
            if name == "duration[" + unit + "]":
                return DataType.duration(unit)
        raise Error("Unknown dtype: " + name)

    @staticmethod
    def is_known(name: String) -> Bool:
        try:
            _ = DataType.parse(name)
            return True
        except:
            return False

    def __eq__(self, other: Self) -> Bool:
        return self._code == other._code and self._unit == other._unit

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def name(self) -> String:
        """The canonical name, as accepted by parse."""
        if self._code == _INT64:
            return "int64"
        if self._code == _FLOAT64:
            return "float64"
        if self._code == _BOOL:
            return "bool"
        if self._code == _DATE:
            return "date"
        if self._code == _TIME:
            return "time"
        if self._code == _DATETIME:
            return "datetime[" + self.unit() + "]"
        if self._code == _DURATION:
            return "duration[" + self.unit() + "]"
        if self._code == _UNTYPED_INT:
            return "integer literal"
        if self._code == _UNTYPED_FLOAT:
            return "float literal"
        if self._code >= _INT8:
            return String(self.storage().value())
        return "string"

    def short_name(self) -> String:
        """The compact name used in table headers (i64, f64, bool, str)."""
        if self._code == _INT64:
            return "i64"
        if self._code == _FLOAT64:
            return "f64"
        if self._code == _BOOL:
            return "bool"
        if self._code == _STRING:
            return "str"
        if self._code >= _INT8:
            var name = self.name()
            if name.startswith("uint"):
                return "u" + String(name[byte=4:])
            if name.startswith("int"):
                return "i" + String(name[byte=3:])
            return "f" + String(name[byte=5:])
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

    def physical(self) -> DataType:
        """The storage type: INT64 for temporal types, otherwise self."""
        if self.is_temporal():
            return DataType.INT64
        return self

    def is_numeric(self) -> Bool:
        return self.is_integer() or self.is_float()

    def is_integer(self) -> Bool:
        return self._code == _INT64 or (
            self._code >= _INT8 and self._code <= _UINT64
        )

    def is_float(self) -> Bool:
        return self._code == _FLOAT64 or self._code == _FLOAT32

    def is_unsigned(self) -> Bool:
        return self._code >= _UINT8 and self._code <= _UINT64

    def is_signed(self) -> Bool:
        """Whether values can be negative (numeric types only)."""
        return self.is_numeric() and not self.is_unsigned()

    def bit_width(self) -> Int:
        """Bits per value for fixed-width types; 0 for variable-width."""
        if self._code == _BOOL:
            return 1
        var physical = self.physical()
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if physical == DataType.of(D):
                return size_of[Scalar[D]]() * 8
        return 0

    def sum_type(self) -> DataType:
        """The result type of sum and cumulative sums (as in Polars): 8-
        and 16-bit integers widen to INT64; other types keep their type."""
        if (
            self._code == _INT8
            or self._code == _INT16
            or self._code == _UINT8
            or self._code == _UINT16
        ):
            return DataType.INT64
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
