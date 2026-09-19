"""Structured logical data types.

`DataType` identifies a column's logical type. Values are compile-time
constants (`DataType.INT64`) and compare with `==`. `String(dtype)` gives the
canonical name used in schemas, casts, and error messages, and
`DataType.parse(name)` is its inverse. Parameterized types (such as datetime
units) keep their parameter in `_unit`.
"""

comptime _INT64 = 0
comptime _FLOAT64 = 1
comptime _BOOL = 2
comptime _STRING = 3
comptime _DATE = 4
comptime _DATETIME = 5
comptime _DURATION = 6
comptime _TIME = 7

# Time units for datetime and duration, kept in DataType._unit.
comptime NS = 1
comptime US = 2
comptime MS = 3


@fieldwise_init
struct DataType(Copyable, Equatable, ImplicitlyCopyable, Writable):
    """A logical column type.

    INT64, FLOAT64, BOOL, STRING, DATE (days since 1970-01-01), TIME
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
        return self._code == _INT64

    def is_float(self) -> Bool:
        return self._code == _FLOAT64

    def is_signed(self) -> Bool:
        """Whether values can be negative (numeric types only)."""
        return self.is_numeric()

    def bit_width(self) -> Int:
        """Bits per value for fixed-width types; 0 for variable-width."""
        if self._code == _INT64 or self._code == _FLOAT64 or self.is_temporal():
            return 64
        if self._code == _BOOL:
            return 1
        return 0

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
