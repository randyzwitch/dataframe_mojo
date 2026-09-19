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


@fieldwise_init
struct DataType(Copyable, Equatable, ImplicitlyCopyable, Writable):
    """A logical column type: INT64, FLOAT64, BOOL, or STRING."""

    var _code: Int
    var _unit: Int

    comptime INT64 = DataType(_INT64, 0)
    comptime FLOAT64 = DataType(_FLOAT64, 0)
    comptime BOOL = DataType(_BOOL, 0)
    comptime STRING = DataType(_STRING, 0)

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
        return "string"

    def short_name(self) -> String:
        """The compact name used in table headers (i64, f64, bool, str)."""
        if self._code == _INT64:
            return "i64"
        if self._code == _FLOAT64:
            return "f64"
        if self._code == _BOOL:
            return "bool"
        return "str"

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
        if self._code == _INT64 or self._code == _FLOAT64:
            return 64
        if self._code == _BOOL:
            return 1
        return 0

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.name())
