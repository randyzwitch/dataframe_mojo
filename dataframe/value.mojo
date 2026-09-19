"""A runtime-tagged scalar for row access and single-cell results."""
from .dtype import DataType, NUMERIC_DTYPES
from .temporal import format as format_temporal


struct AnyValue(Copyable, Equatable, Writable):
    """One nullable cell of a supported dtype.

    Only the payload field matching `dtype` is meaningful. Equality is
    structural: nulls equal nulls of the same dtype and NaN equals NaN.
    """

    var _dtype: DataType
    var _valid: Bool
    var _int: Int64
    var _float: Float64
    var _bool: Bool
    var _string: String

    def __init__[D: DType](out self, value: Scalar[D]):
        """A numeric value. Integers are held in an Int64 slot (UInt64 by
        bit pattern) and floats in a Float64 slot; both are exact."""
        comptime if D.is_floating_point():
            self = Self(DataType.of(D), True, 0, Float64(value), False, "")
        elif D == DType.uint64:
            self = Self(
                DataType.of(D),
                True,
                Int64(value.cast[DType.int64]()),
                0,
                False,
                "",
            )
        else:
            self = Self(DataType.of(D), True, Int64(value), 0, False, "")

    def __init__(out self, value: Bool):
        self = Self(DataType.BOOL, True, 0, 0, value, "")

    def __init__(out self, value: String):
        self = Self(DataType.STRING, True, 0, 0, False, value)

    def __init__(
        out self,
        dtype: DataType,
        valid: Bool,
        integer: Int64,
        floating: Float64,
        boolean: Bool,
        var string: String,
    ):
        self._dtype = dtype
        self._valid = valid
        self._int = integer
        self._float = floating
        self._bool = boolean
        self._string = string^

    @staticmethod
    def null(dtype: DataType) -> Self:
        return Self(dtype, False, 0, 0, False, "")

    @staticmethod
    def null(dtype: String) raises -> Self:
        return Self(DataType.parse(dtype), False, 0, 0, False, "")

    @staticmethod
    def temporal(dtype: DataType, value: Int64) raises -> Self:
        """A date, datetime, duration, or time from its stored Int64."""
        if not dtype.is_temporal():
            raise Error("AnyValue.temporal requires a temporal dtype")
        return Self(dtype, True, value, 0, False, "")

    def to_physical(self) raises -> Int64:
        """The stored Int64 of an Int64 or temporal value."""
        if self._dtype.physical() != DataType.INT64:
            raise Error(
                "Expected an integer-backed value, found " + self._dtype.name()
            )
        if not self._valid:
            raise Error("Cannot read a null value")
        return self._int

    def dtype(self) -> DataType:
        return self._dtype

    def is_null(self) -> Bool:
        return not self._valid

    def _check(self, dtype: DataType) raises:
        if self._dtype != dtype:
            raise Error(
                "Expected "
                + dtype.name()
                + " value, found "
                + self._dtype.name()
            )
        if not self._valid:
            raise Error("Cannot read a null value")

    def numeric[D: DType](self) raises -> Scalar[D]:
        """The value as Scalar[D]; the dtype must match exactly."""
        self._check(DataType.of(D))
        comptime if D.is_floating_point():
            return self._float.cast[D]()
        else:
            return self._int.cast[D]()

    def int64(self) raises -> Int64:
        return self.numeric[DType.int64]()

    def float64(self) raises -> Float64:
        return self.numeric[DType.float64]()

    def int8(self) raises -> Int8:
        return self.numeric[DType.int8]()

    def int16(self) raises -> Int16:
        return self.numeric[DType.int16]()

    def int32(self) raises -> Int32:
        return self.numeric[DType.int32]()

    def uint8(self) raises -> UInt8:
        return self.numeric[DType.uint8]()

    def uint16(self) raises -> UInt16:
        return self.numeric[DType.uint16]()

    def uint32(self) raises -> UInt32:
        return self.numeric[DType.uint32]()

    def uint64(self) raises -> UInt64:
        return self.numeric[DType.uint64]()

    def float32(self) raises -> Float32:
        return self.numeric[DType.float32]()

    def bool(self) raises -> Bool:
        self._check(DataType.BOOL)
        return self._bool

    def string(self) raises -> String:
        self._check(DataType.STRING)
        return self._string

    def __eq__(self, other: Self) -> Bool:
        if self._dtype != other._dtype or self._valid != other._valid:
            return False
        if not self._valid:
            return True
        if self._dtype.physical() == DataType.INT64 or self._dtype.is_integer():
            return self._int == other._int
        if self._dtype.is_float():
            var both_nan = (
                self._float != self._float and other._float != other._float
            )
            return both_nan or self._float == other._float
        if self._dtype == DataType.BOOL:
            return self._bool == other._bool
        return self._string == other._string

    def write_to(self, mut writer: Some[Writer]):
        if not self._valid:
            writer.write("null")
        elif self._dtype == DataType.UINT64:
            writer.write(self._int.cast[DType.uint64]())
        elif self._dtype.is_integer():
            writer.write(self._int)
        elif self._dtype.is_temporal():
            writer.write(format_temporal(self._int, self._dtype))
        elif self._dtype == DataType.FLOAT32:
            writer.write(self._float.cast[DType.float32]())
        elif self._dtype == DataType.FLOAT64:
            writer.write(self._float)
        elif self._dtype == DataType.BOOL:
            writer.write("true" if self._bool else "false")
        else:
            writer.write(self._string)
