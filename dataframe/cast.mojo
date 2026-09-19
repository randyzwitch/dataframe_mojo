"""Explicit dtype conversion. String parsing matches read_csv exactly.

Values pass through an exact intermediate: Int128 for every integer type
(UInt64 included) and Bool, Float64 for both float types. Integer targets
are range-checked; float to integer truncates toward zero and rejects NaN,
infinities, and out-of-range values.
"""
from std.math import isinf, isnan, trunc
from std.sys import size_of
from .bool_column import BoolColumn
from .column import Column
from .string_column import StringColumn, StringBuilder
from .dtype import DataType, NUMERIC_DTYPES
from .parse import parse_bool, parse_float64, parse_integer
from .series import Series
from .temporal_kernels import cast_temporal


def _dtype(series: Series) -> DataType:
    return series.dtype()


def _text(series: Series, row: Int) -> String:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            return String(series._data[Column[Scalar[D]]]._get(row))
    if series._data.isa[BoolColumn]():
        return "true" if series._data[BoolColumn]._get(row) else "false"
    return String(series._data[StringColumn]._get(row))


def _valid(series: Series, row: Int) -> Bool:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            return series._data[Column[Scalar[D]]]._valid(row)
    if series._data.isa[BoolColumn]():
        return series._data[BoolColumn]._valid(row)
    return series._data[StringColumn]._valid(row)


def _to_integer[D: DType](value: Int128) raises -> Scalar[D]:
    if (
        value < Scalar[D].MIN.cast[DType.int128]()
        or value > Scalar[D].MAX.cast[DType.int128]()
    ):
        raise Error("out of " + String(D) + " range")
    return value.cast[D]()


def _float_to_integer[D: DType](value: Float64) raises -> Scalar[D]:
    if isnan(value) or isinf(value):
        raise Error("out of " + String(D) + " range")
    # Exact bounds: 2**(bits - 1) for signed types, 2**bits for unsigned.
    var bits = size_of[Scalar[D]]() * 8 - (1 if D.is_signed() else 0)
    var upper = Float64(1)
    for _ in range(bits):
        upper *= 2
    var lower = -upper if D.is_signed() else Float64(0)
    var whole = trunc(value)
    if whole >= upper or whole < lower:
        raise Error("out of " + String(D) + " range")
    return whole.cast[D]()


struct _Values(Movable):
    """The exact intermediate for one cast: `is_float` picks the list."""

    var ints: List[Int128]
    var floats: List[Float64]
    var is_float: Bool

    def __init__(out self, n: Int, is_float: Bool):
        self.is_float = is_float
        self.ints = List[Int128](length=0 if is_float else n, fill=0)
        self.floats = List[Float64](length=n if is_float else 0, fill=0)


def _read_source(
    input: Series,
    source: DataType,
    target: DataType,
    i: Int,
    mut values: _Values,
) raises:
    """Row i of input into the intermediate, parsing strings for target."""
    if source == DataType.STRING:
        var text = String(input._data[StringColumn]._get(i))
        if target.is_float():
            values.floats[i] = parse_float64(text)
        elif target == DataType.BOOL:
            values.ints[i] = Int128(Int(parse_bool(text)))
        elif target == DataType.UINT64:
            values.ints[i] = parse_integer[DType.uint64](text).cast[
                DType.int128
            ]()
        else:
            values.ints[i] = parse_integer[DType.int64](text).cast[
                DType.int128
            ]()
        return
    if source == DataType.BOOL:
        var bit = Int(input._data[BoolColumn]._get(i))
        if values.is_float:
            values.floats[i] = Float64(bit)
        else:
            values.ints[i] = Int128(bit)
        return
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if input._data.isa[Column[Scalar[D]]]():
            var x = input._data[Column[Scalar[D]]]._get(i)
            comptime if D.is_floating_point():
                if values.is_float:
                    values.floats[i] = x.cast[DType.float64]()
                else:
                    values.floats[i] = x.cast[DType.float64]()
            else:
                if values.is_float:
                    values.floats[i] = x.cast[DType.float64]()
                else:
                    values.ints[i] = x.cast[DType.int128]()


def cast_series(
    input: Series, target: DataType, strict: Bool, offset: Int, mask: List[Bool]
) raises -> Series:
    """Convert every valid, observed row; others become null."""
    var source = _dtype(input)
    if source == target:
        return input.copy()
    if source.is_temporal() or target.is_temporal():
        var observed = input.copy()
        return cast_temporal(observed, source, target, strict)
    var n = len(input)
    var valid = List[Bool](length=n, fill=False)
    if target == DataType.STRING:
        var out = StringBuilder(n)
        for i in range(n):
            if _valid(input, i) and (len(mask) != n or mask[i]):
                out.append(_text(input, i))
            else:
                out.append_null()
        return Series(input.name(), out^.finish())
    # Floats stay floats in the intermediate; everything else is exact Int128.
    var float_source = source.is_float()
    var values = _Values(n, target.is_float() or float_source)
    var targets = List[Bool](length=n, fill=False)
    for i in range(n):
        if not _valid(input, i):
            continue
        if len(mask) == n and not mask[i]:
            continue
        try:
            _read_source(input, source, target, i, values)
            if target == DataType.BOOL:
                if values.is_float:
                    if isnan(values.floats[i]):
                        raise Error("NaN has no Boolean value")
                    targets[i] = values.floats[i] != 0
                else:
                    targets[i] = values.ints[i] != 0
            elif target.is_integer():
                comptime for k in range(len(NUMERIC_DTYPES)):
                    comptime D = NUMERIC_DTYPES[k]
                    comptime if D.is_integral():
                        if target == DataType.of(D):
                            if values.is_float:
                                _ = _float_to_integer[D](values.floats[i])
                            else:
                                _ = _to_integer[D](values.ints[i])
            valid[i] = True
        except e:
            if strict:
                raise Error(
                    "cast from "
                    + source.name()
                    + " to "
                    + target.name()
                    + " failed at row "
                    + String(offset + i)
                    + " for value '"
                    + _text(input, i)
                    + "': "
                    + String(e)
                )
    if target == DataType.BOOL:
        return Series(input.name(), BoolColumn(targets^, valid))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if target == DataType.of(D):
            var out = List[Scalar[D]](length=n, fill=0)
            for i in range(n):
                if not valid[i]:
                    continue
                comptime if D.is_floating_point():
                    out[i] = (
                        values.floats[i]
                        .cast[D]() if values.is_float else values.ints[i]
                        .cast[D]()
                    )
                else:
                    out[i] = _float_to_integer[D](
                        values.floats[i]
                    ) if values.is_float else _to_integer[D](values.ints[i])
            return Series(input.name(), Column[Scalar[D]](out^, valid))
    raise Error("Unsupported cast target " + target.name())
