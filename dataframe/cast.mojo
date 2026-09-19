"""Explicit dtype conversion. String parsing matches read_csv exactly."""
from std.math import isinf, isnan
from .column import Column
from .dtype import DataType
from .parse import parse_bool, parse_float64, parse_int64
from .series import Series
from .temporal_kernels import cast_temporal


def _dtype(series: Series) -> DataType:
    return series.dtype()


def _text(series: Series, row: Int) -> String:
    if series._data.isa[Column[Int64]]():
        return String(series._data[Column[Int64]]._values[row])
    if series._data.isa[Column[Float64]]():
        return String(series._data[Column[Float64]]._values[row])
    if series._data.isa[Column[Bool]]():
        return "true" if series._data[Column[Bool]]._values[row] else "false"
    return series._data[Column[String]]._values[row]


def _valid(series: Series, row: Int) -> Bool:
    if series._data.isa[Column[Int64]]():
        return series._data[Column[Int64]]._valid(row)
    if series._data.isa[Column[Float64]]():
        return series._data[Column[Float64]]._valid(row)
    if series._data.isa[Column[Bool]]():
        return series._data[Column[Bool]]._valid(row)
    return series._data[Column[String]]._valid(row)


def _float_to_int(value: Float64) raises -> Int64:
    # 2**63 is exactly representable; every float in [-2**63, 2**63) fits.
    if (
        isnan(value)
        or isinf(value)
        or value >= 9223372036854775808.0
        or value < -9223372036854775808.0
    ):
        raise Error("out of Int64 range")
    return Int64(value)


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
    var ints = List[Int64](length=n if target == DataType.INT64 else 0, fill=0)
    var floats = List[Float64](
        length=n if target == DataType.FLOAT64 else 0, fill=0
    )
    var bools = List[Bool](
        length=n if target == DataType.BOOL else 0, fill=False
    )
    var strings = List[String](
        length=n if target == DataType.STRING else 0, fill=""
    )
    for i in range(n):
        if not _valid(input, i):
            continue
        if len(mask) == n and not mask[i]:
            continue
        try:
            if target == DataType.STRING:
                strings[i] = _text(input, i)
            elif source == DataType.STRING:
                ref text = input._data[Column[String]]._values[i]
                if target == DataType.INT64:
                    ints[i] = parse_int64(text)
                elif target == DataType.FLOAT64:
                    floats[i] = parse_float64(text)
                else:
                    bools[i] = parse_bool(text)
            elif source == DataType.INT64:
                var x = input._data[Column[Int64]]._values[i]
                if target == DataType.FLOAT64:
                    floats[i] = Float64(x)
                else:
                    bools[i] = x != 0
            elif source == DataType.FLOAT64:
                var x = input._data[Column[Float64]]._values[i]
                if target == DataType.INT64:
                    ints[i] = _float_to_int(x)
                else:
                    if isnan(x):
                        raise Error("NaN has no Boolean value")
                    bools[i] = x != 0
            else:
                var x = input._data[Column[Bool]]._values[i]
                if target == DataType.INT64:
                    ints[i] = Int64(x)
                else:
                    floats[i] = Float64(Int(x))
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
    if target == DataType.INT64:
        return Series(input.name(), Column[Int64](ints^, valid))
    if target == DataType.FLOAT64:
        return Series(input.name(), Column[Float64](floats^, valid))
    if target == DataType.BOOL:
        return Series(input.name(), Column[Bool](bools^, valid))
    return Series(input.name(), Column[String](strings^, valid))
