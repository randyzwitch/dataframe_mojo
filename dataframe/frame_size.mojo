"""Visible buffer accounting, following Polars' Arrow array size estimates."""
from std.sys import size_of
from .frame import DataFrame
from .series import Series, FixedElements
from .column import Column
from .bool_column import BoolColumn
from .string_column import StringColumn
from .nested_column import ListColumn, StructColumn
from .dtype import DataType


def _bitmap_bytes(offset: Int, length: Int, present: Bool) -> Int:
    if not present or length == 0:
        return 0
    return length // 8 + (length % 8 + offset % 8 + 7) // 8


def _series_bytes(series: Series) raises -> Float64:
    if series.dtype() == DataType.NULL:
        return 0
    if series.is_chunked():
        var total = Float64(0)
        for part in series.chunks():
            total += _series_bytes(part)
        return total
    comptime for i in range(len(FixedElements.Ts)):
        comptime E: Copyable & Deinitable = FixedElements.Ts[i]
        if series._data.isa[Column[E]]():
            ref column = series._data[Column[E]]
            return Float64(len(column)) * Float64(size_of[E]()) + Float64(
                _bitmap_bytes(
                    column._offset, len(column), len(column._bits[]) > 0
                )
            )
    if series._data.isa[BoolColumn]():
        ref column = series._data[BoolColumn]
        return Float64(
            _bitmap_bytes(column._offset, len(column), True)
        ) + Float64(
            _bitmap_bytes(column._offset, len(column), len(column._bits[]) > 0)
        )
    if series._data.isa[StringColumn]():
        ref column = series._data[StringColumn]
        if column._is_view_storage():
            var total = Float64(0)
            for i in range(len(column)):
                if column._valid(i):
                    total += Float64(column._byte_length(i))
            return total
        var start = column._start(0)
        var end = Int(column._offsets[][column._offset + len(column)])
        return (
            Float64(end - start)
            + (Float64(len(column)) + 1) * 8
            + Float64(
                _bitmap_bytes(
                    column._offset, len(column), len(column._bits[]) > 0
                )
            )
        )
    if series._data.isa[ListColumn]():
        ref column = series._data[ListColumn]
        var start = column._start(0)
        var end = Int(column._offsets[][column._offset + len(column)])
        return (
            _series_bytes(column.child()._slice_validated(start, end - start))
            + (Float64(len(column)) + 1) * 8
            + Float64(
                _bitmap_bytes(
                    column._offset, len(column), len(column._bits[]) > 0
                )
            )
        )
    ref column = series._data[StructColumn]
    var total = Float64(
        _bitmap_bytes(column._offset, len(column), len(column._bits[]) > 0)
    )
    for i in range(column.field_count()):
        total += _series_bytes(column.field(i))
    return total


def frame_size(frame: DataFrame, unit: String) raises -> Float64:
    """Sum visible buffer estimates in one binary-scaled size unit."""
    var divisor = Float64(1)
    if unit == "b" or unit == "bytes":
        pass
    elif unit == "kb" or unit == "kilobytes":
        divisor = 1024
    elif unit == "mb" or unit == "megabytes":
        divisor = 1024 * 1024
    elif unit == "gb" or unit == "gigabytes":
        divisor = Float64(1024) ** 3
    elif unit == "tb" or unit == "terabytes":
        divisor = Float64(1024) ** 4
    else:
        raise Error("Unknown estimated_size unit: " + unit)
    var total = Float64(0)
    for column in frame._columns:
        total += _series_bytes(column)
    return total / divisor
