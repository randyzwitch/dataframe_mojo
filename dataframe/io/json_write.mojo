"""Buffered JSON table writers with prepared chunk and nested-column cursors."""
from std.collections import Optional
from std.math import isnan, isinf
from std.memory import Pointer
from dataframe.text.byte_encoding import encode_bytes
from dataframe.storage.column import Column, SCALAR_DTYPES, _bit
from dataframe.storage.boolean import BoolColumn
from ..dtype import DataType
from ..frame import DataFrame
from ..series import Series
from dataframe.storage.string import StringColumn
from dataframe.text.bytes import _Bytes
from dataframe.storage.nested import ListColumn, StructColumn
from ..json_decode import validate_json_dtype
from ..json_value import json_quote
from ..json_number import float_text
from ..decimal import format_decimal
from ..temporal import PreparedFormat, default_format, format_prepared_in
from ..timezone import TimeZone
from .json_wire import duration_text

comptime _Number = def(Int, Int) thin -> String


def _number[D: DType](address: Int, row: Int) -> String:
    var value = Pointer[Scalar[D], ImmutAnyOrigin](unsafe_from_address=address)[
        unsafe_offset=row
    ]
    comptime if D.is_floating_point():
        if isnan(value) or isinf(value):
            return "null"
        return float_text(Float64(value), True)
    else:
        return String(value)


def _quote(bytes: Span[UInt8, _], mut out: List[UInt8]):
    """Copy UTF-8 runs; JSON escaping changes only ASCII control bytes."""
    out.append(34)
    var hex = StringSlice("0123456789abcdef").as_bytes()
    var start = 0
    for at in range(len(bytes)):
        var byte = bytes[at]
        if byte < 32 or byte == 34 or byte == 92:
            out.extend(bytes[start:at])
            out.append(92)
            if byte == 34 or byte == 92:
                out.append(byte)
            elif byte == 8:
                out.append(98)
            elif byte == 9:
                out.append(116)
            elif byte == 10:
                out.append(110)
            elif byte == 12:
                out.append(102)
            elif byte == 13:
                out.append(114)
            else:
                out.extend(StringSlice("u00").as_bytes())
                out.append(hex[Int(byte) >> 4])
                out.append(hex[Int(byte) & 15])
            start = at + 1
    out.extend(bytes[start:])
    out.append(34)


struct _Node(Movable):
    var series: Series
    # -1 chunk cursor, 0 number, 1 bool, 2 string, 3 list, 4 struct,
    # 5 temporal, 6 decimal, 7 duration, 8 bytes, 9 categorical, 10 null.
    var kind: Int
    var dtype: DataType
    var children: List[Int]
    var names: List[String]
    var cursor: Int
    var first: Int
    var values: Int
    var bits: Int
    var offset: Int
    var number: _Number
    var strings: Optional[_Bytes]
    var format: PreparedFormat
    var zone: TimeZone

    def __init__(out self, series: Series) raises:
        self.series = series.copy()
        self.dtype = series.dtype()
        self.kind = 0
        self.children = List[Int]()
        self.names = List[String]()
        self.cursor = 0
        self.first = 0
        self.values = 0
        self.bits = 0
        self.offset = 0
        self.number = _number[DType.int64]
        self.strings = None
        self.zone = TimeZone.utc()
        self.format = PreparedFormat("")

    def valid(self, row: Int) -> Bool:
        if self.bits == 0:
            return True
        var at = self.offset + row
        var byte = Pointer[UInt8, ImmutAnyOrigin](
            unsafe_from_address=self.bits
        )[unsafe_offset=at // 8]
        return (byte & (UInt8(1) << UInt8(at % 8))) != 0


def _prepare(mut nodes: List[_Node], series: Series) raises -> Int:
    var index = len(nodes)
    nodes.append(_Node(series))
    if series.is_chunked():
        nodes[index].kind = -1
        for chunk in series.chunks():
            var child = _prepare(nodes, chunk)
            nodes[index].children.append(child)
        return index
    var dtype = series.dtype()
    if dtype == DataType.NULL:
        nodes[index].kind = 10
    elif dtype == DataType.STRING or dtype._is_bytes():
        ref column = series._data[StringColumn]
        nodes[index].kind = 8 if dtype._is_bytes() else 2
        nodes[index].strings = _Bytes(column)
        nodes[index].offset = column._offset
        if len(column._bits[]):
            nodes[index].bits = Int(column._bits[].unsafe_ptr())
    elif dtype == DataType.BOOL:
        ref column = series._data[BoolColumn]
        nodes[index].kind = 1
        nodes[index].offset = column._offset
        nodes[index].values = Int(column._data[].unsafe_ptr())
        if len(column._bits[]):
            nodes[index].bits = Int(column._bits[].unsafe_ptr())
    elif dtype.is_list():
        ref column = series._data[ListColumn]
        nodes[index].kind = 3
        nodes[index].offset = column._offset
        nodes[index].values = Int(column._offsets[].unsafe_ptr())
        if len(column._bits[]):
            nodes[index].bits = Int(column._bits[].unsafe_ptr())
        var child = _prepare(nodes, column.child())
        nodes[index].children.append(child)
    elif dtype.is_struct():
        ref column = series._data[StructColumn]
        nodes[index].kind = 4
        nodes[index].offset = column._offset
        if len(column._bits[]):
            nodes[index].bits = Int(column._bits[].unsafe_ptr())
        for i in range(column.field_count()):
            var field = column.field(i)
            nodes[index].names.append(json_quote(field.name()) + ":")
            var child = _prepare(nodes, field)
            nodes[index].children.append(child)
    else:
        comptime for D in SCALAR_DTYPES:
            if series._data.isa[Column[Scalar[D]]]():
                ref column = series._data[Column[Scalar[D]]]
                nodes[index].values = Int(column._ptr())
                nodes[index].offset = column._offset
                nodes[index].number = _number[D]
                if len(column._bits[]):
                    nodes[index].bits = Int(column._bits[].unsafe_ptr())
        if dtype.is_temporal():
            nodes[index].kind = 7 if dtype.is_duration() else 5
            if dtype.is_datetime() and dtype.time_zone().byte_length():
                # A canonical UTC offset preserves the instant without an
                # Int64 local-offset addition at extreme timestamp values.
                nodes[index].dtype = DataType.datetime(dtype.unit(), "UTC")
            nodes[index].format = PreparedFormat(
                default_format(nodes[index].dtype)
            )
        elif dtype.is_decimal():
            nodes[index].kind = 6
        elif dtype.is_categorical():
            nodes[index].kind = 9
            var dictionary = dtype.dictionary()
            for i in range(len(dictionary[])):
                nodes[index].names.append(
                    json_quote(String(dictionary[].get(i)))
                )
    return index


def _emit(mut nodes: List[_Node], index: Int, row: Int, mut out: List[UInt8]):
    var kind = nodes[index].kind
    if kind == -1:
        while nodes[index].cursor + 1 < len(nodes[index].children):
            var child = nodes[index].children[nodes[index].cursor]
            if row < nodes[index].first + len(nodes[child].series):
                break
            nodes[index].first += len(nodes[child].series)
            nodes[index].cursor += 1
        var child = nodes[index].children[nodes[index].cursor]
        _emit(nodes, child, row - nodes[index].first, out)
        return
    if kind == 10 or not nodes[index].valid(row):
        out.extend(StringSlice("null").as_bytes())
        return
    if kind == 0:
        var text = nodes[index].number(nodes[index].values, row)
        out.extend(text.as_bytes())
    elif kind == 1:
        var at = nodes[index].offset + row
        var byte = Pointer[UInt8, ImmutAnyOrigin](
            unsafe_from_address=nodes[index].values
        )[unsafe_offset=at // 8]
        out.extend(
            StringSlice(
                "true" if byte & (UInt8(1) << UInt8(at % 8)) else "false"
            ).as_bytes()
        )
    elif kind == 2:
        _quote(nodes[index].strings.value().get(row), out)
    elif kind == 8:
        out.append(34)
        encode_bytes(nodes[index].strings.value().get(row), False, out)
        out.append(34)
    elif kind == 3:
        var offsets = Pointer[Int64, ImmutAnyOrigin](
            unsafe_from_address=nodes[index].values
        )
        var at = nodes[index].offset + row
        var start = Int(offsets[unsafe_offset=at])
        var end = Int(offsets[unsafe_offset=at + 1])
        out.append(91)
        for i in range(start, end):
            if i > start:
                out.append(44)
            _emit(nodes, nodes[index].children[0], i, out)
        out.append(93)
    elif kind == 4:
        out.append(123)
        for i in range(len(nodes[index].children)):
            if i:
                out.append(44)
            out.extend(nodes[index].names[i].as_bytes())
            _emit(nodes, nodes[index].children[i], row, out)
        out.append(125)
    elif kind == 9:
        var code = Pointer[UInt32, ImmutAnyOrigin](
            unsafe_from_address=nodes[index].values
        )[unsafe_offset=row]
        out.extend(nodes[index].names[Int(code)].as_bytes())
    else:
        var text: String
        if kind == 6:
            var value = Int128(0)
            if nodes[index].dtype.decimal_width() == 128:
                value = Pointer[Int128, ImmutAnyOrigin](
                    unsafe_from_address=nodes[index].values
                )[unsafe_offset=row]
            elif nodes[index].dtype.decimal_width() == 64:
                value = Int128(
                    Pointer[Int64, ImmutAnyOrigin](
                        unsafe_from_address=nodes[index].values
                    )[unsafe_offset=row]
                )
            else:
                value = Int128(
                    Pointer[Int32, ImmutAnyOrigin](
                        unsafe_from_address=nodes[index].values
                    )[unsafe_offset=row]
                )
            text = format_decimal(value, nodes[index].dtype.scale())
        else:
            var value = Pointer[Int64, ImmutAnyOrigin](
                unsafe_from_address=nodes[index].values
            )[unsafe_offset=row]
            text = duration_text(
                value, nodes[index].dtype
            ) if kind == 7 else format_prepared_in(
                value,
                nodes[index].dtype,
                nodes[index].format,
                nodes[index].zone,
            )
            if (
                kind == 5
                and not nodes[index].dtype.is_time()
                and text.as_bytes()[0] != 45
                and text.find("-") > 4
            ):
                text = "+" + text
        _quote(text.as_bytes(), out)


def _write(
    frame: DataFrame, path: String, array: Bool, buffer_size: Int
) raises:
    if buffer_size <= 0:
        raise Error("JSON buffer_size must be positive")
    var nodes = List[_Node]()
    var roots = List[Int]()
    var names = List[String]()
    for column in frame._columns:
        validate_json_dtype(column.dtype(), io=True)
        names.append(json_quote(column.name()) + ":")
        roots.append(_prepare(nodes, column))
    with open(path, "w") as file:
        var buffer = List[UInt8](capacity=buffer_size)
        if array:
            buffer.append(91)
        for row in range(frame.height()):
            if array and row:
                buffer.append(44)
            buffer.append(123)
            for i in range(len(roots)):
                if i:
                    buffer.append(44)
                buffer.extend(names[i].as_bytes())
                _emit(nodes, roots[i], row, buffer)
            buffer.append(125)
            if not array:
                buffer.append(10)
            if len(buffer) >= buffer_size:
                file.write(String(unsafe_from_utf8=buffer))
                buffer.clear()
        if array:
            buffer.append(93)
        if len(buffer):
            file.write(String(unsafe_from_utf8=buffer))


def write_ndjson(
    frame: DataFrame, path: String, *, buffer_size: Int = 65536
) raises:
    """Write UTF-8 objects, one per line, with bounded output buffering.

    Nested lists/structs retain nulls. Finite floats round-trip exactly;
    nonfinite floats become null. Temporal values are ISO 8601 strings,
    decimals exact strings, and binary/geometry bytes padded base64 strings.
    An explicit read schema restores logical dtypes that JSON cannot encode.
    """
    _write(frame, path, False, buffer_size)


def write_json(
    frame: DataFrame, path: String, *, buffer_size: Int = 65536
) raises:
    """Write a JSON array of objects using the same wire forms as NDJSON."""
    _write(frame, path, True, buffer_size)
