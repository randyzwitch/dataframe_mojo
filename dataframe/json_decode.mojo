"""Typed JSON decoding into column builders; one reusable tape per input."""
from std.collections import Optional
from std.math import trunc, floor
from std.sys import size_of
from std.utils import Variant
from .json_reader import JsonReader, JsonBytes
from .dtype import DataType
from .column import Column, SCALAR_DTYPES
from .bool_column import BoolColumn
from .string_column import StringBuilder, StringColumn
from .string_bytes import _Bytes
from .nested_column import ListColumn, StructColumn
from .series import Series
from .parse import parse_integer, parse_float64
from .temporal import parse_in, zone_of, has_offset
from .timezone import TimeZone
from .decimal import check_precision, pow10

from .json_number import (
    validate_json_numbers,
    json_number_text,
    is_json_fraction as _fraction,
)

comptime _Numbers = Variant[
    List[Int64],
    List[Float64],
    List[Int8],
    List[Int16],
    List[Int32],
    List[UInt8],
    List[UInt16],
    List[UInt32],
    List[UInt64],
    List[Float32],
    List[Int128],
]


def validate_json_dtype(dtype: DataType, depth: Int = 0) raises:
    if depth > 64:
        raise Error("JSON output schema nesting exceeds 64")
    if dtype.is_datetime():
        var zone = dtype.time_zone()
        if zone.byte_length() != 0 and zone != "UTC":
            raise Error("JSON datetime requires a naive or UTC output dtype")
    if dtype.is_list():
        validate_json_dtype(dtype.inner(), depth + 1)
    elif dtype.is_struct():
        var names = dtype.field_names()
        var seen = Dict[String, Bool]()
        for name in names:
            if name in seen:
                raise Error("JSON struct field names must be unique")
            seen[name] = True
        for field in dtype.field_dtypes():
            validate_json_dtype(field, depth + 1)
    elif not (
        dtype.is_numeric()
        or dtype.is_temporal()
        or dtype == DataType.STRING
        or dtype == DataType.BOOL
        or dtype == DataType.NULL
        or dtype.is_categorical()
    ):
        raise Error("JSON decoding does not support " + dtype.name())


@always_inline
def _hash(bytes: Span[UInt8, _]) -> UInt64:
    var hash = UInt64(14695981039346656037)
    for byte in bytes:
        hash = (hash ^ UInt64(byte)) * UInt64(1099511628211)
    return hash


def _key_hash(reader: JsonReader, value: Int) raises -> UInt64:
    ref node = reader.nodes[value]
    if node.key_escaped:
        var key = reader.string_at(node.key_start)
        return _hash(key.as_bytes())
    return _hash(reader.slice(node.key_start + 1, node.key_end - 1).as_bytes())


def _key_matches(reader: JsonReader, value: Int, key: String) raises -> Bool:
    ref node = reader.nodes[value]
    if node.key_escaped:
        return reader.string_at(node.key_start) == key
    var bytes = reader.slice(node.key_start + 1, node.key_end - 1).as_bytes()
    var other = key.as_bytes()
    if len(bytes) != len(other):
        return False
    for i in range(len(bytes)):
        if (
            bytes.unsafe_ptr()[unsafe_offset=i]
            != other.unsafe_ptr()[unsafe_offset=i]
        ):
            return False
    return True


@always_inline
def _valid(mut bits: List[UInt8], length: Int, valid: Bool):
    if not valid and len(bits) == 0:
        bits = List[UInt8](length=(length + 8) // 8, fill=255)
    if len(bits) != 0:
        if len(bits) < (length + 8) // 8:
            bits.append(255)
        if not valid:
            bits[length // 8] &= ~(UInt8(1) << UInt8(length % 8))


def _take[T: Copyable & Deinitable](mut values: List[T]) -> List[T]:
    var result = List[T]()
    swap(result, values)
    return result^


struct _Builder(Movable):
    var dtype: DataType
    var name: String
    # 0 numeric,1 bool,2 text,3 list,4 struct,5 date,6 datetime,7 duration,
    # 8 time,9 decimal,10 categorical,11 null.
    var kind: Int
    var children: List[Int]
    var fields: Dict[UInt64, Int]
    var next_collision: Int
    var length: Int
    var bits: List[UInt8]
    var bools: List[UInt8]
    var offsets: List[Int64]
    var numbers: _Numbers
    var strings: StringBuilder
    var zone: Optional[TimeZone]

    def __init__(out self, dtype: DataType, name: String, capacity: Int) raises:
        self.dtype = dtype
        self.name = name
        self.kind = 0
        self.children = List[Int]()
        self.fields = Dict[UInt64, Int]()
        self.next_collision = -1
        self.length = 0
        self.bits = List[UInt8]()
        self.bools = List[UInt8]()
        self.offsets = List[Int64]()
        self.numbers = _Numbers(List[Int64]())
        self.strings = StringBuilder(0)
        self.zone = None
        if dtype.is_list():
            self.kind = 3
            self.offsets.append(0)
        elif dtype.is_struct():
            self.kind = 4
        elif dtype == DataType.NULL:
            self.kind = 11
            self.numbers = _Numbers(List[Int8](capacity=capacity))
        elif dtype == DataType.BOOL:
            self.kind = 1
        elif dtype == DataType.STRING or dtype.is_categorical():
            self.kind = 10 if dtype.is_categorical() else 2
            self.strings = StringBuilder(capacity, capacity * 8)
        elif dtype.is_decimal():
            self.kind = 9
            self.numbers = _Numbers(List[Int128](capacity=capacity))
        else:
            var storage = dtype.physical().storage().value()
            comptime for D in SCALAR_DTYPES:
                if storage == D:
                    self.numbers = _Numbers(List[Scalar[D]](capacity=capacity))
            if dtype.is_date():
                self.kind = 5
                self.zone = zone_of(dtype)
            elif dtype.is_datetime():
                self.kind = 6
                self.zone = zone_of(dtype)
            elif dtype.is_duration():
                self.kind = 7
            elif dtype.is_time():
                self.kind = 8
                self.zone = zone_of(dtype)

    def mark(mut self, valid: Bool):
        if self.kind != 2 and self.kind != 10:
            _valid(self.bits, self.length, valid)
        self.length += 1


def _compile(
    mut nodes: List[_Builder], dtype: DataType, name: String, capacity: Int
) raises -> Int:
    var index = len(nodes)
    nodes.append(_Builder(dtype, name, capacity))
    if dtype.is_list():
        var child = _compile(nodes, dtype.inner(), "item", capacity)
        nodes[index].children.append(child)
    elif dtype.is_struct():
        var names = dtype.field_names()
        var types = dtype.field_dtypes()
        for i in range(len(names)):
            var child = _compile(nodes, types[i], names[i], capacity)
            nodes[index].children.append(child)
            var hash = _hash(names[i].as_bytes())
            if hash in nodes[index].fields:
                nodes[child].next_collision = nodes[index].fields[hash]
            nodes[index].fields[hash] = child
    return index


@always_inline
def _numeric(kind: Int) -> Bool:
    return kind == 45 or (kind >= 48 and kind <= 57)


def _int128(text: StringSlice) raises -> Int128:
    var data = text.as_bytes()
    var negative = data[0] == 45
    var at = Int(negative)
    var max = Int128(170141183460469231731687303715884105727)
    var limit = max // 10
    var last = 8 if negative else 7
    var value = Int128(0)
    while at < len(data):
        var digit = Int(data[at]) - 48
        if (
            digit < 0
            or digit > 9
            or value > limit
            or (value == limit and digit > last)
        ):
            raise Error("JSON integer overflow")
        if negative and value == limit and digit == 8:
            if at + 1 != len(data):
                raise Error("JSON integer overflow")
            return -max - 1
        value = value * 10 + Int128(digit)
        at += 1
    return -value if negative else value


def _number[
    D: DType
](reader: JsonReader, value: Int, boolean: Bool = True) raises -> Tuple[
    Scalar[D], Bool
]:
    var kind = reader.nodes[value].kind
    if boolean and (kind == 116 or kind == 102):
        return (Scalar[D](Int(kind == 116)), True)
    if not _numeric(kind):
        raise Error("JSON value is not numeric")
    var text = reader.raw(value)
    try:
        comptime if D.is_floating_point():
            if text == "-0":
                return (Scalar[D](0), True)
            return (parse_float64(text).cast[D](), True)
        else:
            if not _fraction(text):
                comptime if D == DType.int128:
                    return (rebind[Scalar[D]](_int128(text)), True)
                else:
                    return (parse_integer[D](text), True)
            var number = trunc(parse_float64(text))
            var high = Float64(1)
            comptime for _ in range(
                size_of[Scalar[D]]() * 8 - Int(D.is_signed())
            ):
                high *= 2
            var low = -high if D.is_signed() else Float64(0)
            if number != number or number < low or number >= high:
                return (Scalar[D](0), False)
            return (number.cast[D](), True)
    except:
        return (Scalar[D](0), False)


def _decimal_text(text: StringSlice, dtype: DataType) raises -> Int128:
    var bytes = text.as_bytes()
    var n = len(bytes)
    var at = 0
    var negative = False
    if n > 0 and (bytes[0] == 45 or bytes[0] == 43):
        negative = bytes[0] == 45
        at += 1
    var first = at
    var digits = 0
    var fraction = 0
    var dot = False
    while at < n and bytes[at] != 101 and bytes[at] != 69:
        var byte = bytes[at]
        if byte == 46 and not dot:
            dot = True
        elif byte >= 48 and byte <= 57:
            digits += 1
            if dot:
                fraction += 1
        else:
            raise Error("Invalid JSON decimal text")
        at += 1
    if digits == 0:
        raise Error("Invalid JSON decimal text")
    var end = at
    var exponent = 0
    if at < n:
        at += 1
        var sign = 1
        if at < n and (bytes[at] == 43 or bytes[at] == 45):
            sign = -1 if bytes[at] == 45 else 1
            at += 1
        var start = at
        while at < n:
            if bytes[at] < 48 or bytes[at] > 57 or exponent > 10000:
                raise Error("Invalid JSON decimal exponent")
            exponent = exponent * 10 + Int(bytes[at]) - 48
            at += 1
        if at == start:
            raise Error("Invalid JSON decimal exponent")
        exponent *= sign
    var keep = digits + exponent + dtype.scale() - fraction
    var value = Int128(0)
    var position = 0
    var round = 0
    var following = False
    for i in range(first, end):
        if bytes[i] == 46:
            continue
        var digit = Int(bytes[i]) - 48
        if position < keep:
            if value > (pow10(dtype.precision()) - 1 - Int128(digit)) // 10:
                raise Error("JSON decimal precision overflow")
            value = value * 10 + Int128(digit)
        elif position == keep:
            round = digit
        elif digit != 0:
            following = True
        position += 1
    if keep > digits and value != 0:
        if keep - digits > dtype.precision():
            raise Error("JSON decimal precision overflow")
        value *= pow10(keep - digits)
    if keep >= 0 and (
        round > 5 or (round == 5 and (following or value % 2 != 0))
    ):
        value += 1
    return check_precision(-value if negative else value, dtype)


def _decimal_value(
    reader: JsonReader, index: Int, dtype: DataType
) raises -> Int128:
    var kind = reader.nodes[index].kind
    if kind == 34:
        var text = reader.string_at(reader.nodes[index].start)
        return _decimal_text(text, dtype)
    if not _numeric(kind):
        raise Error("JSON value is not decimal")
    var raw = reader.raw(index)
    if not _fraction(raw):
        # Exact integer JSON values keep their integral precision.
        var result = _int128(raw)
        if result > (pow10(dtype.precision()) - 1) // pow10(
            dtype.scale()
        ) or result < -(pow10(dtype.precision()) - 1) // pow10(dtype.scale()):
            raise Error("JSON decimal precision overflow")
        return check_precision(result * pow10(dtype.scale()), dtype)
    var scaled = parse_float64(raw) * Float64(pow10(dtype.scale()))
    if abs(scaled) >= Float64(pow10(dtype.precision())):
        raise Error("JSON decimal precision overflow")
    var lower = floor(scaled)
    var rounded = lower
    if scaled - lower > 0.5 or (
        scaled - lower == 0.5 and Int128(lower) % 2 != 0
    ):
        rounded += 1
    return check_precision(Int128(rounded), dtype)


def _append_null(mut nodes: List[_Builder], index: Int) raises:
    var kind = nodes[index].kind
    if kind == 2 or kind == 10:
        nodes[index].strings.append_null()
    elif kind == 3:
        nodes[index].offsets.append(
            Int64(nodes[nodes[index].children[0]].length)
        )
    elif kind == 4:
        for k in range(len(nodes[index].children)):
            _append_null(nodes, nodes[index].children[k])
    elif kind == 1:
        if nodes[index].length % 8 == 0:
            nodes[index].bools.append(0)
    else:
        comptime for D in SCALAR_DTYPES:
            if nodes[index].numbers.isa[List[Scalar[D]]]():
                nodes[index].numbers[List[Scalar[D]]].append(Scalar[D](0))
    nodes[index].mark(False)


def _append(
    mut nodes: List[_Builder], index: Int, reader: JsonReader, value: Int
) raises:
    if value < 0 or reader.nodes[value].kind == 110:
        _append_null(nodes, index)
        return
    var kind = nodes[index].kind
    var source = reader.nodes[value].kind
    if kind == 11:
        raise Error("JSON value is not null")
    if kind == 4:
        if source != 123:
            raise Error("JSON value is not a struct object")
        var child = value + 1
        while child < reader.nodes[value].after:
            var hash = _key_hash(reader, child)
            if hash in nodes[index].fields:
                var field = nodes[index].fields[hash]
                while field >= 0:
                    if _key_matches(reader, child, nodes[field].name):
                        if nodes[field].length == nodes[index].length:
                            _append(nodes, field, reader, child)
                        break
                    field = nodes[field].next_collision
            child = reader.nodes[child].after
        for k in range(len(nodes[index].children)):
            var field = nodes[index].children[k]
            if nodes[field].length == nodes[index].length:
                _append_null(nodes, field)
    elif kind == 3:
        var field = nodes[index].children[0]
        if source == 91:
            var child = value + 1
            while child < reader.nodes[value].after:
                _append(nodes, field, reader, child)
                child = reader.nodes[child].after
        else:
            _append(nodes, field, reader, value)
        nodes[index].offsets.append(Int64(nodes[field].length))
    elif kind == 2 or kind == 10:
        if source == 34:
            reader.append_string(
                reader.nodes[value].start, nodes[index].strings._bytes
            )
        elif source == 116 or source == 102:
            nodes[index].strings._bytes.extend(reader.raw(value).as_bytes())
        elif _numeric(source):
            var text = json_number_text(reader, value)
            nodes[index].strings._bytes.extend(text.as_bytes())
        else:
            raise Error("JSON value is not a string scalar")
        nodes[index].strings._offsets.append(
            Int64(len(nodes[index].strings._bytes))
        )
        nodes[index].strings._push_bit(True)
    elif kind == 1:
        if source != 116 and source != 102:
            raise Error("JSON value is not boolean")
        if nodes[index].length % 8 == 0:
            nodes[index].bools.append(0)
        if source == 116:
            nodes[index].bools[nodes[index].length // 8] |= UInt8(1) << UInt8(
                nodes[index].length % 8
            )
    elif kind == 9:
        nodes[index].numbers[List[Int128]].append(
            _decimal_value(reader, value, nodes[index].dtype)
        )
    elif kind >= 5 and kind <= 8:
        var result = Int64(0)
        var valid = True
        if source == 34 and kind != 7:
            var text = reader.string_at(reader.nodes[value].start)
            try:
                if kind == 6 and (
                    (text.find("T") < 0 and text.find(" ") < 0)
                    or (
                        nodes[index].dtype.time_zone().byte_length() != 0
                        and not has_offset(text)
                    )
                ):
                    valid = False
                else:
                    if (
                        kind == 6
                        and nodes[index].dtype.time_zone().byte_length() == 0
                    ):
                        # A naive JSON timestamp keeps the written wall time,
                        # including when its text has an explicit offset.
                        var time = max(text.find("T"), text.find(" "))
                        var end = text.byte_length()
                        var bytes = text.as_bytes()
                        for at in range(time + 1, end):
                            if (
                                bytes[at] == 90
                                or bytes[at] == 43
                                or bytes[at] == 45
                            ):
                                end = at
                                break
                        var local = String(text[byte=0:end])
                        text = local^
                    result = parse_in(
                        text,
                        nodes[index].dtype,
                        "",
                        nodes[index].zone.value(),
                    )
            except:
                valid = False
        elif kind == 6 and source == 45 and not _fraction(reader.raw(value)):
            # Polars accepts the signed Int64 JSON representation of a
            # timestamp; unsigned and fractional JSON numbers are errors.
            result = parse_integer[DType.int64](reader.raw(value))
        elif (
            _numeric(source) or (kind == 7 and (source == 116 or source == 102))
        ) and kind != 6:
            var number = _number[DType.int64](reader, value, kind == 7)
            result = number[0]
            valid = number[1]
            if kind == 5:
                valid = valid and result >= -2147483648 and result <= 2147483647
            elif kind == 8:
                valid = valid and result >= 0 and result < 86400000000000
        else:
            raise Error("JSON value has incompatible temporal type")
        if not valid:
            _append_null(nodes, index)
            return
        nodes[index].numbers[List[Int64]].append(result)
    else:
        var valid = False
        comptime for D in SCALAR_DTYPES:
            if nodes[index].numbers.isa[List[Scalar[D]]]():
                var number = _number[D](reader, value)
                nodes[index].numbers[List[Scalar[D]]].append(number[0])
                valid = number[1]
        nodes[index].mark(valid)
        return
    nodes[index].mark(True)


def _finish(mut nodes: List[_Builder], index: Int) raises -> Series:
    var kind = nodes[index].kind
    var dtype = nodes[index].dtype
    if kind == 2 or kind == 10:
        var text = StringBuilder(0)
        swap(text, nodes[index].strings)
        var result = Series("", text^.finish())
        return result.cast(dtype) if kind == 10 else result^
    if kind == 3:
        var child = _finish(nodes, nodes[index].children[0])
        return Series(
            "",
            ListColumn(
                _take(nodes[index].offsets), child^, _take(nodes[index].bits)
            ),
        )
    if kind == 4:
        var fields = List[Series]()
        for k in range(len(nodes[index].children)):
            var child = nodes[index].children[k]
            fields.append(_finish(nodes, child).renamed(nodes[child].name))
        return Series(
            "",
            StructColumn(
                fields^, _take(nodes[index].bits), length=nodes[index].length
            ),
        )
    if kind == 1:
        return Series(
            "",
            BoolColumn(
                values=_take(nodes[index].bools),
                bits=_take(nodes[index].bits),
                length=nodes[index].length,
            ),
        )
    if kind == 9:
        var values = _take(nodes[index].numbers[List[Int128]])
        if dtype.decimal_width() == 128:
            return Series(
                "",
                Column[Int128](values=values^, bits=_take(nodes[index].bits)),
            ).with_dtype(dtype)
        if dtype.decimal_width() == 64:
            var narrowed = List[Int64](capacity=len(values))
            for value in values:
                narrowed.append(value.cast[DType.int64]())
            return Series(
                "",
                Column[Int64](values=narrowed^, bits=_take(nodes[index].bits)),
            ).with_dtype(dtype)
        var narrowed = List[Int32](capacity=len(values))
        for value in values:
            narrowed.append(value.cast[DType.int32]())
        return Series(
            "", Column[Int32](values=narrowed^, bits=_take(nodes[index].bits))
        ).with_dtype(dtype)
    comptime for D in SCALAR_DTYPES:
        if nodes[index].numbers.isa[List[Scalar[D]]]():
            return Series(
                "",
                Column[Scalar[D]](
                    values=_take(nodes[index].numbers[List[Scalar[D]]]),
                    bits=_take(nodes[index].bits),
                ),
            ).with_dtype(dtype)
    raise Error("Invalid JSON output builder")


def decode_json(input: Series, dtype: DataType) raises -> Series:
    validate_json_dtype(dtype)
    var builders = List[_Builder]()
    _ = _compile(builders, dtype, "", len(input))
    # The scanner only borrows each source row until that row's values have
    # been appended. Output buffers own their data; no source substring DOM.
    var empty = String("")
    var reader = JsonReader(
        JsonBytes(
            unsafe_ptr=empty.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
            length=0,
        )
    )
    var base = 0
    for chunk in input.chunks():
        ref column = chunk._data[StringColumn]
        var rows = _Bytes(column)
        var nulls = column.null_count() != 0
        for row in range(len(column)):
            if nulls and not column._valid(row):
                _append_null(builders, 0)
                continue
            try:
                reader.parse(rows.get(row))
                validate_json_numbers(reader)
                _append(builders, 0, reader, 0)
            except e:
                raise Error(
                    "str.json_decode at row "
                    + String(base + row)
                    + ": "
                    + String(e)
                )
        base += len(column)
        _ = rows^
    return _finish(builders, 0)
