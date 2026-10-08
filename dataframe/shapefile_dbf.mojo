"""DBASE III/IV attributes and live record positions for shapefiles."""
from .shapefile_binary import read_bytes, require, uint, text, encoding_name
from .series import Series
from .column import Column
from .bool_column import BoolColumn
from .dtype import DataType
from .parse import parse_integer, parse_float64
from .temporal import parse


def read_dbf(
    path: String, encoding: String, expected: Int
) raises -> Tuple[List[Series], List[Int]]:
    var codec = encoding_name(encoding)
    var data = read_bytes(path)
    require(data, 0, 32)
    if data[0] != 3 and data[0] != 4 and data[0] != 131 and data[0] != 139:
        raise Error("Unsupported dBASE version: " + String(data[0]))
    var count = uint(data, 4, 4)
    var header = uint(data, 8, 2)
    var width = uint(data, 10, 2)
    if count != expected:
        raise Error(
            "DBF record count differs from SHP: "
            + String(count)
            + " vs "
            + String(expected)
        )
    if header < 33 or (header - 33) % 32 != 0 or width < 1:
        raise Error("Invalid DBF header or record length")
    require(data, 0, header)
    if data[header - 1] != 13:
        raise Error("Invalid DBF field terminator")
    require(data, header, count * width)
    var end = header + count * width
    if len(data) != end and not (len(data) == end + 1 and data[end] == 26):
        raise Error("Unexpected trailing DBF data")
    var rows = List[Int]()
    for row in range(count):
        var flag = data[header + row * width]
        if flag == 32:
            rows.append(row)
        elif flag != 42:
            raise Error(
                "Invalid DBF deletion marker at record " + String(row + 1)
            )
    var columns = List[Series]()
    var names = List[String]()
    var offset = 1
    for field in range((header - 33) // 32):
        var at = 32 + field * 32
        var name_size = 0
        while name_size < 11 and data[at + name_size] != 0:
            name_size += 1
        var name = text(data, at, name_size, codec)
        if name == "" or name in names:
            raise Error("Empty or duplicate DBF field name: " + name)
        names.append(name)
        var kind = data[at + 11]
        var length = Int(data[at + 16])
        var decimals = Int(data[at + 17])
        if length == 0 or offset + length > width:
            raise Error("Invalid DBF field width: " + name)
        if (
            kind != 67
            and kind != 78
            and kind != 70
            and kind != 68
            and kind != 76
        ):
            raise Error(
                "Unsupported DBF field type for " + name + ": " + String(kind)
            )
        if (kind == 68 and length != 8) or (kind == 76 and length != 1):
            raise Error("Invalid date/logical DBF field width: " + name)
        var strings = List[String]()
        var ints = List[Int64]()
        var floats = List[Float64]()
        var bools = List[Bool]()
        var valid = List[Bool]()
        for row in rows:
            try:
                var value = text(
                    data,
                    header + row * width + offset,
                    length,
                    codec if kind == 67 else "ascii",
                )
                if kind == 67:
                    var stripped = String(value.rstrip(" \x00"))
                    value = stripped^
                    strings.append(value)
                    valid.append(value != "")
                    continue
                var trimmed = String(value.strip())
                value = trimmed^
                var missing = value == ""
                if kind == 68:
                    missing = missing or value == "00000000"
                    ints.append(
                        0 if missing else parse(value, DataType.DATE, "%Y%m%d")
                    )
                elif kind == 76:
                    value = value.upper()
                    missing = missing or value == "?"
                    if not missing and value not in ["T", "F", "Y", "N"]:
                        raise Error("invalid logical value")
                    bools.append(value == "T" or value == "Y")
                else:
                    var stars = value != ""
                    for b in value.as_bytes():
                        stars = stars and b == 42
                    missing = missing or stars
                    if kind == 78 and decimals == 0:
                        ints.append(
                            0 if missing else parse_integer[DType.int64](
                                StringSlice(value)
                            )
                        )
                    else:
                        floats.append(0 if missing else parse_float64(value))
                valid.append(not missing)
            except e:
                raise Error(
                    "DBF record "
                    + String(row + 1)
                    + ", field '"
                    + name
                    + "': "
                    + String(e)
                )
        if kind == 67:
            columns.append(Series(name, Column[String](strings^, valid)))
        elif kind == 76:
            columns.append(Series(name, BoolColumn(bools^, valid)))
        elif kind == 68:
            columns.append(
                Series(name, Column[Int64](ints^, valid)).with_dtype(
                    DataType.DATE
                )
            )
        elif kind == 78 and decimals == 0:
            columns.append(Series(name, Column[Int64](ints^, valid)))
        else:
            columns.append(Series(name, Column[Float64](floats^, valid)))
        offset += length
    if offset != width:
        raise Error("DBF field widths do not match record length")
    return (columns^, rows^)
