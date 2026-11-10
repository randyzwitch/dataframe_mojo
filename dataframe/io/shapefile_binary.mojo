"""Checked binary and character decoding for the shapefile components."""
from dataframe.storage.endian import read_scalar
from std.memory import bitcast
from std.sys import Endian
from dataframe.text.utf8 import append_codepoint


def read_bytes(path: String) raises -> List[UInt8]:
    with open(path, "r") as file:
        return file.read_bytes()


def require(data: List[UInt8], at: Int, size: Int) raises:
    if at < 0 or size < 0 or at > len(data) or size > len(data) - at:
        raise Error("truncated binary data at byte " + String(at))


def uint(
    data: List[UInt8], at: Int, size: Int, big: Bool = False
) raises -> Int:
    require(data, at, size)
    if size == 2:
        return Int(
            read_scalar[DType.uint16, Endian.big](Span(data), at)
        ) if big else (
            Int(read_scalar[DType.uint16, Endian.little](Span(data), at))
        )
    if size == 4:
        return Int(
            read_scalar[DType.uint32, Endian.big](Span(data), at)
        ) if big else (
            Int(read_scalar[DType.uint32, Endian.little](Span(data), at))
        )
    raise Error("Unsupported shapefile integer width")


def number(data: List[UInt8], at: Int) raises -> Float64:
    require(data, at, 8)
    return read_scalar[DType.float64, Endian.little](Span(data), at)


def put_uint(mut out: List[UInt8], value: Int):
    out.extend(UInt32(value).as_bytes[endian=Endian.little]())


def put_number(mut out: List[UInt8], value: Float64):
    out.extend(bitcast[DType.uint64](value).as_bytes[endian=Endian.little]())


def encoding_name(encoding: String) raises -> String:
    var name = encoding.lower()
    if name == "utf-8" or name == "utf8":
        return "utf8"
    if name == "latin1" or name == "latin-1" or name == "iso-8859-1":
        return "latin1"
    if name == "cp1252" or name == "windows-1252":
        return "cp1252"
    if name == "ascii":
        return name
    raise Error(
        "Unsupported DBF encoding: "
        + encoding
        + "; use UTF-8, Latin-1, Windows-1252 or ASCII"
    )


def text(
    data: List[UInt8], at: Int, size: Int, encoding: String
) raises -> String:
    require(data, at, size)
    var bytes = List[UInt8](capacity=size)
    var cp = List[Int](
        [
            8364,
            -1,
            8218,
            402,
            8222,
            8230,
            8224,
            8225,
            710,
            8240,
            352,
            8249,
            338,
            -1,
            381,
            -1,
            -1,
            8216,
            8217,
            8220,
            8221,
            8226,
            8211,
            8212,
            732,
            8482,
            353,
            8250,
            339,
            -1,
            382,
            376,
        ]
    )
    for i in range(size):
        var value = Int(data[at + i])
        if encoding == "ascii" and value > 127:
            raise Error("Invalid ASCII at byte " + String(at + i))
        if encoding == "utf8" or encoding == "ascii":
            bytes.append(UInt8(value))
            continue
        if encoding == "cp1252" and value >= 128 and value < 160:
            value = cp[value - 128]
            if value < 0:
                raise Error("Undefined Windows-1252 byte at " + String(at + i))
        append_codepoint(bytes, value)
    return String(StringSlice(from_utf8=Span(bytes)))
