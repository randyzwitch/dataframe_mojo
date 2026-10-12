"""Independent CSV field grammar and producer, with no dataframe imports.

This deliberately simple state machine consumes well-formed CSV emitted by a
writer. Reader error tests use literal malformed fixtures, not this parser.
Quotedness is retained because quoted empty text and bare null differ.
"""
from std.ffi import external_call
from std.memory import Pointer
from tests.support.reference_model import ReferenceTable


@fieldwise_init
struct CsvCell(Copyable, Movable):
    var text: String
    var quoted: Bool


def csv_records(
    text: String, separator: UInt8 = 44
) raises -> List[List[CsvCell]]:
    var records = List[List[CsvCell]]()
    var record = List[CsvCell]()
    var bytes = List[UInt8]()
    var quoted = False
    var inside = False
    var closed = False
    var started = False
    var at = 0
    while at < text.byte_length():
        var b = text.as_bytes()[at]
        at += 1
        if inside:
            if b == 34:
                if at < text.byte_length() and text.as_bytes()[at] == 34:
                    bytes.append(34)
                    at += 1
                else:
                    inside = False
                    closed = True
            else:
                bytes.append(b)
            continue
        if b == separator or b == 10 or b == 13:
            record.append(CsvCell(String(unsafe_from_utf8=bytes^), quoted))
            bytes = []
            quoted = False
            closed = False
            started = False
            if b != separator:
                records.append(record^)
                record = []
                if (
                    b == 13
                    and at < text.byte_length()
                    and text.as_bytes()[at] == 10
                ):
                    at += 1
            continue
        if closed:
            raise Error("Reference CSV has text after closing quote")
        if b == 34:
            if started:
                raise Error("Reference CSV quote inside bare field")
            quoted = True
            inside = True
            started = True
        else:
            started = True
            bytes.append(b)
    if inside:
        raise Error("Reference CSV unterminated quoted field")
    if started or record:
        record.append(CsvCell(String(unsafe_from_utf8=bytes^), quoted))
        records.append(record^)
    return records^


def csv_escape(
    text: String, separator: String = ",", force: Bool = False
) -> String:
    var quote = force or not text
    for byte in text.as_bytes():
        quote = (
            quote
            or byte == 34
            or byte == 10
            or byte == 13
            or byte == separator.as_bytes()[0]
        )
    return '"' + text.replace('"', '""') + '"' if quote else text.copy()


def csv_produce_model(
    model: ReferenceTable,
    separator: String = ",",
    newline: String = "\n",
    header: Bool = True,
) raises -> String:
    var result = String(
        "i" + separator + "f" + separator + "s" + separator + "b" + newline
    ) if header else String()
    for row in model.rows:
        result += String(row.i) if row.iv else ""
        result += separator
        result += String(row.f) if row.fv else ""
        result += separator
        result += csv_escape(row.s, separator) if row.sv else ""
        result += separator
        result += ("true" if row.b else "false") if row.bv else ""
        result += newline
    return result^


def csv_float(text: String) raises -> Float64:
    """Independent libc consumer, requiring full field consumption."""
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    bytes.append(0)
    var end = Int(0)
    var start = Int(bytes.unsafe_ptr())
    var value = external_call["strtod", Float64](start, Int(Pointer(to=end)))
    _ = bytes^
    if end != start + text.byte_length() or not text:
        raise Error("Reference numeric field was not fully consumed")
    return value
