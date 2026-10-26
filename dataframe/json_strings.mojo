"""JSONPath String kernel; compile selectors once and reuse row parse storage."""
from .json_path import JsonPath, append_json
from .json_reader import JsonReader, json_bytes
from .json_number import validate_json_numbers
from .series import Series
from .string_column import StringColumn, StringBuilder
from .string_bytes import _Bytes


def json_path_match(input: Series, text: String) raises -> Series:
    var path = JsonPath(text)
    var out = StringBuilder(len(input), len(input) * 8)
    var empty = String("")
    var reader = JsonReader(json_bytes(empty))
    var matches = List[Int]()
    for chunk in input.chunks():
        ref column = chunk._data[StringColumn]
        var rows = _Bytes(column)
        var nulls = column.null_count() != 0
        for row in range(len(column)):
            if nulls and not column._valid(row):
                out.append_null()
                continue
            try:
                reader.parse(rows.get(row))
                validate_json_numbers(reader, False)
            except:
                out.append_null()
                continue
            var selected = path.first(reader, matches)
            if selected < 0 or reader.nodes[selected].kind == 110:
                out.append_null()
                continue
            if reader.nodes[selected].kind == 34:
                reader.append_string(reader.nodes[selected].start, out._bytes)
            else:
                append_json(reader, selected, out._bytes)
            out._offsets.append(Int64(len(out._bytes)))
            out._push_bit(True)
        _ = rows^
    return Series("", out^.finish())
