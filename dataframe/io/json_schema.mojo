"""Sample JSON objects into one ordered, recursively promoted schema."""
from ..dtype import DataType
from dataframe.json.reader import JsonReader
from dataframe.json.number import (
    is_json_number,
    is_json_fraction,
    validate_json_numbers,
)
from dataframe.text.parse import parse_int64


def _scalar(dtype: DataType) -> Bool:
    return (
        dtype == DataType.STRING or dtype == DataType.BOOL or dtype.is_numeric()
    )


def merge_json_types(
    left: DataType, right: DataType, depth: Int = 0
) raises -> DataType:
    """Union object fields, promote numbers, and stringify mixed scalars.

    Field order is the first occurrence in the sample. Null contributes no
    type information; an entirely null field retains the Null dtype.
    """
    if depth > 64:
        raise Error("JSON schema nesting exceeds 64")
    if left == DataType.NULL:
        return right
    if right == DataType.NULL or left == right:
        return left
    if left.is_struct() and right.is_struct():
        var names = left.field_names()
        var types = left.field_dtypes()
        var positions = Dict[String, Int]()
        for i in range(len(names)):
            positions[names[i]] = i
        var other_names = right.field_names()
        var other_types = right.field_dtypes()
        for i in range(len(other_names)):
            var name = other_names[i].copy()
            if name in positions:
                var at = positions[name]
                types[at] = merge_json_types(
                    types[at], other_types[i], depth + 1
                )
            else:
                positions[name] = len(names)
                names.append(name^)
                types.append(other_types[i])
        return DataType.struct(names, types)
    if left.is_list() and right.is_list():
        return DataType.list(
            merge_json_types(left.inner(), right.inner(), depth + 1)
        )
    if left.is_numeric() and right.is_numeric():
        return DataType.FLOAT64
    if _scalar(left) and _scalar(right):
        return DataType.STRING
    raise Error(
        "Cannot infer one JSON type from "
        + left.name()
        + " and "
        + right.name()
    )


def infer_json_value(
    reader: JsonReader, index: Int, depth: Int = 0
) raises -> DataType:
    """Infer from the shared parser's tape without constructing value objects."""
    if depth > 64:
        raise Error("JSON schema nesting exceeds 64")
    ref node = reader.nodes[index]
    var kind = node.kind
    if kind == 110:
        return DataType.NULL
    if kind == 116 or kind == 102:
        return DataType.BOOL
    if kind == 34:
        return DataType.STRING
    if is_json_number(kind):
        var text = reader.raw(index)
        if not is_json_fraction(text):
            try:
                _ = parse_int64(text)
                return DataType.INT64
            except:
                pass
        return DataType.FLOAT64
    if kind == 91:
        var inner = DataType.NULL
        var child = index + 1
        while child < node.after:
            inner = merge_json_types(
                inner, infer_json_value(reader, child, depth + 1), depth + 1
            )
            child = reader.nodes[child].after
        return DataType.list(inner)
    if kind == 123:
        var names = List[String]()
        var types = List[DataType]()
        var seen = Dict[String, Bool]()
        var child = index + 1
        while child < node.after:
            var name = reader.string_at(reader.nodes[child].key_start)
            # The shared typed decoder retains the first duplicate field.
            if name not in seen:
                seen[name] = True
                names.append(name^)
                types.append(infer_json_value(reader, child, depth + 1))
            child = reader.nodes[child].after
        return DataType.struct(names, types)
    raise Error("Invalid JSON value while inferring schema")


struct JsonSchema(Movable):
    var dtype: DataType
    var rows: Int

    def __init__(out self):
        self.dtype = DataType.NULL
        self.rows = 0

    def observe(mut self, reader: JsonReader, index: Int = 0) raises:
        if reader.nodes[index].kind != 123:
            raise Error("JSON rows must contain objects")
        validate_json_numbers(reader)
        self.dtype = merge_json_types(
            self.dtype, infer_json_value(reader, index)
        )
        self.rows += 1

    def finish(self) raises -> DataType:
        if self.rows == 0:
            raise Error("Cannot infer JSON schema from empty input")
        return self.dtype
