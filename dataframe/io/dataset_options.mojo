"""Path-only dataset options and Hive metadata; never opens file contents."""
from std.collections import Dict
from dataframe.dtype import DataType
from .file_sources import resolve_paths

comptime HIVE_NULL = "__HIVE_DEFAULT_PARTITION__"


def _signature_part(value: String) -> String:
    return String(value.byte_length()) + ":" + value


def _hex(byte: UInt8) -> Int:
    if byte >= 48 and byte <= 57:
        return Int(byte - 48)
    if byte >= 65 and byte <= 70:
        return Int(byte - 65) + 10
    if byte >= 97 and byte <= 102:
        return Int(byte - 97) + 10
    return -1


def hive_decode(value: String) raises -> String:
    """Decode percent escapes once; reject malformed escapes and UTF-8."""
    var bytes = List[UInt8]()
    var source = value.as_bytes()
    var i = 0
    while i < len(source):
        if source[i] == 37:
            if i + 2 >= len(source):
                raise Error("Invalid Hive percent escape: " + value)
            var high = _hex(source[i + 1])
            var low = _hex(source[i + 2])
            if high < 0 or low < 0:
                raise Error("Invalid Hive percent escape: " + value)
            bytes.append(UInt8(high * 16 + low))
            i += 3
        else:
            bytes.append(source[i])
            i += 1
    return String(StringSlice(from_utf8=Span(bytes)))


def hive_encode(value: String) -> String:
    """Encode path components reversibly, including a literal null sentinel."""
    var digits = "0123456789ABCDEF".as_bytes()
    var bytes = List[UInt8]()
    for byte in value.as_bytes():
        var safe = (
            (byte >= 65 and byte <= 90)
            or (byte >= 97 and byte <= 122)
            or (byte >= 48 and byte <= 57)
            or byte == 45
            or byte == 95
            or byte == 46
            or byte == 126
        )
        if safe and value != HIVE_NULL:
            bytes.append(byte)
        else:
            bytes.append(37)
            bytes.append(digits[Int(byte) // 16])
            bytes.append(digits[Int(byte) % 16])
    return String(unsafe_from_utf8=Span(bytes))


def hive_int64(value: String) raises -> Int64:
    """Strict decimal conversion with explicit signed 64-bit overflow checks."""
    var bytes = value.as_bytes()
    if len(bytes) == 0:
        raise Error("Empty Hive integer")
    var negative = bytes[0] == 45
    var start = 1 if negative or bytes[0] == 43 else 0
    if start == len(bytes):
        raise Error("Invalid Hive integer: " + value)
    var limit = UInt64(9223372036854775808) if negative else UInt64(
        9223372036854775807
    )
    var number = UInt64(0)
    for i in range(start, len(bytes)):
        if bytes[i] < 48 or bytes[i] > 57:
            raise Error("Invalid Hive integer: " + value)
        var digit = UInt64(bytes[i] - 48)
        if number > (limit - digit) // 10:
            raise Error("Hive integer overflow: " + value)
        number = number * 10 + digit
    if negative:
        if number == UInt64(9223372036854775808):
            return Int64.MIN
        return -Int64(number)
    return Int64(number)


struct DatasetOptions(Copyable):
    var hive_partitioning: Bool
    var hive_schema: Dict[String, String]
    var include_file_paths: String
    var missing_columns: String
    var extra_columns: String
    var union_by_name: Bool
    var glob: Bool

    def __init__(
        out self,
        *,
        hive_partitioning: Bool = False,
        hive_schema: Dict[String, String] = Dict[String, String](),
        include_file_paths: String = "",
        missing_columns: String = "raise",
        extra_columns: String = "raise",
        union_by_name: Bool = False,
        glob: Bool = True,
    ) raises:
        if missing_columns != "raise" and missing_columns != "insert":
            raise Error("missing_columns must be 'raise' or 'insert'")
        if extra_columns != "raise" and extra_columns != "ignore":
            raise Error("extra_columns must be 'raise' or 'ignore'")
        for field in hive_schema.items():
            if field.key == "" or not DataType.is_known(field.value):
                raise Error(
                    "Invalid hive_schema field: "
                    + field.key
                    + ": "
                    + field.value
                )
        self.hive_partitioning = hive_partitioning
        self.hive_schema = hive_schema.copy()
        self.include_file_paths = include_file_paths
        self.missing_columns = missing_columns
        self.extra_columns = extra_columns
        self.union_by_name = union_by_name
        self.glob = glob

    def active(self) -> Bool:
        return (
            self.hive_partitioning
            or len(self.hive_schema) > 0
            or self.include_file_paths != ""
            or self.missing_columns != "raise"
            or self.extra_columns != "raise"
            or self.union_by_name
        )


struct DatasetSpec(Copyable):
    var paths: List[String]
    var options: DatasetOptions
    var hive_names: List[String]
    var hive_dtypes: List[DataType]
    var hive_values: List[List[String]]
    var hive_nulls: List[List[Bool]]

    def __init__(out self, paths: List[String], options: DatasetOptions) raises:
        self.paths = resolve_paths(paths, options.glob)
        self.options = options.copy()
        self.hive_names = List[String]()
        self.hive_dtypes = List[DataType]()
        self.hive_values = List[List[String]]()
        self.hive_nulls = List[List[Bool]]()
        var values = List[Dict[String, String]]()
        var nulls = List[Dict[String, Bool]]()
        for path in self.paths:
            var row = Dict[String, String]()
            var missing = Dict[String, Bool]()
            if options.hive_partitioning:
                var parts = path.split("/")
                for p in range(max(0, len(parts) - 1)):
                    var part = String(parts[p])
                    var separator = -1
                    for b in range(part.byte_length()):
                        if part.as_bytes()[b] == 61:
                            separator = b
                            break
                    if separator < 0:
                        continue
                    var key = hive_decode(String(part[byte=0:separator]))
                    var raw = String(
                        part[byte = separator + 1 : part.byte_length()]
                    )
                    if key == "" or key in row:
                        raise Error(
                            "Invalid or duplicate Hive partition key: " + key
                        )
                    if key not in self.hive_names:
                        self.hive_names.append(key)
                    missing[key] = raw == HIVE_NULL
                    row[key] = "" if raw == HIVE_NULL else hive_decode(raw)
            values.append(row^)
            nulls.append(missing^)
        for key in self.hive_names:
            var dtype = DataType.INT64
            var seen = False
            for f in range(len(values)):
                if key not in values[f] or nulls[f][key]:
                    continue
                seen = True
                var value = values[f][key]
                try:
                    _ = hive_int64(value)
                except:
                    dtype = DataType.STRING
            if not seen:
                dtype = DataType.STRING
            if key in options.hive_schema:
                dtype = DataType.parse(options.hive_schema[key])
            self.hive_dtypes.append(dtype)
        for f in range(len(values)):
            var row = List[String]()
            var missing = List[Bool]()
            for k in range(len(self.hive_names)):
                var key = self.hive_names[k]
                var absent = key not in values[f]
                row.append("" if absent else values[f][key])
                missing.append(True if absent else nulls[f][key])
                if not missing[k] and self.hive_dtypes[k] == DataType.INT64:
                    _ = hive_int64(row[k])
            self.hive_values.append(row^)
            self.hive_nulls.append(missing^)

    def active(self) -> Bool:
        return self.options.active()

    def hive_index(self, name: String) -> Int:
        for k in range(len(self.hive_names)):
            if self.hive_names[k] == name:
                return k
        return -1

    def signature(self) -> String:
        """Unambiguous stable identity, independent of override insertion order."""
        var result = _signature_part("dataset-v1")
        result += _signature_part(String(len(self.paths)))
        for path in self.paths:
            result += _signature_part(path)
        result += _signature_part(String(self.options.hive_partitioning))
        result += _signature_part(self.options.include_file_paths)
        result += _signature_part(self.options.missing_columns)
        result += _signature_part(self.options.extra_columns)
        result += _signature_part(String(self.options.union_by_name))
        result += _signature_part(String(self.options.glob))
        var overrides = List[String]()
        for field in self.options.hive_schema.items():
            overrides.append(field.key)
        sort(overrides)
        result += _signature_part(String(len(overrides)))
        for key in overrides:
            result += _signature_part(key)
            for field in self.options.hive_schema.items():
                if field.key == key:
                    result += _signature_part(field.value)
                    break
        result += _signature_part(String(len(self.hive_names)))
        for k in range(len(self.hive_names)):
            result += _signature_part(self.hive_names[k])
            result += _signature_part(self.hive_dtypes[k].name())
        result += _signature_part(String(len(self.hive_values)))
        for f in range(len(self.hive_values)):
            result += _signature_part(String(len(self.hive_values[f])))
            for value in self.hive_values[f]:
                result += _signature_part(value)
            result += _signature_part(String(len(self.hive_nulls[f])))
            for absent in self.hive_nulls[f]:
                result += _signature_part(String(absent))
        return result^
