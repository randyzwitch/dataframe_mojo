"""Strict, explicit-schema CSV ingestion for native Mojo dataframes.

The scalar tokenizer is deliberately separate from typed column decoding. It
keeps state across input buffers, including quoted records, so later SIMD
structural scanning or parallel decoding can replace one stage at a time.
"""
from std.collections import Dict
from std.utils import Variant

from .column import Column
from .frame import DataFrame
from .series import Series
from .parse import parse_int64, parse_float64


comptime CSV_INT64 = 0
comptime CSV_FLOAT64 = 1
comptime CSV_BOOL = 2
comptime CSV_STRING = 3


@fieldwise_init
struct CsvField(Copyable):
    """One CSV output field. Names and dtypes are always explicit."""

    var name: String
    var dtype: Int
    var nullable: Bool

    @staticmethod
    def int64(name: String, nullable: Bool = True) -> CsvField:
        return CsvField(name, CSV_INT64, nullable)

    @staticmethod
    def float64(name: String, nullable: Bool = True) -> CsvField:
        return CsvField(name, CSV_FLOAT64, nullable)

    @staticmethod
    def bool(name: String, nullable: Bool = True) -> CsvField:
        return CsvField(name, CSV_BOOL, nullable)

    @staticmethod
    def string(name: String, nullable: Bool = True) -> CsvField:
        return CsvField(name, CSV_STRING, nullable)


struct CsvSchema(Copyable, Sized):
    """An ordered, non-empty set of uniquely named CSV fields."""

    var _fields: List[CsvField]

    def __init__(out self, var fields: List[CsvField]) raises:
        if len(fields) == 0:
            raise Error("CSV schema must contain at least one field")
        var names = Dict[String, Bool]()
        for field in fields:
            if field.name in names:
                raise Error("Duplicate CSV field name: " + field.name)
            if field.dtype < CSV_INT64 or field.dtype > CSV_STRING:
                raise Error("Unknown CSV dtype for field: " + field.name)
            names[field.name] = True
        self._fields = fields^

    def __len__(self) -> Int:
        return len(self._fields)

    @staticmethod
    def of(frame: DataFrame) raises -> CsvSchema:
        """A nullable schema matching a frame's names and dtypes."""
        var fields = List[CsvField](capacity=frame.width())
        for field in frame.schema():
            if field.dtype == "int64":
                fields.append(CsvField.int64(field.name))
            elif field.dtype == "float64":
                fields.append(CsvField.float64(field.name))
            elif field.dtype == "bool":
                fields.append(CsvField.bool(field.name))
            else:
                fields.append(CsvField.string(field.name))
        return CsvSchema(fields^)

    def field(self, index: Int) raises -> CsvField:
        if index < 0 or index >= len(self):
            raise Error("CSV schema index out of bounds")
        return self._fields[index].copy()


@fieldwise_init
struct _IntBuilder(Copyable):
    var values: List[Int64]
    var valid: List[Bool]


@fieldwise_init
struct _FloatBuilder(Copyable):
    var values: List[Float64]
    var valid: List[Bool]


@fieldwise_init
struct _BoolBuilder(Copyable):
    var values: List[Bool]
    var valid: List[Bool]


@fieldwise_init
struct _StringBuilder(Copyable):
    var values: List[String]
    var valid: List[Bool]


comptime _Builder = Variant[
    _IntBuilder, _FloatBuilder, _BoolBuilder, _StringBuilder
]


struct _CsvColumn(Copyable):
    var field: CsvField
    var builder: _Builder

    def __init__(out self, field: CsvField):
        self.field = field.copy()
        if field.dtype == CSV_INT64:
            self.builder = _Builder(_IntBuilder([], []))
        elif field.dtype == CSV_FLOAT64:
            self.builder = _Builder(_FloatBuilder([], []))
        elif field.dtype == CSV_BOOL:
            self.builder = _Builder(_BoolBuilder([], []))
        else:
            self.builder = _Builder(_StringBuilder([], []))

    def _error(self, record: Int, text: String) -> Error:
        return Error(
            String(
                "CSV record ",
                record,
                ", field '",
                self.field.name,
                "': ",
                text,
            )
        )

    def append(mut self, text: String, quoted: Bool, record: Int) raises:
        # Null markers only match unquoted fields. Consequently `,"",` is a
        # valid empty string while `,,` is null with the default marker.
        if not quoted and text == "":
            if not self.field.nullable:
                raise self._error(record, "null in a non-nullable field")
            if self.builder.isa[_IntBuilder]():
                self.builder[_IntBuilder].values.append(0)
                self.builder[_IntBuilder].valid.append(False)
            elif self.builder.isa[_FloatBuilder]():
                self.builder[_FloatBuilder].values.append(0)
                self.builder[_FloatBuilder].valid.append(False)
            elif self.builder.isa[_BoolBuilder]():
                self.builder[_BoolBuilder].values.append(False)
                self.builder[_BoolBuilder].valid.append(False)
            else:
                self.builder[_StringBuilder].values.append("")
                self.builder[_StringBuilder].valid.append(False)
            return

        if self.builder.isa[_IntBuilder]():
            try:
                self.builder[_IntBuilder].values.append(parse_int64(text))
            except:
                raise self._error(record, "invalid Int64 value '" + text + "'")
            self.builder[_IntBuilder].valid.append(True)
        elif self.builder.isa[_FloatBuilder]():
            try:
                self.builder[_FloatBuilder].values.append(parse_float64(text))
            except e:
                raise self._error(record, String(e))
            self.builder[_FloatBuilder].valid.append(True)
        elif self.builder.isa[_BoolBuilder]():
            if text != "true" and text != "false":
                raise self._error(
                    record, "Boolean must be exactly 'true' or 'false'"
                )
            self.builder[_BoolBuilder].values.append(text == "true")
            self.builder[_BoolBuilder].valid.append(True)
        else:
            self.builder[_StringBuilder].values.append(text)
            self.builder[_StringBuilder].valid.append(True)

    def finish(self) raises -> Series:
        if self.builder.isa[_IntBuilder]():
            return Series(
                self.field.name,
                Column[Int64](
                    self.builder[_IntBuilder].values.copy(),
                    self.builder[_IntBuilder].valid.copy(),
                ),
            )
        if self.builder.isa[_FloatBuilder]():
            return Series(
                self.field.name,
                Column[Float64](
                    self.builder[_FloatBuilder].values.copy(),
                    self.builder[_FloatBuilder].valid.copy(),
                ),
            )
        if self.builder.isa[_BoolBuilder]():
            return Series(
                self.field.name,
                Column[Bool](
                    self.builder[_BoolBuilder].values.copy(),
                    self.builder[_BoolBuilder].valid.copy(),
                ),
            )
        return Series(
            self.field.name,
            Column[String](
                self.builder[_StringBuilder].values.copy(),
                self.builder[_StringBuilder].valid.copy(),
            ),
        )


struct _CsvReader:
    var schema: CsvSchema
    var columns: List[_CsvColumn]
    var has_header: Bool
    var prefix: List[UInt8]
    var prefix_done: Bool
    var field_bytes: List[UInt8]
    var header_fields: List[String]
    var field_index: Int
    var record: Int
    var physical_line: Int
    var in_quotes: Bool
    var after_quote: Bool
    var field_quoted: Bool
    var field_started: Bool
    var record_open: Bool
    var pending_cr: Bool

    def __init__(out self, schema: CsvSchema, has_header: Bool):
        self.schema = schema.copy()
        self.columns = List[_CsvColumn](capacity=len(schema))
        for i in range(len(schema)):
            self.columns.append(_CsvColumn(schema._fields[i]))
        self.has_header = has_header
        self.prefix = List[UInt8](capacity=3)
        self.prefix_done = False
        self.field_bytes = List[UInt8]()
        self.header_fields = List[String]()
        self.field_index = 0
        self.record = 1
        self.physical_line = 1
        self.in_quotes = False
        self.after_quote = False
        self.field_quoted = False
        self.field_started = False
        self.record_open = False
        self.pending_cr = False

    def _location(self, text: String) -> Error:
        return Error(
            String(
                "CSV record ",
                self.record,
                ", field ",
                self.field_index + 1,
                ", physical line ",
                self.physical_line,
                ": ",
                text,
            )
        )

    def _finish_field(mut self) raises:
        if self.field_index >= len(self.schema):
            raise self._location("too many fields")
        var text: String
        try:
            text = String(from_utf8=self.field_bytes)
        except:
            raise self._location("field is not valid UTF-8")
        if self.has_header and self.record == 1:
            self.header_fields.append(text^)
        else:
            self.columns[self.field_index].append(
                text^, self.field_quoted, self.record
            )
        self.field_bytes.clear()
        self.field_index += 1
        self.field_quoted = False
        self.field_started = False
        self.after_quote = False

    def _finish_record(mut self) raises:
        self._finish_field()
        if self.field_index != len(self.schema):
            raise self._location(
                String(
                    "expected ",
                    len(self.schema),
                    " fields, found ",
                    self.field_index,
                )
            )
        if self.has_header and self.record == 1:
            for i in range(len(self.schema)):
                if self.header_fields[i] != self.schema._fields[i].name:
                    raise Error(
                        String(
                            "CSV header field ",
                            i + 1,
                            " is '",
                            self.header_fields[i],
                            "'; expected '",
                            self.schema._fields[i].name,
                            "'",
                        )
                    )
        self.field_index = 0
        self.record += 1
        self.record_open = False

    def _consume(mut self, byte: UInt8) raises:
        if self.pending_cr:
            if byte != 10:
                raise self._location(
                    "bare carriage return outside quoted field"
                )
            self.pending_cr = False
            self._finish_record()
            self.physical_line += 1
            return

        self.record_open = True
        if self.in_quotes:
            if self.after_quote:
                if byte == 34:
                    self.field_bytes.append(34)
                    self.after_quote = False
                elif byte == 44:
                    self.in_quotes = False
                    self._finish_field()
                elif byte == 10:
                    self.in_quotes = False
                    self._finish_record()
                    self.physical_line += 1
                elif byte == 13:
                    self.in_quotes = False
                    self.pending_cr = True
                else:
                    raise self._location("unexpected byte after closing quote")
            elif byte == 34:
                self.after_quote = True
            else:
                self.field_bytes.append(byte)
                if byte == 10:
                    self.physical_line += 1
            return

        if byte == 34:
            if self.field_started:
                raise self._location("quote inside an unquoted field")
            self.in_quotes = True
            self.field_quoted = True
            self.field_started = True
        elif byte == 44:
            self._finish_field()
        elif byte == 10:
            self._finish_record()
            self.physical_line += 1
        elif byte == 13:
            self.pending_cr = True
        else:
            self.field_started = True
            self.field_bytes.append(byte)

    def feed(mut self, bytes: List[UInt8]) raises:
        for byte in bytes:
            if not self.prefix_done:
                self.prefix.append(byte)
                if len(self.prefix) == 3:
                    if (
                        self.prefix[0] != 239
                        or self.prefix[1] != 187
                        or self.prefix[2] != 191
                    ):
                        for prefix_byte in self.prefix:
                            self._consume(prefix_byte)
                    self.prefix.clear()
                    self.prefix_done = True
                continue
            self._consume(byte)

    def finish(mut self) raises -> DataFrame:
        # Files shorter than three bytes never resolved the optional BOM prefix.
        for byte in self.prefix:
            self._consume(byte)
        self.prefix.clear()
        self.prefix_done = True
        if self.pending_cr:
            raise self._location("bare carriage return at end of file")
        if self.in_quotes:
            if self.after_quote:
                self.in_quotes = False
            else:
                raise self._location("unterminated quoted field")
        if self.record_open:
            self._finish_record()
        var output = List[Series](capacity=len(self.columns))
        for column in self.columns:
            output.append(column.finish())
        return DataFrame(output^)


def read_csv(
    path: String,
    schema: CsvSchema,
    *,
    has_header: Bool = True,
    buffer_size: Int = 65536,
) raises -> DataFrame:
    """Read a strict UTF-8 CSV file into typed, nullable columns.

    Empty unquoted fields are null. Quoted empty strings are values, which are
    therefore only valid in String columns. Whitespace is never trimmed.
    """
    if buffer_size <= 0:
        raise Error("CSV buffer_size must be positive")
    var reader = _CsvReader(schema, has_header)
    with open(path, "r") as file:
        while True:
            var bytes = file.read_bytes(buffer_size)
            if len(bytes) == 0:
                break
            reader.feed(bytes^)
    return reader.finish()


def _needs_quotes(text: String, separator: UInt8, null_value: String) -> Bool:
    """Quote when unquoted output would re-read differently."""
    if text.byte_length() == 0 or text == null_value:
        return True
    for b in text.as_bytes():
        if b == separator or b == 34 or b == 10 or b == 13:
            return True
    return False


def _quoted(text: String) -> String:
    return '"' + text.replace('"', '""') + '"'


def _render_field(
    text: String,
    is_string: Bool,
    style: String,
    separator: UInt8,
    null_value: String,
) -> String:
    if style == "always" or (style == "non_numeric" and is_string):
        return _quoted(text)
    if style == "never":
        return text
    if _needs_quotes(text, separator, null_value):
        return _quoted(text)
    return text


def _cell_text(series: Series, row: Int) -> String:
    """Canonical text for a valid cell; floats use the round-trip form."""
    if series._data.isa[Column[Int64]]():
        return String(series._data[Column[Int64]]._values[row])
    if series._data.isa[Column[Float64]]():
        return String(series._data[Column[Float64]]._values[row])
    if series._data.isa[Column[Bool]]():
        return "true" if series._data[Column[Bool]]._values[row] else "false"
    return series._data[Column[String]]._values[row]


def _cell_valid(series: Series, row: Int) -> Bool:
    if series._data.isa[Column[Int64]]():
        return series._data[Column[Int64]]._valid(row)
    if series._data.isa[Column[Float64]]():
        return series._data[Column[Float64]]._valid(row)
    if series._data.isa[Column[Bool]]():
        return series._data[Column[Bool]]._valid(row)
    return series._data[Column[String]]._valid(row)


struct _CsvWriter:
    var separator: String
    var separator_byte: UInt8
    var quote_style: String
    var null_value: String
    var line_terminator: String

    def __init__(
        out self,
        separator: String,
        quote_style: String,
        null_value: String,
        line_terminator: String,
    ) raises:
        if separator.byte_length() != 1:
            raise Error("CSV separator must be a single byte")
        var byte = separator.as_bytes()[0]
        if byte == 34 or byte == 10 or byte == 13:
            raise Error("CSV separator cannot be a quote, CR, or LF")
        if (
            quote_style != "necessary"
            and quote_style != "always"
            and quote_style != "non_numeric"
            and quote_style != "never"
        ):
            raise Error(
                "quote_style must be necessary, always, non_numeric, or never"
            )
        if line_terminator != "\n" and line_terminator != "\r\n":
            raise Error("line_terminator must be LF or CRLF")
        for b in null_value.as_bytes():
            if b == byte or b == 34 or b == 10 or b == 13:
                raise Error(
                    "null_value cannot contain the separator, quotes, or"
                    " line breaks"
                )
        self.separator = separator
        self.separator_byte = byte
        self.quote_style = quote_style
        self.null_value = null_value
        self.line_terminator = line_terminator

    def header(self, frame: DataFrame) -> String:
        var line = String()
        for c in range(frame.width()):
            if c > 0:
                line += self.separator
            line += _render_field(
                frame._columns[c].name(),
                True,
                self.quote_style,
                self.separator_byte,
                self.null_value,
            )
        return line + self.line_terminator

    def row(self, frame: DataFrame, row: Int) -> String:
        var line = String()
        for c in range(frame.width()):
            if c > 0:
                line += self.separator
            ref column = frame._columns[c]
            if not _cell_valid(column, row):
                line += self.null_value
                continue
            var is_string = (
                column.dtype() == "string" or column.dtype() == "bool"
            )
            line += _render_field(
                _cell_text(column, row),
                is_string,
                self.quote_style,
                self.separator_byte,
                self.null_value,
            )
        return line + self.line_terminator


def to_csv_string(
    frame: DataFrame,
    *,
    has_header: Bool = True,
    separator: String = ",",
    quote_style: String = "necessary",
    null_value: String = "",
    line_terminator: String = "\n",
) raises -> String:
    """Render the whole frame as CSV text; use write_csv for large frames."""
    var writer = _CsvWriter(separator, quote_style, null_value, line_terminator)
    var out = String()
    if has_header:
        out += writer.header(frame)
    for row in range(frame.height()):
        out += writer.row(frame, row)
    return out^


def write_csv(
    frame: DataFrame,
    path: String,
    *,
    has_header: Bool = True,
    separator: String = ",",
    quote_style: String = "necessary",
    null_value: String = "",
    line_terminator: String = "\n",
    buffer_size: Int = 65536,
) raises:
    """Stream a frame to a UTF-8 CSV file that read_csv reads back exactly.

    With the defaults, nulls are empty unquoted fields and empty strings are
    written as "" so they stay distinct. Output is flushed in chunks of about
    buffer_size bytes, so memory does not grow with the frame.
    """
    if buffer_size <= 0:
        raise Error("CSV buffer_size must be positive")
    var writer = _CsvWriter(separator, quote_style, null_value, line_terminator)
    with open(path, "w") as file:
        var chunk = String()
        if has_header:
            chunk += writer.header(frame)
        for row in range(frame.height()):
            chunk += writer.row(frame, row)
            if chunk.byte_length() >= buffer_size:
                file.write(chunk)
                chunk = String()
        if chunk.byte_length() > 0:
            file.write(chunk)
