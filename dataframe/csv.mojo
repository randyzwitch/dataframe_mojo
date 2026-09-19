"""Strict, explicit-schema CSV ingestion for native Mojo dataframes.

The scalar tokenizer is deliberately separate from typed column decoding. It
keeps state across input buffers, including quoted records, so later SIMD
structural scanning or parallel decoding can replace one stage at a time.
"""
from std.collections import Dict
from std.utils import Variant

from .column import Column
from .dtype import DataType
from .frame import DataFrame
from .series import Series
from .parse import parse_int64, parse_float64
from .temporal import format as format_temporal, parse as parse_temporal


comptime CSV_INT64 = DataType.INT64
comptime CSV_FLOAT64 = DataType.FLOAT64
comptime CSV_BOOL = DataType.BOOL
comptime CSV_STRING = DataType.STRING


struct CsvField(Copyable):
    """One CSV output field. Names and dtypes are always explicit.

    Temporal fields take an optional strftime-style format; an empty format
    means ISO 8601 (see dataframe/temporal.mojo).
    """

    var name: String
    var dtype: DataType
    var nullable: Bool
    var format: String

    def __init__(
        out self,
        name: String,
        dtype: DataType,
        nullable: Bool = True,
        format: String = "",
    ):
        self.name = name
        self.dtype = dtype
        self.nullable = nullable
        self.format = format

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

    @staticmethod
    def date(
        name: String, format: String = "", nullable: Bool = True
    ) -> CsvField:
        return CsvField(name, DataType.DATE, nullable, format)

    @staticmethod
    def datetime(
        name: String,
        unit: String = "us",
        format: String = "",
        nullable: Bool = True,
    ) raises -> CsvField:
        return CsvField(name, DataType.datetime(unit), nullable, format)

    @staticmethod
    def time(
        name: String, format: String = "", nullable: Bool = True
    ) -> CsvField:
        return CsvField(name, DataType.TIME, nullable, format)


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
            names[field.name] = True
        self._fields = fields^

    def __len__(self) -> Int:
        return len(self._fields)

    @staticmethod
    def of(frame: DataFrame) raises -> CsvSchema:
        """A nullable schema matching a frame's names and dtypes."""
        var fields = List[CsvField](capacity=frame.width())
        for field in frame.schema():
            fields.append(CsvField(field.name, field.dtype, True))
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
    var keep: Bool

    def __init__(out self, field: CsvField, keep: Bool = True):
        self.field = field.copy()
        self.keep = keep
        if field.dtype.physical() == CSV_INT64:
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

    def pop(mut self):
        """Remove the last appended value (record rollback)."""
        if not self.keep:
            return
        if self.builder.isa[_IntBuilder]():
            _ = self.builder[_IntBuilder].values.pop()
            _ = self.builder[_IntBuilder].valid.pop()
        elif self.builder.isa[_FloatBuilder]():
            _ = self.builder[_FloatBuilder].values.pop()
            _ = self.builder[_FloatBuilder].valid.pop()
        elif self.builder.isa[_BoolBuilder]():
            _ = self.builder[_BoolBuilder].values.pop()
            _ = self.builder[_BoolBuilder].valid.pop()
        else:
            _ = self.builder[_StringBuilder].values.pop()
            _ = self.builder[_StringBuilder].valid.pop()

    def append(
        mut self,
        text: String,
        quoted: Bool,
        record: Int,
        null_values: List[String] = List[String](),
    ) raises:
        # Null markers only match unquoted fields. Consequently `,"",` is a
        # valid empty string while `,,` is null with the default marker.
        if not self.keep:
            return
        var is_null = not quoted and text == ""
        if not quoted and not is_null:
            for token in null_values:
                if text == token:
                    is_null = True
                    break
        if is_null:
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

        if self.builder.isa[_IntBuilder]() and self.field.dtype.is_temporal():
            try:
                self.builder[_IntBuilder].values.append(
                    parse_temporal(text, self.field.dtype, self.field.format)
                )
            except e:
                raise self._error(record, String(e))
            self.builder[_IntBuilder].valid.append(True)
        elif self.builder.isa[_IntBuilder]():
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
            ).with_dtype(self.field.dtype)
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
    var fields: List[String]
    var quoted: List[Bool]
    var field_index: Int
    var record: Int
    var physical_line: Int
    var in_quotes: Bool
    var after_quote: Bool
    var field_quoted: Bool
    var field_started: Bool
    var record_open: Bool
    var pending_cr: Bool
    var separator: UInt8
    var quote: UInt8
    var quoting: Bool
    var comment: List[UInt8]
    var comment_match: List[UInt8]
    var in_comment: Bool
    var skip_lines: Int
    var n_rows: Int
    var rows: Int
    var done: Bool
    var null_values: List[String]
    var ignore_errors: Bool
    var truncate_ragged: Bool
    var lossy: Bool
    var skipped: Int
    var sampling: Bool
    var sample_limit: Int
    var sample: List[List[String]]
    var sample_quoted: List[List[Bool]]
    var check_header: Bool

    def __init__(
        out self,
        schema: CsvSchema,
        has_header: Bool,
        options: CsvOptions,
        keep: List[Bool],
    ) raises:
        self.schema = schema.copy()
        self.columns = List[_CsvColumn](capacity=len(schema))
        for i in range(len(schema)):
            self.columns.append(_CsvColumn(schema._fields[i], keep[i]))
        self.has_header = has_header
        self.prefix = List[UInt8](capacity=3)
        self.prefix_done = False
        self.field_bytes = List[UInt8]()
        self.header_fields = List[String]()
        self.fields = List[String]()
        self.quoted = List[Bool]()
        self.field_index = 0
        self.record = 1
        self.physical_line = 1
        self.in_quotes = False
        self.after_quote = False
        self.field_quoted = False
        self.field_started = False
        self.record_open = False
        self.pending_cr = False
        self.separator = options.separator.as_bytes()[0]
        self.quoting = options.quote_char.byte_length() == 1
        self.quote = options.quote_char.as_bytes()[0] if self.quoting else 0
        self.comment = List[UInt8]()
        for b in options.comment_prefix.as_bytes():
            self.comment.append(b)
        self.comment_match = List[UInt8]()
        self.in_comment = False
        self.skip_lines = options.skip_rows
        self.n_rows = options.n_rows
        self.rows = 0
        self.done = options.n_rows == 0
        self.null_values = options.null_values.copy()
        self.ignore_errors = options.ignore_errors
        self.truncate_ragged = options.truncate_ragged_lines
        self.lossy = options.encoding == "utf8-lossy"
        self.skipped = 0
        self.sampling = False
        self.sample_limit = -1
        self.sample = List[List[String]]()
        self.sample_quoted = List[List[Bool]]()
        self.check_header = True

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
        var text: String
        if self.lossy:
            text = String(from_utf8_lossy=self.field_bytes)
        else:
            try:
                text = String(from_utf8=self.field_bytes)
            except:
                raise self._location("field is not valid UTF-8")
        if (
            not self.sampling
            and self.field_index >= len(self.schema)
            and not self.truncate_ragged
        ):
            if not self.ignore_errors or (self.has_header and self.record == 1):
                raise self._location("too many fields")
        self.fields.append(text^)
        self.quoted.append(self.field_quoted)
        self.field_bytes.clear()
        self.field_index += 1
        self.field_quoted = False
        self.field_started = False
        self.after_quote = False

    def _finish_record(mut self) raises:
        self._finish_field()
        if self.sampling:
            self.sample.append(self.fields.copy())
            self.sample_quoted.append(self.quoted.copy())
            self.fields.clear()
            self.quoted.clear()
            self.field_index = 0
            self.record += 1
            self.record_open = False
            if self.sample_limit >= 0 and len(self.sample) >= self.sample_limit:
                self.done = True
            return
        var header = self.has_header and self.record == 1
        var count = len(self.fields)
        if header and not self.check_header:
            pass
        elif header:
            if count != len(self.schema):
                raise self._location(
                    String(
                        "expected ",
                        len(self.schema),
                        " fields, found ",
                        count,
                    )
                )
            for i in range(len(self.schema)):
                if self.fields[i] != self.schema._fields[i].name:
                    raise Error(
                        String(
                            "CSV header field ",
                            i + 1,
                            " is '",
                            self.fields[i],
                            "'; expected '",
                            self.schema._fields[i].name,
                            "'",
                        )
                    )
        elif count != len(self.schema) and not self.truncate_ragged:
            if not self.ignore_errors:
                raise self._location(
                    String(
                        "expected ",
                        len(self.schema),
                        " fields, found ",
                        count,
                    )
                )
            self.skipped += 1
        else:
            var appended = 0
            try:
                for i in range(len(self.schema)):
                    if i < count:
                        self.columns[i].append(
                            self.fields[i],
                            self.quoted[i],
                            self.record,
                            self.null_values,
                        )
                    else:
                        self.columns[i].append(
                            "", False, self.record, self.null_values
                        )
                    appended += 1
                self.rows += 1
                if self.n_rows >= 0 and self.rows >= self.n_rows:
                    self.done = True
            except e:
                if not self.ignore_errors:
                    raise e^
                for i in range(appended):
                    self.columns[i].pop()
                self.skipped += 1
        self.fields.clear()
        self.quoted.clear()
        self.field_index = 0
        self.record += 1
        self.record_open = False

    def _consume(mut self, byte: UInt8) raises:
        if self.done:
            return
        if self.skip_lines > 0:
            if byte == 10:
                self.skip_lines -= 1
                self.physical_line += 1
            return
        if self.in_comment:
            if byte == 10:
                self.in_comment = False
                self.physical_line += 1
            return
        # A comment prefix only counts at the start of a record.
        if (
            len(self.comment) > 0
            and not self.record_open
            and not self.pending_cr
        ):
            if byte == self.comment[len(self.comment_match)]:
                self.comment_match.append(byte)
                if len(self.comment_match) == len(self.comment):
                    self.comment_match.clear()
                    self.in_comment = True
                return
            if len(self.comment_match) > 0:
                var replay = self.comment_match.copy()
                self.comment_match.clear()
                for b in replay:
                    self._consume_record_byte(b)
        self._consume_record_byte(byte)

    def _consume_record_byte(mut self, byte: UInt8) raises:
        if self.done:
            return
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
                if byte == self.quote:
                    self.field_bytes.append(self.quote)
                    self.after_quote = False
                elif byte == self.separator:
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
            elif byte == self.quote:
                self.after_quote = True
            else:
                self.field_bytes.append(byte)
                if byte == 10:
                    self.physical_line += 1
            return

        if self.quoting and byte == self.quote:
            if self.field_started:
                raise self._location("quote inside an unquoted field")
            self.in_quotes = True
            self.field_quoted = True
            self.field_started = True
        elif byte == self.separator:
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
        if len(self.comment_match) > 0:
            var replay = self.comment_match.copy()
            self.comment_match.clear()
            for b in replay:
                self._consume_record_byte(b)
        if self.done:
            return self._frame()
        if self.pending_cr:
            raise self._location("bare carriage return at end of file")
        if self.in_quotes:
            if self.after_quote:
                self.in_quotes = False
            else:
                raise self._location("unterminated quoted field")
        if self.record_open and not self.in_comment:
            self._finish_record()
        return self._frame()

    def _frame(self) raises -> DataFrame:
        var output = List[Series](capacity=len(self.columns))
        for column in self.columns:
            if column.keep:
                output.append(column.finish())
        return DataFrame(output^, height=self.rows)


@fieldwise_init
struct CsvOptions(Copyable):
    """Validated reader options; see read_csv for their meaning."""

    var separator: String
    var quote_char: String
    var comment_prefix: String
    var skip_rows: Int
    var n_rows: Int
    var null_values: List[String]
    var ignore_errors: Bool
    var truncate_ragged_lines: Bool
    var encoding: String

    def validate(self) raises:
        if self.separator.byte_length() != 1:
            raise Error("CSV separator must be a single byte")
        var sep = self.separator.as_bytes()[0]
        if sep == 10 or sep == 13:
            raise Error("CSV separator cannot be CR or LF")
        if self.quote_char.byte_length() > 1:
            raise Error("CSV quote_char must be one byte or empty")
        if self.quote_char.byte_length() == 1:
            var q = self.quote_char.as_bytes()[0]
            if q == sep or q == 10 or q == 13:
                raise Error("CSV quote_char cannot be the separator, CR, or LF")
        for b in self.comment_prefix.as_bytes():
            if b == 10 or b == 13:
                raise Error("comment_prefix cannot contain CR or LF")
        if self.skip_rows < 0:
            raise Error("skip_rows must be nonnegative")
        if self.n_rows < -1:
            raise Error("n_rows must be nonnegative or -1")
        if self.encoding != "utf8" and self.encoding != "utf8-lossy":
            raise Error("encoding must be 'utf8' or 'utf8-lossy'")


def _options(
    separator: String,
    quote_char: String,
    comment_prefix: String,
    skip_rows: Int,
    n_rows: Int,
    null_values: List[String],
    ignore_errors: Bool,
    truncate_ragged_lines: Bool,
    encoding: String,
    buffer_size: Int,
) raises -> CsvOptions:
    if buffer_size <= 0:
        raise Error("CSV buffer_size must be positive")
    var options = CsvOptions(
        separator,
        quote_char,
        comment_prefix,
        skip_rows,
        n_rows,
        null_values.copy(),
        ignore_errors,
        truncate_ragged_lines,
        encoding,
    )
    options.validate()
    return options^


def _projection(schema: CsvSchema, columns: List[String]) raises -> List[Bool]:
    var keep = List[Bool](length=len(schema), fill=len(columns) == 0)
    var requested = Dict[String, Bool]()
    for name in columns:
        if name in requested:
            raise Error("CSV column listed twice: " + name)
        requested[name] = True
        var found = False
        for i in range(len(schema)):
            if schema._fields[i].name == name:
                keep[i] = True
                found = True
        if not found:
            raise Error("Unknown CSV column: " + name)
    return keep^


def _stream(
    path: String, mut reader: _CsvReader, buffer_size: Int
) raises -> DataFrame:
    with open(path, "r") as file:
        while not reader.done:
            var bytes = file.read_bytes(buffer_size)
            if len(bytes) == 0:
                break
            reader.feed(bytes^)
    return reader.finish()


def read_csv(
    path: String,
    schema: CsvSchema,
    *,
    has_header: Bool = True,
    separator: String = ",",
    quote_char: String = '"',
    comment_prefix: String = "",
    skip_rows: Int = 0,
    n_rows: Int = -1,
    columns: List[String] = List[String](),
    null_values: List[String] = List[String](),
    ignore_errors: Bool = False,
    truncate_ragged_lines: Bool = False,
    encoding: String = "utf8",
    buffer_size: Int = 65536,
) raises -> DataFrame:
    """Read a strict UTF-8 CSV file into typed, nullable columns.

    Empty unquoted fields are null, as are unquoted fields equal to one of
    null_values. Quoted empty strings are values, which are therefore only
    valid in String columns. Whitespace is never trimmed. Every option is
    applied by the streaming tokenizer, so results do not depend on buffer
    boundaries. See docs/csv.md for the option contract.
    """
    var options = _options(
        separator,
        quote_char,
        comment_prefix,
        skip_rows,
        n_rows,
        null_values,
        ignore_errors,
        truncate_ragged_lines,
        encoding,
        buffer_size,
    )
    var reader = _CsvReader(
        schema, has_header, options, _projection(schema, columns)
    )
    return _stream(path, reader, buffer_size)


def _is_integer_text(text: String) -> Bool:
    var bytes = text.as_bytes()
    var start = (
        1 if len(bytes) > 0 and (bytes[0] == 43 or bytes[0] == 45) else 0
    )
    if len(bytes) == start:
        return False
    for i in range(start, len(bytes)):
        if bytes[i] < 48 or bytes[i] > 57:
            return False
    return True


def _has_leading_zero(text: String) -> Bool:
    var bytes = text.as_bytes()
    var start = (
        1 if len(bytes) > 0 and (bytes[0] == 43 or bytes[0] == 45) else 0
    )
    return len(bytes) - start > 1 and bytes[start] == 48


def infer_dtype(
    values: List[String], quoted: List[Bool], null_values: List[String]
) raises -> DataType:
    """Narrowest of Bool, Int64, Float64, Date, Datetime[us], String that
    reads every sample value (dates and datetimes in ISO 8601 form).

    Nulls (empty unquoted fields and null tokens) are ignored; a column with
    no values is String. Integers with leading zeros, such as identifiers
    like 007, and integers outside Int64 infer as String rather than losing
    information.
    """
    var boolean = True
    var integer = True
    var floating = True
    var seen = False
    for i in range(len(values)):
        ref text = values[i]
        if not quoted[i]:
            if text == "":
                continue
            var is_null = False
            for token in null_values:
                is_null = is_null or text == token
            if is_null:
                continue
        elif text == "":
            return CSV_STRING
        seen = True
        if boolean and text != "true" and text != "false":
            boolean = False
        if _is_integer_text(text):
            # Leading zeros and values outside Int64 would lose information
            # as numbers, so they are text.
            if _has_leading_zero(text):
                return CSV_STRING
            try:
                _ = parse_int64(text)
            except:
                return CSV_STRING
        else:
            integer = False
        if floating and not integer:
            try:
                _ = parse_float64(text)
            except:
                floating = False
    if not seen:
        return CSV_STRING
    if boolean:
        return CSV_BOOL
    if integer:
        return CSV_INT64
    if floating:
        return CSV_FLOAT64
    # Temporal candidates follow the numeric ones: ISO dates, then datetimes.
    for candidate in [DataType.DATE, DataType.datetime("us")]:
        var fits = True
        for i in range(len(values)):
            if not quoted[i] and (values[i] == "" or values[i] in null_values):
                continue
            try:
                _ = parse_temporal(values[i], candidate)
            except:
                fits = False
                break
        if fits:
            return candidate
    return CSV_STRING


def read_csv(
    path: String,
    *,
    infer_schema_length: Int = 10000,
    schema_overrides: Dict[String, String] = Dict[String, String](),
    has_header: Bool = True,
    separator: String = ",",
    quote_char: String = '"',
    comment_prefix: String = "",
    skip_rows: Int = 0,
    n_rows: Int = -1,
    columns: List[String] = List[String](),
    null_values: List[String] = List[String](),
    ignore_errors: Bool = False,
    truncate_ragged_lines: Bool = False,
    encoding: String = "utf8",
    buffer_size: Int = 65536,
) raises -> DataFrame:
    """Read a CSV file, inferring a nullable schema from a sample.

    The first infer_schema_length data records (-1 means every record) are
    tokenized and each column takes the narrowest type of Bool, Int64,
    Float64, or String that reads every sampled value. schema_overrides maps
    column names to int64, float64, bool, or string and wins over inference.
    The whole file is then read strictly with that schema: a later value that
    does not fit raises with its record and field instead of changing type.
    Header names are kept, with repeats renamed name_1, name_2, ...; without
    a header, columns are column_1, column_2, ...
    """
    if infer_schema_length < -1:
        raise Error("infer_schema_length must be nonnegative or -1")
    var options = _options(
        separator,
        quote_char,
        comment_prefix,
        skip_rows,
        -1,
        null_values,
        ignore_errors,
        truncate_ragged_lines,
        encoding,
        buffer_size,
    )
    var sampler = _CsvReader(
        CsvSchema([CsvField.string("_")]), False, options, [False]
    )
    sampler.sampling = True
    var header_rows = 1 if has_header else 0
    sampler.sample_limit = (
        -1 if infer_schema_length < 0 else infer_schema_length + header_rows
    )
    _ = _stream(path, sampler, buffer_size)
    var sample = sampler.sample.copy()
    var quoted = sampler.sample_quoted.copy()
    var width = 0
    if len(sample) > 0:
        width = len(sample[0])
    for row in sample:
        if (
            len(row) != width
            and not truncate_ragged_lines
            and not ignore_errors
        ):
            raise Error(
                "CSV records have different field counts; expected "
                + String(width)
                + ", found "
                + String(len(row))
            )
    if width == 0:
        return DataFrame([])
    var names = List[String]()
    var used = Dict[String, Int]()
    for i in range(width):
        var name = sample[0][i] if has_header else "column_" + String(i + 1)
        if name in used:
            used[name] += 1
            var candidate = name + "_" + String(used[name])
            while candidate in used:
                used[name] += 1
                candidate = name + "_" + String(used[name])
            name = candidate
        used[name] = 0
        names.append(name)
    for item in schema_overrides.items():
        var known = False
        for name in names:
            known = known or name == item.key
        if not known:
            raise Error("schema_overrides names unknown column: " + item.key)
    var fields = List[CsvField]()
    for i in range(width):
        var dtype: DataType
        if names[i] in schema_overrides:
            var requested = schema_overrides[names[i]]
            if not DataType.is_known(requested):
                raise Error("Unknown dtype in schema_overrides: " + requested)
            dtype = DataType.parse(requested)
        else:
            var values = List[String]()
            var flags = List[Bool]()
            for r in range(header_rows, len(sample)):
                if i < len(sample[r]):
                    values.append(sample[r][i])
                    flags.append(quoted[r][i])
            dtype = infer_dtype(values, flags, null_values)
        fields.append(CsvField(names[i], dtype, True))
    var schema = CsvSchema(fields^)
    var strict = options.copy()
    strict.n_rows = n_rows
    var reader = _CsvReader(
        schema, has_header, strict, _projection(schema, columns)
    )
    reader.check_header = False
    try:
        return _stream(path, reader, buffer_size)
    except e:
        var message = String(e)
        if (
            message.find("invalid") >= 0
            or message.find("overflow") >= 0
            or message.find("Boolean") >= 0
        ):
            raise Error(
                message
                + " (the schema was inferred from the first "
                + String(infer_schema_length)
                + " records; pass schema_overrides or a larger"
                + " infer_schema_length)"
            )
        raise e^


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
        if series.dtype().is_temporal():
            return format_temporal(
                series._data[Column[Int64]]._values[row], series.dtype()
            )
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
                column.dtype() == DataType.STRING
                or column.dtype() == DataType.BOOL
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
