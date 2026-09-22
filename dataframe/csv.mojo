"""Strict, explicit-schema CSV ingestion for native Mojo dataframes.

The scalar tokenizer is deliberately separate from typed column decoding. It
keeps state across input buffers, including quoted records, so later SIMD
structural scanning or parallel decoding can replace one stage at a time.
"""
from std.bit import count_trailing_zeros
from std.ffi import external_call
from std.memory import ArcPointer, Pointer, bitcast
from std.collections import Dict
from std.utils import Variant

from .bool_column import BoolColumn
from .column import Column
from .string_column import StringColumn, StringBuilder
from .dtype import DataType, NUMERIC_DTYPES
from .frame import DataFrame
from .series import Series
from .parse import parse_int64, parse_float64, parse_integer
from .frame import concat
from .parallel import Job, Pool, _ProducedJobs, worker_count
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


comptime _Builder = Variant[
    _IntBuilder, _FloatBuilder, _BoolBuilder, StringBuilder
]

# Which builder a column holds. The answer is fixed for the whole file, so it
# is decided once and compared as an integer; asking the Variant per field
# means a chain of runtime tag tests on the hottest path in the reader.
comptime _KIND_INT = 0
comptime _KIND_TEMPORAL = 1
comptime _KIND_FLOAT = 2
comptime _KIND_BOOL = 3
comptime _KIND_STRING = 4


struct _CsvColumn(Copyable):
    var field: CsvField
    var builder: _Builder
    var keep: Bool
    var kind: Int

    def __init__(out self, field: CsvField, keep: Bool = True):
        self.field = field.copy()
        self.keep = keep
        # Every integer width parses into Int64 slots (UInt64 by bit
        # pattern) and every float into Float64; finish() narrows exactly.
        if field.dtype.physical() == CSV_INT64 or field.dtype.is_integer():
            self.builder = _Builder(_IntBuilder([], []))
            self.kind = (
                _KIND_TEMPORAL if field.dtype.is_temporal() else _KIND_INT
            )
        elif field.dtype.is_float():
            self.builder = _Builder(_FloatBuilder([], []))
            self.kind = _KIND_FLOAT
        elif field.dtype == CSV_BOOL:
            self.builder = _Builder(_BoolBuilder([], []))
            self.kind = _KIND_BOOL
        else:
            self.builder = _Builder(StringBuilder())
            self.kind = _KIND_STRING

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
        if self.kind == _KIND_INT or self.kind == _KIND_TEMPORAL:
            _ = self.builder[_IntBuilder].values.pop()
            _ = self.builder[_IntBuilder].valid.pop()
        elif self.kind == _KIND_FLOAT:
            _ = self.builder[_FloatBuilder].values.pop()
            _ = self.builder[_FloatBuilder].valid.pop()
        elif self.kind == _KIND_BOOL:
            _ = self.builder[_BoolBuilder].values.pop()
            _ = self.builder[_BoolBuilder].valid.pop()
        else:
            self.builder[StringBuilder]._pop()

    def append(
        mut self,
        text: StringSlice,
        quoted: Bool,
        record: Int,
        null_values: List[String] = List[String](),
    ) raises:
        # Null markers only match unquoted fields. Consequently `,"",` is a
        # valid empty string while `,,` is null with the default marker.
        if not self.keep:
            return
        var is_null = not quoted and text.byte_length() == 0
        # The marker scan only matters when markers exist, and unquoted
        # non-empty fields are the common case.
        if not quoted and not is_null and len(null_values) > 0:
            for token in null_values:
                if text == token:
                    is_null = True
                    break
        if is_null:
            if not self.field.nullable:
                raise self._error(record, "null in a non-nullable field")
            if self.kind == _KIND_INT or self.kind == _KIND_TEMPORAL:
                self.builder[_IntBuilder].values.append(0)
                self.builder[_IntBuilder].valid.append(False)
            elif self.kind == _KIND_FLOAT:
                self.builder[_FloatBuilder].values.append(0)
                self.builder[_FloatBuilder].valid.append(False)
            elif self.kind == _KIND_BOOL:
                self.builder[_BoolBuilder].values.append(False)
                self.builder[_BoolBuilder].valid.append(False)
            else:
                self.builder[StringBuilder].append_null()
            return

        if self.kind == _KIND_TEMPORAL:
            try:
                self.builder[_IntBuilder].values.append(
                    parse_temporal(
                        String(text), self.field.dtype, self.field.format
                    )
                )
            except e:
                raise self._error(record, String(e))
            self.builder[_IntBuilder].valid.append(True)
        elif self.kind == _KIND_INT:
            try:
                self.builder[_IntBuilder].values.append(
                    _parse_int_slot(text, self.field.dtype)
                )
            except:
                raise self._error(
                    record,
                    "invalid "
                    + _display_name(self.field.dtype)
                    + " value '"
                    + String(from_utf8_lossy=text.as_bytes())
                    + "'",
                )
            self.builder[_IntBuilder].valid.append(True)
        elif self.kind == _KIND_FLOAT:
            try:
                self.builder[_FloatBuilder].values.append(parse_float64(text))
            except e:
                raise self._error(record, String(e))
            self.builder[_FloatBuilder].valid.append(True)
        elif self.kind == _KIND_BOOL:
            var is_true = text.as_bytes() == "true".as_bytes()
            if not is_true and text.as_bytes() != "false".as_bytes():
                raise self._error(
                    record, "Boolean must be exactly 'true' or 'false'"
                )
            self.builder[_BoolBuilder].values.append(is_true)
            self.builder[_BoolBuilder].valid.append(True)
        else:
            self.builder[StringBuilder].append(text)

    def finish(self) raises -> Series:
        var dtype = self.field.dtype
        if dtype.is_numeric() and dtype != CSV_INT64 and dtype != CSV_FLOAT64:
            comptime for k in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[k]
                if dtype == DataType.of(D):
                    var values = List[Scalar[D]]()
                    var valid: List[Bool]
                    comptime if D.is_floating_point():
                        ref builder = self.builder[_FloatBuilder]
                        valid = builder.valid.copy()
                        values.reserve(len(builder.values))
                        for x in builder.values:
                            values.append(x.cast[D]())
                    else:
                        ref builder = self.builder[_IntBuilder]
                        valid = builder.valid.copy()
                        values.reserve(len(builder.values))
                        for x in builder.values:
                            values.append(x.cast[D]())
                    return Series(
                        self.field.name, Column[Scalar[D]](values^, valid)
                    )
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
                BoolColumn(
                    self.builder[_BoolBuilder].values.copy(),
                    self.builder[_BoolBuilder].valid.copy(),
                ),
            )
        return Series(
            self.field.name, self.builder[StringBuilder].copy().finish()
        )


struct _CsvReader:
    var schema: CsvSchema
    var columns: List[_CsvColumn]
    var has_header: Bool
    var prefix: List[UInt8]
    var prefix_done: Bool
    var field_bytes: List[UInt8]
    var header_fields: List[String]
    # Every field of the current record, end to end, with one end offset
    # per field. A String per field was 849 ms of a 1,336 ms single-threaded
    # read of 1M rows -- 8 million allocations -- and a slice over this
    # buffer needs none.
    var record_bytes: List[UInt8]
    var field_ends: List[Int]
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
        self.record_bytes = List[UInt8]()
        self.field_ends = List[Int]()
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
        if (
            not self.sampling
            and self.field_index >= len(self.schema)
            and not self.truncate_ragged
        ):
            if not self.ignore_errors or (self.has_header and self.record == 1):
                raise self._location("too many fields")
        var start = (self.field_ends[len(self.field_ends) - 1]) if len(
            self.field_ends
        ) > 0 else 0
        # Only a field whose bytes become text needs to be valid UTF-8: a
        # string column, a header name, or an inference sample. A number,
        # Boolean or timestamp is parsed from ASCII, and its parser rejects
        # any byte above 0x7F, so validating it first is work that changes
        # no outcome -- 238 ms of a 925 ms single-threaded read of 1M rows
        # of 8 columns. Polars does not validate per field either; its
        # parser takes raw bytes and only builds a (lossy) String to
        # describe a field it could not parse.
        var becomes_text = (
            self.sampling
            or (self.has_header and self.record == 1)
            or (
                self.field_index < len(self.columns)
                and self.columns[self.field_index].kind == _KIND_STRING
            )
        )
        if self.lossy and becomes_text:
            # Lossy decoding substitutes U+FFFD, so the bytes change and a
            # String has to be built; it is the rare path.
            var text = String(
                from_utf8_lossy=Span(self.record_bytes)[
                    start : len(self.record_bytes)
                ]
            )
            self.record_bytes.resize(start, 0)
            self.record_bytes.extend(text.as_bytes())
        elif becomes_text:
            # Validation only, over bytes already in place: no allocation
            # and no copy, the field having been written here directly.
            try:
                _ = StringSlice(
                    from_utf8=Span(self.record_bytes)[
                        start : len(self.record_bytes)
                    ]
                )
            except:
                raise self._location("field is not valid UTF-8")
        self.field_ends.append(len(self.record_bytes))
        if self.sampling or (self.has_header and self.record == 1):
            # Inference keeps its sample, and a header's names are compared
            # and stored, so those records still materialise Strings.
            self.fields.append(
                String(self._field_text(len(self.field_ends) - 1))
            )
        self.quoted.append(self.field_quoted)
        self.field_index += 1
        self.field_quoted = False
        self.field_started = False
        self.after_quote = False

    def _field_text(self, i: Int) -> StringSlice[ImmutAnyOrigin]:
        """Field i of the current record, borrowed from `record_bytes`."""
        var start = self.field_ends[i - 1] if i > 0 else 0
        return StringSlice[ImmutAnyOrigin](
            unsafe_from_utf8=Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=self.record_bytes.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
                .unsafe_offset(start),
                length=self.field_ends[i] - start,
            )
        )

    def _finish_record(mut self) raises:
        self._finish_field()
        if self.sampling:
            self.sample.append(self.fields.copy())
            self.sample_quoted.append(self.quoted.copy())
            self.fields.clear()
            self.quoted.clear()
            self.record_bytes.clear()
            self.field_ends.clear()
            self.field_index = 0
            self.record += 1
            self.record_open = False
            if self.sample_limit >= 0 and len(self.sample) >= self.sample_limit:
                self.done = True
            return
        var header = self.has_header and self.record == 1
        var count = len(self.field_ends)
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
                            self._field_text(i),
                            self.quoted[i],
                            self.record,
                            self.null_values,
                        )
                    else:
                        self.columns[i].append(
                            StringSlice[ImmutAnyOrigin](),
                            False,
                            self.record,
                            self.null_values,
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
        self.record_bytes.clear()
        self.field_ends.clear()
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
                    self.record_bytes.append(self.quote)
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
                self.record_bytes.append(byte)
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
            self.record_bytes.append(byte)

    def _mask(self, matches: SIMD[DType.bool, 64]) -> UInt64:
        """Pack 64 comparison lanes into one bit per input byte.

        Each byte is zero or one. Multiplication gathers eight such bytes
        into the high byte of each word; distinct bit weights prevent carries.
        The library requires little endian storage on every supported target.
        """
        var words = bitcast[DType.uint64, 8](matches.cast[DType.uint8]())
        var packed = (words * SIMD[DType.uint64, 8](0x0102040810204080)) >> 56
        var shifts = SIMD[DType.uint64, 8](0, 8, 16, 24, 32, 40, 48, 56)
        return (packed << shifts).reduce_or()

    def _structural_mask(
        self, bytes: Span[UInt8, ImmutAnyOrigin], start: Int
    ) -> UInt64:
        """Structural positions in one 64-byte block, low position first."""
        var block = bytes.unsafe_ptr().unsafe_load[width=64](start)
        var separator = SIMD[DType.uint8, 64](self.separator)
        var newline = SIMD[DType.uint8, 64](10)
        var carriage_return = SIMD[DType.uint8, 64](13)
        # With quoting disabled, use the separator again so literal quotes
        # remain ordinary content while this comparison stays uniform.
        var quote = SIMD[DType.uint8, 64](
            self.quote if self.quoting else self.separator
        )
        return self._mask(
            block.eq(separator)
            | block.eq(newline)
            | block.eq(carriage_return)
            | block.eq(quote)
        )

    def _append_ordinary(
        mut self,
        bytes: Span[UInt8, ImmutAnyOrigin],
        start: Int,
        stop: Int,
    ):
        """Append bytes known not to transition tokenizer state."""
        if stop > start:
            self.record_open = True
            self.field_started = True
            self.record_bytes.extend(bytes[start:stop])

    def feed(mut self, bytes: Span[UInt8, ImmutAnyOrigin]) raises:
        var i = 0
        var n = len(bytes)
        # Resolve the optional BOM before scanning the remaining buffer.
        while i < n and not self.prefix_done:
            self.prefix.append(bytes[i])
            i += 1
            if len(self.prefix) == 3:
                if (
                    self.prefix[0] != 239
                    or self.prefix[1] != 187
                    or self.prefix[2] != 191
                ):
                    for byte in self.prefix:
                        self._consume(byte)
                self.prefix.clear()
                self.prefix_done = True
        if self.done:
            return
        if self.skip_lines == 0 and len(self.comment) == 0:
            while i + 64 <= n:
                var base = i
                var end = base + 64
                var mask = self._structural_mask(bytes, base)
                while mask != 0:
                    var stop = base + Int(count_trailing_zeros(mask))
                    # Ordinary content after CR or a closing quote is an
                    # error; let the scalar state machine diagnose it.
                    if i < stop and (self.pending_cr or self.after_quote):
                        self._consume(bytes[i])
                        i += 1
                    self._append_ordinary(bytes, i, stop)
                    self._consume(bytes[stop])
                    i = stop + 1
                    mask &= mask - 1
                    if self.done:
                        return
                if i < end and (self.pending_cr or self.after_quote):
                    self._consume(bytes[i])
                    i += 1
                self._append_ordinary(bytes, i, end)
                i = end
        while i < n and not self.done:
            self._consume(bytes[i])
            i += 1

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


comptime _SCAN_WIDTH = 32


def _splat(byte: UInt8) -> SIMD[DType.uint8, _SCAN_WIDTH]:
    return SIMD[DType.uint8, _SCAN_WIDTH](byte)


@fieldwise_init
struct _Split(Copyable, Movable):
    """Where a record starts, and how many records precede it."""

    var offset: Int
    var record: Int


@fieldwise_init
struct _Layout(Movable):
    """How a buffer divides into ranges that each begin on a record."""

    var splits: List[_Split]
    # Offset just after the last record terminator, so a caller reading in
    # blocks knows what to carry forward. 0 when no record is complete.
    var complete: Int
    var records: Int


def record_splits(
    bytes: Span[UInt8, ImmutAnyOrigin],
    quote: UInt8,
    quoting: Bool,
    parts: Int,
) raises -> _Layout:
    """Byte offsets at which records start, splitting into at most `parts`.

    A chunk boundary may land inside a quoted field, and a quoted field may
    contain newlines, so a newline is only a record boundary when an even
    number of quotes precede it. Toggling on every quote byte handles CSV's
    doubled-quote escape without a special case: `""` toggles twice, so a
    newline after it still sees the enclosing field as open.

    The returned splits always begin with offset 0, are strictly increasing,
    and are followed by a terminator whose offset is the buffer length. Each
    carries the number of complete records before it, so a worker can report
    record numbers that match a serial read.
    """
    var n = len(bytes)
    var splits = List[_Split]()
    splits.append(_Split(0, 0))
    if n == 0:
        splits.append(_Split(0, 0))
        return _Layout(splits^, 0, 0)

    var stride = max(1, n // max(parts, 1))
    var target = stride
    var inside = False
    var records = 0
    var complete = 0
    var i = 0
    var pointer = bytes.unsafe_ptr()
    var quotes = _splat(quote)
    var newlines = _splat(10)

    @__parameter
    def scan_byte(at: Int):
        """The definition of the scan: one byte, in the state machine's terms."""
        var byte = bytes[at]
        if quoting and byte == quote:
            inside = not inside
        elif byte == 10 and not inside:
            records += 1
            complete = at + 1
            if parts > 1 and at + 1 >= target and at + 1 < n:
                splits.append(_Split(at + 1, records))
                target = at + 1 + stride

    while i < n:
        # A block with no quote can be summarised rather than walked: every
        # newline in it is a record boundary, because parity cannot change.
        # Only a block that carries a quote, or that holds a split target or
        # the running last boundary, needs the byte loop.
        if i + _SCAN_WIDTH <= n:
            var block = pointer.unsafe_load[width=_SCAN_WIDTH](i)
            var quoted_here = quoting and block.eq(quotes).reduce_or()
            if not quoted_here and not inside:
                var found = block.eq(newlines)
                var count = Int(found.cast[DType.uint8]().reduce_add())
                if count == 0:
                    i += _SCAN_WIDTH
                    continue
                var crosses_target = parts > 1 and i + _SCAN_WIDTH > target
                if not crosses_target:
                    # Nothing here needs a position, only the totals, except
                    # the last boundary, which the tail pass recovers.
                    records += count
                    complete = -1
                    i += _SCAN_WIDTH
                    continue
        scan_byte(i)
        i += 1

    if complete < 0:
        # A summarised block held the final boundary; find it by scanning
        # back for the last newline outside quotes, which is the last byte
        # the forward pass would have marked.
        complete = 0
        var back = n
        while back > 0:
            back -= 1
            if bytes[back] == 10:
                complete = back + 1
                break
    splits.append(_Split(n, records))
    return _Layout(splits^, complete, records)


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
            reader.feed(
                Span[UInt8, ImmutAnyOrigin](
                    unsafe_ptr=bytes.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    length=len(bytes),
                )
            )
            _ = bytes^
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
    var keep = _projection(schema, columns)
    if _parallel_is_safe(options, worker_count(1 << 40)):
        var mapping = _map_file(path)
        if mapping.address == 0:
            # Not every path can be mapped; read it in blocks instead.
            return _stream_parallel(
                path, schema, has_header, options, keep, buffer_size
            )
        return _read_mapped(mapping^, schema, has_header, options, keep)
    var reader = _CsvReader(schema, has_header, options, keep)
    return _stream(path, reader, buffer_size)


struct _RangeJob(Job):
    """Decode one record-aligned byte range with a reader of its own."""

    var reader: _CsvReader
    var bytes: ArcPointer[List[UInt8]]
    var start: Int
    var end: Int
    var frame: DataFrame

    def __init__(
        out self,
        var reader: _CsvReader,
        bytes: ArcPointer[List[UInt8]],
        start: Int,
        end: Int,
    ) raises:
        self.reader = reader^
        self.bytes = bytes.copy()
        self.start = start
        self.end = end
        self.frame = DataFrame(List[Series](), height=0)

    def run(mut self) raises:
        # A view into the shared buffer. Copying the range out first cost a
        # second pass over every byte of the file, on top of reading it.
        ref buffer = self.bytes[]
        self.reader.feed(
            Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=buffer.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
                .unsafe_offset(self.start),
                length=self.end - self.start,
            )
        )
        self.frame = self.reader.finish()

    def into_frame(deinit self) -> DataFrame:
        return self.frame^


# Worker sizing for CSV is by bytes, not rows: `worker_count` measures rows
# and a block is bytes, so handing it a byte count silently yields one
# worker. A quarter megabyte per worker keeps the per-range fixed costs
# (a reader, its column builders, and the concatenation) small next to the
# decoding.
comptime _MIN_BYTES_PER_WORKER = 262144

# A CSV range owns one set of column builders. Keep enough small ranges to
# smooth out scheduler stalls and cache pressure, without making a very wide
# frame allocate an unbounded number of builder sets. These are deliberately
# byte based: the tokenizer has not decoded rows yet when it makes the plan.
comptime _CHUNKS_PER_WORKER = 8
comptime _CHUNK_ALLOCATION_BUDGET = 500000
comptime _MIN_CHUNK_BYTES = 4096
comptime _MAX_CHUNK_BYTES = 16 << 20

# Parallel reads pull larger blocks than the streaming default, because a
# block is what gets divided: a 64 KiB block cannot usefully be split.
comptime _PARALLEL_BLOCK = 32 << 20


def _csv_workers(bytes: Int) -> Int:
    """Threads for a block of this many bytes, honouring DATAFRAME_THREADS."""
    return max(1, min(worker_count(1 << 40), bytes // _MIN_BYTES_PER_WORKER))


def _csv_chunks(bytes: Int, columns: Int, workers: Int) -> Int:
    """How many record-aligned chunks to make for a parallel CSV read.

    A worker claims several chunks instead of owning one large range. A slow
    core then delays at most one cache-sized decode. The column cap matters
    because every chunk creates one reader and therefore a builder per output
    column.
    """
    if bytes <= 0 or workers <= 1:
        return 1
    var max_chunks = max(workers, _CHUNK_ALLOCATION_BUDGET // max(columns, 1))
    var wanted = min(workers * _CHUNKS_PER_WORKER, max_chunks)
    var chunk_bytes = min(
        _MAX_CHUNK_BYTES, (bytes + max(wanted, 1) - 1) // max(wanted, 1)
    )
    chunk_bytes = max(_MIN_CHUNK_BYTES, chunk_bytes)
    return max(1, (bytes + chunk_bytes - 1) // chunk_bytes)


# A read-only whole-file mapping, so a parallel read neither copies the file
# into a buffer nor reads it twice. Polars does the same
# (`polars-io/src/mmap.rs`, `MMapSemaphore::new_from_file`), and it is why
# its whole 32-thread read finishes faster than a plain read of the same
# bytes. Mapping this 50 MB file costs 2.4 ms against 58 ms to read it.
comptime _PROT_READ = 1
comptime _MAP_PRIVATE = 2
comptime _SEEK_END = 2  # whence for seeking to the end


struct _Mapping(Movable):
    """A file mapped for reading, unmapped when it goes out of scope."""

    var address: Int
    var length: Int

    def __init__(out self, address: Int, length: Int):
        self.address = address
        self.length = length

    def __deinit__(deinit self):
        if self.address != 0:
            _ = external_call["munmap", Int32](self.address, self.length)

    def span(self) -> Span[UInt8, ImmutAnyOrigin]:
        return Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=self.address
            )
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            length=self.length,
        )


def _map_file(path: String) -> _Mapping:
    """Map `path` whole, or return a mapping with address 0 if it cannot be.

    A pipe or a character device has no length to map. Reporting that as a
    value rather than an exception keeps the caller's fallback to reading in
    blocks separate from a decoding error, which must not be swallowed.
    """
    try:
        return _try_map(path)
    except:
        return _Mapping(0, 0)


def _try_map(path: String) raises -> _Mapping:
    with open(path, "r") as file:
        # The handle's own seek, not lseek through FFI: the standard library
        # already declares that symbol with another signature.
        var length = Int(file.seek(0, _SEEK_END))
        if length <= 0:
            raise Error("nothing to map")
        var address = external_call["mmap", Int](
            0,
            length,
            Int32(_PROT_READ),
            Int32(_MAP_PRIVATE),
            Int32(file._get_raw_fd()),
            0,
        )
        if address == 0 or address == -1:
            raise Error("mmap failed")
        # The mapping outlives the descriptor, so closing the file here is
        # fine and is what leaving this block does.
        return _Mapping(address, length)


struct _MappedRangeJob(Job):
    """Decode one record-aligned range of a mapped file.

    The mapping outlives every job, being held by the frame-building call
    below, so a range reads straight out of it.
    """

    var reader: _CsvReader
    var base: Int
    var start: Int
    var end: Int
    var frame: DataFrame

    def __init__(
        out self, var reader: _CsvReader, base: Int, start: Int, end: Int
    ) raises:
        self.reader = reader^
        self.base = base
        self.start = start
        self.end = end
        self.frame = DataFrame(List[Series](), height=0)

    def run(mut self) raises:
        self.reader.feed(
            Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=self.base + self.start
                )
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                length=self.end - self.start,
            )
        )
        self.frame = self.reader.finish()

    def into_frame(deinit self) -> DataFrame:
        return self.frame^


def _read_mapped_produced(
    var mapping: _Mapping,
    schema: CsvSchema,
    has_header: Bool,
    options: CsvOptions,
    keep: List[Bool],
    workers: Int,
    chunks: Int,
) raises -> DataFrame:
    """Scan one mapped file while workers decode each range as it appears.

    Quote parity stays a single left-to-right state machine. The only change
    from `record_splits` is that reaching a safe record boundary immediately
    publishes the preceding range, hiding all but the first boundary scan
    behind decoding.
    """
    var span = mapping.span()
    var quoting = options.quote_char.byte_length() > 0
    var quote = options.quote_char.as_bytes()[0] if quoting else UInt8(0)
    var pool = Pool(workers)
    # At most one boundary is emitted per target, plus the final range.
    var produced = _ProducedJobs[_MappedRangeJob](chunks + 1)
    pool.run_produced(produced)

    var n = len(span)
    var stride = max(1, n // max(chunks, 1))
    var target = stride
    var start = 0
    var start_record = 0
    var records = 0
    var inside = False
    var i = 0
    var pointer = span.unsafe_ptr()
    var quotes = _splat(quote)
    var newlines = _splat(10)

    @__parameter
    def publish(stop: Int) raises:
        var leading = start == 0
        var reader = _CsvReader(
            schema, has_header and leading, options, keep.copy()
        )
        reader.record += start_record
        if not leading:
            reader.prefix_done = True
        produced.submit(_MappedRangeJob(reader^, mapping.address, start, stop))
        start = stop
        start_record = records
        target = stop + stride

    @__parameter
    def scan_byte(at: Int) raises:
        var byte = span[at]
        if quoting and byte == quote:
            inside = not inside
        elif byte == 10 and not inside:
            records += 1
            if at + 1 >= target and at + 1 < n:
                publish(at + 1)

    while i < n:
        # The same fast path as `record_splits`: a quote-free SIMD block
        # outside a field can update counts in bulk until it approaches the
        # next target, where byte positions become necessary to publish.
        if i + _SCAN_WIDTH <= n:
            var block = pointer.unsafe_load[width=_SCAN_WIDTH](i)
            var quoted_here = quoting and block.eq(quotes).reduce_or()
            if not quoted_here and not inside:
                var found = block.eq(newlines)
                var count = Int(found.cast[DType.uint8]().reduce_add())
                if count == 0:
                    i += _SCAN_WIDTH
                    continue
                if i + _SCAN_WIDTH <= target:
                    records += count
                    i += _SCAN_WIDTH
                    continue
        scan_byte(i)
        i += 1
    if start < n:
        publish(n)

    var jobs = produced.finish()
    # The scanner is finished. Join the idle workers before concat starts so
    # their between-round spin cannot contend with column assembly.
    pool.release()
    var frames = List[DataFrame](capacity=len(jobs))
    jobs.reverse()
    while len(jobs) > 0:
        frames.append(jobs.pop().into_frame())
    var result = concat(frames) if len(frames) > 1 else frames.pop(0)
    _ = mapping^
    return result^


def _read_mapped(
    var mapping: _Mapping,
    schema: CsvSchema,
    has_header: Bool,
    options: CsvOptions,
    keep: List[Bool],
) raises -> DataFrame:
    """Read a mapped file: split it once, decode the ranges in parallel.

    There is no block loop and no carried remainder, because the whole file
    is addressable at once -- which is the other half of what mapping buys.
    """
    var span = mapping.span()
    var quoting = options.quote_char.byte_length() > 0
    var quote = options.quote_char.as_bytes()[0] if quoting else UInt8(0)
    var workers = _csv_workers(len(span))
    var chunks = _csv_chunks(len(span), len(keep), workers)
    if workers > 1:
        return _read_mapped_produced(
            mapping^, schema, has_header, options, keep, workers, chunks
        )
    var layout = record_splits(span, quote, quoting, chunks)

    var jobs = List[_MappedRangeJob]()
    for s in range(len(layout.splits) - 1):
        var start = layout.splits[s].offset
        var stop = min(layout.splits[s + 1].offset, len(span))
        if stop <= start:
            continue
        var leading = s == 0
        var reader = _CsvReader(
            schema, has_header and leading, options, keep.copy()
        )
        reader.record += layout.splits[s].record
        # A byte-order mark means something only at the very start of the
        # file; elsewhere those bytes are data.
        if not leading:
            reader.prefix_done = True
        jobs.append(_MappedRangeJob(reader^, mapping.address, start, stop))

    if len(jobs) == 0:
        var empty = _CsvReader(schema, has_header, options, keep.copy())
        return empty.finish()
    if len(jobs) == 1:
        jobs[0].run()
    else:
        # Ranges vary with field widths and allocator stalls. Claiming a
        # small range dynamically keeps one delayed core from setting the
        # read's tail while the result list preserves file order.
        var pool = Pool(workers)
        pool.run(jobs, claim=True)
    var frames = List[DataFrame]()
    while len(jobs) > 0:
        frames.append(jobs.pop(0).into_frame())
    var result = concat(frames) if len(frames) > 1 else frames.pop(0)
    # The mapping must outlive every read of it.
    _ = mapping^
    return result^


def _stream_parallel(
    path: String,
    schema: CsvSchema,
    has_header: Bool,
    options: CsvOptions,
    keep: List[Bool],
    buffer_size: Int,
) raises -> DataFrame:
    """Read a file in blocks, decoding each block's records in parallel.

    Blocks are read sequentially, so memory stays bounded by the block size
    rather than the file size. Within a block, `record_splits` finds offsets
    that begin a record, each range gets its own reader, and the partial
    frames are concatenated in range order, which is what makes the output
    identical to a serial read. Bytes after the last complete record are
    carried into the next block.
    """
    var block_size = max(buffer_size, _PARALLEL_BLOCK)
    var quoting = options.quote_char.byte_length() > 0
    var quote = options.quote_char.as_bytes()[0] if quoting else UInt8(0)
    var frames = List[DataFrame]()
    var carry = List[UInt8]()
    var first_block = True
    var base = 0
    with open(path, "r") as file:
        while True:
            var block = file.read_bytes(block_size)
            var last = len(block) == 0
            var buffer = carry^
            carry = List[UInt8]()
            buffer.extend(Span(block))
            _ = block^
            if len(buffer) == 0:
                break
            var workers = _csv_workers(len(buffer))
            var chunks = _csv_chunks(len(buffer), len(keep), workers)
            var span = Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=buffer.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                length=len(buffer),
            )
            var layout = record_splits(span, quote, quoting, chunks)
            # Whatever follows the last record terminator belongs to the next
            # block; on the final block there is nothing more to read, so the
            # remainder is decoded here.
            var upto = len(buffer) if last else layout.complete
            if upto == 0:
                carry = buffer^
                if last:
                    break
                continue
            if not last and upto < len(buffer):
                carry.extend(Span(buffer)[upto : len(buffer)])
            var shared = ArcPointer(buffer^)
            var jobs = List[_RangeJob]()
            for s in range(len(layout.splits) - 1):
                var start = layout.splits[s].offset
                var stop = min(layout.splits[s + 1].offset, upto)
                if stop <= start:
                    continue
                var leading = first_block and s == 0
                var reader = _CsvReader(
                    schema, has_header and leading, options, keep.copy()
                )
                reader.record += base + layout.splits[s].record
                # A byte-order mark means something only at the very start of
                # the file. Every other range begins mid-file, where those
                # bytes are data, so only the leading reader looks for one.
                if not leading:
                    reader.prefix_done = True
                jobs.append(_RangeJob(reader^, shared, start, stop))
            if len(jobs) == 1:
                jobs[0].run()
            else:
                var pool = Pool(workers)
                pool.run(jobs, claim=True)
            while len(jobs) > 0:
                frames.append(jobs.pop(0).into_frame())
            base += layout.records
            first_block = False
            if last:
                break
    if len(frames) == 0:
        var empty = _CsvReader(schema, has_header, options, keep.copy())
        return empty.finish()
    if len(frames) == 1:
        return frames.pop(0)
    return concat(frames)


def _parallel_is_safe(options: CsvOptions, workers: Int) -> Bool:
    """Whether a file can be split without changing the result.

    Row-limited and row-skipping reads, and comment prefixes, all depend on
    counting records from the start of the file, which a range cannot do on
    its own. Those stay serial until they are handled explicitly.
    """
    return (
        workers > 1
        and options.n_rows < 0
        and options.skip_rows == 0
        and options.comment_prefix.byte_length() == 0
        and not options.ignore_errors
        and not options.truncate_ragged_lines
    )


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


def _parse_int_slot(text: StringSlice, dtype: DataType) raises -> Int64:
    """Parse an integer field range-checked for dtype into an Int64 slot
    (UInt64 keeps its bit pattern)."""
    if dtype == CSV_INT64 or dtype.is_temporal():
        return parse_int64(text)
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if D.is_integral():
            if dtype == DataType.of(D):
                return parse_integer[D](text).cast[DType.int64]()
    raise Error("not an integer dtype")


def _display_name(dtype: DataType) -> String:
    if dtype == CSV_INT64:
        return "Int64"
    return dtype.name()


def _cell_text(series: Series, row: Int) -> String:
    """Canonical text for a valid cell; floats use the round-trip form."""
    if series.dtype().is_temporal():
        return format_temporal(
            series._data[Column[Int64]]._get(row), series.dtype()
        )
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            return String(series._data[Column[Scalar[D]]]._get(row))
    if series._data.isa[BoolColumn]():
        return "true" if series._data[BoolColumn]._get(row) else "false"
    return String(series._data[StringColumn]._get(row))


def _cell_valid(series: Series, row: Int) -> Bool:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            return series._data[Column[Scalar[D]]]._valid(row)
    if series._data.isa[BoolColumn]():
        return series._data[BoolColumn]._valid(row)
    return series._data[StringColumn]._valid(row)


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
