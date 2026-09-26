"""CSV reading and writing for native Mojo dataframes.

Both public read_csv overloads use the same Polars-derived pipeline.
"""
from std.collections import Dict
from .csv_types import (
    CsvField,
    CsvSchema,
    CsvOptions,
    CSV_INT64,
    CSV_FLOAT64,
    CSV_BOOL,
    CSV_STRING,
)
from .csv_reader import read_csv_explicit, read_csv_inferred
from .bool_column import BoolColumn
from .column import Column
from .string_column import StringColumn
from .dtype import DataType, NUMERIC_DTYPES
from .frame import DataFrame
from .series import Series
from .temporal import format as format_temporal


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
    """Read CSV through the Polars-derived scanner and typed builders.

    Schema names label columns positionally. Null and conversion behavior
    follows Polars; CsvField.nullable is metadata, not a parsing guard.
    buffer_size remains a positive compatibility argument; mapped reads use
    the pipeline's row-aligned chunk sizing. See docs/csv.md.
    """
    return read_csv_explicit(
        path,
        schema,
        has_header=has_header,
        separator=separator,
        quote_char=quote_char,
        comment_prefix=comment_prefix,
        skip_rows=skip_rows,
        n_rows=n_rows,
        columns=columns,
        null_values=null_values,
        ignore_errors=ignore_errors,
        truncate_ragged_lines=truncate_ragged_lines,
        encoding=encoding,
        buffer_size=buffer_size,
    )


def read_csv(
    path: String,
    *,
    infer_schema_length: Int = 100,
    schema_overrides: Dict[String, String] = Dict[String, String](),
    try_parse_dates: Bool = False,
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
    """Infer a CSV schema and decode it through the single CSV pipeline.

    Sample 100 rows by default; -1 scans all rows for inference. Temporal
    inference is enabled by try_parse_dates. schema_overrides sets explicit
    output dtypes. See docs/csv.md for Polars-compatible inference rules.
    """
    return read_csv_inferred(
        path,
        infer_schema_length=infer_schema_length,
        schema_overrides=schema_overrides,
        try_parse_dates=try_parse_dates,
        has_header=has_header,
        separator=separator,
        quote_char=quote_char,
        comment_prefix=comment_prefix,
        skip_rows=skip_rows,
        n_rows=n_rows,
        columns=columns,
        null_values=null_values,
        ignore_errors=ignore_errors,
        truncate_ragged_lines=truncate_ragged_lines,
        encoding=encoding,
        buffer_size=buffer_size,
    )


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


def _cell_text(series: Series, row: Int) raises -> String:
    """Canonical text for a valid cell; floats use the round-trip form."""
    if series.is_chunked():
        var part = series._chunk_at(row)
        return _cell_text(part[0], part[1])
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


def _cell_valid(series: Series, row: Int) raises -> Bool:
    if series.is_chunked():
        var part = series._chunk_at(row)
        return _cell_valid(part[0], part[1])
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

    def row(self, frame: DataFrame, row: Int) raises -> String:
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
    _reject_nested(frame)
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
    _reject_nested(frame)
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


def _reject_nested(frame: DataFrame) raises:
    for field in frame.schema():
        if field.dtype.is_nested():
            raise Error(
                "CSV cannot hold list or struct columns: "
                + field.name
                + " is "
                + field.dtype.name()
            )
