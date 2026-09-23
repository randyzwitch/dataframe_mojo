"""CSV schema inference for the reader.

The clean reader supplies the resulting schema
and ``data_offset`` to its chunk pipeline; this module deliberately does not
open files or decode rows.

The current CSV API infers Bool, Int64, Float64, and String. With
``try_parse_dates=True``, it also infers the supported ISO Date,
Datetime[us], and Time forms. Date inference is opt-in and defaults to false.
Decimal-comma, Null, optional Int128, per-column
null values, and column-name replacement are outside this API; ``CsvOptions``
has all-column null tokens and ``schema_overrides`` is the existing
name-to-dtype adapter. As in Polars, header names are always converted with
UTF-8 loss replacement; body UTF-8 checking belongs to the later decoder path.
"""
from std.collections import Dict

from .csv_types import CsvField, CsvOptions, CsvSchema
from .csv_splitfields import CsvSplitFields
from .dtype import DataType
from .temporal import parse as parse_temporal


comptime _BOOL = 1
comptime _INT = 2
comptime _FLOAT = 4
comptime _STRING = 8
comptime _DATE = 16
comptime _DATETIME = 32
comptime _TIME = 64


@fieldwise_init
struct CsvInference(Copyable):
    """Inferred schema plus the first content-record offset.

    ``data_offset`` is exactly the byte position left by streaming.rs after
    SkipEmpty, SkipRowsBeforeHeader, and SkipHeader.  ``record_start`` counts
    physical records through that prelude for decoder diagnostics.
    """

    var schema: CsvSchema
    var data_offset: Int
    var record_start: Int


def _quote(options: CsvOptions) -> Tuple[UInt8, Bool]:
    if options.quote_char.byte_length() == 1:
        return (options.quote_char.as_bytes()[0], True)
    return (UInt8(34), False)


def _record_end(
    bytes: Span[UInt8, ImmutAnyOrigin], start: Int, options: CsvOptions
) -> Int:
    """Find one source record with the same SplitFields parity as Polars."""
    var quote = _quote(options)
    var fields = CsvSplitFields(
        options.separator.as_bytes()[0], quote[0], quote[1]
    )
    var input = bytes[start:]
    while True:
        var next = fields.next(input)
        if not next:
            return len(bytes)
        var field = next.value().copy()
        if field.ends_record:
            return start + field.end + Int(field.terminator != 0)


def _comment_at(
    bytes: Span[UInt8, ImmutAnyOrigin], start: Int, prefix: String
) -> Bool:
    var marker = prefix.as_bytes()
    if len(marker) == 0 or start + len(marker) > len(bytes):
        return False
    for i in range(len(marker)):
        if bytes[start + i] != marker[i]:
            return False
    return True


def _empty_record(
    bytes: Span[UInt8, ImmutAnyOrigin], start: Int, end: Int
) -> Bool:
    var length = end - start
    if length > 0 and bytes[end - 1] == 10:
        length -= 1
    return length == 0 or (length == 1 and bytes[start] == 13)


def _trim_cr(bytes: Span[UInt8, ImmutAnyOrigin]) -> Span[UInt8, ImmutAnyOrigin]:
    # streaming.rs hands schema inference records without LF and with a
    # possible CR. Our record ranges include LF, so normalize to that input.
    var end = len(bytes)
    if end > 0 and bytes[end - 1] == 10:
        end -= 1
    if end > 0 and bytes[end - 1] == 13:
        end -= 1
    return bytes[0:end]


def _unwrapped(
    raw: Span[UInt8, ImmutAnyOrigin], needs_escaping: Bool
) -> Span[UInt8, ImmutAnyOrigin]:
    # Exact infer_headers/infer_types_from_line behavior: remove wrappers for
    # inference, but retain doubled quotes rather than unescaping them.
    if needs_escaping and len(raw) >= 2:
        return raw[1 : len(raw) - 1]
    return raw


def _ascii_lower(byte: UInt8) -> UInt8:
    if byte >= 65 and byte <= 90:
        return byte + UInt8(32)
    return byte


def _ascii_equal_ci(bytes: Span[UInt8, ImmutAnyOrigin], text: String) -> Bool:
    var target = text.as_bytes()
    if len(bytes) != len(target):
        return False
    for i in range(len(bytes)):
        if _ascii_lower(bytes[i]) != target[i]:
            return False
    return True


def _ascii_equal(bytes: Span[UInt8, ImmutAnyOrigin], text: String) -> Bool:
    var target = text.as_bytes()
    if len(bytes) != len(target):
        return False
    for i in range(len(bytes)):
        if bytes[i] != target[i]:
            return False
    return True


def _digits(bytes: Span[UInt8, ImmutAnyOrigin], start: Int, end: Int) -> Bool:
    if start >= end:
        return False
    for i in range(start, end):
        if bytes[i] < 48 or bytes[i] > 57:
            return False
    return True


def _float_regex(bytes: Span[UInt8, ImmutAnyOrigin]) -> Bool:
    """Byte equivalent of Polars' FLOAT_RE in utils/other.rs.

    ``^[-+]?((\\d*\\.\\d+)([eE][-+]?\\d+)?|inf|NaN|(\\d+)[eE][-+]?\\d+|\\d+\\.)$``
    is intentionally narrower than the numeric parser: integers are handled
    by INTEGER_RE, while signed inf/NaN retain the regex's exact spelling.
    """
    var n = len(bytes)
    var start = 0
    if n > 0 and (bytes[0] == 43 or bytes[0] == 45):
        start = 1
    if start >= n:
        return False
    var rest = bytes[start:]
    if _ascii_equal(rest, "inf") or _ascii_equal(rest, "NaN"):
        return True

    var dot = -1
    var exponent = -1
    for i in range(start, n):
        if bytes[i] == 46:
            if dot >= 0:
                return False
            dot = i
        elif bytes[i] == 69 or bytes[i] == 101:
            if exponent >= 0:
                return False
            exponent = i
    if exponent >= 0:
        var exp_start = exponent + 1
        if exp_start < n and (bytes[exp_start] == 43 or bytes[exp_start] == 45):
            exp_start += 1
        if not _digits(bytes, exp_start, n):
            return False
        # The mantissa alternatives are \d*\.\d+ or \d+.
        if dot >= 0:
            return _digits(bytes, dot + 1, exponent) and (
                dot == start or _digits(bytes, start, dot)
            )
        return _digits(bytes, start, exponent)
    if dot >= 0:
        # The no-exponent alternatives are \d*\.\d+ and \d+\.
        if dot == n - 1:
            return _digits(bytes, start, dot)
        return _digits(bytes, dot + 1, n) and (
            dot == start or _digits(bytes, start, dot)
        )
    return False


def _infer_temporal(raw: Span[UInt8, ImmutAnyOrigin]) -> Int:
    """The ISO subset of polars_time infer_pattern_single.

    Polars reaches this branch only after Boolean, Float, and Integer regexes.
    It tries Date, Time, then Datetime. The existing temporal parser supplies
    that ISO subset; non-ISO DMY and compact chrono patterns stay String until
    the public CSV API exposes matching format support.
    """
    var text = String(from_utf8_lossy=raw)
    try:
        _ = parse_temporal(text, DataType.DATE)
        return _DATE
    except:
        pass
    try:
        _ = parse_temporal(text, DataType.TIME)
        return _TIME
    except:
        pass
    try:
        _ = parse_temporal(text, DataType.datetime("us"))
        return _DATETIME
    except:
        return _STRING


def _infer_field(
    raw: Span[UInt8, ImmutAnyOrigin], try_parse_dates: Bool
) -> Int:
    """Port infer_field_schema Boolean, Float, Integer, temporal ordering."""
    if _ascii_equal_ci(raw, "true") or _ascii_equal_ci(raw, "false"):
        return _BOOL
    if _float_regex(raw):
        return _FLOAT
    var start = 0
    if len(raw) > 0 and raw[0] == 45:
        start = 1
    if _digits(raw, start, len(raw)):
        # With no dtype-i128 feature Polars preserves INTEGER_RE as Int64
        # even when parse::<i64>() overflows. The clean API has no Int128.
        return _INT
    if try_parse_dates:
        return _infer_temporal(raw)
    return _STRING


def _is_null(raw: Span[UInt8, ImmutAnyOrigin], options: CsvOptions) -> Bool:
    for marker in options.null_values:
        var target = marker.as_bytes()
        if len(raw) != len(target):
            continue
        var same = True
        for i in range(len(raw)):
            if raw[i] != target[i]:
                same = False
                break
        if same:
            return True
    return False


def _header_names(
    line: Span[UInt8, ImmutAnyOrigin], options: CsvOptions
) raises -> List[String]:
    var trimmed = _trim_cr(line)
    var quote = _quote(options)
    var fields = CsvSplitFields(
        options.separator.as_bytes()[0], quote[0], quote[1]
    )
    var names = List[String]()
    var counts = Dict[String, Int]()
    var used = Dict[String, Bool]()
    while True:
        var next = fields.next(trimmed)
        if not next:
            break
        var field = next.value().copy()
        var raw = _unwrapped(field.bytes(trimmed), field.needs_escaping)
        # infer_headers uses String::from_utf8_lossy independently of the
        # configured body encoding.
        var original = String(from_utf8_lossy=raw)
        var count = counts[original] if original in counts else 0
        var name = original.copy()
        if count != 0:
            name += "_duplicated_" + String(count - 1)
        if name in used:
            raise Error(
                "CSV header de-duplication produced an existing name: " + name
            )
        used[name] = True
        counts[original] = count + 1
        names.append(name^)
    return names^


def _apply_overrides(
    names: List[String], overrides: Dict[String, String]
) raises -> Dict[String, DataType]:
    var result = Dict[String, DataType]()
    for item in overrides.items():
        var known = False
        for name in names:
            if name == item.key:
                known = True
                break
        if not known:
            continue
        if not DataType.is_known(item.value):
            raise Error("Unknown dtype in schema_overrides: " + item.value)
        result[item.key] = DataType.parse(item.value)
    return result^


def infer_csv_schema(
    bytes: Span[UInt8, ImmutAnyOrigin],
    options: CsvOptions,
    *,
    has_header: Bool = True,
    infer_schema_length: Int = 100,
    try_parse_dates: Bool = False,
    schema_overrides: Dict[String, String] = Dict[String, String](),
) raises -> CsvInference:
    """Infer the clean CSV schema and return the prelude's content offset.

    ``infer_schema_length`` follows Polars' ``Option<usize>`` adaptation:
    ``-1`` samples every data record, ``0`` samples enough width but forces
    String, and a positive value caps sampled non-comment records.
    ``try_parse_dates`` is Polars' opt-in temporal inference switch.
    """
    # Options are validated by the reader before inference.
    if infer_schema_length < -1:
        raise Error("infer_schema_length must be nonnegative or -1")

    var start = 0
    var record = 1
    if (
        len(bytes) >= 3
        and bytes[0] == 239
        and bytes[1] == 187
        and bytes[2] == 191
    ):
        start = 3

    # streaming.rs State::SkipEmpty is entered only with a header.
    if has_header:
        while start < len(bytes):
            var end = _record_end(bytes, start, options)
            if not _empty_record(bytes, start, end):
                break
            start = end
            record += 1

    # State::SkipRowsBeforeHeader. Comments never reduce skip_rows.
    var remaining = options.skip_rows
    while start < len(bytes):
        if _comment_at(bytes, start, options.comment_prefix):
            start = _record_end(bytes, start, options)
            record += 1
            continue
        if remaining == 0:
            break
        start = _record_end(bytes, start, options)
        record += 1
        remaining -= 1

    var names = List[String]()
    if has_header and start < len(bytes):
        var header_end = _record_end(bytes, start, options)
        names = _header_names(bytes[start:header_end], options)
        start = header_end
        record += 1

    var candidates = List[Int](capacity=len(names))
    for _ in range(len(names)):
        candidates.append(0)
    # `read_until_start_and_infer_schema` returns leftover at this position;
    # sampling below must not consume the handoff offset exposed to read_impl.
    var data_offset = start
    var record_start = record
    var sample_rows = 0
    # `ContentInspect` falls through to `InferCollect` for the first record.
    # Therefore a zero inference length still observes one row's width.
    var limit = infer_schema_length
    if limit == 0:
        limit = 1
    while start < len(bytes) and (
        infer_schema_length < 0 or sample_rows < limit
    ):
        if _comment_at(bytes, start, options.comment_prefix):
            start = _record_end(bytes, start, options)
            record += 1
            continue
        var end = _record_end(bytes, start, options)
        var line = _trim_cr(bytes[start:end])
        var quote = _quote(options)
        var fields = CsvSplitFields(
            options.separator.as_bytes()[0], quote[0], quote[1]
        )
        var column = 0
        while True:
            var next = fields.next(line)
            if not next:
                break
            var field = next.value().copy()
            if column >= len(names):
                if has_header:
                    break
                names.append("column_" + String(column + 1))
                candidates.append(0)
            if infer_schema_length == 0:
                candidates[column] |= _STRING
            else:
                var raw = field.bytes(line)
                if len(raw) == 0:
                    # Polars records nullable but adds no type possibility.
                    pass
                else:
                    raw = _unwrapped(raw, field.needs_escaping)
                    if not _is_null(raw, options):
                        candidates[column] |= _infer_field(raw, try_parse_dates)
            column += 1
        start = end
        record += 1
        sample_rows += 1

    if len(names) == 0:
        raise Error("CSV inference found no columns")
    var overrides = _apply_overrides(names, schema_overrides)
    var fields = List[CsvField](capacity=len(names))
    for i in range(len(names)):
        var dtype = DataType.STRING
        if names[i] in overrides:
            dtype = overrides[names[i]]
        elif candidates[i] == _BOOL:
            dtype = DataType.BOOL
        elif candidates[i] == _INT:
            dtype = DataType.INT64
        elif candidates[i] == _FLOAT or candidates[i] == (_INT | _FLOAT):
            dtype = DataType.FLOAT64
        elif candidates[i] == _DATE:
            dtype = DataType.DATE
        elif candidates[i] == _DATETIME:
            dtype = DataType.datetime("us")
        elif candidates[i] == _TIME:
            dtype = DataType.TIME
        elif candidates[i] == _STRING:
            dtype = DataType.STRING
        # Polars' Null/unsupported result becomes the clean API's String.
        fields.append(CsvField(names[i], dtype, True))
    return CsvInference(CsvSchema(fields^), data_offset, record_start)
