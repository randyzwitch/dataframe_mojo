"""CSV orchestration port of Polars py-1.44.2 read_impl::parse_csv.

Both public read_csv overloads use this pipeline. This module owns mapping,
header removal, CountLines range discovery, immediate decode publication, and
array-reference reassembly.
"""
from std.atomic import Atomic
from std.collections import Dict
from .csv_infer import infer_csv_schema
from std.memory import ArcPointer, Pointer
from .csv_types import (
    CsvSchema,
    CsvOptions,
    _Mapping,
    _map_file,
    _options,
    _projection,
)
from .csv_decode import decode_chunk
from .csv_scan import CountLines, chunk_size
from .csv_splitfields import CsvSplitFields
from .frame import DataFrame, concat
from .series import Series
from .parallel import Job, Pool, _ProducedJobs, configured_workers


struct _ReadContext(Movable):
    var schema: CsvSchema
    var options: CsvOptions
    var keep: List[Bool]
    var decoded_rows: Atomic[Int64]

    def __init__(
        out self, schema: CsvSchema, options: CsvOptions, keep: List[Bool]
    ):
        self.schema = schema.copy()
        self.options = options.copy()
        self.keep = keep.copy()
        self.decoded_rows = Atomic[Int64](0)


struct _DecodeJob(Job):
    var context: ArcPointer[_ReadContext]
    var address: Int
    var length: Int
    var rows: Int
    var record: Int
    var result: DataFrame

    def __init__(
        out self,
        context: ArcPointer[_ReadContext],
        address: Int,
        length: Int,
        rows: Int,
        record: Int,
    ) raises:
        self.context = context.copy()
        self.address = address
        self.length = length
        self.rows = rows
        self.record = record
        self.result = DataFrame(List[Series](), height=0)

    def run(mut self) raises:
        var bytes = Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=self.address
            )
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            length=self.length,
        )
        self.result = decode_chunk(
            bytes,
            self.context[].schema,
            self.context[].options,
            self.context[].keep,
            self.rows,
            self.record,
        )
        # read_impl validates CountLines' estimate after decoding. Comments
        # make an undershoot expected, but an overshoot is always malformed.
        var actual = self.result.height()
        var malformed = actual > self.rows or (
            actual < self.rows
            and self.context[].options.comment_prefix.byte_length() == 0
        )
        if malformed:
            var message = String(
                "CSV malformed: expected ",
                self.rows,
                " rows, actual ",
                actual,
                " rows in range length ",
                self.length,
            )
            if self.context[].options.ignore_errors:
                # This runtime has no warning subsystem; retain Polars'
                # observable permissive-mode warning rather than dropping it.
                print("warning: " + message)
            else:
                raise Error(message)
        # Exactly as read_impl, this counter exists only when n_rows requests
        # its probabilistic early stop.
        if self.context[].options.n_rows >= 0:
            _ = self.context[].decoded_rows.fetch_add(Int64(actual))

    def into_frame(deinit self) -> DataFrame:
        return self.result^


def _line_end(
    bytes: Span[UInt8, ImmutAnyOrigin], start: Int, options: CsvOptions
) -> Int:
    """Header/prelude record boundary using the same field iterator."""
    var quoted = options.quote_char.byte_length() != 0
    var quote = options.quote_char.as_bytes()[0] if quoted else UInt8(34)
    var fields = CsvSplitFields(options.separator.as_bytes()[0], quote, quoted)
    var input = bytes[start:]
    while True:
        var field = fields.next(input)
        if not field:
            return len(bytes)
        var value = field.value().copy()
        if value.ends_record:
            return start + value.end + Int(value.terminator != 0)


def _comment_at(
    bytes: Span[UInt8, ImmutAnyOrigin], start: Int, prefix: String
) -> Bool:
    """Port parser.rs ``is_comment_line`` at a record boundary."""
    var value = prefix.as_bytes()
    if len(value) == 0 or start + len(value) > len(bytes):
        return False
    for i in range(len(value)):
        if bytes[start + i] != value[i]:
            return False
    return True


def _empty_record(
    bytes: Span[UInt8, ImmutAnyOrigin], start: Int, end: Int
) -> Bool:
    """The SkipEmpty state accepts an empty line and a single CR line."""
    var length = end - start
    if length > 0 and bytes[end - 1] == 10:
        length -= 1
    if length == 0:
        return True
    return length == 1 and bytes[start] == 13


def _before_header(
    bytes: Span[UInt8, ImmutAnyOrigin], options: CsvOptions, has_header: Bool
) -> Tuple[Int, Int]:
    """Return the record boundary immediately before the optional header."""
    var start = 0
    var record = 1
    if (
        len(bytes) >= 3
        and bytes[0] == 239
        and bytes[1] == 187
        and bytes[2] == 191
    ):
        start = 3

    # `SkipEmpty` exists only when a header is requested.
    if has_header:
        while start < len(bytes):
            var end = _line_end(bytes, start, options)
            if not _empty_record(bytes, start, end):
                break
            start = end
            record += 1

    # `SkipRowsBeforeHeader`: comments are discarded but do not consume a
    # skipped record. The next non-comment is the header (or data without one).
    var remaining = options.skip_rows
    while start < len(bytes):
        if _comment_at(bytes, start, options.comment_prefix):
            start = _line_end(bytes, start, options)
            record += 1
            continue
        if remaining == 0:
            break
        start = _line_end(bytes, start, options)
        record += 1
        remaining -= 1
    return (start, record)


def _prelude(
    bytes: Span[UInt8, ImmutAnyOrigin], options: CsvOptions, has_header: Bool
) -> Tuple[Int, Int]:
    """Port streaming.rs SkipEmpty/SkipRowsBeforeHeader/SkipHeader.

    ``skip_rows`` counts valid CSV records, never comments.  It therefore
    uses the quote-aware field iterator, whereas comment detection occurs
    only at the start of a record.  ``record`` is the source record number
    passed into decode jobs after the prelude.
    """
    var before = _before_header(bytes, options, has_header)
    var start = before[0]
    var record = before[1]
    if has_header and start < len(bytes):
        start = _line_end(bytes, start, options)
        record += 1
    return (start, record)


def _validate_explicit_header(
    bytes: Span[UInt8, ImmutAnyOrigin],
    schema: CsvSchema,
    options: CsvOptions,
    has_header: Bool,
) raises:
    """Match Polars' explicit-schema width check without legacy name checks.

    An explicit schema labels output positionally, so file header spelling and
    order are ignored. Polars still rejects a header that defines more input
    columns than that schema; a shorter header is permitted because data rows
    can legitimately supply trailing null fields.
    """
    if not has_header:
        return
    var before = _before_header(bytes, options, has_header)
    var start = before[0]
    if start >= len(bytes):
        return
    var end = _line_end(bytes, start, options)
    var quote = options.quote_char.byte_length() != 0
    var quote_byte = options.quote_char.as_bytes()[0] if quote else UInt8(34)
    var fields = CsvSplitFields(
        options.separator.as_bytes()[0], quote_byte, quote
    )
    var count = 0
    var input = bytes[start:end]
    while True:
        var field = fields.next(input)
        if not field:
            break
        count += 1
    if count > len(schema):
        raise Error(
            String(
                "provided schema does not match number of columns in file (",
                len(schema),
                " != ",
                count,
                " in file)",
            )
        )


def _decode_unmapped(
    var input: List[UInt8],
    schema: CsvSchema,
    options: CsvOptions,
    keep: List[Bool],
    has_header: Bool,
) raises -> DataFrame:
    """Fallback for sources _map_file cannot map, with the same prelude.

    The bytes remain alive for the synchronous borrowed decode. This path is
    intentionally serial; mmap range jobs remain the parallel fast path.
    """
    var bytes = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=input.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(input),
    )
    _validate_explicit_header(bytes, schema, options, has_header)
    var prelude = _prelude(bytes, options, has_header)
    var offset = prelude[0]
    var result = decode_chunk(
        bytes[offset:], schema, options, keep, 0, prelude[1]
    )
    if options.n_rows >= 0 and result.height() > options.n_rows:
        result = result.head(options.n_rows)
    _ = input^
    return result^


def read_csv_explicit(
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
    """Read an explicit schema through the source-mapped CSV pipeline.

    ``buffer_size`` remains a validated public compatibility keyword. Mapped
    input is range-split directly, as in Polars, so it does not set the mmap
    chunk size; unmapped input is decoded as one owned byte span.
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
    var mapping = _map_file(path)
    if mapping.address == 0:
        # Open establishes the same missing-file error. Apply the exact mmap
        # prelude before synchronous decoding; decoding the raw file here
        # would incorrectly emit a BOM/header/skipped records as data.
        with open(path, "r") as file:
            var input = file.read_bytes()
            return _decode_unmapped(input^, schema, options, keep, has_header)
    _validate_explicit_header(mapping.span(), schema, options, has_header)
    var prelude = _prelude(mapping.span(), options, has_header)
    return _read_mapped_body(
        mapping^, schema, options, keep, prelude[0], prelude[1]
    )


def _read_mapped_body(
    var mapping: _Mapping,
    schema: CsvSchema,
    options: CsvOptions,
    keep: List[Bool],
    var offset: Int,
    var record: Int,
) raises -> DataFrame:
    var n_rows = options.n_rows
    var quote_char = options.quote_char
    var bytes = mapping.span()
    if offset == len(bytes):
        return decode_chunk(
            bytes[len(bytes) :], schema, options, keep, 0, record
        )
    var workers = configured_workers()
    var projected_width = 0
    for selected in keep:
        projected_width += Int(selected)
    var size = chunk_size(len(bytes) - offset, workers, projected_width)
    var quoting = quote_char.byte_length() != 0
    var quote = quote_char.as_bytes()[0] if quoting else UInt8(34)
    var counter = CountLines(quote, quoting)
    # read_impl's EOF branch asks whether the whole post-prelude body, not
    # the final chunk, begins with a comment prefix.
    var body_is_comment = _comment_at(bytes, offset, options.comment_prefix)
    var context = ArcPointer(_ReadContext(schema, options, keep))
    # Each pair of adjacent ranges spans at least one initial window;
    # otherwise find_next would have included the second boundary in the
    # first range. Doubling the hint only reduces the maximum job count.
    var capacity = 2 * ((len(bytes) - offset + size - 1) // size) + 2
    # Polars uses a persistent Rayon pool. This repository's scoped Pool is
    # an explicit remaining runtime deviation (startup once per read).
    var pool = Pool(workers)
    var produced = _ProducedJobs[_DecodeJob](capacity)
    pool.run_produced(produced)
    while offset < len(bytes):
        var boundary = counter.find_next(bytes[offset:], size)
        size = boundary.chunk_size
        var count = boundary.rows
        var end: Int
        if count == 0:
            end = len(bytes)
            count = 0 if body_is_comment else 1
        else:
            end = offset + boundary.last_newline + 1
        produced.submit(
            _DecodeJob(
                context, mapping.address + offset, end - offset, count, record
            )
        )
        offset = end
        record += count
        # Match read_impl: this is an asynchronous, probabilistic stop. The
        # final ordered head remains authoritative for exact n_rows output.
        if n_rows >= 0 and context[].decoded_rows.load() > Int64(n_rows):
            break
    var jobs = produced.finish()
    pool.release()
    var frames = List[DataFrame](capacity=len(jobs))
    # Jobs retain source order; this is equivalent to Polars sorting by input
    # byte address, without introducing any output-value sort.
    jobs.reverse()
    while len(jobs) > 0:
        frames.append(jobs.pop().into_frame())
    # Decode jobs borrow mmap bytes through an address. Keep the mapping alive
    # until every worker is joined and its frame owns its output buffers.
    _ = mapping^
    var result = concat(frames) if len(frames) > 1 else frames.pop(0)
    if n_rows >= 0 and result.height() > n_rows:
        return result.head(n_rows)
    return result^


def read_csv_inferred(
    path: String,
    *,
    infer_schema_length: Int = 100,
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
    try_parse_dates: Bool = False,
) raises -> DataFrame:
    """Infer and decode using one mapped input and one prelude traversal.

    ``buffer_size`` is validated for public compatibility; mapped input is
    split by record-aligned byte ranges. ``try_parse_dates`` is forwarded to
    the inference port, which decides whether its supported temporal types
    can be inferred.
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
    var mapping = _map_file(path)
    if mapping.address != 0:
        var inferred = infer_csv_schema(
            mapping.span(),
            options,
            has_header=has_header,
            infer_schema_length=infer_schema_length,
            schema_overrides=schema_overrides,
            try_parse_dates=try_parse_dates,
        )
        if len(inferred.schema) == 0:
            return DataFrame([])
        var keep = _projection(inferred.schema, columns)
        return _read_mapped_body(
            mapping^,
            inferred.schema,
            options,
            keep,
            inferred.data_offset,
            inferred.record_start,
        )
    with open(path, "r") as file:
        var input = file.read_bytes()
        var bytes = Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=input.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            length=len(input),
        )
        var inferred = infer_csv_schema(
            bytes,
            options,
            has_header=has_header,
            infer_schema_length=infer_schema_length,
            schema_overrides=schema_overrides,
            try_parse_dates=try_parse_dates,
        )
        if len(inferred.schema) == 0:
            return DataFrame([])
        var keep = _projection(inferred.schema, columns)
        var result = decode_chunk(
            bytes[inferred.data_offset :],
            inferred.schema,
            options,
            keep,
            0,
            inferred.record_start,
        )
        if n_rows >= 0 and result.height() > n_rows:
            result = result.head(n_rows)
        _ = input^
        return result^
