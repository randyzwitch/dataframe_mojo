"""CSV reader orchestration.

Both public read_csv overloads use this pipeline. This module owns mapping,
header removal, CountLines range discovery, immediate decode publication, and
array-reference reassembly.
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.collections import Dict, Optional
from .csv_infer import infer_csv_schema, _CsvInferenceState, _header_names
from .csv_input import CsvBuffer
from .csv_source import (
    _CsvCompressedInput,
    _CsvRecords,
    _csv_compression,
    _csv_span,
)
from std.memory import ArcPointer, Pointer
from .csv_types import (
    CsvSchema,
    CsvOptions,
    _CsvScanOptions,
    _Mapping,
    _map_file,
    _options,
    _projection,
)
from .csv_decode import decode_chunk
from .csv_scan import CountLines, chunk_size
from .csv_splitfields import CsvSplitFields
from ..frame import DataFrame, concat
from ..expr import Expr
from ..series import Series
from ..parallel import Job, Pool, _ProducedJobs, configured_workers


struct _ReadContext(Movable):
    var schema: CsvSchema
    var options: CsvOptions
    var keep: List[Bool]
    var predicate: Optional[Expr]
    var decoded_rows: Atomic[Int64]

    def __init__(
        out self,
        schema: CsvSchema,
        options: CsvOptions,
        keep: List[Bool],
        predicate: Optional[Expr],
    ):
        self.schema = schema.copy()
        self.options = options.copy()
        self.keep = keep.copy()
        self.predicate = predicate.copy()
        self.decoded_rows = Atomic[Int64](0)


struct _DecodeJob(Job):
    var context: ArcPointer[_ReadContext]
    var address: Int
    var length: Int
    var rows: Int
    var record: Int
    var result: DataFrame
    var owned: List[UInt8]

    def __init__(
        out self,
        context: ArcPointer[_ReadContext],
        address: Int,
        length: Int,
        rows: Int,
        record: Int,
    ) raises:
        self.owned = List[UInt8]()
        self.context = context.copy()
        self.address = address
        self.length = length
        self.rows = rows
        self.record = record
        self.result = DataFrame(List[Series](), height=0)

    @staticmethod
    def from_owned(
        context: ArcPointer[_ReadContext],
        var bytes: List[UInt8],
        rows: Int,
        record: Int,
    ) raises -> Self:
        var job = Self(context, 0, len(bytes), rows, record)
        job.owned = bytes^
        return job^

    def run(mut self) raises:
        var address = Int(self.owned.unsafe_ptr()) if len(
            self.owned
        ) else self.address
        var bytes = Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)
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
        if self.context[].predicate:
            self.result = self.result.filter(self.context[].predicate.value())

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
    """Check for a comment line at a record boundary."""
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
    """Validate explicit-schema width without legacy name checks.

    An explicit schema labels output positionally, so file header spelling and
        order are ignored. Reject a header that defines more input columns than
        the schema; a shorter header is permitted because data rows can
        legitimately supply trailing null fields.
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
    predicate: Optional[Expr] = Optional[Expr](),
) raises -> DataFrame:
    """Read an explicit schema through the source-mapped CSV pipeline.

    ``buffer_size`` remains a validated public compatibility keyword. Mapped
    input is range-split directly, so it does not set the mmap chunk size;
    unmapped input is decoded as one owned byte span.
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
    var codec = _csv_compression(mapping.span(), path)
    if codec:
        return _read_compressed_csv(
            _CsvCompressedInput(path, codec),
            Optional(schema.copy()),
            columns,
            options,
            has_header,
            100,
            Dict[String, String](),
            False,
            predicate,
        )
    var keep = _projection(schema, columns)
    if mapping.address == 0:
        # Open establishes the same missing-file error. Apply the exact mmap
        # prelude before synchronous decoding; decoding the raw file here
        # would incorrectly emit a BOM/header/skipped records as data.
        with open(path, "r") as file:
            var input = file.read_bytes()
            var result = _decode_unmapped(
                input^, schema, options, keep, has_header
            )
            return result.filter(predicate.value()) if predicate else result^
    _validate_explicit_header(mapping.span(), schema, options, has_header)
    var prelude = _prelude(mapping.span(), options, has_header)
    return _read_mapped_body(
        mapping^, schema, options, keep, prelude[0], prelude[1], predicate
    )


def _read_mapped_body(
    var mapping: _Mapping,
    schema: CsvSchema,
    options: CsvOptions,
    keep: List[Bool],
    var offset: Int,
    var record: Int,
    predicate: Optional[Expr] = Optional[Expr](),
) raises -> DataFrame:
    var n_rows = options.n_rows
    var quote_char = options.quote_char
    var bytes = mapping.span()
    if offset == len(bytes):
        var empty = decode_chunk(
            bytes[len(bytes) :], schema, options, keep, 0, record
        )
        return empty.filter(predicate.value()) if predicate else empty^
    # Give each decode worker enough input to amortize pool startup.
    # Large mapped files still use every configured worker.
    var workers = min(
        configured_workers(), max(1, (len(bytes) - offset) // (256 * 1024))
    )
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
    var context = ArcPointer(_ReadContext(schema, options, keep, predicate))
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
    predicate: Optional[Expr] = Optional[Expr](),
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
    var codec = _csv_compression(mapping.span(), path)
    if codec:
        return _read_compressed_csv(
            _CsvCompressedInput(path, codec),
            Optional[CsvSchema](),
            columns,
            options,
            has_header,
            infer_schema_length,
            schema_overrides,
            try_parse_dates,
            predicate,
        )
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
            predicate,
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
        if predicate:
            result = result.filter(predicate.value())
        _ = input^
        return result^


def read_csv_buffer_explicit(
    source: CsvBuffer,
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
    predicate: Optional[Expr] = Optional[Expr](),
) raises -> DataFrame:
    """Decode owned CSV bytes through the same typed builders as files."""
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
    var codec = _csv_compression(source._span())
    if codec:
        return _read_compressed_csv(
            _CsvCompressedInput(source, codec),
            Optional(schema.copy()),
            columns,
            options,
            has_header,
            100,
            Dict[String, String](),
            False,
            predicate,
        )
    var keep = _projection(schema, columns)
    var bytes = source._span()
    _validate_explicit_header(bytes, schema, options, has_header)
    var prelude = _prelude(bytes, options, has_header)
    return _decode_buffer_body(
        bytes, schema, options, keep, prelude[0], prelude[1], predicate
    )


def read_csv_buffer_inferred(
    source: CsvBuffer,
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
    predicate: Optional[Expr] = Optional[Expr](),
) raises -> DataFrame:
    """Infer and decode a reusable in-memory source with file-reader options."""
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
    var codec = _csv_compression(source._span())
    if codec:
        return _read_compressed_csv(
            _CsvCompressedInput(source, codec),
            Optional[CsvSchema](),
            columns,
            options,
            has_header,
            infer_schema_length,
            schema_overrides,
            try_parse_dates,
            predicate,
        )
    var bytes = source._span()
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
    return _decode_buffer_body(
        bytes,
        inferred.schema,
        options,
        keep,
        inferred.data_offset,
        inferred.record_start,
        predicate,
    )


def _decode_buffer_body(
    bytes: Span[UInt8, ImmutAnyOrigin],
    schema: CsvSchema,
    options: CsvOptions,
    keep: List[Bool],
    offset: Int,
    record: Int,
    predicate: Optional[Expr],
) raises -> DataFrame:
    # Decoding is synchronous. The caller retains its byte owner through
    # this call; numeric and string builders own the resulting columns.
    var result = decode_chunk(bytes[offset:], schema, options, keep, 0, record)
    if options.n_rows >= 0 and result.height() > options.n_rows:
        result = result.head(options.n_rows)
    if predicate:
        return result.filter(predicate.value())
    return result^


struct _MappedCsvBatches(Movable):
    """A mapped CSV cursor; only a bounded wave of decode jobs is retained."""

    var mapping: _Mapping
    var context: ArcPointer[_ReadContext]
    var offset: Int
    var record: Int
    var remaining: Int
    var batch_size: Int
    var emitted: Bool
    # The quote byte line ends are found outside of, if quoting.
    var quote: UInt8
    var quoting: Bool

    def __init__(
        out self,
        var mapping: _Mapping,
        schema: Optional[CsvSchema],
        columns: List[String],
        limit: Int,
        batch_size: Int,
        scan: _CsvScanOptions = _CsvScanOptions.defaults(),
    ) raises:
        self.mapping = mapping^
        if self.mapping.address == 0:
            raise Error("Streaming CSV requires a nonempty regular mapped file")
        var options = scan.reader()
        self.quoting = options.quote_char.byte_length() != 0
        self.quote = options.quote_char.as_bytes()[
            0
        ] if self.quoting else UInt8(34)
        var selected: CsvSchema
        if schema:
            selected = schema.value().copy()
            _validate_explicit_header(
                self.mapping.span(), selected, options, scan.has_header
            )
            var prelude = _prelude(
                self.mapping.span(), options, scan.has_header
            )
            self.offset = prelude[0]
            self.record = prelude[1]
        else:
            var inferred = infer_csv_schema(
                self.mapping.span(),
                options,
                has_header=scan.has_header,
                infer_schema_length=scan.infer_schema_length,
                try_parse_dates=scan.try_parse_dates,
                schema_overrides=scan.schema_overrides,
            )
            selected = inferred.schema.copy()
            self.offset = inferred.data_offset
            self.record = inferred.record_start
        self.context = ArcPointer(
            _ReadContext(
                selected,
                options,
                _projection(selected, columns),
                Optional[Expr](),
            )
        )
        # Batches are cut at line ends, and a comment line is a line but
        # no row, so with comments a count of lines bounds no row count:
        # read on and leave a positive limit to the slice above the scan.
        var comments = options.comment_prefix.byte_length() > 0
        self.remaining = -1 if comments and limit > 0 else limit
        self.batch_size = batch_size
        self.emitted = False

    def next(mut self) raises -> Optional[_DecodeJob]:
        if self.emitted and (
            self.offset >= self.mapping.length or self.remaining == 0
        ):
            return None
        self.emitted = True
        var size = self.batch_size if self.remaining < 0 else min(
            self.batch_size, self.remaining
        )
        var count = 0
        var end = self.offset
        if size > 0:
            var found = CountLines(self.quote, self.quoting).take_rows(
                self.mapping.span()[self.offset :], size
            )
            count = found.rows
            end = self.offset + found.last_newline + 1
        var job = _DecodeJob(
            self.context,
            self.mapping.address + self.offset,
            end - self.offset,
            count,
            self.record,
        )
        self.offset = end
        self.record += count
        if self.remaining >= 0:
            self.remaining -= count
        return job^

    def discard(mut self):
        # Every worker has copied its range. Unmap consumed whole pages,
        # rather than advising DONTNEED (Darwin may retain those pages in RSS).
        # Keep only the contiguous remaining mapping for the next wave and
        # for _Mapping's destructor, including on an early stop or error.
        var page = Int(external_call["getpagesize", Int32]())
        var end = self.offset // page * page
        if end > 0:
            var status = external_call["munmap", Int32](
                self.mapping.address, end
            )
            if status == 0:
                self.mapping.address += end
                self.mapping.length -= end
                self.offset -= end
                if self.mapping.length == 0:
                    self.mapping.address = 0


@fieldwise_init
struct _CsvReplayRecord(Copyable, Movable):
    var bytes: List[UInt8]
    var number: Int


def _compressed_prelude(
    mut records: _CsvRecords, options: CsvOptions, has_header: Bool
) raises -> Tuple[Optional[_CsvReplayRecord], Int]:
    """Discard skipped input incrementally; retain only the header/first row."""
    var number = 1
    var first = True
    var skip_empty = has_header
    var remaining = options.skip_rows
    while True:
        var next = records.next()
        if not next:
            return (Optional[_CsvReplayRecord](), number)
        var raw = next.take()
        if first:
            first = False
            var bytes = _csv_span(raw)
            if (
                len(bytes) >= 3
                and bytes[0] == 239
                and bytes[1] == 187
                and bytes[2] == 191
            ):
                var stripped = List[UInt8]()
                stripped.extend(bytes[3:])
                raw = stripped^
        var bytes = _csv_span(raw)
        if skip_empty and _empty_record(bytes, 0, len(bytes)):
            number += 1
            continue
        skip_empty = False
        if _comment_at(bytes, 0, options.comment_prefix):
            number += 1
            continue
        if remaining > 0:
            remaining -= 1
            number += 1
            continue
        return (_CsvReplayRecord(raw^, number), number + 1)


def _next_compressed_record(
    mut records: _CsvRecords,
    mut pending: Optional[_CsvReplayRecord],
    mut number: Int,
) raises -> Optional[_CsvReplayRecord]:
    if pending:
        return pending.take()
    var next = records.next()
    if not next:
        return None
    var at = number
    number += 1
    return _CsvReplayRecord(next.take(), at)


struct _CompressedCsvBatches(Movable):
    """Bounded owned CSV jobs over an incremental decompressed transport.

    Finite inference retains sampled data records, never skipped/comment
    prefixes. Full inference reduces type masks on the first pass and reopens
    the same compressed source for decoding on the second pass.
    """

    var records: _CsvRecords
    var context: ArcPointer[_ReadContext]
    var replay: List[_CsvReplayRecord]
    var replay_offset: Int
    var pending: Optional[_CsvReplayRecord]
    var live_record: Int
    var remaining: Int
    var batch_size: Int
    var emitted: Bool
    var ended: Bool

    def __init__(
        out self,
        var input: _CsvCompressedInput,
        schema: Optional[CsvSchema],
        columns: List[String],
        options: CsvOptions,
        has_header: Bool,
        infer_schema_length: Int,
        schema_overrides: Dict[String, String],
        try_parse_dates: Bool,
        batch_size: Int,
    ) raises:
        if infer_schema_length < -1:
            raise Error("infer_schema_length must be nonnegative or -1")
        self.records = _CsvRecords(input^, options.quote_char)
        self.replay = List[_CsvReplayRecord]()
        self.replay_offset = 0
        self.pending = Optional[_CsvReplayRecord]()
        var prelude = _compressed_prelude(self.records, options, has_header)
        self.live_record = prelude[1]
        var first = prelude[0].take() if prelude[0] else _CsvReplayRecord(
            List[UInt8](), prelude[1]
        )
        var names = List[String]()
        var selected: CsvSchema
        if has_header:
            if schema:
                var header_options = options.copy()
                header_options.skip_rows = 0
                _validate_explicit_header(
                    _csv_span(first.bytes), schema.value(), header_options, True
                )
            elif len(first.bytes):
                names = _header_names(_csv_span(first.bytes), options)
        elif len(first.bytes):
            self.pending = first.copy()
        if schema:
            selected = schema.value().copy()
        else:
            var inference = _CsvInferenceState(
                names, has_header, infer_schema_length, try_parse_dates
            )
            var sampled = 0
            var limit = max(1, infer_schema_length)
            while infer_schema_length < 0 or sampled < limit:
                var next = _next_compressed_record(
                    self.records, self.pending, self.live_record
                )
                if not next:
                    break
                var item = next.take()
                var bytes = _csv_span(item.bytes)
                if _comment_at(bytes, 0, options.comment_prefix):
                    continue
                inference.add(bytes, options)
                if infer_schema_length >= 0:
                    self.replay.append(item^)
                sampled += 1
            selected = inference.finish(schema_overrides)
            if infer_schema_length < 0:
                self.records.reset()
                var again = _compressed_prelude(
                    self.records, options, has_header
                )
                self.live_record = again[1]
                if not has_header and again[0]:
                    self.pending = again[0].take()
            else:
                # During sampling, _next_record consumes live records only.
                self.replay_offset = 0
        self.context = ArcPointer(
            _ReadContext(
                selected,
                options,
                _projection(selected, columns),
                Optional[Expr](),
            )
        )
        var comments = options.comment_prefix.byte_length() > 0
        self.remaining = (
            -1 if comments and options.n_rows > 0 else options.n_rows
        )
        self.batch_size = max(1, batch_size)
        self.emitted = False
        self.ended = False

    def _next_record(mut self) raises -> Optional[_CsvReplayRecord]:
        if self.pending:
            return self.pending.take()
        if self.replay_offset < len(self.replay):
            var item = self.replay[self.replay_offset].copy()
            self.replay_offset += 1
            return item^
        return self._next_live_record()

    def _next_live_record(mut self) raises -> Optional[_CsvReplayRecord]:
        return _next_compressed_record(
            self.records, self.pending, self.live_record
        )

    def next(mut self) raises -> Optional[_DecodeJob]:
        if self.ended:
            return None
        var bytes = List[UInt8]()
        var count = 0
        var start = self.live_record
        var size = self.batch_size if self.remaining < 0 else min(
            self.batch_size, self.remaining
        )
        # A very wide record is indivisible; otherwise each job is also
        # bounded in bytes, even when the requested row batch is enormous.
        while count < size and len(bytes) < 1024 * 1024:
            var next = self._next_record()
            if not next:
                self.ended = True
                break
            var item = next.take()
            if count and item.number != start + count:
                # Sampled comments were discarded. Split at their gap to
                # preserve original CSV record numbers in decoder errors.
                self.pending = item^
                break
            if count == 0:
                start = item.number
            bytes.extend(_csv_span(item.bytes))
            count += 1
        if self.remaining >= 0:
            self.remaining -= count
            if self.remaining == 0:
                self.finish()
        if count == 0 and self.emitted:
            return None
        self.emitted = True
        return _DecodeJob.from_owned(self.context, bytes^, count, start)

    def finish(mut self) raises:
        self.records.close()
        self.replay.clear()
        self.pending = Optional[_CsvReplayRecord]()
        self.ended = True


def _read_compressed_csv(
    var input: _CsvCompressedInput,
    schema: Optional[CsvSchema],
    columns: List[String],
    options: CsvOptions,
    has_header: Bool,
    infer_schema_length: Int,
    schema_overrides: Dict[String, String],
    try_parse_dates: Bool,
    predicate: Optional[Expr],
) raises -> DataFrame:
    var batches = _CompressedCsvBatches(
        input^,
        schema,
        columns,
        options,
        has_header,
        infer_schema_length,
        schema_overrides,
        try_parse_dates,
        65536,
    )
    var frames = List[DataFrame]()
    var rows = 0
    while True:
        var next = batches.next()
        if not next:
            break
        var job = next.take()
        job.run()
        var frame = job^.into_frame()
        rows += frame.height()
        frames.append(frame^)
        if options.n_rows >= 0 and rows >= options.n_rows:
            break
    batches.finish()
    var result = concat(frames) if len(frames) > 1 else frames.pop(0)
    if options.n_rows >= 0 and result.height() > options.n_rows:
        result = result.head(options.n_rows)
    if predicate:
        return result.filter(predicate.value())
    return result^


struct _CsvBatches(Movable):
    """Mapped or compressed source, sharing the existing worker-job interface."""

    var mapped: List[_MappedCsvBatches]
    var compressed: List[_CompressedCsvBatches]

    def __init__(
        out self,
        var mapping: _Mapping,
        schema: Optional[CsvSchema],
        columns: List[String],
        limit: Int,
        batch_size: Int,
        scan: _CsvScanOptions = _CsvScanOptions.defaults(),
    ) raises:
        self.mapped = List[_MappedCsvBatches]()
        self.compressed = List[_CompressedCsvBatches]()
        self.mapped.append(
            _MappedCsvBatches(
                mapping^, schema, columns, limit, batch_size, scan
            )
        )

    def __init__(out self, var compressed: _CompressedCsvBatches):
        self.mapped = List[_MappedCsvBatches]()
        self.compressed = List[_CompressedCsvBatches]()
        self.compressed.append(compressed^)

    def next(mut self) raises -> Optional[_DecodeJob]:
        if len(self.compressed):
            return self.compressed[0].next()
        return self.mapped[0].next()

    def project_none(mut self):
        # Dataset scans can request only virtual/path columns. Preserve the
        # physical record count without parsing any unneeded typed values.
        if len(self.compressed):
            ref keep = self.compressed[0].context[].keep
            for index in range(len(keep)):
                keep[index] = False
        else:
            ref keep = self.mapped[0].context[].keep
            for index in range(len(keep)):
                keep[index] = False

    def discard(mut self):
        if len(self.mapped):
            self.mapped[0].discard()

    def finish(mut self) raises:
        if len(self.compressed):
            self.compressed[0].finish()


def _open_csv_batches(
    path: String,
    schema: Optional[CsvSchema],
    columns: List[String],
    limit: Int,
    batch_size: Int,
    scan: _CsvScanOptions = _CsvScanOptions.defaults(),
) raises -> Optional[_CsvBatches]:
    """Open either transport; None retains the plain unmapped fallback."""
    var mapping = _map_file(path)
    var codec = _csv_compression(mapping.span(), path)
    if codec:
        var options = scan.reader()
        options.n_rows = limit
        return _CsvBatches(
            _CompressedCsvBatches(
                _CsvCompressedInput(path, codec),
                schema,
                columns,
                options,
                scan.has_header,
                scan.infer_schema_length,
                scan.schema_overrides,
                scan.try_parse_dates,
                batch_size,
            )
        )
    if mapping.address == 0:
        return None
    return _CsvBatches(mapping^, schema, columns, limit, batch_size, scan)
