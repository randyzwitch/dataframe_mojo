"""CSV orchestration port of Polars py-1.44.2 read_impl::parse_csv.

The public reader is switched only after differential validation. This module
owns mapping, header removal, CountLines range discovery, immediate decode
publication, and array-reference reassembly. The legacy reader remains the
reference during the port.
"""
from std.atomic import Atomic
from std.memory import ArcPointer, Pointer
from .csv import CsvSchema, CsvOptions, _map_file, _options, _projection
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


def _prelude(
    bytes: Span[UInt8, ImmutAnyOrigin], options: CsvOptions, has_header: Bool
) -> Tuple[Int, Int]:
    """Port streaming.rs SkipEmpty/SkipRowsBeforeHeader/SkipHeader.

    ``skip_rows`` counts valid CSV records, never comments.  It therefore
    uses the quote-aware field iterator, whereas comment detection occurs
    only at the start of a record.  ``record`` is the source record number
    passed into decode jobs after the prelude.
    """
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

    var remaining = options.skip_rows
    # `SkipRowsBeforeHeader`: comments are always discarded but do not reduce
    # the count; the first non-comment after it becomes the header (or data
    # when has_header=False).
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

    if has_header and start < len(bytes):
        start = _line_end(bytes, start, options)
        record += 1
    return (start, record)


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
) raises -> DataFrame:
    """Source-mapped CSV pipeline under differential validation."""
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
        65536,
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
    var bytes = mapping.span()
    var prelude = _prelude(bytes, options, has_header)
    var offset = prelude[0]
    var record = prelude[1]
    if offset == len(bytes) or n_rows == 0:
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
