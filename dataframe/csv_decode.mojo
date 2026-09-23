"""Chunk-local CSV decoding.

``decode_chunk`` receives record-aligned, header-free input from read_impl.
It holds only projected CsvBuffer instances, forwards borrowed SplitFields
spans to typed buffers, and delegates omitted tails to a quote-aware skip rule. Prelude removal and global n_rows bounds
belong to orchestration, never to an individual chunk.
"""
from std.bit import count_trailing_zeros
from .csv_bits import _mask64
from .csv_types import CsvOptions, CsvSchema
from .csv_buffers import CsvBuffer, CsvCell
from .csv_splitfields import CsvSplitFields
from .dtype import DataType
from .frame import DataFrame
from .series import Series


def _skip_projected_tail(
    bytes: Span[UInt8, ImmutAnyOrigin], start: Int, quote: UInt8, quoting: Bool
) -> Int:
    """Skip unprojected fields after the final projected input."""
    var in_quotes = False
    var i = start
    while i < len(bytes):
        var byte = bytes[i]
        if quoting and byte == quote:
            in_quotes = not in_quotes
        elif byte == 10 and not in_quotes:
            return i + 1
        i += 1
    return len(bytes)


def _comment_at(
    bytes: Span[UInt8, ImmutAnyOrigin], start: Int, prefix: String
) -> Bool:
    var prefix_bytes = prefix.as_bytes()
    if len(prefix_bytes) == 0 or start + len(prefix_bytes) > len(bytes):
        return False
    for i in range(len(prefix_bytes)):
        if bytes[start + i] != prefix_bytes[i]:
            return False
    return True


@always_inline
def _append_cell(
    bytes: Span[UInt8, ImmutAnyOrigin],
    mut buffers: List[CsvBuffer],
    mut staged: List[List[CsvCell]],
    index: Int,
    cell: CsvCell,
    staged_decode: Bool,
    ignore_errors: Bool,
) raises:
    if staged_decode:
        staged.unsafe_ptr().unsafe_offset(index)[].append(cell.copy())
        return
    if cell.start < 0:
        buffers.unsafe_ptr().unsafe_offset(index)[].add_null()
        return
    var raw = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=bytes.unsafe_ptr().unsafe_offset(cell.start),
        length=cell.length,
    )
    try:
        buffers.unsafe_ptr().unsafe_offset(index)[].add(
            raw, cell.needs_escaping, ignore_errors
        )
    except error:
        raise Error(
            "CSV record "
            + String(cell.record)
            + ", field '"
            + buffers.unsafe_ptr().unsafe_offset(index)[].field.name
            + "': "
            + String(error)
        )


def _flush_staged(
    bytes: Span[UInt8, ImmutAnyOrigin],
    mut buffers: List[CsvBuffer],
    mut staged: List[List[CsvCell]],
    ignore_errors: Bool,
) raises:
    var first_record = -1
    var first_column = -1
    var first_message = String()
    for i in range(len(buffers)):
        var result = (
            buffers.unsafe_ptr()
            .unsafe_offset(i)[]
            .add_many(
                bytes, staged.unsafe_ptr().unsafe_offset(i)[], ignore_errors
            )
        )
        if result[0] >= 0 and (first_record < 0 or result[0] < first_record):
            first_record = result[0]
            first_column = i
            first_message = result[1]
        staged.unsafe_ptr().unsafe_offset(i)[].clear()
    if first_record >= 0:
        raise Error(
            "CSV record "
            + String(first_record)
            + ", field '"
            + buffers.unsafe_ptr().unsafe_offset(first_column)[].field.name
            + "': "
            + first_message
        )


def decode_chunk(
    bytes: Span[UInt8, ImmutAnyOrigin],
    schema: CsvSchema,
    options: CsvOptions,
    keep: List[Bool],
    rows: Int,
    record_start: Int = 1,
) raises -> DataFrame:
    """Decode one record-aligned source range.

    ``rows`` is CountLines' capacity hint. Buffers reserve ``rows + 1`` like
    typed buffers. `skip_rows` and `n_rows` are intentionally ignored here:
    the reader has already removed the prelude and limits the concatenated
    result, avoiding a chunk-boundary-dependent global row limit.
    """
    # The reader validates options once before chunk publication.
    if len(keep) != len(schema):
        raise Error("CSV projection mask does not match schema")

    var selected = 0
    var needs_utf8 = False
    for i in range(len(schema)):
        # read_impl's `check_utf8` inspects the complete source schema, not
        # its projection. It runs before read_chunk, so an omitted text field
        # still makes a strict UTF-8 chunk invalid.
        if schema._fields[i].dtype == DataType.STRING:
            needs_utf8 = True
        if keep[i]:
            selected += 1
    if selected == 0:
        raise Error("decode_chunk requires at least one selected column")

    # Polars validates the record-aligned chunk once before borrowed Utf8
    # handoff. Numeric-only projections retain raw byte parsing.
    if needs_utf8 and options.encoding == "utf8":
        try:
            _ = StringSlice(from_utf8=bytes)
        except:
            raise Error("CSV input is not valid UTF-8")

    # Polars parse_lines receives compiled projection indices, not a mask
    # scanned again for every record. Compile the API mask once per chunk.
    var quoting = options.quote_char.byte_length() == 1
    var quote = options.quote_char.as_bytes()[0] if quoting else UInt8(34)
    var lossy = options.encoding == "utf8-lossy"
    var projection = List[Int](capacity=selected)
    var buffers = List[CsvBuffer](capacity=selected)
    var staged = List[List[CsvCell]](capacity=selected)
    for i in range(len(schema)):
        if keep[i]:
            projection.append(i)
            buffers.append(CsvBuffer(schema._fields[i], rows + 1, quote, lossy))
            staged.append(List[CsvCell](capacity=256 if selected > 1 else 0))

    # parser.rs treats any projection as implicit ragged truncation.
    var partial_projection = selected != len(schema)
    var staged_decode = selected > 1
    var offset = 0
    var record = record_start
    var output_rows = 0
    var separator = options.separator.as_bytes()[0]
    var simd_separator = SIMD[DType.uint8, 64](separator)
    var simd_eol = SIMD[DType.uint8, 64](10)
    var simd_quote = SIMD[DType.uint8, 64](quote)
    var has_comments = options.comment_prefix.byte_length() != 0
    while offset < len(bytes):
        if has_comments and _comment_at(bytes, offset, options.comment_prefix):
            while offset < len(bytes) and bytes[offset] != 10:
                offset += 1
            if offset < len(bytes):
                offset += 1
            continue

        # A record ending in this vector can use its separator mask directly.
        # A quote before LF sends the whole record to the general splitter,
        # which handles quoted separators and embedded newlines.
        if len(bytes) - offset >= 64:
            var block = bytes.unsafe_ptr().unsafe_load[width=64](offset)
            var eols = _mask64(block.eq(simd_eol))
            if eols != 0:
                var last = Int(count_trailing_zeros(eols))
                var before_eol = (UInt64(1) << UInt64(last)) - 1
                if (
                    not quoting
                    or _mask64(block.eq(simd_quote)) & before_eol == 0
                ):
                    var delimiters = (
                        _mask64(block.eq(simd_separator)) & before_eol
                    )
                    var field_start = offset
                    var source_index = 0
                    var processed = 0
                    var next_selected = projection.unsafe_ptr()[]
                    var ended_with_separator = False
                    while processed < selected:
                        var end = offset + last
                        ended_with_separator = delimiters != 0
                        if ended_with_separator:
                            var bit = Int(count_trailing_zeros(delimiters))
                            end = offset + bit
                            delimiters &= delimiters - 1
                        if source_index == next_selected:
                            var field_end = end
                            if (
                                field_end > field_start
                                and bytes[field_end - 1] == 13
                            ):
                                field_end -= 1
                            var raw = Span[UInt8, ImmutAnyOrigin](
                                unsafe_ptr=bytes.unsafe_ptr().unsafe_offset(
                                    field_start
                                ),
                                length=field_end - field_start,
                            )
                            var is_null = False
                            if len(options.null_values) != 0:
                                for marker in options.null_values:
                                    if raw == marker.as_bytes():
                                        is_null = True
                                        break
                            _append_cell(
                                bytes,
                                buffers,
                                staged,
                                processed,
                                CsvCell(
                                    -1, 0, False, record
                                ) if is_null else CsvCell(
                                    field_start,
                                    field_end - field_start,
                                    False,
                                    record,
                                ),
                                staged_decode,
                                options.ignore_errors,
                            )
                            processed += 1
                            if processed < selected:
                                next_selected = (
                                    projection.unsafe_ptr().unsafe_offset(
                                        processed
                                    )[]
                                )
                        source_index += 1
                        field_start = end + 1
                        if not ended_with_separator:
                            break
                    if (
                        processed == selected
                        and ended_with_separator
                        and not partial_projection
                        and not options.truncate_ragged_lines
                    ):
                        _flush_staged(
                            bytes, buffers, staged, options.ignore_errors
                        )
                        raise Error("found more fields than defined in schema")
                    while processed < selected:
                        _append_cell(
                            bytes,
                            buffers,
                            staged,
                            processed,
                            CsvCell(-1, 0, False, record),
                            staged_decode,
                            options.ignore_errors,
                        )
                        processed += 1
                    offset += last + 1
                    output_rows += 1
                    record += 1
                    if staged_decode and output_rows % 256 == 0:
                        _flush_staged(
                            bytes, buffers, staged, options.ignore_errors
                        )
                    continue

        var record_offset = offset
        # `offset` is initially zero and thereafter is `record_offset +
        # fields.consumed()`, where SplitFields proves consumed <= len(input).
        # This is parser.rs' trusted cursor advance, rather than a checked
        # slice reconstruction for every record.
        var input = Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=bytes.unsafe_ptr().unsafe_offset(record_offset),
            length=len(bytes) - record_offset,
        )
        var fields = CsvSplitFields(separator, quote, quoting)
        var source_index = 0
        # `selected > 0` above, and projection/buffers both have this exact
        # length. These mirror parser.rs' projection iterator and unchecked
        # builder indexing inside its processed_fields < projection.len proof.
        var next_selected = projection.unsafe_ptr()[]
        var processed = 0
        var complete = False
        while not complete and offset < len(bytes):
            var next = fields.next(input)
            if not next:
                break
            var field = next.value().copy()
            var consumed = fields.consumed()
            if source_index == next_selected:
                # parser.rs removes CR directly before Builder::add.
                var raw = field._unsafe_bytes(input)
                # parse_lines trims a selected field's trailing CR before
                # Builder::add regardless of how SplitFields terminated it.
                # This includes an unterminated final record at EOF.
                if len(raw) > 0 and raw[len(raw) - 1] == 13:
                    raw = raw[0 : len(raw) - 1]
                # parser.rs matches compiled null tokens before Builder::add.
                var is_null = False
                if len(options.null_values) != 0:
                    var null_candidate = raw
                    if field.needs_escaping and len(raw) >= 2:
                        null_candidate = raw[1 : len(raw) - 1]
                    for marker in options.null_values:
                        if null_candidate == marker.as_bytes():
                            is_null = True
                            break
                _append_cell(
                    bytes,
                    buffers,
                    staged,
                    processed,
                    CsvCell(-1, 0, False, record) if is_null else CsvCell(
                        record_offset + field.start,
                        len(raw),
                        field.needs_escaping,
                        record,
                    ),
                    staged_decode,
                    options.ignore_errors,
                )
                processed += 1
                if processed < selected:
                    next_selected = projection.unsafe_ptr().unsafe_offset(
                        processed
                    )[]

            offset = record_offset + consumed
            if processed == selected:
                if field.terminator == 10:
                    complete = True
                else:
                    # Exact parse_lines branch: a full projection may reject
                    # remaining source fields; partial projection skips them.
                    if (
                        not partial_projection
                        and not options.truncate_ragged_lines
                        and offset < len(bytes)
                    ):
                        _flush_staged(
                            bytes, buffers, staged, options.ignore_errors
                        )
                        raise Error("found more fields than defined in schema")
                    offset = _skip_projected_tail(bytes, offset, quote, quoting)
                    complete = True
            elif field.ends_record:
                complete = True
            source_index += 1

        if not complete:
            break
        # parser.rs fills unvisited projected buffers with null for short rows.
        while processed < selected:
            _append_cell(
                bytes,
                buffers,
                staged,
                processed,
                CsvCell(-1, 0, False, record),
                staged_decode,
                options.ignore_errors,
            )
            processed += 1
        output_rows += 1
        record += 1
        if staged_decode and output_rows % 256 == 0:
            _flush_staged(bytes, buffers, staged, options.ignore_errors)

    if staged_decode:
        _flush_staged(bytes, buffers, staged, options.ignore_errors)

    var output = List[Series](capacity=selected)
    while len(buffers) > 0:
        var buffer = buffers.pop()
        output.append(buffer.finish())
    output.reverse()
    return DataFrame(output^, height=output_rows)
