"""Chunk-local CSV decoding mapped directly to Polars 1.44.2 parse_lines.

``decode_chunk`` receives record-aligned, header-free input from read_impl.
It holds only projected CsvBuffer instances, forwards borrowed SplitFields
spans to builder.rs-style ``add``, and delegates omitted tails to the same
quote-aware ``skip_this_line`` rule. Prelude removal and global n_rows bounds
belong to orchestration, never to an individual chunk.
"""
from .csv_types import CsvOptions, CsvSchema
from .csv_buffers import CsvBuffer
from .csv_splitfields import CsvSplitFields
from .dtype import DataType
from .frame import DataFrame
from .series import Series


def _skip_projected_tail(
    bytes: Span[UInt8, ImmutAnyOrigin], start: Int, quote: UInt8, quoting: Bool
) -> Int:
    """Port parser.rs ``skip_this_line`` after final projected input."""
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


def decode_chunk(
    bytes: Span[UInt8, ImmutAnyOrigin],
    schema: CsvSchema,
    options: CsvOptions,
    keep: List[Bool],
    rows: Int,
    record_start: Int = 1,
) raises -> DataFrame:
    """Port parser.rs ``parse_lines`` over one record-aligned source range.

    ``rows`` is CountLines' capacity hint. Buffers reserve ``rows + 1`` like
    Polars builders. `skip_rows` and `n_rows` are intentionally ignored here:
    read_impl has already removed the prelude and limits the concatenated
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
    for i in range(len(schema)):
        if keep[i]:
            projection.append(i)
            buffers.append(CsvBuffer(schema._fields[i], rows + 1, quote, lossy))

    # parser.rs treats any projection as implicit ragged truncation.
    var partial_projection = selected != len(schema)
    var offset = 0
    var record = record_start
    var output_rows = 0
    var separator = options.separator.as_bytes()[0]
    while offset < len(bytes):
        if _comment_at(bytes, offset, options.comment_prefix):
            while offset < len(bytes) and bytes[offset] != 10:
                offset += 1
            if offset < len(bytes):
                offset += 1
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
                if is_null:
                    buffers.unsafe_ptr().unsafe_offset(processed)[].add_null()
                else:
                    try:
                        buffers.unsafe_ptr().unsafe_offset(processed)[].add(
                            raw, field.needs_escaping, options.ignore_errors
                        )
                    except error:
                        raise Error(
                            "CSV record "
                            + String(record)
                            + ", field '"
                            + schema._fields[source_index].name
                            + "': "
                            + String(error)
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
            buffers.unsafe_ptr().unsafe_offset(processed)[].add_null()
            processed += 1
        output_rows += 1
        record += 1

    var output = List[Series](capacity=selected)
    while len(buffers) > 0:
        var buffer = buffers.pop()
        output.append(buffer.finish())
    output.reverse()
    return DataFrame(output^, height=output_rows)
