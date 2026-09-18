# CSV ingestion contract

`read_csv(path, schema, has_header=True, buffer_size=65536)` reads a local file
into the existing eager `DataFrame`. It is native Mojo code and has no Python,
Arrow, pandas, or Polars runtime dependency.

## Schema and records

The schema is non-empty, ordered, and uniquely named. Supported fields are
`CsvField.int64`, `.float64`, `.bool`, and `.string`; each is nullable by
default. With a header, names and order must match exactly. Without one, schema
names are assigned positionally. Empty and header-only files produce zero rows
with the requested schema. Every data record must have exactly the schema width.

The parser accepts LF and CRLF record endings and an absent final newline. A
bare CR outside quotes is invalid. Quoted fields may contain commas, CR/LF, and
doubled quote escapes. A quote may only open at the start of a field, and no
bytes may follow its closing quote before a delimiter or record ending.
Whitespace is data and is never trimmed. The UTF-8 BOM is ignored only at byte
zero; elsewhere it is ordinary Unicode text. Every completed field is validated
as UTF-8, even across input-buffer boundaries.

## Null and conversion rules

An empty unquoted field is the sole null marker. Null markers never match quoted
fields, so `""` is a valid empty String and is a conversion error for numeric or
Boolean fields. Null in a non-nullable field raises.

Int64 accepts an optional leading `+` or `-` followed by ASCII decimal digits.
It checks magnitude before every operation and supports the exact range, without
passing through Float64. Float64 uses Mojo's parser but rejects surrounding ASCII
space/tab/newline, and rejects overflow-to-infinity unless the token is one of
the documented explicit infinity spellings. Valid NaN remains a value, distinct
from null. Boolean accepts only lowercase `true` and `false`. String preserves
decoded content exactly after CSV unquoting.

Errors identify the logical record and field where practical. Structural errors
also report the physical line; a logical record can span several physical lines.
Parsing is strict and never skips malformed records.

## Buffering and performance direction

File reads are bounded by `buffer_size`; tokenizer state survives arbitrary
boundaries, including BOM bytes, CRLF pairs, escaped quotes, UTF-8 sequences, and
multiline records. Temporary parsing memory is bounded by the input buffer and
the current field, apart from typed output builders. Final column construction
currently copies builder storage once; immutable/movable column builders can
remove that copy later.

The scalar tokenizer separates structural scanning from typed decoding. Future
SIMD scanning must produce the same state transitions. Parallel partitions must
begin at verified record boundaries or reconcile quote state; raw newline splits
are incorrect for multiline fields.

`pixi run bench-csv` generates a deterministic 100,000-row mixed dataset outside
the timed region, performs one warmup, verifies every result height, and prints
machine-readable timing and throughput metrics. For peak resident memory on
Linux, run `/usr/bin/time -v pixi run bench-csv`; this intentionally remains an
external measurement so the parser has no platform-specific runtime dependency.

Deferred features include schema inference, custom null tokens/dialects, dates,
compression, remote URLs, permissive error skipping, and lazy `scan_csv`.
