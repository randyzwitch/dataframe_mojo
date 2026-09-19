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
Linux, run `/usr/bin/time -v pixi run bench-csv` (on macOS, `/usr/bin/time -l`); this intentionally remains an
external measurement so the parser has no platform-specific runtime dependency.

## Schema inference

`read_csv(path)` without a schema infers one. It tokenizes the header and the
first `infer_schema_length` data records (default 10,000; `-1` reads every
record) with the same options, and keeps those fields in memory as text. Each
column takes the first type in Bool, Int64, Float64, String that reads every
sampled value:

- Nulls (empty unquoted fields and `null_values` tokens) are ignored; a column
  with no values is String, and a quoted empty string makes the column String.
- Integer-shaped text with a leading zero (`007`) or outside the Int64 range
  infers as String, so identifiers and huge numbers are never altered.
- Int64 and Float64 values together infer Float64.

`schema_overrides={"name": "int64"}` fixes a column's dtype and wins over
inference; unknown names or dtypes raise. The file is then read strictly with
the resulting nullable schema. A value after the sample that does not fit
raises with its record and field, plus a hint to pass `schema_overrides` or a
larger `infer_schema_length`; types never change mid-file.

Header names are kept, with repeats renamed `name_1`, `name_2`, and so on.
Without a header, columns are `column_1`, `column_2`, .... Rows in the sample
must have one field count unless `truncate_ragged_lines` or `ignore_errors` is
set. An empty file returns a frame with no columns, and a header-only file
returns String columns with zero rows.

## Reader options

All options are applied by the streaming tokenizer, one byte at a time, so
results never depend on buffer boundaries; tests read every scenario with every
buffer size from one byte to the whole file.

| Option | Default | Meaning |
|---|---|---|
| `separator` | `","` | one byte other than CR or LF |
| `quote_char` | `'"'` | one byte, or `""` to treat quote bytes as data |
| `comment_prefix` | `""` | records starting with this text (outside quotes) are skipped to the end of the line; a partial match is ordinary data |
| `skip_rows` | `0` | raw physical lines skipped before parsing (and before the header), ignoring quotes |
| `n_rows` | `-1` | stop after this many data records; the rest of the file is not read |
| `columns` | all | decode and return only these fields, in schema order; other fields are tokenized but never converted |
| `null_values` | none | extra unquoted tokens read as null in every column; quoted text never matches |
| `ignore_errors` | `False` | drop data records with a conversion error or the wrong field count instead of raising; header errors still raise |
| `truncate_ragged_lines` | `False` | pad short records with nulls and drop extra fields |
| `encoding` | `"utf8"` | `"utf8-lossy"` replaces invalid sequences with U+FFFD instead of raising |

A dropped record is removed atomically: fields already converted are rolled
back, so column lengths stay aligned. Records are buffered as field text until
they end, which keeps parser memory bounded by the current record. Projection
cut ingestion time of the 100,000-row benchmark from 86 ms to 58 ms when
reading one of four columns (`pixi run bench-csv`, Linux x86-64).
Per-column null tokens and dropped-record counts are not yet available.

## Writing

`write_csv(frame, path, has_header=True, separator=",",
quote_style="necessary", null_value="", line_terminator="\n",
buffer_size=65536)` streams a frame to UTF-8 CSV in chunks of about
`buffer_size` bytes; `to_csv_string(frame, ...)` returns the same text.
`CsvSchema.of(frame)` builds a nullable schema from a frame's names and dtypes.

With the defaults the writer is the inverse of `read_csv`: for every supported
dtype, `read_csv(path, CsvSchema.of(frame))` after `write_csv(frame, path)`
equals the frame, including NaN, infinities, `-0.0`, Int64 extremes, empty
strings, nulls, embedded quotes, separators, and line breaks.

- Nulls are written as `null_value`, unquoted (empty by default).
- `necessary` quotes a field (and header name) when it contains the separator,
  a double quote, CR, or LF, is empty, or equals `null_value`, so empty strings
  and literal null tokens stay distinct from nulls. Embedded quotes are doubled.
- `always` quotes every non-null field; `non_numeric` quotes strings, Booleans,
  and header names; `never` writes raw text and may not re-read correctly.
- Int64 uses exact decimal digits, Float64 the shortest round-trippable form
  (`nan`, `inf`, `-inf`, `-0.0`, `1e+300`), and Bool `true`/`false`.
- `separator` must be one byte other than a quote, CR, or LF;
  `line_terminator` is LF or CRLF; `null_value` cannot contain the separator,
  quotes, or line breaks.

Deferred features include dates, compression, remote URLs, and lazy
`scan_csv`.
