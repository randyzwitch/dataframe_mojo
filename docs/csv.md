# CSV ingestion contract

`read_csv(path, schema, ...)` reads typed columns; `read_csv(path, ...)`
infers their types. Both use one native Mojo pipeline derived from Polars
1.44.2: mapped input, quote-aware record scanning, borrowed field splitting,
typed builders, and chunk-preserving assembly. There is no alternate legacy
reader and no Python, Arrow, Polars, Rayon or jemalloc runtime dependency.

## Schema and records

`CsvSchema` is ordered, non-empty and has unique names. Construct fields with
`CsvField.int64`, `.float64`, `.bool`, `.string`, `.date`, `.datetime`, `.time`,
or `CsvField(name, dtype)` for any supported numeric width. Explicit schema
names label columns by position, including when the file has different header
names. `CsvField.nullable` remains accepted metadata; it does not prohibit null
CSV values, matching Polars' nullable columns.

Empty and header-only files produce zero rows with an explicit schema. Short
records are padded with nulls. Extra fields raise unless truncation or a partial
projection permits skipping the remaining source fields. LF, CRLF, missing
final newlines, quoted separators, doubled quotes and multiline fields are
supported. Quote handling follows Polars' splitter and builders; the old
strict tokenizer's rejection rules and error priority are not retained.
The UTF-8 BOM is removed only at the beginning of the input.

Temporal fields accept an optional format; an empty format selects the existing
ISO temporal converter. The writer emits ISO text. Temporal conversion uses
this library's existing parser; it is not a complete port of Polars' chrono
format inference. `CsvSchema.of(frame)` retains temporal types.

## Nulls and conversion

Bare empty fields are null. A quoted empty string remains an empty String;
empty numeric or Boolean fields are null. `null_values` tokens match quoted
as well as unquoted contents, in every column. Consequently, quoting a literal
null token does not protect it when that token is configured as null.

Integers use the pinned `atoi_simd` algorithm, with exact destination-width
checks and no floating-point intermediate. Signed fields accept `+` and `-`;
unsigned fields reject all negative spellings, including `-0`. Floats use the
pinned fast-float algorithm, preserving Float32/Float64 rounding and accepting
its NaN/infinity spellings and overflow behavior. Numeric builders strip
leading ASCII spaces and tabs; String content is preserved after CSV unquoting.
Booleans recognize case-insensitive `true` and `false`.

`ignore_errors=True` replaces failed conversions with null in the affected
field; it does not drop the entire record. Structural errors may still raise.
Errors include record/field context where conversion provides it; their text
is not a compatibility interface.

Strict UTF-8 validation checks each chunk once when the complete source schema
contains a String column, even if that column is not projected. `utf8-lossy`
replaces invalid String byte sequences with U+FFFD. Numeric-only input still
passes through the numeric grammar checks.

## Schema inference

The default sample is **100 rows**; `infer_schema_length=-1` samples the whole
input. Inference and decoding share the same mapped input. Boolean, Int64,
Float64 and String candidates follow Polars' lexical inference rules. Leading
zeroes do not force String, and an integer-shaped value can infer Int64 then
fail range checking during decoding. Null fields are ignored; an all-null
column defaults to String. Mixed integer/float candidates become Float64.

`try_parse_dates=True` enables temporal inference; the default is False.
`schema_overrides` maps names to dtype strings and overrides inferred types.
Unknown names are ignored, as in Polars; invalid dtype names for known columns
raise. Duplicate header names use Polars' `_duplicated_N` suffixes; headerless
columns are `column_1`, `column_2`, and so on. Empty inferred input raises an error. Use explicit types or a longer sample when later values
cannot be represented by the inferred dtype.

## Options and input ownership

| Option | Default | Meaning |
|---|---|---|
| `has_header` | `True` | consume the first content record as a header |
| `separator` | `","` | one byte other than CR or LF |
| `quote_char` | `'"'` | one byte, or `""` to disable quoting |
| `comment_prefix` | `""` | skip matching comment lines |
| `skip_rows` | `0` | skip CSV records before the header, respecting quoted newlines |
| `n_rows` | `-1` | limit returned rows; asynchronous decoding can read beyond the limit |
| `columns` | all | select source columns for decoding |
| `null_values` | none | additional null tokens for every column |
| `ignore_errors` | `False` | null-fill failed conversions |
| `truncate_ragged_lines` | `False` | ignore fields beyond the schema width |
| `encoding` | `"utf8"` | strict UTF-8 or `"utf8-lossy"` |
| `buffer_size` | `65536` | positive compatibility argument; it does not select a tokenizer or bound memory |

Regular files are mapped. Sources that cannot be mapped are read into owned
bytes and passed to the same decoder. That fallback is not a bounded-memory
streaming API. Chunk boundaries come from the record scanner, independently
of `buffer_size`. Numeric buffers and StringView storage transfer into output
columns. Results may retain multiple chunks; `rechunk()` explicitly requests
contiguous storage.

## Writing

`write_csv(frame, path, has_header=True, separator=",", quote_style="necessary",
null_value="", line_terminator="\n", buffer_size=65536)` writes UTF-8 CSV in
bounded output batches; `to_csv_string(frame, ...)` returns the same text.
The writer's `buffer_size` still controls its output batching.

With default null markers, writing and then reading with `CsvSchema.of(frame)`
round-trips supported values, including NaN, infinities, signed zero, empty
strings, nulls, temporal values and multiline text. Custom null tokens that
also occur as literal values cannot be distinguished by quoting alone.

- Nulls are written as unquoted `null_value`.
- `necessary` quotes empty strings, separators, quotes, CR/LF and null-token
  collisions. Embedded quotes are doubled.
- `always` quotes every non-null field; `non_numeric` quotes strings, Booleans
  and headers; `never` writes raw text and may not re-read correctly.
- Output line endings may be LF or CRLF.

Compression, remote URLs, per-column null tokens, custom input EOL bytes and
Arrow stream export are not exposed by this API.
