# JSON and NDJSON

`read_ndjson(path)` reads one UTF-8 object per nonblank line.
`read_json(path)` reads an array of objects. Both return a `DataFrame` and
reuse the parser used by the string JSON expressions. They need no Python
or external JSON library at runtime.

```mojo
from dataframe import DataType, col, read_ndjson, scan_ndjson, write_ndjson

var events = read_ndjson("events.ndjson")
var totals = (
    scan_ndjson("events.ndjson")
    .filter(col("amount") > 0)
    .group_by("region")
    .agg(col("amount").sum())
    .collect(engine="cpu")
)
write_ndjson(totals, "totals.ndjson")

var schema = DataType.struct(
    ["region", "amount"], [DataType.STRING, DataType.FLOAT64]
)
var typed = read_ndjson("totals.ndjson", schema)
```

## Schema and errors

Pass an optional `DataType.struct(names, types)` schema to either reader or
`scan_ndjson`. Otherwise `infer_schema_length=100` samples the first 100
objects; `-1` samples all objects. The sampled schema is the union of keys
in first-seen order, recursively through objects and arrays. Missing fields
are null. Objects become Struct, arrays List, integers that fit Int64 stay
Int64, and fractional or larger numbers become Float64. Integer/float
mixtures become Float64; mixed Boolean/numeric/string scalars become String.
Incompatible container/scalar mixtures raise. Entirely null fields retain
`DataType.NULL`; empty or entirely null arrays infer List(Null).

A field outside the sampled or explicit schema raises by default, including
inside nested objects whose root columns were omitted by projection.
`ignore_errors=True` drops unknown fields and turns incompatible cells into nulls. A bad list element becomes a null
element, while a scalar where a list is required becomes a null list. Syntax
errors, invalid Unicode escapes, invalid numbers and nonobject rows always
raise. NDJSON errors identify the physical line, including blank lines.

These two inference choices follow the issue's explicit contract and differ
from Polars 2.0: Polars silently drops keys outside its sample, and a
Boolean/integer mixture can infer Int64 and then fail conversion.

An empty NDJSON file needs an explicit schema. An empty JSON array is valid.
Objects with no fields retain their rows: two `{}` records produce a frame
with two rows and zero columns. LF, CRLF, blank lines and a UTF-8 BOM are
accepted. A literal newline inside a quoted NDJSON value is invalid; use
the JSON `\n` escape instead.

Both eager readers accept `columns=[...]` and `batch_size=65536`. Projection
selects root fields in schema order and skips their unselected conversions;
the complete JSON syntax is still validated. NDJSON additionally accepts
`n_rows=-1`; a nonnegative limit reads only that many records after schema
discovery. Schema discovery can sample beyond the limit.

## Execution and ownership

Eager NDJSON reading maps the source, splits it into LF-aligned ranges, and
uses the existing scoped CPU worker pool. Results retain input order.
`DATAFRAME_THREADS` sets the process-wide worker limit. A positive `n_rows`
uses a bounded prefix cursor. Decoded output owns its buffers and outlives
the input mapping.

`scan_ndjson` defers opening until collection or schema discovery. Projection
and a direct `head` push into the source. Streaming hands bounded decode
jobs to the existing executor; downstream filters and reductions consume
their batches. Plans can be collected repeatedly. `read_json` is eager and
serial, retaining a parser tape for one object at a time rather than for
the complete array.

## Writing and round trips

`write_ndjson(frame, path)` writes one object per line;
`write_json(frame, path)` writes an array. Both accept `buffer_size=65536`
and flush after complete rows, so one unusually large row can exceed that
buffer size. Column and nested field order are retained.

| Logical type | JSON representation |
|---|---|
| Null / null cell | `null` |
| Integer / finite Float32 or Float64 | Number; finite floats and signed zero round-trip |
| NaN / positive or negative infinity | `null`, as in Polars |
| Bool / String | Boolean / escaped UTF-8 string |
| Date / Datetime / Time | ISO 8601 string; aware datetimes use a canonical UTC offset |
| Duration | Fixed ISO 8601 duration, such as `"-PT0.000000001S"` |
| Decimal | Quoted exact decimal, as in Polars |
| Categorical | String label |
| Binary / Geometry | Padded base64 string containing the original bytes |
| List / Struct | Array / object, recursively with parent and child nulls |

JSON does not store dtype metadata. For a typed round trip, pass the original
struct schema when reading: it restores numeric widths, temporal units and
time zones, decimal precision/scale, categorical dictionaries and opaque-byte
metadata. Without a schema, quoted logical values infer String. Nonfinite
floating-point values lose their distinction from null when written.
Expanded ISO years retain full Date and Datetime storage bounds; positive
years beyond four digits carry a leading `+`. This I/O parsing extension
keeps the existing string expression parser's year policy unchanged.

Time, Duration and Binary wire forms extend the pinned Polars reader, which
does not round-trip those types. Duration parsing accepts fixed days, hours,
minutes and seconds; calendar months and years are not fixed tick counts.
