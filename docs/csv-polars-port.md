# CSV source-level port

Reference: Polars Python **1.44.2**, tag py-1.44.2, commit
1bd8ec12f42d40fcec62badf32ef2177d2377d8d. The source root is
https://github.com/pola-rs/polars/tree/py-1.44.2/crates/polars-io/src/csv/read.
Polars-derived code is covered by the retained
[license](../third_party/POLARS_LICENSE).

## Current integration state

The public read_csv overloads now use the single Polars-derived native Mojo
pipeline. The former scalar streaming reader has been removed from the public
path; there is no compatibility fallback or competing reader. Both explicit
and inferred reads map regular files, find quote-aware record boundaries, split
borrowed fields, decode typed buffers, and assemble chunk-preserving columns.

This is an integration status, not a performance-parity claim. Local
validation has passed across 69 test modules, including the new public CSV
consumer test and reruns of five modules corrected after the initial suite. Historical comparison measurements below describe earlier branches
and must not be treated as paired measurements of the integrated public API.

## Source mapping

| Pinned Polars source | Public Mojo implementation | Current scope |
|---|---|---|
| polars-io mmap.rs | csv_types._Mapping and _map_file | Regular files map read-only; unmappable input uses owned bytes through the same decoder. |
| csv/read/read_impl.rs parse_csv | csv_reader read_csv_explicit, read_csv_inferred, _read_mapped_body | Public facade dispatches to one chunk pipeline. |
| csv/read/parser.rs CountLines | csv_scan.CountLines | Quote-parity masks, newline counting, and record-aligned chunk boundaries. |
| csv/read/splitfields.rs SplitFields | csv_splitfields.CsvSplitFields | Borrowed field spans, quoted-field end caching, and scalar tails. |
| csv/read/parser.rs parse_lines | csv_decode.decode_chunk | Projection-aware decoding, conversion-error null fill, and record context. |
| csv/read/builder.rs | csv_buffers.CsvBuffer | Typed builders, validity, StringView descriptors, temporal conversion, and output chunks. |
| csv/read/schema_inference.rs | csv_infer.infer_csv_schema | Polars lexical candidate order, lossy headers, duplicate suffixes, null tokens, and 100-row default. |
| csv/read/streaming.rs | csv_infer prelude traversal | BOM, empty prelude, quote-aware skip_rows, comments, header removal, and retained body offset. |
| fast_float2 and atoi_simd | csv_numeric.mojo and csv_integer.mojo | Float32/Float64 and integer source paths; x86 integer SIMD plus SWAR fallback. |
| Arrow validity/builders | CsvBuffer and column validity helpers | Bitmap absent until the first null and chunk-preserving output assembly. |
| Rayon task publication | csv_reader plus scoped Pool | Similar scan/publish staging, with a scoped shared-queue Mojo runtime rather than persistent Rayon workers. |

## Public behavior changes

The replacement deliberately follows the source-derived reader contract where
it differs from the removed scalar reader.

- Explicit schema names apply by position and need not match file header names.
  CsvField nullable is metadata; null input is accepted for every field.
- Short records receive null tail fields. Conversion failure with ignore_errors
  fills that field with null instead of dropping the whole record.
- Numeric parsing uses the pinned fast-float and atoi_simd ports. Unsigned
  negative zero is rejected. Boolean text is case-insensitive, and leading-zero
  integer text remains an integer candidate.
- Inferred reads sample 100 rows by default. Integer-shaped overflow can infer
  Int64 and then fail decoding, matching the non-Int128 Polars branch.
  Unknown name-keyed schema overrides are ignored.
- Date, Datetime[us], and Time inference is opt-in through try_parse_dates.
  Supported ISO values use the existing temporal converter. DMY, slash/dot,
  compact, and per-field chrono-format inference remain outside the current
  decoder surface.
- Empty inferred input raises. Header-only explicit input remains a zero-row
  frame. The public buffer_size keyword remains validated for compatibility but
  does not control mapped chunking.

The complete supported API contract is in [csv.md](csv.md). Unsupported source
surface includes named per-column null values, missing_is_null=False,
skip_lines, skip_rows_after_header, custom EOL bytes, decimal-comma, Int128
inference, compression, remote URLs, positional projection, and row indices.

## Local validation

The replacement has been checked at the public facade and internal stage
boundaries:

- scanner, structural bits, splitter, numeric, buffers, decoder, reader,
  chunked-consumer, inference, temporal, options, fuzz, and parallel tests;
- default and disabled-SIMD integer paths, Float32/Float64 bit fixtures, and
  pinned Polars differentials;
- quoted, escaped, multiline, Unicode, null, projection, comment, ragged,
  schema, inference, row-limit, and EOF cases;
- API-document generation, examples, package smoke tests, and full CI suite.

Public CSV differential checks passed 24 comparisons against Polars 1.44.2.
A separate 100,000-row fixture matched all eight integer widths at 1 and 32
workers. The general oracle matched 150 randomized cases and detected all 20
injected faults; Arrow interoperability, package smoke, the CSV example, and
the benchmark suite also passed. GitHub Linux/macOS CI remains separate from
these local checks.

Current public-reader timing and raw samples are recorded in
[wholesale-public](../experiments/csv_port/results/wholesale-public/README.md). Correctness tests passing do
not establish parity with Polars performance or every unsupported option.

## Historical comparison evidence

Before the public switch, the source-port branch used the old reader as a
comparison reference. Those experiments informed implementation choices but
are not current public-reader measurements.

The follow-up source-alignment round recorded 247.91 ms full and 147.55 ms
projected single-thread medians on a million-row mixed fixture, versus 381.62
and 290.58 ms for its initial source-port checkpoint. Its final 32-thread
medians were 19.82 ms full and 22.58 ms projected. Pinned Polars reference
runs in that historical round were 138.11 ms and 93.74 ms single-thread.
The projected 32-thread result regressed from an earlier checkpoint, so that
round explicitly made no parity claim.

Raw samples, setup, and caveats remain under
[experiments/csv_port/results](../experiments/csv_port/results/), including
[initial-source-port](../experiments/csv_port/results/initial-source-port/README.md),
[typed-frontends](../experiments/csv_port/results/typed-frontends/README.md),
[splitfield-cache](../experiments/csv_port/results/splitfield-cache/README.md),
[scheduler phases](../experiments/csv_port/results/scheduler-phases/README.md),
[matched Rayon](../experiments/csv_port/results/matched-rayon/README.md), and
[allocator parity](../experiments/csv_port/results/allocator-parity/README.md).

The isolated Rayon bridge and pinned jemalloc allocator probe were valid
experiments but are not production dependencies. The runtime still differs
from Polars persistent Rayon work stealing, and no scheduler or allocator
experiment alone closed the remaining gap.

## Remaining implementation gaps

- A dedicated Neon integer reduction backend is absent; non-x86 targets use
  the packed SWAR fallback.
- Temporal decoding uses the repository parser and allocates String input.
  It is not a full chrono or timezone-aware parser port.
- Single-array Arrow export rechunks StringView data into legacy large_utf8;
  native Arrow BinaryView export is not complete.
- The scoped pool and shared queue differ materially from persistent Rayon,
  especially under small projected parallel reads.
- The public option and type surface listed above remains intentionally
  narrower than Polars.

No end-to-end performance parity is claimed. Future measurements must compare
the integrated public reader with same-base binaries and pinned Polars on the
same input, options, machine, and worker limits.
