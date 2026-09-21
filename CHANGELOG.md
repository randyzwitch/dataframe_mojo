# Changelog

All notable changes are recorded here. The API is pre-1.0 and provisional:
breaking changes can happen in any release and are listed under **Breaking**.

## Unreleased

### Changed

- A parallel CSV read maps the file instead of reading it into a buffer.
  Mapping the 50 MB benchmark file costs 2.4 ms against 58 ms to read it,
  and the whole file being addressable at once removes the block loop and
  the partial record carried between blocks. A range also reads straight
  out of the mapping rather than copying itself out first, which was a
  second pass over every byte. Reading 1M rows of 8 columns drops from a
  median of 103 ms to 88 ms across 32 threads. A path with no length to map
  -- a pipe, a character device -- still reads, through the block reader
  (#107).

## 0.2.0 - 2026-09-21

### Breaking

- Mojo 1.1 is no longer supported; this release requires the 1.2 series.
  A compiled Mojo package loads only in the version that produced it, so a
  consumer on 1.1 must stay on 0.1.3. The package was built and its suite
  run against 1.2.0.dev2026092105, which is what the bound names: 1.2 is a
  nightly series today.

- Development tracks the same nightly. The nightly channel is now a
  workspace channel and there is one environment again, so plain
  `pixi run test` is the supported toolchain; the separate `nightly`
  environment added in 0.1.3's cycle has nothing left to do.

### Added

- `Pool`, worker threads reused across the rounds of one operation instead
  of created and joined per round. A pool is **scoped**: it joins its
  workers when released, and never outlives the operation that made it. A
  process-wide pool is not possible here -- its workers would still be
  parked in JIT-compiled code when the process exits, which crashes about
  one run in three under `mojo run`, and no Mojo code can run at exit to
  join them first. Sorting uses one for its run pass and merge rounds: a
  two-key sort of 1M rows drops from 156 ms to 150 ms, and at 100k rows
  from 38 ms to 33 ms (#103).

### Changed

- The CSV reader no longer builds a `String` for every field. Fields of a
  record are written end to end into one buffer and passed on as slices
  over it, which needs no allocation; a 1M-row file of 8 columns was
  allocating 8 million Strings. Reading that file single-threaded drops
  from 1,336 ms to 992 ms, and across 32 threads from 155 ms to 118 ms.
  Strings are still built where they are kept rather than parsed: schema
  inference's sample, a header's names, and lossy decoding, which
  substitutes U+FFFD and so changes the bytes (#107).

- The CSV reader copies runs of ordinary field bytes in bulk, finding the
  next separator, newline or quote a block at a time rather than
  dispatching on every byte. Reading 1M rows of 8 columns on one thread
  drops from 990 ms to 925 ms. Across many threads the read is bound by
  something else and the difference is within run-to-run noise, so this is
  for machines with few cores (#107).

- `concat` builds its output columns on worker threads. Each output column
  is assembled from that column of every input frame and touches nothing
  else, so there is nothing to coordinate. A parallel CSV read concatenates
  one frame per range per block -- 64 of them for a 50 MB file -- and doing
  that one column after another was a third of the read: 1M rows of 8
  columns drops from 158 ms to 133 ms (#107).

- A join assembles its output across worker threads. It gathered one
  column at a time on the calling thread, which was about half of a join:
  98 ms of 198 ms for 1M x 500k inner. `take_parallel` already writes
  disjoint output ranges, and now carries the -1 that a join uses for
  "no row on this side", so every column of both sides is gathered at
  once. The join drops from 174 ms to 119 ms at 1M rows, and from 19 ms
  to 15 ms at 100k (#105).

- Encoding a single key column no longer builds a hash map to combine it
  with the others, there being no others: the distinct values are
  renumbered in row order through an array indexed by code. This is on the
  path every single-key join and group_by takes. A 1M-row join drops from
  119 ms to 116 ms and a 100k-row one from 15 ms to 13 ms; grouping 100k
  rows on a high-cardinality key from 4.2 ms to 3.6 ms (#105).

- Sorting by fixed-width keys no longer ranks its key columns. Each value
  maps to an Int whose signed order is the value's order, in one linear
  pass, which is what the sort compares anyway; ranking existed to give
  strings an order, and string keys still take that path. An n-column sort
  did n sorts before the one that orders the rows, and now does none: for
  1M rows and two keys, building the keys drops from 77 ms to 11 ms and the
  whole sort from 147 ms to 85 ms, which is 3.0x to 1.5x of Polars. Row
  order is unchanged, including -0.0 equal to 0.0, NaN after the numbers in
  either direction, and null placement independent of direction (#108).

- String sort keys are encoded the same way, as their first 24 bytes plus
  their length, so a sort by a string key no longer ranks it either.
  Padding with zeros and comparing the length last is exact for any two
  values that fit, including one that is a prefix of the other and
  including embedded NUL bytes. A value longer than 24 bytes cannot be
  compared from its prefix alone, so such a column falls back to ranking.
  Sorting 1M rows by a string key drops from 331 ms to 75 ms (#108).

- Sorting ranks its numeric key columns by sorting `(value, row)` pairs and
  walking them, instead of reducing the values to the distinct ones and
  binary-searching every row back in. That search was the largest single
  cost of a sort -- 114 ms of the 182 ms spent ranking two key columns of
  1M rows -- and the string path already ranked by walking a sorted order.
  A two-key sort of 1M rows drops from 341 ms to 261 ms; ranking a
  high-cardinality Int64 column drops from 195 ms to 72 ms. Row order is
  unchanged for every dtype, direction and null placement (#108).

- A sort's merge rounds are split across threads. Each round halves the
  number of merges, so the last round was one thread merging the whole
  array; every merge is now cut into output slices, located by binary
  search so that a slice starts at the same place in both runs. Runs are
  also formed one per thread rather than one per 65,536 rows, which that
  minimum -- sized for a linear scan -- had capped at 15 on a 32-core
  machine, and which left sorts below 131,072 rows entirely serial.
  Merging and run-sorting 1M rows drops from 114 ms to 31 ms; a whole
  two-key sort from 261 ms to 191 ms, and at 100k rows from 43 ms to
  38 ms. Row order is unchanged: ranks break ties by row index, so the
  comparison is a total order and a slice boundary falls in exactly one
  place (#108).

- A sort with several key columns ranks them at once rather than one after
  another, which is worth doing because ranking a column is itself serial
  and is the largest part of a sort. A two-key sort of 1M rows spends
  107 ms ranking before and 76 ms after, for 191 ms to 155 ms overall
  (#108).

- CSV reads decode records on worker threads. A block is split at record
  boundaries -- decided by quote parity, so a newline inside a quoted field
  is never mistaken for one -- each range is decoded by its own reader, and
  the partial frames are concatenated in order, which keeps the output
  identical to a serial read. Blocks are read sequentially with the
  trailing partial record carried forward, so memory stays bounded by the
  block size rather than the file size. 1M rows of 8 columns drops from
  1,081 ms to 145 ms, and a 100k-row file from 66 ms to 18 ms. Reads using
  `n_rows`, `skip_rows`, `comment_prefix`, `ignore_errors` or
  `truncate_ragged_lines` stay serial, since those count records from the
  start of the file (#107).

- Appending one column to another copies the window in bulk instead of one
  bounds-checked element at a time. That is how `concat`, `vstack`, batch
  reassembly and every parallel stage reassemble their output, so it is not
  specific to CSV; concatenating the 64 partial frames of a 1M-row CSV read
  drops from 69 ms to 51 ms (#107).

## 0.1.3 - 2026-09-20

### Added

- `pixi run -e oracle bench-polars`: a head-to-head benchmark against Polars
  on identical CSV inputs with thread counts pinned equal, covering CSV
  read, elementwise arithmetic and comparison, filter, global sum, grouping
  at low, high and skewed cardinality on Int64 and String keys, inner join,
  and multi-column sort. Each workload's row count and a column total must
  agree between the engines before their times are compared (#102).

### Changed

- `sort`, `arg_sort`, `top_k` and `bottom_k` sort one row range per worker
  and merge the runs, instead of one serial mergesort. The result is the
  same stable order for any worker count: ranges are cut in row order and
  merges prefer the earlier run on ties. Sorting 1M rows by two keys drops
  from 594 ms to 346 ms (#108).

- Joins encode their key columns one hash bucket at a time when keys are
  many, instead of building one dictionary over both sides. A join's row
  order comes from iterating rows rather than from the id numbering, so the
  ids may be assigned in any consistent order; the documented match order
  is unchanged. The rows per key id also moved from one list per key to a
  flat index. A 1M-row inner join on 500k distinct keys drops from 283 ms
  to 188 ms (#105).

- Grouping is parallel at every cardinality. Rows are hashed by key and
  partitioned into buckets, and each bucket is encoded and reduced on its
  own: equal keys always share a bucket, so no dictionaries or reduction
  states are ever merged, which is what the earlier merge-based attempts
  (#8) lost at high cardinality. Output order is bucket order unless
  `maintain_order=True`, which sorts groups by their first input row in
  O(groups). Frames below the parallel threshold keep the serial path,
  and `GroupBy.len` and `group_indices` are unchanged (#104).

- CSV reads plain decimal Float64 fields in one pass that validates and
  computes together, instead of checking the grammar with one scan and
  converting with another. Values are unchanged and exact: the fast path
  only runs when the mantissa and the power of ten are both exact, so the
  single division is correctly rounded, and anything else (signs,
  exponents, `nan`, infinities, long mantissas) uses the original parser.
  Float parsing was about 35% of a CSV read; a 4-column file drops from
  80.5 ms to 66 ms and a 1M-row 8-column file from 1,260 ms to 1,081 ms
  (#107).

### Fixed

- CSV and temporal parsing accept an ISO 8601 zone designator on a
  datetime: a trailing `Z` means UTC, and `+HH:MM`, `-HH:MM`, `+HHMM` or
  `+HH` are converted to UTC, since datetimes here are naive and hold UTC.
  Both forms previously raised "unexpected trailing text", which rejected
  the most common datetime spelling in real data. Dates and times still
  reject designators (#107).

- Format directives with no separator between them take their exact width,
  so `%Y%m%d` reads `20240228` instead of letting `%Y` consume six digits
  and then failing. A directive followed by a literal still accepts
  one-digit months and days as before (#107).

## 0.1.2 - 2026-09-20

### Added

- Public access to a column's Arrow buffers, so a consumer reads values in
  place instead of one tagged `AnyValue` per element: `unsafe_values()`,
  `unsafe_validity()` and `validity_offset()` on `Column`, `BoolColumn` and
  `StringColumn` (plus `unsafe_bytes()` / `unsafe_offsets()` for the
  `large_utf8` layout), a non-raising `is_valid(i)`, and `to_list()`
  promoted from `_to_list()`. Summing a million-row Float64 column drops
  from about 3,500 us through `Series.get` to about 700 us through the
  buffers. Null slots hold whatever the buffer holds, as in Arrow, where
  they are undefined (#91).

- `DataFrame.group_indices(keys)` returns which rows belong to which group
  without aggregating them, for callers that want each group's rows rather
  than one summary row: `count`, `ids`, `rows(g)`, `all_rows`, `sizes`,
  and `representative(g)` for where to read a group's key values. Groups are
  numbered in first-occurrence order and null keys form their own group,
  both as in `group_by`. Building sub-frames from the indices is left to the
  caller, who may prefer to read the shared buffers directly (#92).

## 0.1.1 - 2026-09-20

Everything below is the initial feature set. v0.1.0 was tagged the same day and
is identical except that it was missing bit-packed Boolean columns, whose merge
had been stranded on a branch; prefer 0.1.1.

### Breaking

- `Series.bool()` returns a `BoolColumn` (bit-packed values) instead of
  `Column[Bool]`, with the same `value`, `is_null`, `null_count`, `take`,
  and `slice` methods. `Series(name, Column[Bool](...))` and
  `DataFrame.filter(Column[Bool])` still work and pack on construction (#80).
- Arrow import keeps narrow integer and float32 types instead of widening
  them to Int64 / Float64, and accepts UInt64.
- `Series.string()` returns a `StringColumn` (Arrow `large_utf8`: one UTF-8
  buffer plus Int64 offsets) instead of `Column[String]`. It has the same
  `value`, `is_null`, `null_count`, `take`, and `slice` methods.
  `Series(name, Column[String](...))` still works and converts (#33).
- Dtypes are structured `DataType` values instead of strings:
  `Series.dtype()`, `DataFrame.dtypes()`, `Field.dtype`, `AnyValue.dtype()`,
  and `CsvField.dtype` return `DataType`. Compare with constants
  (`s.dtype() == DataType.INT64`) or use `String(dtype)` / `dtype.name()`
  for the old string. Name-taking APIs still accept strings.

- Removed the legacy column kernels `sum_int64`, `sum_float64`,
  `greater_than`, and `multiply`, and `DataFrame.group_by_sum`. Use
  expressions instead: `col("x").sum(min_count=1)` for a null-for-empty sum,
  `col("x") > lit(...)`, `col("x") * lit(...)`, and
  `group_by(key).agg(col("v").sum(min_count=1))`. Expression sums check
  overflow on the exact final total rather than on each prefix, so inputs such
  as `[MAX, 1, -1]` now succeed.
- `group_by` and `join` accept keys of any dtype, and `join` supports `how="full"`;
  code that relied on these raising must be updated.

### Changed

- Columns are windows (offset, length) onto reference-counted, immutable
  buffers, as in Arrow arrays. `select`, `rename`, `drop`, `head`, `slice`,
  `column()`, typed extraction, GroupBy snapshots, and expression batch slices
  share storage in O(1) instead of copying. Global reductions are 1.6-2x and
  low-cardinality grouping 1.5x faster at 1M rows (#34).

### Fixed

- Float64 text parsing (CSV, casts, inference) no longer accepts malformed
  numbers that Mojo's parser tolerates, such as `2024-02-28` (read as
  2024002028.0), `1-2`, or `1.5.5`.

### Added

- Installable as a Mojo package straight from GitHub (`[package]` with the
  `pixi-build-mojo` backend), so another Pixi workspace can depend on
  `dataframe_mojo` by git tag or path and `from dataframe import ...`. Adds
  a LICENSE file (MIT).

- Boolean columns store one bit per value (Arrow layout): 8x less memory,
  zero-copy Arrow export, and faster Boolean results (nullable compare
  4.7 -> 2.7 ms, filter 10.7 -> 8.7 ms at 1M rows) (#80).

- Row-wise expressions and filters run on worker threads for large inputs:
  arithmetic and comparisons ~4x faster and filter ~3.5x faster at 1M rows,
  with identical row order and results (#5, #7).

- Parallel reductions: global and grouped reductions over large inputs run
  on worker threads (POSIX threads through the C FFI; no new dependency) with
  worker-private states merged in row order. `DATAFRAME_THREADS` caps the
  thread count (1 disables). Global sums are ~3.7x faster at 1M rows
  (#6, #8).
- Unfused float kernels read contiguous SIMD vectors directly from shared
  column buffers (about 1.45x faster for Float32 arithmetic, `%`, and math
  functions) (#3).
- `DataFrame.fill_null(0)` / `fill_null(0.5)`: a bare number fills every
  column it can adopt (integers: all numeric columns; floats: float
  columns), each in its own dtype; `fill_null("x")` fills string columns.

- Expression sugar: bare numbers and Bools work wherever an `Expr` is
  expected (`col("x") > 0`, `col("x") * 2.5`, `1 + col("x")`,
  `.fill_null(0)`, `when(...).then(1)`), `==` / `!=` build expressions, and
  strings are accepted by comparisons, `is_in`, `fill_null`, and
  `then`/`otherwise`. Bare numbers are untyped and adopt the other operand's
  dtype at bind time (range-checked), keeping the no-implicit-promotion
  rule (#75).

- Numeric dtypes Int8, Int16, Int32, UInt8, UInt16, UInt32, UInt64, and
  Float32 alongside Int64 and Float64, through every operation: checked
  arithmetic at each width, Float32 SIMD kernels, exact 128-bit sums (8/16-bit
  sums produce Int64, as in Polars), sorting, grouping and join keys, casts
  with exact range checks, CSV fields, display, Arrow (zero-copy, native
  formats), and typed literals (`lit(Int32(1))`, generic `lit[D](Scalar[D])`).
  `Series.numeric[D]()` / `.int8()` ... `.float32()` and matching `AnyValue`
  accessors; `Expr.cast` also takes a `DataType` (#30).

- Arrow C Data Interface: `export_arrow` / `import_arrow` (frames as struct
  arrays) and `export_arrow_series` / `import_arrow_series`, with
  `ArrowArray` / `ArrowSchema` structs. Int64, Float64, String, Datetime,
  Duration, and Time export zero-copy; Bool and Date convert. Import copies
  and accepts narrower integer, float32, utf8, date64, and time32 inputs.
  See docs/arrow.md (#35).

- Date, Datetime, Duration, and Time types with a `.dt()` namespace, temporal
  arithmetic and casts, strptime/strftime-style formats, CSV support,
  inference, and `date_range`/`datetime_range`.

- Frame utilities, display, concatenation, multi-column sort, multi-key
  grouping and joins, reshaping, deduplication, and a Series API.
- Expression operators, Kleene logic, conditionals, reductions, casts, string
  and window operations, selectors, and `over()` partitions.
- `write_csv`, `to_csv_string`, and `CsvSchema.of`.
