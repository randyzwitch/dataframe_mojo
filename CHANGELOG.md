# Changelog

All notable changes are recorded here. The API is pre-1.0 and provisional:
breaking changes can happen in any release and are listed under **Breaking**.

## Unreleased

### Added

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
