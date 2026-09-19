# Changelog

All notable changes are recorded here. The API is pre-1.0 and provisional:
breaking changes can happen in any release and are listed under **Breaking**.

## Unreleased

### Breaking

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

### Fixed

- Float64 text parsing (CSV, casts, inference) no longer accepts malformed
  numbers that Mojo's parser tolerates, such as `2024-02-28` (read as
  2024002028.0), `1-2`, or `1.5.5`.

### Added

- Date, Datetime, Duration, and Time types with a `.dt()` namespace, temporal
  arithmetic and casts, strptime/strftime-style formats, CSV support,
  inference, and `date_range`/`datetime_range`.

- Frame utilities, display, concatenation, multi-column sort, multi-key
  grouping and joins, reshaping, deduplication, and a Series API.
- Expression operators, Kleene logic, conditionals, reductions, casts, string
  and window operations, selectors, and `over()` partitions.
- `write_csv`, `to_csv_string`, and `CsvSchema.of`.
