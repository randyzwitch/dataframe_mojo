# Changelog

All notable changes are recorded here. The API is pre-1.0 and provisional:
breaking changes can happen in any release and are listed under **Breaking**.

## Unreleased

### Breaking

- Removed the legacy column kernels `sum_int64`, `sum_float64`,
  `greater_than`, and `multiply`, and `DataFrame.group_by_sum`. Use
  expressions instead: `col("x").sum(min_count=1)` for a null-for-empty sum,
  `col("x") > lit(...)`, `col("x") * lit(...)`, and
  `group_by(key).agg(col("v").sum(min_count=1))`. Expression sums check
  overflow on the exact final total rather than on each prefix, so inputs such
  as `[MAX, 1, -1]` now succeed.
- `group_by` and `join` accept keys of any dtype, and `join` supports `how="full"`;
  code that relied on these raising must be updated.

### Added

- Frame utilities, display, concatenation, multi-column sort, multi-key
  grouping and joins, reshaping, deduplication, and a Series API.
- Expression operators, Kleene logic, conditionals, reductions, casts, string
  and window operations, selectors, and `over()` partitions.
- `write_csv`, `to_csv_string`, and `CsvSchema.of`.
