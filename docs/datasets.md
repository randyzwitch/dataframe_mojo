# Multiple local files

`read_csv`, `scan_csv`, `read_parquet`, and `scan_parquet` accept a path or a
list of paths. Paths can contain `*`, `?`, bracket character classes, and
recursive `**` components:

```mojo
var rows = read_csv(["day-02.csv", "day-01.csv"])
var plan = scan_parquet("sales/**/*.parquet").filter(col("amount") > 0)
```

Files are processed in sorted path order, preserving row order within each
file. Explicit repeated sources produce repeated rows. Schemas must match
in output names, column order and dtypes; a mismatch raises. CSV schema,
inference, conversion and projection options retain their single-file
meaning. Every file has its own header and skipped-record prelude.

CSV `n_rows` limits the combined output across all files, including when the
limit falls within a file. Parquet `row_groups` applies to each file. Lazy
plans keep file scans separate, so projection, filtering and head pushdown
retain their existing reader behavior.

Use `glob=False` for literal filenames containing pattern characters, such
as `read_parquet("part[1].parquet", glob=False)`. Recursive globs span zero
or more directories and do not traverse directory symlinks. Hidden files
and directories need an explicit dot component. An empty source list or a
pattern matching no files raises.

Path lists and globs are resolved when constructing a lazy scan; file contents
remain deferred until collection or schema inference. The resolved source
list is stable if more files appear before collection.

Hive columns, file-path columns, configurable cross-file schema alignment,
compressed CSV, streaming sinks and partitioned output are separate remaining
parts of #796.
