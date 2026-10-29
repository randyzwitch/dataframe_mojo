# Multiple local files

`read_csv`, `scan_csv`, `read_parquet`, and `scan_parquet` accept a path or a
list of paths. Paths support `*`, `?`, bracket classes, and recursive `**`:

```mojo
var rows = read_csv(["day-02.csv", "day-01.csv"])
var plan = scan_parquet("sales/**/*.parquet").filter(col("amount") > 0)
```

Files are processed in sorted path order, preserving row order within each
file. Repeated sources produce repeated rows. Use `glob=False` for literal
pattern characters. Recursive globs do not traverse directory symlinks;
hidden paths need an explicit dot component. Empty lists and unmatched globs
raise. Lazy scans resolve paths at construction and defer file contents until
execution or schema inference.

CSV options apply to each file, including its header and skipped-record
prelude. CSV `n_rows` limits the combined output. Parquet `row_groups` applies
to every file, preserving requested group order and repetitions.

## Hive partitions and source paths

```mojo
var plan = scan_parquet(
    "sales/year=*/month=*/*.parquet",
    hive_partitioning=True,
    include_file_paths="source",
).filter(col("year") == 2024)
```

Hive mode adds columns from `key=value` directory components. A field is
Int64 when all present, non-null path values parse as Int64; otherwise it is
String. Override inference with `hive_schema=Dict[String, String]`, mapping
field names to `"Int64"` or `"String"`. Missing partition values and the raw
`__HIVE_DEFAULT_PARTITION__` marker become null. Names and values are percent
decoded once; an encoded literal marker remains a string value.

Path values override same-named physical fields while retaining their column
positions. Other partition fields are appended. `include_file_paths` adds a
String field with each resolved source path; a conflicting name raises.
Supported predicates over partition/path fields skip irrelevant file payloads.
Schema discovery may still inspect file headers or Parquet footers. Filters
retain ordinary row semantics, including when an earlier `head` limits input.

## Cross-file schemas

The default requires matching names, order, and required dtypes. Set
`missing_columns="insert"` to fill absent fields with typed nulls, and
`extra_columns="ignore"` to discard fields absent from the first file.
These controls align fields by name. Their default is `"raise"`.

`union_by_name=True` instead includes every field in first-seen order and
fills absent values with nulls. It takes precedence over the missing/extra
controls. Shared required fields must have compatible types; no automatic
numeric promotion is performed. Projection can avoid reading and validating
unused fields. Explicit CSV schemas retain their positional interpretation.

## Compressed CSV

CSV path readers and owned byte buffers accept gzip and zstd. Content magic
is checked before filename extensions. Concatenated gzip members and zstd
frames are supported. Full reads validate compressed trailers and truncation;
early row limits close the input without draining its unread tail.

## Output

`LazyFrame.sink_csv(path, ...)` and `sink_parquet(path, ...)` consume execution
batches directly. Row-local plans do not accumulate their complete output.
Sorts, joins, grouping, and other stateful operators keep their existing
execution memory requirements. `batch_size` controls execution batches;
Parquet `row_group_size` controls output row groups across batch boundaries.

File sinks write to a temporary file beside the destination and publish it
on successful completion. An error preserves an existing destination. This
also permits a sink to replace a file used by its own input scan.

`write_parquet(frame, root, partition_by=["year", "month"])` writes an eager
Hive directory tree with one `part-00000.parquet` per partition. Partition
fields remain in the files, and rows retain their input order within each
partition. Names and values are escaped, including literal null markers.
The destination must be absent or empty. Output is staged beside the root
and published only after every file succeeds. An empty frame creates an
empty root directory.
