# Architecture and development direction

## Current layers

- `column.mojo`: typed windows (offset, length) onto reference-counted value
  buffers and LSB-first validity bitmaps, as in Arrow arrays.
- `string_column.mojo`: string columns in the Arrow `large_utf8` layout (one
  UTF-8 byte buffer, Int64 offsets, validity), read as borrowed `StringSlice`s
  and built with `StringBuilder`.
- `bool_column.mojo`: Boolean columns with bit-packed values (one bit per
  row), sharing the bitmap helpers with validity.
- `arrow.mojo`: Arrow C Data Interface export (zero-copy where layouts
  agree) and import.
- `series.mojo`: named, runtime-tagged columns and batch slicing/concatenation.
- `expr.mojo`: flat expression nodes and composition, independent of data.
- `binding.mojo`: schema resolution, dtypes, shape, and aggregate dependencies.
- `execution.mojo`: bounded eager evaluation and grouped accumulator scheduling.
- `expr_kernels.mojo`: specialized binary kernels, including explicit Float64 SIMD.
- `reductions.mojo`: mergeable exact-integer and reassociable floating-point states.
- `frame.mojo`: validated eager dataframe operations and expression entry points.

The only runtime dependency is the Mojo standard library. This is an eager CPU
implementation, not a production query engine. Buffers are immutable once
shared, so copying a column, `select`, `rename`, `drop`, `head`, `slice`,
`column()`, typed extraction, GroupBy snapshots, and expression batch slices are
O(1) per column and share storage. The only in-place growth (batch reassembly)
copies first unless the column owns its buffers outright. Column lookup at binding
is O(schema width), but execution uses resolved source indices.

Sorting is O(n log n) with O(n) index workspace. Hash grouping is expected O(n).
Join index construction is expected O(left + right + output); materialization
also scales with column count. Variable-length string costs depend on their length.

See [expression semantics](expressions.md) for the broadcasting, ordering, and
reduction contracts that deliberately leave room for parallelism and SIMD.
No universal dataframe trait or foreign-backend abstraction is imposed.

## Next milestones, with CPU performance central

1. **Measurements and copies:** benchmark representative expressions, scans,
   grouping, joins, and sorting across row counts/null densities. (Immutable
   shared buffers with zero-copy windows are done.) Measure allocations and peak
   memory, not just kernel throughput.
2. **Fusion and SIMD:** fuse compatible elementwise nodes within a batch, release
   dead intermediates, share subexpressions, load contiguous SIMD vectors directly,
   and optimize validity handling. Add vector overflow detection for Int64 kernels.
3. **Parallel execution:** (reductions, row-wise expressions, and filter
   compaction are done: contiguous row partitions on POSIX threads via
   `parallel.mojo`, merged in partition order.) Hash grouping with
   worker-local mappings remains. Apply `maintain_order` as an explicit requirement rather
   than an accidental default. Validate against scalar and partitioned oracles.
4. **Planning:** add a lazy relational plan using the same expressions, predicate
   and projection pushdown, multi-aggregate scans, and an explicit logical-type
   representation. Keep runtime schemas supported.
5. **Storage and interchange:** (contiguous UTF-8 strings, bit-packed
   Booleans, and Arrow C Data import/export are done.) Chunked arrays and
   zero-copy import that retains foreign buffers remain.
6. **Broader operations and input:** additional expressions/reductions, multiple
   and numeric grouping/join keys, casts, multi-column sorting, and CSV input.
   Parquet remains a separate integration project.

GPU execution, distributed execution, arbitrary Python-object columns, pandas
index compatibility, and cross-engine adapters remain outside the initial scope.
