# Architecture and development direction

## Current layers

- `column.mojo`: typed storage and validity bitmaps.
- `series.mojo`: named, runtime-tagged columns and batch slicing/concatenation.
- `expr.mojo`: flat expression nodes and composition, independent of data.
- `binding.mojo`: schema resolution, dtypes, shape, and aggregate dependencies.
- `execution.mojo`: bounded eager evaluation and grouped accumulator scheduling.
- `expr_kernels.mojo`: specialized binary kernels, including explicit Float64 SIMD.
- `reductions.mojo`: mergeable exact-integer and reassociable floating-point states.
- `frame.mojo`: validated eager dataframe operations and expression entry points.

The only runtime dependency is the Mojo standard library. This is an eager CPU
implementation, not a production query engine. All public extraction APIs copy;
expression execution reads bounded slices rather than repeatedly copying complete
source columns. GroupBy currently owns a copied snapshot. Column lookup at binding
is O(schema width), but execution uses resolved source indices.

Sorting is O(n log n) with O(n) index workspace. Hash grouping is expected O(n).
Join index construction is expected O(left + right + output); materialization
also scales with column count. Variable-length string costs depend on their length.

See [expression semantics](expressions.md) for the broadcasting, ordering, and
reduction contracts that deliberately leave room for parallelism and SIMD.
No universal dataframe trait or foreign-backend abstraction is imposed.

## Next milestones, with CPU performance central

1. **Measurements and copies:** benchmark representative expressions, scans,
   grouping, joins, and sorting across row counts/null densities. Add read-only
   buffer views and immutable shared storage. Measure allocations and peak memory,
   not just kernel throughput.
2. **Fusion and SIMD:** fuse compatible elementwise nodes within a batch, release
   dead intermediates, share subexpressions, load contiguous SIMD vectors directly,
   and optimize validity handling. Add vector overflow detection for Int64 kernels.
3. **Parallel execution:** schedule disjoint row batches; use worker-private
   reduction states with explicit merges. Share a group-key mapping or reconcile
   worker-local mappings. Apply `maintain_order` as an explicit requirement rather
   than an accidental default. Validate against scalar and partitioned oracles.
4. **Planning:** add a lazy relational plan using the same expressions, predicate
   and projection pushdown, multi-aggregate scans, and an explicit logical-type
   representation. Keep runtime schemas supported.
5. **Storage and interchange:** contiguous UTF-8 buffers, chunked arrays, Arrow
   C Data import/export with ownership/release/offset tests. Current storage is
   not yet Arrow-compatible for every dtype.
6. **Broader operations and input:** additional expressions/reductions, multiple
   and numeric grouping/join keys, casts, multi-column sorting, and CSV input.
   Parquet remains a separate integration project.

GPU execution, distributed execution, arbitrary Python-object columns, pandas
index compatibility, and cross-engine adapters remain outside the initial scope.
