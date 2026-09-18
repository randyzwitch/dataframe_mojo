# Architecture and development direction

## Current implementation

`column.mojo` owns typed lists and validity bitmaps. `series.mojo` holds a named
variant of the four supported column types, keeping runtime type dispatch at the
column level. `frame.mojo` validates schemas and implements eager relational
operations. `kernels.mojo` contains scalar CPU arithmetic and reduction kernels.

This establishes a correctness baseline. It is intentionally not yet an optimized
query engine. The only dependency is the Mojo standard library. Owned copies make
lifetimes straightforward but increase peak memory use; explicit moves into
constructors avoid copies where possible. Column lookup is currently O(width).

Sorting is O(n log n) with O(n) index workspace. Grouping is expected O(n) with
hash tables. Joins are expected O(left + right + output) for their index-building
phase; full output materialization also scales with the number of columns.
Many-to-many joins can produce much larger output than either input. Hashing and
copying variable-length strings also depend on string length.

No universal DataFrame trait is imposed. Narrow schema/batch traits should be
introduced when a second real consumer or implementation establishes the need.
A portable operation API would also require a separately tested semantic contract.

## Next milestones

1. **Storage and measurements:** benchmarks for filter/group/join/sort across row
   counts and null densities; borrowed read-only column views; shared immutable
   buffers; an explicit logical-type enum; contiguous UTF-8 offsets/data storage;
   chunked columns and batch consumption. Preserve the current contract tests.
2. **Expressions:** `col`, literals, comparisons, arithmetic, named aggregates;
   schema/type validation before execution; a small logical plan with a reference
   executor. Add laziness and optimizations after the representation is useful.
3. **Interchange:** Arrow C Data import/export with tested release callbacks,
   ownership, offsets, null masks, and dtype validation. Current bitmaps are a
   building block, not a claim that current arrays can be exported zero-copy.
   String and Boolean payload layouts must be addressed first.
4. **Broader operations and input:** numeric/multiple join and grouping keys,
   additional reductions, cast rules, multi-column sorting, and a CSV reader.
   Parquet can remain an optional integration initially; making it native is a
   separate substantial project.
5. **CPU performance:** specialize dtype dispatch outside hot loops, bitmap-aware
   iteration, SIMD kernels, and parallel execution where contract semantics permit.
   Retain scalar kernels as correctness references. Revisit checked integer sums
   and floating-point accumulation semantics before parallelizing reductions.

GPU execution, distributed execution, Python-object columns, automatic host/device
migration, pandas index compatibility, and cross-engine adapters are outside these
initial milestones. No CPU/GPU fallback is implicit.
