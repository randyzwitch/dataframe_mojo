# Changelog

All notable changes are listed here, newest release first. Each bullet gives
the pull requests that made the change; the pull request has the measurements
and tests. Release tags are cut by the maintainer.

## Unreleased (becoming 0.3.0)

This release was built against Mojo 1.2 nightlies. It is not yet tagged.

### Executor and performance

- Lazy plans over in-memory frames run as thread-owned pipelines over morsels, with joins as probe steps and group-by, top-k and distinct as pipeline sinks (#539, #543, #548, #553, #555, #557).
- Group-by uses a radix-partitioned aggregate table with inline states, partitioned row blocks, a parallel finish and DuckDB's partition sizing (#641, #650, #671, #711, #782, #886).
- Group-by keys that fit a small integer domain are numbered by perfect hash or direct lookup, as DuckDB does (#843, #858).
- Hash joins gained a tag byte per slot, open addressing, key-range and bitmap rejection of absent probe keys, and direct routes for dense, progression and perfect-hash keys (#494, #698, #721, #787, #829).
- Joins choose their build side from observed sizes, and join graphs are ordered by estimated cost after DuckDB's join order optimizer (#421, #725, #733, #747).
- DuckDB's distinct-count statistics are kept for loaded tables and used to order joins; filter estimates between joins become join graph edges (#759, #760, #838).
- Join key sets, key ranges and aggregates are passed sideways from one join input to the other, including semi-join reduction of aggregates (#503, #508, #690, #709, #717, #839).
- Repeated subplans run once, are found by hash and are pruned to the columns their readers need (#531, #653, #847, #891, #892).
- Sorting packs keys into integers, ranks in parallel, uses stable radix passes and cuts short a sort followed by a slice (#133, #135, #358, #359).
- Top-k keeps the k best rows in the sink and skips morsels by their key's zone maps, as DuckDB's Top-N filter does (#499, #557, #867).
- Window functions run in one pass, number their partitions with the aggregate table's hashing and evaluate partitions contiguously (#592, #659, #679, #692).
- Filters run on selection trees narrowed in place, with null tests read from bitmaps and comparisons written straight to packed bits (#350, #429, #745, #718).
- Comparisons, Boolean logic, conditionals and narrow-integer arithmetic run in SIMD blocks (#208, #399, #493, #712).
- Strings are Arrow string views in memory, with byte-level comparison, a SIMD search filter and views gathered without copying (#302, #394, #397, #686).
- Categorical columns and Parquet dictionary codes stay coded through filters, sorts, joins and group-bys (#370, #430, #461, #495).
- Hash tables and partition buffers sit on transparent huge pages, and all parallel work of a query runs on one crew of threads (#514, #536, #663, #704).
- The planner caches node schemas across optimizer passes and skips work that cannot change a plan, cutting fixed cost on small queries (#687, #762, #845, #893).
- Gathers, concatenation, join output and chunk merges run on every worker and avoid copying all-valid validity (#137, #139, #366, #836).
- Decimal arithmetic follows DuckDB's result types and avoids 128-bit division on every row (#398, #415, #694, #702).

### Lazy frames and planner

- Added `LazyFrame` methods for tail, reverse, shift, with_row_index, rename, cast, unpivot, pivot, describe, gather, concat and cross join (#479, #567, #788, #813, #823, #824, #826).
- Lazy `group_by_dynamic` and `rolling` are available (#812).
- Lazy joins accept differently named keys, and `scan_csv` passes `read_csv`'s options to every read path (#356, #825).
- Lazy plans execute as bounded ordered batch pipelines and stream group-bys only when keys repeat (#296, #367).
- Filters are pushed through renames, unions, shared subplans and the other side of an equi-join (#688, #689).
- Whole-input selections run ahead of a join stream, and a stream's computed inputs run at once (#708, #818, #821).
- Sort ties are unordered unless `maintain_order=True`, as in Polars, which lets joins reorder under sorts (#817).
- `explain()` prints the plan alone, and execution reports show observed counters and time per node (#484, #510, #554).

### API features

- Added decimal columns with Arrow-width storage, correct rounding and decimal results that match DuckDB (#339, #346, #425, #702).
- Added list and struct columns, with split, explode, unnest, implode, as_struct and struct keys (#283, #285).
- Added time-zone-aware datetimes, binary columns and categorical columns (#352, #353, #370).
- Added geospatial storage with GeoJSON, GeoArrow and GeoParquet interchange, WKB accessors and a native shapefile reader (#500, #502, #504, #515).
- Added eager and lazy as-of joins, and calendar-aware time windows with rolling std and var (#506, #507).
- Added statistics and shape expressions: arg_min, arg_max, mode, skew, kurtosis, corr, cov, describe, sample, value_counts, interpolate, cut and qcut (#324, #325, #345).
- Added horizontal sum, min, max, mean, any and all, plus `replace` and `replace_strict` (#831, #832).
- Added log, trig, sign, cbrt, truncate, product and cum_prod expressions (#828).
- Added string functions that need no regex, such as titlecase, strip_prefix, find and split_exact (#822).
- Added a linear-time regex engine in the RE2 and Rust regex family and used it in the `.str` namespace (#835, #841, #882).
- Added `str.like` and string-with-literal comparisons over raw bytes (#392, #397).
- Boolean columns sum, average and cumulative-sum as in Polars, and NaN sorts first in descending order as in Polars and DuckDB (#815, #827).
- Added concise tuple schemas to `read_csv` and write NaN as NaN in CSV (#293, #406).
- Added a `LazyFrame.concat` (UNION ALL) plan node (#567).
- Join column names round-trip losslessly, and `coalesce=False` keeps right keys for every join kind (#588, #589).
- Fixed UTF-8 temporal format literals and escaped percent signs (#730).

### IO

- `read_csv` is a Polars-derived pipeline with parallel decoding, memory-mapped input and fast integer and float parsing (#153, #159, #160, #161).
- Parquet is read through a minimal Arrow build behind a swappable backend, streamed by row group into chunked columns (#275, #291).
- Added `write_parquet` with codec and row-group options (#294).
- Parquet decimals load at the smallest width their precision allows (#558).
- Arrow import runs in parallel and keeps field metadata and unknown extensions (#295, #501).
- Dropped all-valid bitmaps on Arrow import so bitmap-free paths stay active (#587).
- File-format IO moved into a `dataframe/io` subpackage and the unused `dfq_read_parquet` entry point was removed (#728, #749).

### Metal and accelerators

- Added `engine` selection with `cpu`, `auto` and `accel` (#541).
- Added an accelerator provider interface with row planning, capability checks and memory preflight shared by every backend (#545, #551, #561).
- Added an optional NVIDIA provider with column storage, transfer ownership and float reductions (#542, #544, #546).
- Added an Apple Metal accelerator for selections, projections, sorts, aggregates, joins, windows and distinct counts, then rebuilt it in Mojo and retired the native runtime (#565, #577, #578, #581, #684).
- Metal matches CPU results for Float32 and Float64 arithmetic, rounding, modulo, power, exp and log (#601, #603, #610, #668, #672).
- Metal runs exact decimal arithmetic, comparisons, sums and means for Decimal32, 64 and 128 (#618, #637, #642, #643, #644).
- Metal supports rolling, shifted and interpolated windows, duration and time-zone windows, and calendar arithmetic (#645, #646, #669, #676, #678).
- Metal supports variance, covariance, correlation, skew, kurtosis, median and quantiles (#647, #649, #655, #660).
- On any data fault Metal reruns the query on the CPU (#819).
- Added a Mojo GPU path for Lists, explode, cut and qcut, time-zone mutations and hash joins (#699, #726, #741, #901).
- Added a Metal baseline on the development suites with a forced-Metal grid, and archived paired reduction, grouping and join evidence (#849, #889, #895).
- Metal contracts run on a local Apple silicon Mac before PRs, and the hosted CI job was retired (#710, #786).

### Benchmarks and CI

- Benchmarks run against H2O, PDS-H and TPC-DS with answers checked against Polars and DuckDB, and ClickBench is held out (#322, #467, #757).
- Added a TPC-DS translation for 89 of 99 queries for this library and Polars (#470, #473, #476, #481, #575, #879).
- Reports include provenance, the CPU governor, per-query execution traces and a Mermaid plan explorer (#349, #446, #471, #534).
- Benchmarks moved to Polars 2.0 and DuckDB 2.0 (#907, #910).
- Retired micro-benchmarks that the external suites supersede (#323, #743).
- CI builds libdfparquet on Linux, runs contract areas as parallel jobs and drops benchmark and package-smoke steps from pull requests (#284, #497, #522).
- The test suite was rebuilt around maintained dataframe contracts with independent expected values (#624).
- Suite policy entries live as markers in the test modules they describe (#784).
- Test modules compile into a few driver programs to fit CI memory (#361, #475, #482, #638).
- Cleared every compiler warning in the library and its tests (#685, #703, #909, #919, #922).

### Repo hygiene

- Benchmark evidence stays out of the tree, and a CI check fails when result files are tracked (#166, #748, #918).
- Only runnable programs live under `tests/`, and a CI check fails when one is unreachable (#917, #920).
- Docs are split into user docs and design notes (#921).
- Benchmark scripts moved under `benchmarks/` and dead code was removed (#746, #753).
- The Mojo toolchain moved to the newest nightly (#591, #906).

## 0.2.0 - 2026-09-21

Mojo 1.2 only. See the [GitHub release notes](https://github.com/randyzwitch/dataframe_mojo/releases/tag/v0.2.0).
