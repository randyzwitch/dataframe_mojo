# Matching the pinned Polars allocator (isolated experiment)

Polars 1.44.2 Linux uses tikv-jemallocator 0.7.0 with the patched
jemalloc-sys revision 0d683dfb157097e2075d5e0eaf25f71f514a7552.
Source: crates/polars-python/src/c_api/allocator.rs and
crates/polars-ooc/src/global_alloc.rs. The installed runtime contains the
prefixed jemalloc implementation. Mojo AOT binaries import glibc malloc/free.
This is an established implementation difference, not an assumed bottleneck.

An isolated Rust interposition library forwards ordinary malloc/calloc/realloc/
free/aligned allocation calls to the exact pinned private _rjem functions.
It uses disable_initial_exec_tls and background_threads features, with
_RJEM_MALLOC_CONF=dirty_decay_ms:500,muzzy_decay_ms:1000, matching
py-polars/src/polars/__init__.py without optional POLARS_THP overrides.
Only the Mojo child is preloaded. Python/Polars keeps its own allocator.

All 24 actual Polars differential cases pass under preload for each of the
scoped and Rayon readers (48 comparisons). Preloading the entire CPython/
Polars process crashed in the earlier smoke; no claim is made that this
interposer is a generally safe package dependency or equivalent static linkage.
The production repository has no new allocator dependency.

Same parser binaries and fixtures as ../matched-rayon, seven samples,
warm cache, explicit matched worker counts, sequential processes, no builds
or tests during timing. Output equality is checked outside each read timer.
Medians in milliseconds:

| Fixture | Threads | Read | Scoped/glibc | Scoped/jemalloc | Rayon/glibc | Rayon/jemalloc | Polars |
|---|---:|---|---:|---:|---:|---:|---:|
| mixed | 1 | full | 249.16 | 249.19 | 259.58 | 261.16 | 140.77 |
| mixed | 1 | projected | 148.46 | 146.01 | 164.88 | 167.23 | 97.46 |
| mixed | 32 | full | 16.77 | 18.84 | 26.44 | 22.53 | 11.10 |
| mixed | 32 | projected | 23.49 | 23.83 | 18.17 | 15.93 | 10.88 |
| short-ascii | 32 | full | 15.74 | 16.83 | 21.03 | 21.39 | 8.66 |
| short-ascii | 32 | projected | 18.59 | 18.18 | 13.06 | 13.37 | 6.96 |
| long-ascii | 32 | full | 29.64 | 29.50 | 27.38 | 25.16 | 15.69 |
| long-ascii | 32 | projected | 30.80 | 31.39 | 22.81 | 22.50 | 14.24 |

The allocator helps some parallel Rayon cases, most noticeably mixed full
and projected reads, but is not a general win. Single-thread results barely
change; short ASCII gains do not materialize. The source-matched scheduler
experiment still does not achieve Polars parity. Do not combine the fastest
cell from each runtime into an imaginary reader or attribute every gap to
allocation.

Reproduction sources retained in this result directory under probe/: use
cargo build --release --locked in that directory, then apply the resulting .so only
to the Mojo benchmark process via LD_PRELOAD, with the configuration above.
The scope/decoder binaries are the same immutable snapshots identified in
../matched-rayon. The original local probe is /tmp/polars_jemalloc_probe.
The bridge matches allocator version/features/decay settings; it intentionally
differs from Polars' private static linkage and is not adopted for production.
