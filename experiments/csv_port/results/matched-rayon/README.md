# Identical-parser Rayon comparison

The scoped and native Rayon binaries contain the same dataframe sources from
2637cbc except csv_reader.mojo, which wraps the same decode path in pinned
Rayon 1.12.0 / rayon-core 1.13.0 scope/spawn through the isolated native bridge.
The native dependency remains outside production. Source equality was checked
before building; the scoped binary is the static-table candidate retained at
e21d6ea. Only docs changed between that commit and 2637cbc.

One million rows per fixture; identical explicit schema, file, projection,
thread limits and seven warm-cache read-only samples. All processes ran
sequentially, without simultaneous builds/tests. Full output equality is checked
outside timing. Rayon reader tests pass 7/7 and actual Polars comparisons 24/24.
Polars 1.44.2 reference samples are freshly collected in the same window.

| Fixture | Threads | Read | Scoped ms | Rayon ms | Polars ms |
|---|---:|---|---:|---:|---:|
| mixed | 1 | full | 250.78 | 264.98 | 139.80 |
| mixed | 1 | projected | 148.08 | 164.67 | 93.07 |
| mixed | 32 | full | 20.78 | 28.02 | 11.71 |
| mixed | 32 | projected | 22.56 | 17.31 | 9.71 |
| short-ascii | 1 | full | 197.78 | 216.31 | 108.58 |
| short-ascii | 1 | projected | 114.21 | 126.45 | 75.01 |
| short-ascii | 32 | full | 17.10 | 19.90 | 8.49 |
| short-ascii | 32 | projected | 19.88 | 12.94 | 6.95 |
| long-ascii | 1 | full | 225.16 | 241.88 | 121.15 |
| long-ascii | 1 | projected | 137.48 | 158.86 | 86.45 |
| long-ascii | 32 | full | 29.88 | 26.69 | 15.83 |
| long-ascii | 32 | projected | 31.47 | 18.37 | 13.40 |

Rayon improves projected 32-thread reads on every fixture, but regresses full 32-thread reads
on mixed and short ASCII, and regresses all single-thread reads. This supports
the shared-queue contention diagnosis but does not establish an overall
scheduler improvement or justify adopting the native dependency. No runtime
choice is selected dynamically from workload or benchmark results.

Reproduce fixtures with prepare.py --rows 1000000 --profile mixed, short-ascii,
or long-ascii (with spaces between flags and values). The mixed fixture is
unchanged. Short ASCII labels fit in BinaryView's inline 12 bytes; long ASCII
labels require backing buffers. Both remove the mixed fixture's quoted
separators, embedded newlines, escapes and Unicode, retaining its numeric
values, numeric nulls and Boolean distribution.

Binaries for this local experiment: /tmp/bench_csv_port_9ed7249_static_tables
(scoped) and /tmp/csv_bench_rayon_current (Rayon). The latter snapshot is
/tmp/dataframe-mojo-rayon-current; its only dataframe source difference is
csv_reader.mojo. Native bridge provenance: ../rayon-isolated/README.md.
These temporary Rayon build paths are diagnostic, not a supported package.
