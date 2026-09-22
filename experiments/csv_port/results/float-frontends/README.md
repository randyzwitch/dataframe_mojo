# Addressable float lookup tables

The retained change maps fast-float2 0.2.4 float.rs fixed power-of-ten tables
to addressable Mojo Array constants. Values, masked indices, scanning,
rounding and fallback control flow are unchanged. Assembly now loads the
selected scalar directly from a static table instead of copying SIMD table
constants to the caller stack.

Same machine, one-million-row mixed fixture and seven-sample warm-cache
method as ../initial-source-port; no concurrent builds/tests. Paired medians
in milliseconds, based on 9ed7249 plus only the table representation change:

| Read | Threads | Before | Static arrays |
|---|---:|---:|---:|
| Full | 1 | 260.65 | 247.91 |
| Projected | 1 | 145.75 | 147.55 |
| Full | 32 | 18.96 | 19.82 |
| Projected | 32 | 22.56 | 22.58 |

Full single-thread reads improved ~4.9%; projected reads exclude Float64 and
are a control. Multithreaded results do not show a gain. Retained raw samples
are static-{baseline,only}-*.csv. Numeric tests pass 4/4, covering all 419
expected Float32/Float64 bit patterns independently verified with Polars 1.44.2.
The final static-table reader also passes all 24 actual Polars differential
comparisons (explicit/inferred/projection/options, one and four threads).

A larger candidate also inlined the scanner and outlined slow fallback code.
That changed full 1T 259.20 to 241.23 ms and projected 1T 147.46 to 142.98 ms,
but full 32T regressed 17.50 to 21.24 ms; projected 32T was 22.73 to 22.51 ms.
Those samples are tables-{typed-trusted,float-tables}-*.csv. The larger
candidate was not retained. It passed numeric and differential tests but its
mixed timings did not justify the additional control-flow changes.

The standalone driver is experiments/csv_port/bench_csv_port.mojo. Build with
`pixi run mojo build -I . experiments/csv_port/bench_csv_port.mojo -o /tmp/bench`.
Run `DATAFRAME_THREADS=N /tmp/bench explicit FILE 7 full` (or projected).
The base source commit is 9ed7249; the static-only commit containing this report
is the candidate. Initial fixture/environment metadata are in
../initial-source-port/metadata.json. This remains one workload, not parity.
