# Public CSV integer parser comparison

This benchmark uses the existing public read_csv in both builds. The candidate
changes only its integer conversion to the atoi_simd 0.18.1 source port used by
Polars 1.44.2. There is no alternate CSV reader. Generic casts, schema inference,
floats, temporal parsing, chunking, scheduling and string storage are unchanged.

The source port maps atoi_simd short.rs, fallback.rs and simd/sse_avx.rs to
Mojo. x86 SSSE3/SSE4.1 and AVX2 use source-shaped SIMD reductions; unsupported
SIMD targets use the packed fallback. ARM has no dedicated Neon port yet.
The MIT license is retained in third_party/ATOI_SIMD_LICENSE. See the module
header for the exact source mapping.

One deliberate CSV behavior change: unsigned negative zero is rejected, as by
Polars CSV/atoi_simd. Generic dataframe.parse cast semantics still accept it.
The public reader continues to wrap parse failures in its existing CSV errors.

## Measurements

One million rows, seven warm-cache samples per separate process, same host,
compiler and thread limits. No simultaneous tests/builds. Times are median ms.
The wide fixture contains alternating signed 19-digit and unsigned 20-digit
values; mixed has ordinary short integer IDs, Float64, Bool, quoted/escaped
Unicode strings and numeric nulls. Projection selects the signed column in
wide and id,label in mixed. Both engines validate output equality outside timing.

| Fixture | Threads | Read | Main | Candidate |
|---|---:|---|---:|---:|
| Mixed | 1 | Full | 408.49 | 404.89 |
| Mixed | 1 | Projected | 347.78 | 332.53 |
| Mixed | 32 | Full | 35.96 | 36.10 |
| Mixed | 32 | Projected | 30.45 | 33.40 |
| Wide integers | 1 | Full | 119.55 | 68.83 |
| Wide integers | 1 | Projected | 76.94 | 49.24 |
| Wide integers | 32 | Full | 20.51 | 17.93 |
| Wide integers | 32 | Projected | 16.68 | 19.17 |

Wide integer full reads improve 42% at 1 thread and 13% at 32 threads. Mixed full
reads are effectively unchanged. Both projected 32-thread cases regress: this is not
an across-the-board speedup. The shared producer queue remains unchanged;
faster parsing exposing queue contention is a hypothesis for this reader,
not a phase-measured conclusion from these samples. No parity claim is made.
All raw samples and environment/input provenance are in results/.

## Reproduction

Generate each input before timing:

```sh
python3 experiments/csv_integer/prepare.py mixed /tmp/mixed.csv
python3 experiments/csv_integer/prepare.py wide /tmp/wide.csv
```

Compile the same driver separately against main and the candidate checkout:

```sh
pixi run mojo build -I . experiments/csv_integer/bench_wide.mojo -o /tmp/bench-wide
DATAFRAME_THREADS=1 /tmp/bench-wide legacy /tmp/wide.csv 7 full
DATAFRAME_THREADS=32 /tmp/bench-wide legacy /tmp/wide.csv 7 projected
```

Use bench_mixed.mojo and mixed.csv for the mixed scenario. `legacy` is the
historical engine label in the standalone driver; both binaries call public
read_csv. For the baseline, compile the driver with -I pointing at an unchanged
b742ff2 checkout. Compile before measurement; retain every CSV output sample.


## Validation

The full repository suite passes: 365 tests in 55 modules. Direct parser tests
also pass with x86 SIMD disabled. The public-reader regression tests cover all
eight widths, exact boundaries, overflow/invalid text, unsigned negative-zero
rejection and unchanged temporal conversion. The pinned Polars 1.44.2 CSV oracle
checks 31 static fixtures and unsigned negative-zero rejection.

The full-path differential generates 100,000 seeded rows across all eight
integer widths, with boundary values, explicit signs, leading zeroes and nulls.
Mojo public read_csv -> write_csv output matches Polars parsing of the original
input at both 1 and 32 workers:

```sh
pixi run mojo build -I . experiments/csv_integer/public_integer_roundtrip.mojo -o /tmp/csv-integer-roundtrip
pixi run -e oracle python experiments/csv_integer/public_integer_differential.py /tmp/csv-integer-roundtrip
pixi run -e oracle python experiments/csv_integer/verify_integer_oracle.py
```
