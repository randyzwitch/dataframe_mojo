# Numeric parsing follow-up to #153

This branch starts at `b5dd9c5`, the final implementation commit in #153.
It advances #149 and #152; neither issue is complete.

## Implementation

The plain-decimal scan already accumulates up to 19 digits in a UInt64.
For mantissas above 2^53, pass that integer and the decimal exponent directly
to Mojo's `lemire_algorithm` instead of constructing a String and parsing
its digits again. The exact 2^53 boundary uses the same Clinger path as the
standard converter. Apply the sign after conversion, preserving signed zero.

Exponent, special, and longer inputs retain strict grammar validation, then
call the standard library's `_atof(StringSlice)` entry point. Successful
conversion no longer constructs an owned String at this boundary. Error
messages are still constructed only when needed. This is not a claim that
all standard-library internals have been audited for allocation.

Both entry points are private APIs from the Mojo 1.2 nightly in `pixi.lock`.
An upgrade can break their signatures or behavior; the independent
`_parse_float64_strict` reference still uses `Float64(text)` so comparisons
can detect that. This change intentionally preserves the current converter's
rounding, including known differences from correctly rounded conversion.
It does not introduce a new Float32 rounding implementation.

## Measurement method

Compare separate binaries built from #153 and this branch, on an idle host.
CSV timings use the best of seven reads per process, repeated in three paired
processes with matching thread counts. File contents and checksums match.
The plain fixture is the existing 49.7 MB, 1-million-row, eight-column CSV;
the quoted fixture quotes its string key. No builds or tests run during timing.

`benchmarks/bench_numeric_parse.mojo` isolates ten million conversions of
sixteen varying inputs per family. It checks an accumulated checksum to
prevent constant folding. Parser timings are not CSV ablations.

```sh
mkdir -p /tmp/dataframe-base-149
git archive b5dd9c5 | tar -x -C /tmp/dataframe-base-149
pixi run mojo build -I /tmp/dataframe-base-149 benchmarks/bench_vs_polars.mojo -o build/bench_csv_base149
pixi run mojo build -I . benchmarks/bench_vs_polars.mojo -o build/bench_csv_numeric149
pixi run -e oracle bench-polars --csv-only --threads 32 --reps 7 --runner build/bench_csv_numeric149
pixi run mojo build -I /tmp/dataframe-base-149 benchmarks/bench_numeric_parse.mojo -o build/bench_numeric_base149
pixi run mojo build -I . benchmarks/bench_numeric_parse.mojo -o build/bench_numeric149
build/bench_numeric_base149
build/bench_numeric149
```

## CSV results

AMD Ryzen Threadripper 3970X, Linux x86-64, pinned Mojo 1.2 nightly,
Polars 1.44.2. Ranges below are the three process minima, not confidence
intervals. Polars values are from the after-change comparison processes.

| Fixture | Rows | Threads | #153 (ms) | This branch (ms) | Polars (ms) |
|---|---:|---:|---:|---:|---:|
| Plain | 100,000 | 1 | 38.14–38.61 | 30.96–31.13 | 21.65–22.05 |
| Plain | 100,000 | 32 | 5.72–5.85 | 5.16–5.25 | 2.54–2.62 |
| Plain | 1,000,000 | 1 | 379.82–388.22 | 317.24–318.71 | 214.20–215.12 |
| Plain | 1,000,000 | 32 | 25.84–26.44 | 22.28–23.43 | 15.89–20.59 |
| Quoted | 1,000,000 | 1 | 423.19–425.48 | 358.26–361.26 | 214.11–219.79 |
| Quoted | 1,000,000 | 32 | 29.97–31.05 | 26.84–31.36 | 14.84–15.57 |

Paired plain 1M reads improved 11–16% at 32 threads and 16–18% at one
thread. Quoted reads improved 15–16% on one thread. At 32 threads, two
quoted trials improved 11–14%, but the third regressed about 5%; this remains
noisy. Neither fixture reaches Polars parity. The 100k parallel workload
still takes about twice as long as Polars.

## Parser results

Ten million conversions per family, three paired processes:

| Family | #153 (ms) | This branch (ms) |
|---|---:|---:|
| Short plain decimal | 126.67–132.36 | 129.32–133.67 |
| Wide plain decimal | 2017.36–2110.87 | 287.29–288.79 |
| Exponent | 1799.41–1855.38 | 1840.44–1850.05 |

Wide decimals improve about 7.0–7.3x. Short decimals range from essentially
unchanged to about 4% slower in the paired runs; exponent results show no
consistent speedup. Removing owned-string construction on the exponent
route does not remove its grammar scan and standard-library digit scan.
The end-to-end improvement is driven by the wide-decimal path.

## Validation

The float tests compare result bits, acceptance, and borrowed-slice error
messages with the unchanged strict reference. They cover generated decimal
scales, signs, 2^53 boundaries, subnormals, overflow, long inputs, special
values, malformed grammar, and known rounding-sensitive examples. An
additional independent deterministic audit checked 235,948 conversions
without an acceptance or bit mismatch.

All 49 test modules passed. The 100-case Polars oracle (seeds 1–100),
API documentation, version, dtype
literal, and changed-file formatting checks passed.

## Remaining work

SIMD integer prototypes improved 18-digit fields by about 43%, but regressed
single-digit fields by 10–20% or more. They are not included. Most integers
in the CSV fixture are short, so selecting a long-input microbenchmark win
would not establish an improvement to this workload.

#149 still needs integer improvements without short-field regressions,
Float32 work, and a fuller single-pass exponent/long-mantissa implementation
with documented ambiguous-case behavior. #151's chunked-column storage
and the overall performance target in #152 remain open.
