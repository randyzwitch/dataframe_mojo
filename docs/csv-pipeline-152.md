# CSV pipeline experiment (#152)

Comparison base: `0030f0e877499978eed68a9cbc4cf3dd8dfa474c`.
All changes are on `perf/csv-pipeline-152`.

## Reproduce the comparison

The benchmark driver now accepts `--csv-only` and `--runner`. The first skips
unrelated dataframe workloads; the second uses an existing binary without
rebuilding it against the current checkout. Both engines still read the same
seeded files and check row counts and the sum of `x` before reporting timings.

To build a baseline with the same benchmark interface while retaining the
baseline library:

```bash
mkdir -p /tmp/dataframe-baseline-152
git archive 0030f0e | tar -x -C /tmp/dataframe-baseline-152
cp benchmarks/bench_vs_polars.mojo /tmp/dataframe-baseline-152/benchmarks/
pixi run mojo build -I /tmp/dataframe-baseline-152 \
  /tmp/dataframe-baseline-152/benchmarks/bench_vs_polars.mojo \
  -o build/bench_csv_before152
pixi run -e oracle bench-polars --csv-only --threads 32 --reps 7 \
  --runner build/bench_csv_before152
pixi run -e oracle bench-polars --csv-only --threads 32 --reps 7
```

Repeat with `--threads 1` for scaling. Alternate the two commands, with no
concurrent builds or test processes, and compare several runs. The data files
are cached in `build/bench_polars`; each process warms up before timing.
The binaries are local build artifacts and are not committed.

## First checkpoint: `f0f00c3`

- Mapped reads publish record-aligned ranges as the boundary scan proceeds.
  Worker claims are dynamic, with roughly 8 chunks per worker and limits
  based on file size and schema width. The caller counts toward the thread
  limit. Results and errors retain file order.
- Tokenization loads 64 bytes once and drains their structural positions from
  a cached mask. Quotes and CRLF still pass through the strict state machine;
  comments, skipped lines, and partial buffers keep the scalar path.
- Numeric grammar checks consume borrowed text. The standard Float64
  conversion still materializes an owned string internally.
- Result collection moves chunks in linear time instead of repeatedly removing
  the first element of a list. Concatenation jobs share one frame metadata list
  instead of copying that list for every output column.

At this checkpoint the epic remained incomplete: there is no Eisel–Lemire parser or SIMD
integer parser, borrowed fields still copy into the record buffer, and columns
remain contiguous after concatenation. The strict numeric fallback can still
allocate an owned string, including on failure. The general chunked-column
change in #151 is not included.

An exact UInt128 decimal-rounding prototype was discarded: it disagreed with
the existing standard parser on `4441829661224440.750`, `9651576341293971.0`,
and `41900288798.355793`. The first two are halfway cases, but excluding only
halfway cases was insufficient. The regression tests retain these cases and
compare thousands of 17-digit decimals at every decimal-point position against
the reference. The final branch keeps the original float conversion semantics.

### First-checkpoint measurements

AMD Ryzen Threadripper 3970X, Linux x86-64, pinned Mojo 1.2 nightly,
Polars 1.44.2. Two interleaved before/after runs at each thread count,
seven repetitions per run after warmup. No concurrent builds or tests.
The table gives the range of each run's minimum, not a confidence interval.

| Rows | Threads | Before (ms) | After (ms) | Polars during after runs (ms) |
|---|---:|---:|---:|---:|
| 100,000 | 1 | 65.73–65.80 | 57.35–57.49 | 21.72 |
| 100,000 | 32 | 11.29–11.40 | 9.28–9.32 | 2.69–2.70 |
| 1,000,000 | 1 | 663.81–668.47 | 576.83–580.74 | 212.61–213.59 |
| 1,000,000 | 32 | 66.49–80.70 | 40.70–42.68 | 16.39–17.59 |

Using the minimum across runs, the 1M-row parallel read is 39% faster and
scales 14.2x from one to 32 threads (baseline: 10.0x). The first-checkpoint comparisons are 2.3–2.6x Polars. The working target
is now parity or better; within 2x is only an intermediate milestone. At 100k rows,
thread startup and chunk overhead still dominate much of the read.

Eight chunks per worker were selected by paired measurements. With the final
numeric conversion, 16 chunks per worker took 43.97/43.69 ms and eight took
41.26/39.35 ms in two tuning runs. The final report above comes from separate
runs. Eight produces 256 chunks of roughly 194 KB for the reference file.

### Phase breakdown

A separate build instrumented `_read_mapped_produced` and each range's `run`
with `monotonic()`. Instrumentation is absent from normal library builds.
One late warmed read of 1M rows at 32 threads measured:

| Phase | ms |
|---|---:|
| Pool and producer setup | 1.24 |
| Boundary scan, reader creation, and publication (overlaps decoding) | 8.38 |
| Remaining decoding, draining results, and joining workers | 20.25 |
| Moving frames into the result list | 1.47 |
| Concatenation | 8.53 |

These phases start after the file is mapped. Their total is 39.87 ms; the
instrumented end-to-end minimum, including timing/reporting overhead, was
41.42 ms. Scan time is elapsed producer time, not an isolated scan CPU cost;
it overlaps workers' decoding and must not be added to a separate full decode
measurement.

For that read, 256 ranges had a median decode time of 2.63 ms and a maximum of
6.86 ms (2.61x). Across the seven timed reads, maxima ranged from 6.86 to
8.80 ms and medians from 2.63 to 3.07 ms. This bounds the cost of a delayed
worker to a much smaller range, but does not eliminate scheduler variability.
Concatenation still copies values and remains a material cost.

Local raw results are in `build/csv_comparison152.txt` and
`build/csv_phases152.txt` (untracked build artifacts).

### First-checkpoint validation

- Full suite: 48 modules passed. The final changes also passed the focused
  CSV parallel/fuzz, worker-pool, and float-parser modules.
- Polars oracle: 100/100 cases matched, seeds 1 through 100, on the final code.
- Formatting: all 92 Mojo files passed. Public API documentation, dtype
  literal checks, version consistency, and Python syntax checks passed.
- CSV comparisons cover several worker counts, quoted newlines, CRLF, missing
  final terminators, and exact record numbers for late parse errors.
- Fuzz and alignment tests force streaming with a finite row limit so that
  one-byte buffers really exercise the scalar reader; otherwise mmap would
  bypass `buffer_size`. Valid quoted records are slid across all SIMD offsets.
- Pool tests check dynamic and produced rounds, exactly-once execution,
  ordered errors, capacity failure during production, reuse, and serial fallback.

Only Linux x86-64 was executed locally; macOS/arm64 still need CI coverage.

## Second checkpoint: borrowed records and consuming builders

This checkpoint builds on `f0f00c3` on the same branch. The target is parity
with Polars or better; crossing below twice its runtime is progress, not
completion of #152.

- Complete unquoted LF-terminated records decode directly from input spans.
  Their field offsets no longer require a copy into `record_bytes`. String
  builders still copy retained output text into owned column storage; no
  input borrow outlives `feed`.
- Headers and partial records finish through the existing state machine.
  Quotes, CR, and wrong field counts replay the whole current record through
  that path, preserving error order and locations. The remainder of that
  buffer also uses the existing path. Sampling, comments, skipped lines,
  lossy UTF-8, ignored errors, and ragged-row truncation retain their previous
  paths. Projected-out String fields still receive UTF-8 validation.
- Column finalization consumes builders, transferring numeric buffers and
  string storage instead of copying builders and their values. Narrow types
  still allocate converted values; validity still packs into bitmaps.
  Builders are separate fields because Variant projections cannot transfer
  their payloads. This adds unused empty builders, including a StringBuilder
  offset allocation for numeric columns, which may matter on very wide schemas.
- Int64 fields with at most 18 digits validate and accumulate without a
  per-digit overflow check. Longer fields retain the exact range-checked
  parser. Float conversion is unchanged.

### Paired measurements

Same machine, data, and seven-repetition method as above. Two interleaved
runs per version and thread count, with no concurrent builds or tests.
Each interval contains the two runs' minimum timings.

| Rows | Threads | `f0f00c3` (ms) | Second checkpoint (ms) | Polars during second-checkpoint runs (ms) |
|---|---:|---:|---:|---:|
| 100,000 | 1 | 57.58–58.08 | 39.08–39.42 | 21.58–21.83 |
| 100,000 | 32 | 9.62–10.33 | 6.91–7.10 | 2.53–2.60 |
| 1,000,000 | 1 | 574.91–590.50 | 390.33–397.17 | 210.89–212.36 |
| 1,000,000 | 32 | 41.60–41.67 | 28.72–29.07 | 16.72–17.40 |

The 1M-row parallel read takes about 31% less time than the first checkpoint,
and about 57% less than the original base's best 66.49 ms measurement.
Its runtime remains 1.67–1.72x Polars. The 100k-row parallel read remains
2.66–2.81x Polars, so even the intermediate 2x milestone is workload-specific.

An isolated borrowed-record build, before integer and builder changes,
measured 31.18 ms at 32 threads and 408.59 ms at one thread; the paired
first-checkpoint timings were 41.42 and 582.34 ms. Borrowed record decoding
accounts for most of this batch's improvement on this workload.

A late warmed read from the separately instrumented second-checkpoint build:

| Phase | ms |
|---|---:|
| Pool and producer setup | 1.28 |
| Boundary scan, reader creation, and publication (overlaps decoding) | 8.63 |
| Remaining decoding, draining results, and joining workers | 9.04 |
| Moving frames into the result list (includes diagnostic printing) | 0.76 |
| Concatenation | 8.13 |

Instrumented end-to-end minimum: 28.84 ms. Remaining decode/drain/join time
fell from roughly 20 ms to 9 ms; concatenation still copies the result and
costs about 8 ms. Next candidates are eliminating that assembly copy,
reducing producer/reader setup overhead, and improving numeric conversion.
The producer phase overlaps decoding and is not an independent additive
CPU cost. Correct float rounding and broad CSV coverage remain required.

Reproduce using the existing benchmark driver and a binary built at either
commit. The local saved runners are `build/bench_csv_checkpoint152` and
`build/bench_csv_next152`; raw paired results and phase traces are
`build/csv_comparison_next152.txt` and `build/csv_phases_next152.txt`.

### Validation

- All 49 test modules pass on the final source.
- All 100 Polars oracle cases match, seeds 1 through 100.
- Six new borrowed-record differential tests compare one-byte scalar reads
  with 64-, 65-, 128-, and 4096-byte buffers. They cover structural-mask
  offsets, partial headers/rows, nulls, quotes, CRLF, row limits, malformed
  records, exact error messages, and invalid projected-out String fields.
  These tests caught and fixed an empty-buffer transition that cleared
  partially collected field offsets.
- Integer tests cover signs, leading zeros, and malformed short inputs;
  existing dtype and Int64 boundary tests also pass.
- Changed Mojo files pass formatting checks. Public API documentation,
  dtype literal checks, version consistency, and `git diff --check` pass.
- Only Linux x86-64 was executed locally; macOS/arm64 need CI coverage.
