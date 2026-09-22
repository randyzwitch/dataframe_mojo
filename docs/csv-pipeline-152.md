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

## Third checkpoint: quote scanning and buffer assembly

Built on `d92c063` on the same branch. The main changes are:

- Reader construction moves from the producer to workers, which reserve
  retained column capacity from the boundary scanner's row counts.
- Quote-free blocks retain the existing short SIMD path. Quote-bearing and
  target-crossing blocks use 64-bit masks and prefix XOR to identify record
  terminators without repeatedly loading overlapping blocks. Doubled quotes
  still toggle parity twice; the decoder remains responsible for syntax.
- Validity packing writes eight Boolean values per output byte. Bitmap
  appends bulk-copy aligned bytes and SIMD-shift unaligned destinations.
  String offset rebasing uses SIMD, and parallel concat schedules separate
  byte, offset, and validity jobs for each string column in one global pool.
- Fully consumed plain decimals with a wide mantissa call the standard
  Float64 converter directly, skipping a redundant grammar scan. Conversion
  still allocates internally and preserves the existing rounding behavior.

### Paired measurements

Same hardware and versions as previous checkpoints, measured September 22.
Two interleaved runs, seven timed repetitions after warmup, with no concurrent
builds or tests. Ranges contain each run's minimum.

| Input | Rows | Threads | `d92c063` (ms) | Third checkpoint (ms) | Polars during third-checkpoint runs (ms) |
|---|---:|---:|---:|---:|---:|
| Plain | 100,000 | 1 | 39.65–39.77 | 38.87–39.44 | 21.62–21.77 |
| Plain | 100,000 | 32 | 7.07–7.09 | 6.09–6.44 | 2.42–2.58 |
| Plain | 1,000,000 | 1 | 395.41–406.05 | 394.99–395.89 | 213.68–214.84 |
| Plain | 1,000,000 | 32 | 29.15–29.64 | 26.40–27.59 | 16.66–17.53 |
| Quoted strings | 1,000,000 | 32 | 90.42–92.26 | 42.55–42.75 | 15.49–16.42 |

The quoted fixture contains the same values as the plain file, with only
`key_str` surrounded by quotes. These results represent about 9% less time
for the plain parallel read and 53% less for the quoted read, comparing the
best run minima. Single-threaded improvements are small. Parity is still
unmet: approximately 1.6x Polars on plain input and 2.6–2.7x on quoted input.

A late warmed read in a separate instrumented build measured:

| Phase | ms |
|---|---:|
| Pool and producer setup | 1.26 |
| Overlapping scan and publication | 7.93 |
| Remaining decoding, drain, and join | 9.06 |
| Frame collection, including diagnostic output | 0.61 |
| Concatenation | 6.25 |

Instrumented end-to-end minimum was 26.12 ms. Concatenation remains material;
this checkpoint parallelizes copies rather than eliminating them. Quoted
fields still fall back to the copying decoder after boundary discovery.

### Experiments and validation

- Chunk factors 1, 2, 4, 8, 16, and 32 were compared. Eight remains the default;
  fewer chunks increased tail variability and more increased overhead.
- A wider 256-byte quote-free scan did not reliably improve runtime and was
  omitted. A parity-only scanner helped quoted input but not consistently
  plain input, motivating the hybrid path.
- Assembly inspection confirmed that `List.extend(Span[UInt8])` already uses
  AVX bulk copying. Replacing it with another byte-copy loop was not pursued.
- All 49 test modules and all 100 Polars oracle cases (seeds 1–100) pass on
  the final source. Changed Mojo formatting, API docs, dtype literal checks,
  version consistency, and `git diff --check` pass.
- New tests exercise bitmap source/destination alignments 0–7 and counts
  around 128/256-bit boundaries, large sliced nullable string concatenation,
  and a malformed quote after enough records to ensure actual parallel CSV
  decoding. The concat suite also passes with `DATAFRAME_THREADS=1`.

Saved comparison runners are `build/bench_csv_checkpoint2_152` and
`build/bench_csv_final_third152`. Raw results are
`build/csv_comparison_final_third152.txt`, `build/csv_phases_final_third152.txt`,
and `build/csv_tuning_third152.txt`. The quoted fixture is in
`/tmp/dataframe_mojo_quoted_keystr152`; use `--data-dir` and `--sizes 1000000`
with the benchmark driver to compare it. These are local untracked artifacts.

## Fourth checkpoint: borrowed simple quoted fields

Built on `696fb9e`. Once a plain-record scan encounters a quote, a second
SIMD structural-mask loop handles simple quoted fields directly in the
input span. Closing-quote offsets are encoded in the temporary field-end
list; a compile-time specialization strips the quote wrappers and preserves
quoted-empty and quoted-null-token semantics. The plain specialization is
unchanged. No borrow survives `feed`.

Escaped quotes, embedded CR/LF, incomplete records, and malformed quote
sequences replay the entire current record through the existing state
machine. UTF-8 validation still precedes typed conversion and still applies
to projected-out String fields. This is not a general zero-copy CSV reader:
retained string values still copy into output column storage.

### Paired measurements

Same machine and versions, September 22, two interleaved runs with seven
repetitions after warmup and no concurrent builds/tests. Each range contains
the two run minima, not a confidence interval.

| Input | Rows | Threads | `696fb9e` (ms) | Fourth checkpoint (ms) | Polars during fourth-checkpoint runs (ms) |
|---|---:|---:|---:|---:|---:|
| Plain | 100,000 | 1 | 38.80–39.16 | 37.88–38.01 | 21.35–21.51 |
| Plain | 100,000 | 32 | 5.85–5.94 | 5.68–5.99 | 2.37–2.77 |
| Plain | 1,000,000 | 1 | 386.64–391.59 | 381.41–387.36 | 211.71–211.85 |
| Plain | 1,000,000 | 32 | 26.03–27.11 | 26.43–27.16 | 15.74–16.05 |
| Quoted strings | 1,000,000 | 1 | 579.08–579.68 | 419.63–421.53 | 208.71–213.48 |
| Quoted strings | 1,000,000 | 32 | 41.71–42.04 | 29.43–29.92 | 13.73–13.76 |

The quoted parallel read takes about 29% less time and the single-threaded
read about 28% less time, comparing best minima. Plain parallel timings
remain within the observed run-to-run variation. Polars was faster on the
quoted fixture in this measurement session too: parity is still unmet,
with roughly 1.7x its runtime for plain parallel input and 2.1–2.2x for
quoted parallel input.

A scalar borrowed-quote prototype took 32.14–32.18 ms versus the SIMD
prototype's 28.81–29.31 ms in a separate paired experiment, so the SIMD
version was selected. Those are exploratory numbers; the table above is
the final integrated-source comparison.

All 49 test modules and 100 Polars oracle cases (seeds 1–100) pass. The nine
borrowed-record tests now include quote wrappers at all mask offsets,
separators inside quotes, quoted empty strings/null tokens, malformed
suffixes, escapes, incomplete quotes, and invalid quoted UTF-8 before a
bad numeric conversion. Formatting, API docs, dtype literal checks,
version consistency, and `git diff --check` pass.

Raw results: `build/csv_comparison_fourth152.txt` and
`build/csv_quoted_prototypes152.txt`. The final runner is
`build/bench_csv_fourth152`, compared with `build/bench_csv_final_third152`.
To recreate the quoted fixture after generating the normal benchmark data:

```python
from pathlib import Path
import shutil

source = Path("build/bench_polars")
target = Path("build/bench_polars_quoted")
target.mkdir(exist_ok=True)
with (source / "left_1000000.csv").open("rb") as reader:
    with (target / "left_1000000.csv").open("wb") as writer:
        writer.write(reader.readline())
        for line in reader:
            fields = line.split(b",")
            fields[3] = b'"' + fields[3] + b'"'
            writer.write(b",".join(fields))
shutil.copyfile(source / "right_1000000.csv", target / "right_1000000.csv")
```

Run the benchmark driver with `--csv-only --sizes 1000000 --threads 32
--reps 7 --data-dir build/bench_polars_quoted --runner build/bench_csv_fourth152`.
The next remaining costs include typed decoding and final contiguous-buffer
assembly. These checkpoints do not close #152 or establish performance on
other machines or platforms.

## Public reader after #161: source-stage recheck (2026-09-22)

The preceding checkpoints describe the retired reader. Public `read_csv` now
uses the source-mapped pipeline documented in [csv-polars-port.md](csv-polars-port.md),
with Polars 1.44.2 pinned for the comparison. The old 65.5 ms / 18.1 ms
headline and the serial concat-copy cost in #152 are no longer current.

| #152 Polars stage | Current public Mojo path | Remaining difference |
|---|---|---|
| mmap | `_map_file`, then one bounded byte span | Unmappable inputs use owned bytes through the same decoder. |
| chunk sizing and `CountLines` | `chunk_size` and `CountLines.find_next` produce 513 ranges of about 97 KB on the 1M-row reference fixture | Same size class as Polars' 512 ranges; count includes the final tail. |
| publish while scanning | `Pool.run_produced` with a shared produced-job queue | Pool is scoped to one read; Rayon workers persist and work steal. |
| UTF-8 and builders | `decode_chunk` validates once when String is projected and reserves `rows + 1` | Uses Mojo's validator and builders. |
| `SplitFields` and projection | Cached structural masks, borrowed fields, and quote-aware tail skipping | See the supported options in `csv-polars-port.md`. |
| numeric conversion | `csv_numeric` ports pinned fast-float2 for Float32/Float64; `csv_integer` ports pinned atoi_simd | ARM integers use a packed fallback. |
| vertical assembly | `concat` retains chunked output without copying values | The generic `Series._from_chunks` previously allocated a one-element chunk list for each contiguous input. This branch appends its array reference directly. |

The 49.7 MB, 1M-row, eight-column reference CSV was measured with
`pixi run -e oracle bench-polars --csv-only --sizes 1000000 --threads N --reps 7`.
On this machine, unmodified `main` measured 361.68 ms against Polars 212.84 ms
at one thread, and 19.08 ms against Polars 15.19 ms at 32 threads. This is
18.9x Mojo scaling and 1.26x Polars' time at 32 threads. These runs are a
current checkpoint, not a claim that every workload reaches the same ratio.

A separate temporary worktree instrumented `_read_mapped_body` and each decode
job with `monotonic()`. It did not modify the production branch. Four warmed
32-thread reads produced 513 jobs each. In the fastest of those instrumented
reads, setup was 1.19 ms, producer scan/publication 7.18 ms, remaining
join/drain 8.95 ms, frame gathering 0.41 ms, and chunk assembly 2.94 ms.
Producer and worker decoding overlap, so the producer elapsed time is not
an isolated CPU cost. Across the four reads, range medians fell in 0.8–1.0 ms
100-µs buckets and maxima were 1.47–1.97 ms, within about 2.2x of the
corresponding median bucket. Instrumentation adds overhead; use the unmodified
binaries for the performance comparison.

### Owned-array assembly change

Polars' `accumulate_dataframes_vertical` appends Arrow chunks in
`vstack_mut_owned`. Mojo's `concat` already preserves chunked output, but
`Series._from_chunks` called `part.chunks()` for every input part. For the
usual one-array decode result, that constructed a temporary one-element List,
a temporary Series, and another storage reference before appending the array.
The new single-chunk branch appends its storage reference directly. Existing
multi-chunk inputs retain the old flattening path and checks.

Two optimized binaries were built from `5e05b84` and this branch with the same
pinned Mojo environment. Four pairs of seven-repetition reads alternated
baseline and branch at 32 threads; the reported value from each process is its
best warmed read. Median baseline was 18.93 ms, median branch 18.17 ms, a
0.76 ms (4.0%) improvement. Every pair improved. At one thread, three paired
runs were within noise (baseline median 362.12 ms, branch 362.61 ms).
Correctness checks included `test_chunked_series`, `test_concat`,
`test_csv_consumers`, and `test_csv_parallel`.

#152's original semantic criterion refers to an unchanged scalar reference
reader. #161 removed that reader and deliberately adopted Polars' semantics
where they differed. Current correctness is checked against the documented
public contract, public CSV fixtures, and the Polars oracle, as described in
`csv-polars-port.md`. Further performance work should compare the current
pipeline rather than the retired reader's stage costs.
