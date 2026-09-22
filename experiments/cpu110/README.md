# CPU performance evaluation

Base: `7a01b37` (#154), now merged into `main`. Retained implementation: `be91deb`.
See [the evaluation report](../../docs/cpu-performance-evaluation.md) for final measurements and decisions.
The user requested the complete remaining CPU workplan, not just CSV.
A proposal is evaluated only after a reproducible experiment or verified
prior experiment; source inspection alone is not a performance result.

| Issue / idea | Evidence so far | Remaining evaluation |
|---|---|---|
| #149 float parsing | Wide mantissas in #154; exponent fallback specialization ~6.5x micro / ~40% CSV improvement; Float32 subset rejected | 53 modules +150 oracle cases pass; long/ambiguous inputs use borrowed standard fallback |
| #149 packed integers | Short/packed generic path wins on 16/32/64-bit fields; 8-bit and signed Int64 retain existing path | Outlined helper retains 1.2–2.3x long-field wins; final oracle passes |
| #151 chunked columns | Prototype avoids final copy: 1M read ~22→15 ms; 10M ~166→127 ms; lower comparable RSS | Full column/consumer/Arrow stream integration remains a separate feature |
| #152 projection | Selected-field append experiment regresses projected reads ~6–8%; rejected | Early-stop changes strict unprojected validation; preserve contract |
| #147/#150 scheduling | 4/8/16/32 chunks per worker measured; 16 helps 10M but hurts 1M | 1.15GB size-adaptive result mixed; keep current8 |
| #148 splitter | Scan/decode/concat and per-chunk tails instrumented on plain and quoted input | Instrumentation perturbs timings; complex quote paths still cost more |
| #103 reusable pool | Explicit scoped reuse: 1M two-stage dispatch ~2.93→0.50 ms | No unsafe process-global lifetime change; execution-context integration needed |
| #105 parallel join | Parallel ordered match expansion and adaptive CSR: earlier 10M ~2.56→1.41 s; a raw-address lifetime bug found by the forced CSR test is now fixed | Corrected final head: 1M56.2ms /10M576.8ms at32; bounded-key and parallel tests pass. Sparse/string encoding still costs more |
| #106 dictionary encoding | Repeated code grouping saves ~17–25 ms/1M; encoding ~28–42 ms; 4–7x smaller logical payload | Reconciled Int32 joins save13–22ms, encode/remap29–60ms; categorical/Arrow API remains unimplemented |
| #108 parallel sort | Co-ranked merge already exists; serial radix wins at1 thread; parallel radix loses; parallel partial selection wins in isolation | End-to-end rank encoding/gather still dominate; prototypes not shipped |
| #109 streaming | 20M/1.15GB bounded parallel pipeline validated; ~33MiB vs4.6GiB eager RSS; slower than Polars | Full lazy plan integration and diagnostics/options remain unimplemented |
| #115 bare-column reductions | Direct numeric buffers + bitmap COUNT; Float64 SIMD SUM/MEAN ~5–8x faster in tuned microbench; differential and targeted tests pass | 50 dtype/operation combinations improve at1/32; 1Msum0.34ms vsPolars0.30ms |
| #39 native Parquet in #110 | PyArrow decode + Mojo Arrow-copy bridge measured across 4 codecs and dictionary on/off, against Polars | This evaluates format/import costs, **not a native Parquet implementation** |
| #110 final yardstick | Complete 1M/10M, 1/32-thread baseline in baseline.md | Final corrected measurements and recommendations in the report |

Retain exact integer accumulation, strict CSV grammar, stable sorting and
existing row-order contracts. Unshipped prototypes must be labelled, and a
lower bound that skips work must not be presented as an implementation win.

## #109 bounded CSV streaming prototype

`streaming_109_prototype.patch` is an isolated experiment against `7a01b37`,
not a proposed public lazy API. It proves one useful shape: CSV scan,
`x > 0`, `x * 2`, and a global Float64 sum. The serial form carries one
stateful tokenizer across drained record batches. The parallel form holds one
record-aligned input window, decodes its ranges concurrently, and retains only
one scalar partial sum per range. A record larger than the configured window
is deliberately carried until its terminator, so the actual bound is the
window plus the longest record.

The isolated test covers quoted multiline records and malformed input for the
serial and parallel variants. The parallel scanner passes each range's global
record base to its reader, but concurrent jobs can surface an error in worker
completion order rather than the earliest malformed record. It also fixes the
dialect to the standard comma/quote/header/default-null configuration and
does not expose `read_csv` options such as comments, projection, encoding,
row limits, or error suppression. These are prototype limitations, not
contracts to carry into a full implementation.

A production #109 implementation still needs a plan IR that can represent a
scan source and row-preserving streaming transforms, an aggregate-state merge
protocol, ordered sinks for non-reduction consumers, bounded pipeline
backpressure, and optimizer/explain support. It must retain eager semantics
for nullable predicates, all CSV options and diagnostic order. Blocking
operators (sort, joins, global order-sensitive operations, arbitrary gather)
need explicit materialization or a compatible external/stateful algorithm.

The reproducible one-GiB fixture is generated by
`generate_streaming_1gb.py`. It repeats the established one-million-row CSV
body twenty times while appending `extra_a:Int64` and `extra_b:Float64`; the
header is written once. `bench_streaming_1gb.py` runs the same
CSV -> filter -> select -> sum pipeline in Polars eager or streaming mode for
either the original eight-column or generated ten-column schemas. Use one
process per mode when reporting RSS.

Window-five one-process measurements on the generated 20-million-row,
1,153,606,359-byte fixture (32 threads) were:

| Engine / pipeline | Peak RSS KiB | Reported internal time | Wall time |
|---|---:|---:|---:|
| dataframe_mojo eager | 4,786,908 | 1.477 s (rerun) | 1.90 s |
| dataframe_mojo bounded parallel, 4 MiB window | 34,112 | 1.190 s (rerun) | 1.25 s |
| Polars eager | 2,926,700 | 0.379 s | 0.90 s |
| Polars streaming | 1,549,524 | 0.566 s | 0.82 s |

The internal timings cover the pipeline; outer wall time also includes process
startup/teardown (including Python for Polars). Mojo internal durations were
collected in window six after making the harness print its one-pass timing;
RSS and wall times in this table are the window-five runs. RSS is process peak,
not an allocation counter. Both engines produce the same sum; Polars streaming
can prune unused columns, while this Mojo prototype decodes the full schema.

## Dictionary-code join experiment

`dictionary_join_experiment.mojo` evaluates the minimum reconciliation a
categorical join needs when each input assigned codes independently. It uses a
large repeated fact table and a one-row-per-key dimension table, so cardinality
100 does not create a quadratic many-to-many output. The right dictionary is
reverse-discovered and contains one quarter right-only keys; both sides have a
null. Before timing, the harness compares the joined left/right payload pairs
and their order with the string-key baseline. It reports encoding and
right-code remapping separately from repeated joins. Suggested inputs are
`1000000 100` and `1000000 100000`.

Current Arrow support is an explicit blocker for a categorical dtype:
`dataframe/arrow.mojo` rejects any non-null `array.dictionary` or
`schema.dictionary` pointer, and the PyArrow oracle deliberately expects a
dictionary array to fail. Import/export therefore neither preserves nor
materializes Arrow dictionaries today. A production categorical design must
define Arrow dictionary import/export, code width, null handling, dictionary
lifetime, and reconciliation/cache ownership before a code-based join can be
an observable optimization.

## Reproduction and artifact scope

`baseline.md` records the exact baseline and head-to-head commands. Production
changes are in `dataframe/`; files here are independent experiments, not a new
public API or a replacement test suite. Benchmark only after builds and tests
have finished; pin `DATAFRAME_THREADS` and `POLARS_MAX_THREADS` equally.

Use a disposable baseline source tree when applying an experiment patch:

```sh
cpu110_base=$(mktemp -d)
git archive 7a01b37 | tar -x -C "$cpu110_base"
patch -d "$cpu110_base" -p1 < experiments/cpu110/projection.patch
pixi run mojo build -I "$cpu110_base" experiments/cpu110/projection_bench.mojo -o /tmp/cpu110_projection_bench
DATAFRAME_THREADS=32 /tmp/cpu110_projection_bench build/bench_polars/left_1000000.csv x
```

Patches headed `a/dataframe/...` use `-p1`. Join patches headed
`dataframe/frame.mojo` use `-p0`. The CSR patch includes the inner-expansion
patch; do not apply both. Apply `join_parallel_inner_overflow_guard.patch` and
`join_csr_order_lifetime.patch` after the CSR patch. The latter preserves the
local scatter-order buffer until fill jobs finish; measurements from before
that fix are historical; the final-head report uses the corrected code. The dense-range and
fast-bucket patches are separate extensions of the CSR variant. Exponent inline/outline, packed integer
generic/short, and reduction SIMD/extended patches are alternatives, not a
stack. The retained production variants may differ from these intermediate
experiments; the PR's source and focused tests are authoritative.

Standalone sort, dictionary, pool, reduction-type and numeric microbenchmarks
can compile with `-I` set to the desired source tree. Their module docstrings
and raw command logs specify inputs. `results/` preserves historical quiet
windows, including rejected candidates and failed intermediate experiments;
use the report's decisions rather than interpreting every number as a shipped
result. RSS comparisons require separate processes and equal retained inputs.
