# CSV source-level port

Reference: Polars Python **1.44.2**, tag `py-1.44.2`, commit
`1bd8ec12f42d40fcec62badf32ef2177d2377d8d`. Repository baseline: `main`
`b742ff2777a3faf3db86663efba72ed8249f3c33` (merged PR #155).
This effort is CSV only. Drafts #156–#158 were closed as superseded.

The new implementation remains internal until correctness and timing checks
pass. Existing `read_csv` is the reference during development. Algorithm names
alone do not establish equivalence; the mapping below records actual functions.

| Polars source at pinned commit | Mojo port | Status / remaining difference |
|---|---|---|
| `polars-io/src/mmap.rs` | Existing `_Mapping` / `_map_file` | Shared read-only file mapping |
| `csv/read/read_impl.rs:parse_csv` chunk sizing | `csv_scan.chunk_size` | 16 parts/thread; 500,000-column-buffer budget; 4 KiB minimum, 16 MiB maximum |
| `polars-utils/src/clmul.rs` / SIMD mask packing | `csv_bits` | `pack_bits`; target-gated x86 PCLMUL and AArch64 PMULL; portable fallback |
| `csv/read/parser.rs:CountLines::count/find_next` | `csv_scan.CountLines` | 64-byte quote-parity masks, newline count, last boundary, window doubling |
| `csv/read/splitfields.rs:SplitFields::next` (SIMD build) | `csv_splitfields.CsvSplitFields` | Borrowed offsets; cached quoted-field structural ends; scalar tail |
| `csv/read/parser.rs:parse_lines` | `csv_decode.decode_chunk` | Borrowed fields, projected-only buffers, null-on-error; focused semantic and explicit-reader orchestration tests pass |
| `csv/read/builder.rs:validate_utf8` | `StringSlice(from_utf8=chunk)` | Whole-chunk validator; generated assembly verified SIMD on x86-64 |
| `fast_float2` / `atoi_simd` dispatch | `csv_numeric.mojo` / existing integer parser | Float32/Float64 source dispatch, packed fractional digits and fixed-array batched decimal fallback; integer SIMD backend not yet ported |
| Rayon scoped task publication | `csv_reader` + `Pool.run_produced` | Same scan/publish overlap; scoped pthread pool and shared queue remain explicit runtime differences from persistent Rayon/work stealing |
| `accumulate_dataframes_vertical` / `vstack_mut_owned` | `Series._from_chunks`, `frame.concat` | Append immutable Arrow array references; chunk storage and consumer tests pass |
| Chunk-aware downstream kernels | `Series.chunks/slice`, kernel adapters | Row access, slicing, display and validity retain chunks; some consumers explicitly rechunk; reductions iterate chunks directly |

Source root:
<https://github.com/pola-rs/polars/tree/py-1.44.2/crates/polars-io/src/csv/read>.
Polars-derived code is covered by [its retained license](../third_party/POLARS_LICENSE).

## Evidence required before switching the public reader

- New vs reference reader on supported CSV contracts, plus Polars differentials
  where behavior intentionally follows Polars instead of the old strict reader.
- Quoted/escaped/multiline, Unicode, nulls, projection, inference, comments,
  malformed input, row limits, and every numeric dtype.
- Chunked result consumers compared with contiguous equivalents, including
  validity, slicing, gather, expressions, reductions and single-array export.
- Same-base paired timings and Polars 1.44.2 on the same files, options, machine
  and thread limits; no concurrent builds or benchmarks.
- Report remaining copies, allocations and runtime deviations. A faster read
  that simply transfers its copy cost to the next operation is insufficient.
- Atomic PRs, each with its source mapping and validation; no performance claim
  or issue closure until supported by measurements.

## Remaining source differences

These are unfinished work, not alternative optimization choices:

- Integer parsing is not yet a verified translation of the enabled atoi_simd
  architecture-specific backend.
- Primitive/bool builders still write an eager validity bitmap. Polars creates
  validity lazily at the first null; this requires matching the column storage
  contract and its consumers, not just changing CSV append calls.
- String builders still produce offset/byte arrays instead of Polars BinaryView.
- The pool is scoped per read and uses a shared queue, not persistent Rayon
  workers with work stealing.
- Temporal conversion still uses the existing Mojo parser and constructs a
  String; the Polars temporal parser has not been translated.
- New schema inference is not implemented. The public reader remains unchanged.
- Single-array Arrow export explicitly rechunks; Arrow stream export is pending.

No end-to-end performance parity is claimed for this intermediate state.

The internal reader follows Polars nullable-column behavior even if a legacy
`CsvField` carries `nullable=False`; it does not add the legacy non-null guard.
The public reader has not been switched to these new semantics.

## Validation commands

From this branch's checkout with its Pixi environment, run the dedicated tests:

```sh
for name in csv_bits csv_scan csv_splitfields csv_decimal csv_numeric csv_buffers csv_decode csv_reader chunked_series chunked_consumers; do
    pixi run mojo run -I . tests/test_${name}.mojo
done
pixi run mojo run --target-features -pclmul -I . tests/test_csv_bits.mojo
```

The numeric fixture verifier and untimed dataset generator are under
`experiments/csv_port/`. Its README separates build/setup from timing and notes
that the branch's legacy reader is not an untouched-main performance baseline.

## Current validation snapshot

Focused checks passed on the main-based comparison branch: scanner 4/4,
splitter 8/8, structural bits 2/2 (hardware and `-pclmul` fallback), numeric
4/4, typed buffers 3/3, decoder 9/9, explicit reader 6/6 with four threads,
chunked consumers 2/2, and Arrow 6/6. The numeric oracle verifier independently
confirmed 419 stored Float32/Float64 bit patterns with installed Polars 1.44.2.
Earlier chunk storage and concat checks also passed. No throughput timings
have been collected for this new pipeline.
