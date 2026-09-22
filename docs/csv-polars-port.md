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
| `csv/read/schema_inference.rs` | `csv_infer.infer_csv_schema` | Lossy UTF-8 headers, `_duplicated_N` names, all-column null tokens, Bool/Float/Int candidate sets, and Polars' 100-record default |
| `csv/read/streaming.rs:read_until_start_and_infer_schema` | `csv_infer.infer_csv_schema` | BOM, SkipEmpty, quote-aware `skip_rows`, header removal, comments, and retained post-prelude mmap offset |
| `csv/read/read_impl.rs:parse_csv` | `csv_reader._read_mapped_body` | Source-ordered decode jobs, schema-wide UTF-8 check, CountLines estimate validation, EOF-comment rule, probabilistic `n_rows` stop, final head |
| Arrow mutable primitive/boolean validity | `CsvBuffer`, column validity helpers | Absent bitmap until first null; prior valid prefix initialized once; consumers and Arrow preserve absence |
| `csv/read/builder.rs:validate_utf8` | `StringSlice(from_utf8=chunk)` | Whole-chunk validator; generated assembly verified SIMD on x86-64 |
| `fast_float2` / `atoi_simd` dispatch | `csv_numeric.mojo` / `csv_integer.mojo` | Float32/Float64 source dispatch, packed fractional digits and fixed-array batched decimal fallback; source-dispatched SSE/AVX2 integer reductions and SWAR fallback |
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

The source mapping above covers the implemented hot path. It does **not** mean
that the internal reader exposes every Polars CSV option. The items below are
split between hot-path/representation differences and public-option scope so
that unsupported configuration is not presented as a tokenizer regression.

### Hot path and representation

- Integer SSE/AVX2 dispatch and reductions are ported and instruction-checked
  on x86; the dedicated Neon integer backend remains unported (SWAR fallback).
- CSV strings now retain native 16-byte BinaryView descriptors and Arc-owned
  blocks through slicing, gathering, and batch rechunking. Single-array Arrow
  export currently materializes the legacy large_utf8 ABI; native `vu` export
  and its final variadic buffer-size array remain unimplemented.
- The pool is scoped per read and uses a shared queue, not persistent Rayon
  workers with work stealing.
- Temporal conversion still uses the existing Mojo parser and constructs a
  String; the Polars temporal parser has not been translated.
- Inference supports Bool/Int64/Float64/String, with explicit dtype overrides;
  temporal inference remains unported. The internal default sample is 100,
  matching Polars; the legacy public reader remains unchanged.
- Single-array Arrow export explicitly rechunks; Arrow stream export is pending.

### Unsupported reader options and type surface

These are source features not yet exposed by the clean reader API, rather than
alternate semantics for its existing arguments:

- `skip_lines` (naive, quote-agnostic) and `skip_rows_after_header` are absent.
- Custom `eol_char` is absent; the clean scanner and splitter use LF.
- Named per-column null tokens and `missing_is_null=False` are absent. The
  current `null_values` list applies to every column and bare empty fields are
  null.
- Date/time inference, `decimal_comma`, and optional Int128 inference are not
  implemented. Unsupported inferred candidates become String.
- Column-name replacement, positional projection, row indices, compression,
  and Polars' `raise_if_empty` controls are outside this internal API.

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
4/4, typed buffers 3/3, decoder 9/9, explicit/inferred reader 7/7 with four threads,
chunked consumers 2/2, and Arrow 6/6. The numeric oracle verifier independently
confirmed 419 stored Float32/Float64 bit patterns with installed Polars 1.44.2.
Inference passed 7/7 focused tests and 11 differential fixtures against Polars
1.44.2. Inferred reads retain the same mapping through sampling and decode.
Earlier chunk storage and concat checks also passed. No throughput timings
have been collected for this new pipeline.

Integer validation: default x86, explicit SSE/AVX2, and SIMD-disabled fallback
all pass 3/3 tests, with 31 pinned Polars oracle cases. Emitted x86 assembly
contains the expected `vpmaddubsw`, `vpmaddwd`, and `vpackusdw` reductions.

StringView integration: storage 3/3, StringColumn 10/10, Arrow 7/7, typed CSV
buffers 4/4, and reader 7/7 passed. Actual end-to-end outputs match Polars
1.44.2 in eight runs: explicit/inferred schemas, full/projected reads, and
one/four threads on 6,000 mixed rows with nulls, escapes, Unicode, and multiline
strings. The oracle driver and verifier live in `experiments/csv_port`.
