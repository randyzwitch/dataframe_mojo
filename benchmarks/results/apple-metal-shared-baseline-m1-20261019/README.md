# Shared Apple Metal development baseline and correctness audit

This bundle records the completed 302-case development coverage run on a Mac mini with Apple M1. It is correctness and provenance evidence only; it contains no benchmark timings and makes no speed claim.

The immutable receipt is `coverage.json` (SHA-256 `087e08b97d18ae2ef0490e3429ab300d1f37c67cf759bca51c47e493daabe874`). `integrity.json` records successful integrity verification: all 77 prepared input files and 2,108 written answer files were rehashed against the receipt, with no integrity failures. The strict exact correctness gate returned 1 because 68 cases remain `wrong_answer`; the receipt records 212 `ok`, 68 `wrong_answer`, 20 `reference_unsupported`, and 2 `unsupported`. Integrity passing does not mean correctness passing. The checker and its exact comparison policy were not relaxed, and the diagnostics below do not change any classification.

## Scope and provenance

The coverage run used development scale, 8 threads, 3 repetitions, Mojo implementation, and the compiler version recorded in the receipt: Mojo `1.2.0.dev2026100905` (`3b71af75`). The receipt records macOS `26.5.2` arm64 and revision `50137ecc2e27f76298559a25e0aa4ecca31d4e96`. At that harness revision, `pixi.toml` and `pixi.lock` pin the Max SDK package (`max-core`) to `26.7.0.dev2026100905` for `osx-arm64`.

The benchmark query and engine source files are identified by the receipt's per-file SHA-256 values. The eight original suite/runtime source files tracked at `a52334e758b7138c547883eb776d744442a1a0db` match that source tree byte for byte; `benchmarks/metal_coverage_audit.mojo` is the captured audit harness source at `50137ecc2e27f76298559a25e0aa4ecca31d4e96`. The `a52334e7` source commit is an ancestor of the PR #883 audit branch's `9ab3d20b6819a11d431903daa2780251c5df0b63` commit, and the run harness was based on the same `a52334e7` tree. The receipt preserves all original and derived source hashes. Every file in `raw/sources/` was rehashed against `derived_source_sha256`; no executable build binaries are included.

The host was a Mac mini `Macmini9,1`, Apple M1, 8 CPU cores and 16 GiB RAM, as recorded in the receipt. The original benchmark outputs are retained on the authorized Mac at:

`/Users/randyzwitch/github/dataframe_mojo-metal-868-coverage-audit/build/metal-868-dev-coverage-a52334e7/answers`

The Parquet answer exports and schema sidecars total about 1.2 GB and are intentionally not copied into Git. Their full paths, SHA-256 hashes, schema hashes, and sizes remain in `coverage.json`; the integrity report records rehashing all 2,108 written files. The copied `raw/logs/` contains all 302 worker logs, and `raw/sources/` contains all 17 captured Mojo source files. `raw/SHA256SUMS` records their local digests.

## Wrong-answer diagnostics

The four detailed comparison reports and their scripts cover all 68 wrong-answer cases: 25 TPC-DS base, 9 TPC-DS decimal, 23 H2O group-by variants, and 11 PDS-H. `metal_868_wrong_answer_manifest_a523.json` indexes all 68 classifications and points to each detailed report. The comparison tools read only completed logs and answer exports using the locked oracle Python environment; they did not run the compiler, benchmark workers, or GPU.

The diagnostics keep the strict outcome intact: exact floating-point differences remain failures. All 34 TPC-DS comparisons have matching logical and encoded schemas, row counts, and non-float column multisets. Where a unique non-float signature exists, rows are compared by that signature. Duplicate signatures are kept as multisets, with float columns matched independently within each duplicate group; this does not assert cross-column row association. Scalar outputs without non-float columns are compared as per-column value multisets. CPU-repeat labels come only from logged `answer_export` records.

All 23 H2O cases have matching logical and encoded schemas, row counts, and equal sets of unique query-declared group keys, so values are aligned by those keys. Group-key order differences are reported separately. The PDS-H report records its key-based or scalar matching method case by case. Missing Metal repetitions after the first mismatch are expected: the strict runner stops a case after a failing comparison. For the coverage runtime wrapper, `no_marker` means only that the returned report contained no CPU data-fault adjudication marker; it is not a GPU-only certificate.

| Report | Cases | Contents |
|---|---:|---|
| `tpcds_a523_comparison.json` | 25 | TPC-DS base wrong answers |
| `tpcds_decimal_a523_comparison.json` | 9 | TPC-DS decimal wrong answers |
| `h2o_all_a523_comparison.json` | 23 | All H2O group-by wrong-answer variants |
| `pdsh_a523_comparison.json` | 11 | PDS-H wrong answers |

Each report includes source Mac paths and hashes for available logs, schemas, and exports. The scripts are included so the analysis can be inspected; rerunning them requires access to the original Mac answer paths and the pinned oracle environment.

## Integrity verification

`audit_coverage_integrity.py` is the standalone strict receipt validator used for the run; its source is retained unchanged. The integrity summary is retained in `integrity.json`, including its receipt SHA, 302-case grid, status counts, and rehashed artifact counts. The summary's `correctness_passed` is `false`, and `driver_returncode` is `1`, as expected for the 68 exact mismatches. The validator requires the original answer and input trees at their recorded Mac paths; the large answer files were not vendored here.

The existing coverage, typed-answer-export, and materialization-adapter Python harness tests passed under the locked oracle environment: 23 tests passed. No benchmark run was repeated during this evidence packaging.

To verify the copied receipt and evidence hashes from this directory:

```sh
sha256sum -c raw/SHA256SUMS
sha256sum coverage.json integrity.json audit_coverage_integrity.py
```
