# Forced Metal development-suite baseline on Apple M1

This package records the completed development-scale baseline for #869 and #868.
It measures complete query calls returning host DataFrames, including staging and
result construction. Input loading, answer summaries and result destruction are
outside the timed call. It is not a full-scale headline benchmark or a GPU-only
execution certificate.

All 1,812 scheduled workers ran: 302 query/variant cases, two engines and three
fresh-process rounds. Each worker requested one first-use collection, one extra
warmup and three warm samples. There are 1,686 successful workers and 126
unsupported workers (120 untranslated queries; six Metal concat-capacity
rejections). No compiler overlap or busy host was detected by the collector's
before/after observations; those observations cannot exclude all competing work.

The independent report analysis admits 210 cases covering 110 query IDs for
CPU/Metal ratios. It excludes 70 cases with unverified or hybrid exact evidence
and 22 with unverified runtime adjudication. Across admitted cases, the geometric
mean of CPU warm median / Metal warm median is 0.0306575: forced Metal takes about
32.6 times as long on this subset. The repository policy statistic using median
per-round minima is 0.0277221. These aggregates describe this development suite
and scale, not a placement policy or a claim about all supported queries.

## Identity and scope

- Machine: Apple M1 Mac mini, 16 GiB; eight configured workers.
- Compiler: Mojo 1.2.0.dev2026100905 (3b71af75).
- Recorded checkout and harness: `50137ecc2e27f76298559a25e0aa4ecca31d4e96`.
- All 249 recorded engine source hashes match commit
  `a52334e758b7138c547883eb776d744442a1a0db`; the checkout adds measurement tooling.
- Development inputs: H2O 1,000,000 rows; PDS-H and TPC-DS scale 0.1.
- 302 planned cases / 136 query IDs; catalog denominator 345 cases / 179 query IDs.
  Unplanned cases are not counted as supported or passing.
- Input, compiler, binary, SDK, source and environment identities are retained in
  the raw receipts. Binary executables and input datasets are not included.
- Exact CPU/Metal multiset receipts and independent DuckDB summary checks are
  separate evidence. Timed samples use summary validation, not full multiset
  comparison per sample. Absence of a CPU-adjudication marker does not prove
  GPU-only execution.

## Files and verification

`performance.json.gz` and `coverage.json.gz` preserve the original JSON bytes.
`worker-evidence.tar.gz` contains all raw timed-worker logs, generated query
sources, input manifest and build receipts. `measurement-harness.tar.gz` preserves
all seven harness files named in the timing receipt. `cases.csv` keeps every
planned case and its admission reason; excluded cases have no ratio.
`analysis.json` and `audit.json` summarize the report and hash audit.

Run `python3 verify.py` in this directory. It verifies package hashes, original
receipt identities, all 1,812 worker-log hashes and the archived harness hashes.
The collection itself is not rerun.

To regenerate the interactive report, decompress the two JSON files into a
scratch directory, then run the repository's
`benchmarks/tools/apple_performance_html.py PERFORMANCE_JSON --strict-coverage
COVERAGE_JSON --output REPORT_HTML`. The renderer recomputes admission from the
exact receipt and independent summaries; it reports 210 admitted pairs. The
required case-metadata SHA256 is
`9bae080d8b3ba576448fed603a8c7058224b7b5d9b46ba146c030f1dd4329176`.

#583 remains deferred. This evidence establishes a comparison baseline; it does
not establish replacement coverage or justify retiring any legacy route.
