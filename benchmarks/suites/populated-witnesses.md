# Supplementary populated query witnesses

These ten small input fixtures exercise unchanged canonical queries whose
smoke-scale answers were empty. They are separate workloads, not a replacement
for official generated data or a change to the 345-case canonical denominator.
The tracked backlog contains all 49 prior nonhybrid empty matches. This increment
selects eight: TPC-DS q3/q19/q82 in base and decimal, and ClickBench q19/q41 in
base. The other 41 remain unresolved. Two additional TPC-DS q24 variants retain
their known GPU → eager CPU mean/item → GPU boundary and are labeled hybrid.

Generate using the oracle environment and an existing official data root:

```sh
python benchmarks/tools/populated_query_witnesses.py generate \
  --canonical-data-root /path/to/dataframe_mojo_benchdata \
  --output build/populated-witnesses
```

A published output manifest makes its fixture directory immutable; choose a
fresh output directory for a new version. The generator only reads official Parquet schemas. It writes complete schemas
with unchanged types and column order; unused fields are null. The canonical
query modules, data generator, inputs and comparator remain unchanged. Each
supplementary query has its own table files, hashes, schema receipt, literal
ordered answer, independent DuckDB SQL and expected-result Parquet. All ten
SQL answers must equal the literals and be nonempty before a manifest is written.
This checks fixture construction, not Mojo CPU or Apple execution.

| Query | Variants | Prior canonical rows | Supplementary expected rows per variant |
| --- | --- | --- | --- |
| TPC-DS q3 | base, decimal | 0 | 3 |
| TPC-DS q19 | base, decimal | 0 | 2 |
| TPC-DS q82 | base, decimal | 0 | 2 |
| ClickBench q19 | base | 0 | 2 |
| ClickBench q41 | base | 0 | 1 |
| TPC-DS q24 (hybrid) | base, decimal | 0 | 1 |

The q3/q19 inputs include joined qualifying sales, duplicates, nulls and excluded
rows. q82 covers both price/date/stock endpoints, duplicate inventory and sales,
nulls and excluded manufacturers/stock/dates. q24 includes a peach group, a blue
baseline group and an all-null paid group; its mean cutoff is nonnull and its
answer is one paid row. q19 ClickBench contains duplicate exact UserID matches,
an excluded ID and a null ID. q41 is bounded at exactly 20,001 rows: 10,000 groups
have count two, and one group has count one. OFFSET 10000 therefore returns
`(WindowClientWidth=20000, WindowClientHeight=2, PageViews=1)` without a tie at the
answer boundary.

## Target validation recipe

The measurement owner stages forced CPU/Metal query modules with the unchanged
`benchmarks/tools/bench_metal_suites.py` source derivation. Compile the existing
suite workers and `benchmarks/suites/populated_witness_oracle.mojo` with the
repository and staged source directory in their include paths. The latter
imports the staged query functions; it does not reimplement query expressions.
No compiler or worker is launched by the generator or audit command.

For each manifest case, run the existing `suite_tpcds` or `suite_clickbench`
worker with `worker_argv`, retaining a separate supplemental coverage receipt.
Run the independent typed oracle entry twice, using `oracle_worker_argv.cpu`
and `.metal`. It checks complete ordered schemas and cells against the expected
Parquet and prints `witness_oracle<TAB>ID<TAB>ENGINE<TAB>ROWS<TAB>True` only on
success. When copying the fixture tree, rebuild argv paths from each table's
`relative_path` and `oracle_result_relative_path`; hashes remain unchanged.
Do not append these cells to a canonical grid with the same suite/variant/query
keys. The separate `case-metadata.json` uses the measurement adapter's
`dataframe-metal-case-metadata-v1` format and labels q24 hybrid in both variants.

Audit retained supplemental receipts and oracle logs:

```sh
python benchmarks/tools/populated_query_witnesses.py audit \
  --manifest build/populated-witnesses/manifest.json \
  --receipt build/populated-results.json \
  --oracle-log build/populated-oracles.log \
  --output build/populated-witnesses/audit.json
```

The audit requires source and input hashes, rejects duplicate/unexpected cells,
checks actual nonempty row counts, and records CPU and Metal literal-oracle
checks separately. Missing runs stay untested. Existing `collection_ns` is
retained only as a diagnostic GPU duration; no CPU timing is guessed and no
performance evidence is claimed. The measurement owner's performance collector
must record complete cold/warm collection samples for each engine, round and
repetition on the same fixture files and thread settings. Hybrid q24 timing must
include its eager CPU mean/item stage. Canonical and supplementary population
counts are not a speedup comparison across different datasets.
