# Benchmark correctness validation

This audits the correctness checklist in issue #1041. Benchmark and
checker changes belong in benchmark-only PRs. Failures exposed in `dataframe/`
or another engine package require separate fixes; correctness failures must
remain visible in the intervening reports. No performance thresholds or
engine routes are selected by this work.

## Audit of the summary protocol

`suite_common.mojo` and `engines.py` reduce strings to byte lengths, numeric
values to Float64 sums, timestamps to whole seconds, and other values to
non-null counts. `bench_suites.same_answer` compares these with relative
tolerance 1e-6. Adjacent-column product sums add another lossy statistic.

Consequently, equal-length strings collide; integers above 2**53 lose bits;
nulls can match zeros; subsecond timestamps disappear; row permutation is
invisible; duplicate multiplicities and arbitrary row correlations are not
established by the sums. Column names and dtypes are not checked. Cached
answers without product sums bypass that additional comparison. The report
selects one successful answer per engine, leaving later rounds unchecked and
potentially hiding failed rounds behind a successful one. None of these
summaries proves equality, even when all summary fields agree.

Regression cases in `tests/tooling/test_answer_checks.py` preserve concrete
counterexamples. These tests initially document false acceptance by the
legacy checker; complete-result tests must require rejection of the same
wrong answers. Summary-only historical reports cannot become newly verified
without rerunning workers.

## Exception inventory and replacement contracts

The existing exceptions cover all of the following queries. Every selected
row must belong to an independently computed full relation, including its
aggregate values, and cannot be used more often than its available
multiplicity. All selections must have the required cardinality. For sorted
queries the complete rank sequence, LIMIT, and OFFSET must also be valid;
ties permit different members only at equal ranks.

| Existing exception | Replacement contract |
| --- | --- |
| ClickBench q8–q16, q18, q21–q22, q27–q28, q30–q37 | Descending aggregate rank; q27–q28 limit 25, others limit 10 |
| ClickBench q17 | Any ten rows of the complete grouped relation, or all rows if fewer than ten |
| ClickBench q23 | Ascending EventTime, limit 10, full source-row membership |
| ClickBench q24 | Ascending EventTime, limit 10; validate projected SearchPhrase against source rows, retaining hidden sort-key evidence |
| ClickBench q38–q41 | Descending PageViews with offsets 1000, 1000, 100, and 10000 respectively, limit 10 |
| TPC-DS base q67 | Independently recompute grouped values and rank semantics; audit floating subtotal ties, rather than omitting rk |

The generic full-result checker also needs to honor ordering in queries
without an exception, including ClickBench q7, q25–q26 and q42. Queries with
no output ORDER BY compare multisets, retaining duplicates. A tied ordering
does not impose an extra deterministic order among peers.

## Complete-result protocol and oracle independence

Export original schemas and every cell outside the measurement interval.
Preserve exact integer, decimal, date/time, boolean, string and null values.
Schema normalization, if needed for intentional SQL/API output aliases or
aggregate widening, must be explicit and reviewed per query; do not infer
equivalence from coincidentally matching data. Retain original schemas.

Use independent DuckDB SQL over the original inputs as the CPU oracle. The
existing Metal selection audit demonstrates full-relation membership and
rank validation for nine ClickBench queries; its definitions do not yet cover
every exception or null/nonfinite value. The Metal numerical comparator is
reusable for duplicate-preserving approximate matching, but CPU/Metal
agreement does not independently establish CPU correctness.

Numeric tolerances apply only to declared aggregate/arithmetic floating
fields. Copied source values and grouping/order keys remain exact, as do
integers and decimals. Null matches only null; NaN only NaN; infinities must
have the same sign. Ambiguous matching requires one-to-one assignment;
exhausted comparison limits, missing exports, and unavailable independent
references must report unverified rather than pass. Every measured round
must be validated before its timing can enter a ratio.

Exports, reference relations, receipts, and generated audit evidence stay
outside the repository. Reference caches must include the checker/protocol
identity and input identity, and must reject summary-only entries for
complete-result validation.
