# Benchmark correctness validation

This implements the correctness checklist in issue #1041. Benchmark and
checker changes belong in benchmark-only PRs. Failures exposed in `dataframe/`
or another engine package require separate fixes; correctness failures must
remain visible in the intervening reports. No performance thresholds or
engine routes are selected by this work.

## Audit of the summary protocol

`suite_common.mojo` and `engines.py` reduce strings to byte lengths, numeric
values to Float64 sums, timestamps to whole seconds, and other values to
non-null counts. The former `bench_suites.same_answer` compared these with relative
tolerance 1e-6. Adjacent-column product sums add another lossy statistic.

Consequently, equal-length strings collide; integers above 2**53 lose bits;
nulls can match zeros; subsecond timestamps disappear; row permutation is
invisible; duplicate multiplicities and arbitrary row correlations are not
established by the sums. Column names and dtypes are not checked. Cached
answers without product sums bypass that additional comparison. The report
selects one successful answer per engine, leaving later rounds unchecked and
potentially hiding failed rounds behind a successful one. None of these
summaries proves equality, even when all summary fields agree. That behavior
now survives only as `legacy_summary_agrees`, a diagnostic regression witness
which CPU performance admission does not call. The existing Apple collector
retains its legacy comparator until the next stacked change.

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

## Implemented contracts

`answer_exports.mojo` and the independent Python workers export every timed
result after stopping the timer. `suite_answers.py` validates hash-pinned
schemas and cells. `oracle_sql.py` extends only the outer independent SQL
projection with hidden sort keys and removes outer ordering and slicing to
produce the full reference relation. Nested queries remain intact; unsupported
outer clause forms fail closed. SQL ordering aliases are expanded with the
same output-name precedence as the original ORDER BY. Every current
ClickBench, PDS-H and TPC-DS SQL definition has been bound against the loaded
input schemas; live reference workers also check their original selections
against these full relations.

The former CHECKS/VARIANT_CHECKS omissions do not participate in correctness
evaluation. Membership, duplicate counts, rank sequences and offsets are
checked for all ordered/limited queries, including q24's hidden EventTime
and TPC-DS base q67's rank column. Queries without output ordering use
multisets. Approximate field matching reuses the Metal comparator, while the
reference relation always comes from independent SQL, not another execution
of the native engine. Complete-result adoption by the Apple timing collector
is the next stacked change.

`suite_schema.py` records intentional translation aliases and narrow integer
transport equivalences. Decimal storage width and string offset width are
transport details; decimal precision/scale, integer source widths, temporal
units/timezones and column order remain schema facts. The suite numerical
policy extends the reviewed Metal field policy after reviewing the additional
aggregate/arithmetic outputs. It pins native sources and the independent SQL
for the added fields. Copied floating attributes (`s_acctbal`, `c_acctbal`,
`o_totalprice`, `i_current_price`, `i_wholesale_cost`, `ca_gmt_offset`) remain
exact. The historical Metal policy and its retained receipts are unchanged.

## Failures exposed for separate follow-up

The smoke inputs expose native empty/all-null aggregate outputs that the
legacy zero-valued summaries accepted: PDS-H q17 and TPC-DS q13, q32 and q92
return zero where independent SQL returns null. The TPC-DS q99 translation
projects lowercased call-center names where SQL projects their original case;
equal string byte lengths hid this discrepancy. Decimal variants also expose
PDS-H q11's floating result in place of an exact decimal result, and differing
decimal scales in TPC-DS q23/q93. These cases remain failures; this benchmark
change does not alter engine semantics or query definitions to conceal them.
H2O's null-containing group-by variant also exposes q9's NaN-versus-null
correlation results and q10's zero-versus-null sums in all-null groups.
Retain the concrete exports and reruns outside the repository when addressing
them in follow-up changes.
