# Intermediate relation costs

Implementation and validation for #1002 are in progress. This work also
supports #1041's requirement to explain strategy choices through general
data properties rather than query identities.

## Cost facts and semantic facts

`planning/cost/intermediate.mojo` owns request-scoped estimates of rows,
column distinct counts and payload bytes. Every row estimate records its
reason and whether it is exact. These facts select execution strategies;
they do not establish buffer capacity, uniqueness, join eligibility or the
absence of output rows. Existing `plan_known_height` and
`plan_height_bound` remain separate semantic interfaces.

A request memo visits a shared logical node once. Source column distinct
counts observe at most 4,096 rows, using one deterministically jittered row
per stratum for larger inputs. Exact sample-code frequencies feed a
bias-corrected singleton/doubleton correction, following the
[Chao1 formula](https://scikit.bio/docs/latest/generated/skbio.diversity.alpha.chao1.html).
This is adapted as a cost heuristic: stratified finite-population sampling
does not establish a semantic lower or upper bound. A sample with no
repeated valid values extrapolates its valid fraction. This remains an estimate on large inputs,
including skewed or correlated data. A grouping estimate caps the product
of available key domains by its input estimate. Multi-key joins use the
largest available key domain, avoiding multiplication of correlated
predicates as if they were independent.

Row-preserving projections, aliases and sorts retain input cardinality;
slices and concatenations propagate it. The existing bounded filter sample
is retained directly over frame scans. A filter over an intermediate with
unknown selectivity retains an explicitly inexact input estimate. Unknown
computed-column distributions are not discovered by executing expressions.

Payload widths use source buffer metadata, including average string bytes,
and preserve plain-column aliases. Width lookup does not sample distinct
values. Unsupported computed or nested payloads use a declared cost-model
fallback, rather than inventing a physical layout guarantee.

## Join build decisions

Once a join's right input has executed, its actual height and payload width
are available. The left input is compared using propagated row estimates
and payload width. Only an exact known height may skip the statistics
request: a large upper bound cannot establish that a filtered result is
large. Existing runtime joins still operate on actual frames and preserve
their output-order contracts.

The existing `DATAFRAME_TRACE_PATHS` setting emits a `dataframe-strategy`
join-build record with selected side, row estimate, exactness, estimate
reason, left payload width and observed right payload cost.

## Materialized join inputs and grouping finish costs

An estimator can receive already-executed join inputs before any estimate
is requested. Those inputs supply exact observed heights and bounded
column statistics without executing a subplan again. Observation after a
request has started is rejected, preventing stale ancestor facts.

The late-key grouping model now uses these observed builds and propagated
input cardinality. Reached build rows include duplicate-key multiplicity.
It compares avoided key payload traffic against the row ids that replace
it, bounded preparation reads, an optional uniqueness proof, final key
gathering or regrouping, state copies and output assembly. These are work
estimates in byte-weighted units, not calibrated nanoseconds or allocation
bounds. A full uniqueness proof is only attempted when its optimistic route
could repay the proof. The trace includes input rows, finish groups, saved
work, paid work, the selected key strategy and its reason.

## Remaining validation

- Validate nested joins, filters, aliases, grouping, correlated keys,
  variable string widths, nulls and complete output ordering.
- Retain before/after measurements on independently specified distributions
  at multiple worker counts, including small-input overhead and peak memory.
- Run complete development-suite SQL validation and affected CPU contracts
  on Linux and the live M1 host. Exercise shared Metal paths where affected.
- Keep workload definitions, profiles and generated evidence outside the
  repository, and keep benchmark changes separate from engine changes.
