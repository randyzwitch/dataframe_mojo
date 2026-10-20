# Apple Metal replacement coverage

This matrix tracks the incremental engine-replacement work in [#877](https://github.com/randyzwitch/dataframe_mojo/issues/877), part of the [Metal overhaul in #868](https://github.com/randyzwitch/dataframe_mojo/issues/868). This is a source-and-evidence snapshot recorded on 2026-10-20 UTC. It describes the **open PR #888 candidate** at [`6398713d`](https://github.com/randyzwitch/dataframe_mojo/commit/6398713df4a8f31268f5d6064d47d676b9d7bec1); it does not describe `main` as having adopted that candidate. CPU/Metal automatic placement remains deferred to [#583](https://github.com/randyzwitch/dataframe_mojo/issues/583).

“Legacy” below means the existing Metal route retained for compatibility while the new family-specific implementation is validated. PR #888 replaces only the direct row/scalar slices described below. All other operation families continue using the current Metal implementation and public APIs; they are not silently removed or newly supported by this change.

| Family | Current/legacy route | Replacement in the PR #888 candidate | Semantic limits and status | Validation evidence |
| --- | --- | --- | --- | --- |
| Runtime resources and ownership ([#871](https://github.com/randyzwitch/dataframe_mojo/issues/871)) | Direct collections create per-call execution resources; resident DAG/operator routes use their existing ownership. | Direct row and scalar collections lazily lease a reusable context, bounded scratch storage and fused-pipeline handles. An overlapping collection gets a private context; a failed/faulted lease is not reused. Other resident DAG/operator routes still create fresh contexts. | Focused lifecycle coverage includes reuse, budgets, overlap and poison recovery. A DAG-wide shared device service is planned/in progress; it is not implemented by this candidate. | [PR #888](https://github.com/randyzwitch/dataframe_mojo/pull/888), focused `mojo_gpu_native_reusable` gate on an Apple M1 Mac mini at the PR's recorded candidate revision. |
| Public API, profiles and fault behavior | Existing collect/profile/explain calls, forced-engine errors, data-fault adjudication and CPU-only import/build behavior. | Preserved behind the existing API and provider entry points; the narrow native paths do not create a new placement policy. | Unsupported forced work still raises. Only designated data faults are adjudicated by the CPU; budget, planning and device failures still propagate. | Retained public query contracts run in the reported rows and focused gates. The broad-gate qualification below remains part of the status. |
| Typed storage and direct row output ([#870](https://github.com/randyzwitch/dataframe_mojo/issues/870)) | Fixed-width row values use the generic 21-byte-per-cell matrix and typed host export. | Certified direct Float32 projection/filter output can retain native values and validity and materialize from the completed native output. | The certificate is narrow. It preserves Float32 storage and nullable/sliced/chunked input; it does not replace matrix storage across resident DAGs, joins or other dtypes. | [PR #888](https://github.com/randyzwitch/dataframe_mojo/pull/888); focused [`mojo_gpu_native_reusable`](https://github.com/randyzwitch/dataframe_mojo/blob/6398713df4a8f31268f5d6064d47d676b9d7bec1/tests/tooling/mojo_gpu_native_reusable.mojo) coverage includes native storage and lifetime/accounting checks. |
| Fused scalar integer expression and SUM ([#872](https://github.com/randyzwitch/dataframe_mojo/issues/872)) | Generic row expressions write matrix intermediates consumed by reduction kernels. | Host-certified Int32 `x` or `x > Int32 literal` with optional checked `(x + Int32 literal) * Int32 literal`, explicit Int64 cast and SUM; or one Int64/UInt64 source with one or more SUM siblings over `x`, `x + literal`, `x * literal`, or `(x + literal) * literal`. These shapes use fused kernels and bounded partial state. | Only certified plan shapes and dtypes use these kernels. Nulls, `min_count`, checked row/final overflow, CPU data-fault adjudication and budget bounds remain part of the certificate. Other expression, reduction, and grouped-feed combinations use existing routes. | [PR #888 focused gate](https://github.com/randyzwitch/dataframe_mojo/pull/888) passed the reported 14 native reusable groups and the independent [`mojo_gpu_rows`](https://github.com/randyzwitch/dataframe_mojo/blob/6398713df4a8f31268f5d6064d47d676b9d7bec1/tests/tooling/mojo_gpu_rows.mojo) runner on Apple M1. The [#889 evidence archive](https://github.com/randyzwitch/dataframe_mojo/pull/889) records the paired Int64/UInt64 SUM comparison for candidate ancestor `52efeb3a`; see its [summary](https://github.com/randyzwitch/dataframe_mojo/blob/29890833076c5dc6cd711e685376d1377ccbd7c9/benchmarks/results/metal-native64/a52334e7-52efeb3a/summary.md), [validation log](https://github.com/randyzwitch/dataframe_mojo/blob/29890833076c5dc6cd711e685376d1377ccbd7c9/benchmarks/results/metal-native64/a52334e7-52efeb3a/validation.log), and [retained qualification logs](https://github.com/randyzwitch/dataframe_mojo/tree/29890833076c5dc6cd711e685376d1377ccbd7c9/benchmarks/results/metal-native64/a52334e7-52efeb3a/validation). |
| General grouping and reductions ([#873](https://github.com/randyzwitch/dataframe_mojo/issues/873)) | Existing grouping sorts row indices before grouped reduction; established narrow dictionary SUM paths remain available. Scalar reductions use their existing per-family kernels. | No general grouping replacement is in PR #888. | Multiple general key/reducer types, null/key identity, ordering-sensitive states and feed-fault order still require a family-level proof and budget tests. | Existing maintained execution contracts remain; PR #888's focused gate is not a grouped-replacement gate. The broader candidate gate retained known failures noted below. |
| Sorting, top-K and ranking ([#874](https://github.com/randyzwitch/dataframe_mojo/issues/874)) | Existing stable sort and ranking implementations remain. | No sort/top-K/ranking replacement is in PR #888. | Stable ties, mixed directions, null/NaN ordering, checked-key errors and work bounded by K need explicit evidence before removing current paths. | Existing ordering/ranking tests remain regression coverage; no #874 replacement result is claimed here. |
| Joins and resident join pipelines ([#875](https://github.com/randyzwitch/dataframe_mojo/issues/875)) | Existing join executor and resident DAG routes remain. | No join replacement is in PR #888. | Join style, duplicate expansion, null policy, key coalescing, error order, branch reuse and output-memory bounds remain with current routes pending a replacement gate. | Existing join and DAG tests remain; no #875 replacement result is claimed here. |
| Windows and ordered operations ([#876](https://github.com/randyzwitch/dataframe_mojo/issues/876)) | Existing ordering/window kernels and row-plan machinery remain. | No window replacement is in PR #888. | Partition boundaries, reverse order, cumulative overflow, rolling/null semantics, stable ordering and large/skewed partitions need explicit replacement coverage. | Existing window/ordered tests remain regression coverage; no #876 replacement result is claimed here. |
| Sources, text, nested and temporal values | Existing scan staging and specialized text, nested-value and temporal paths remain. | No broad source/text/nested/temporal replacement is in PR #888. | Their current operation-specific restrictions remain documented in [Apple Metal](apple-metal.md#remaining-cpu-coverage). | The preserved contracts are regression coverage, not evidence of a new replacement in this PR. |

Separate implementations for grouping, ordering, windows and native join inputs are not present in the PR #888 candidate. Their independently pinned validation results are recorded below as they become available. The earlier archived Int32 affine comparison is pinned to its own earlier candidate and is not evidence for the current PR head. The Int64/UInt64 archive above is pinned to `52efeb3a`, an ancestor of the PR #888 head, and remains a comparison for those recorded cases only.

## Follow-up evidence recorded on 2026-10-20

The table above remains pinned to `6398713d`; subsequent branch updates do not
retroactively change its validation scope.

- [PR #894](https://github.com/randyzwitch/dataframe_mojo/pull/894) extends the
  reusable device service to resident DAG queries. Its four focused Mac tests
  passed at `e3434a9fe971020db5399b32cefb4f7108861409`, covering branched joins,
  grouped fields, overlap rejection, poisoned contexts and budget rejection.
  That evidence applies to the recorded revision; the later merge of the updated
  #888 base does not acquire an implicit hardware pass.
- [PR #895](https://github.com/randyzwitch/dataframe_mojo/pull/895) archives the
  completed development-suite baseline: 1,812 worker records across 302 planned
  cases, with 210 cases admitted for CPU/Metal timing ratios. Its package retains
  exact receipts, raw logs, build identities and exclusions. All 249 recorded
  engine-source hashes match frozen baseline `a52334e7`. These are baseline
  measurements, not replacement coverage or a full-scale headline result.

- [PR #896](https://github.com/randyzwitch/dataframe_mojo/pull/896) adds stable
  radix passes and tiled comparison merges. Its four-test ordering gate passed
  on the M1 Mac mini at `a03f7edfb797e02a7446aecb663790dbd96419ee`, including
  mixed keys, stable ties, floating encodings, errors and budgets. Bounded top-k
  is a separate follow-up. The PR distinguishes its earlier algorithm timing
  revision from this validated integration revision.
- [PR #897](https://github.com/randyzwitch/dataframe_mojo/pull/897) adds
  heterogeneous native column gathers and the one-column Float32 join-input
  bridge. Its 24-call typed gate passed at
  `3703624512d3a24a49d8215be45b5f8240f84327`, including exact Float64 bits,
  deferred errors, ownership and budgets. The bridge still converts into the
  generic join representation; packed Bool host transport remains unsupported.
- [PR #898](https://github.com/randyzwitch/dataframe_mojo/pull/898) adds native
  payload gathering from two device row maps. Its four-test focused gate passed
  at `40c4e6c2eabb9ebd51c03a4057007541b8af3c06`, covering coalescing precedence,
  exact Float64 bits, missing rows, bounds/errors and budgets with a shared
  final drain. This is a join foundation, not public native hash-join dispatch.
- [PR #900](https://github.com/randyzwitch/dataframe_mojo/pull/900) adds bounded
  top-k selection for eligible terminal raw sorts with
  `0 < K <= 4096` and K smaller than the input row count. Its five-test ordering gate passed at
  `83e1988db1815b4cc2a29c87d4310a94e68f34e1`, including the four full-sort
  regression groups. Candidate selection is bounded by K; input storage and
  initial indices remain O(N), so total query memory is not O(K).
- [PR #901](https://github.com/randyzwitch/dataframe_mojo/pull/901) adds public
  native single-key hash-join dispatch for Int32, UInt32 and Float32 keys.
  Its eight-test gate passed at
  `b426cc27f0d87df9dcd469c1843a0ef1e578ad45`, covering keyed styles, null
  equality, exact payloads, empty sides, collision fallback, profile counts
  and budget rejection/recovery, with bounded cross-join regression coverage.
  Composite, wide and dictionary keys and ordered merge joins remain follow-ups.

- [PR #902](https://github.com/randyzwitch/dataframe_mojo/pull/902) adds bounded
  partial states and parallel merges for eligible grouped integer reductions.
  Its nine-call gate passed at `0a9297c3f15cae9984ddf5eb9f261df75ada161b`.
  The initial measurements identified two regressions against #865. Corrected
  revision `752eb23322ef9d18e7f2d6b950f733d5e0d8a340` passes a ten-call Mac
  gate and the three-reference measurements below. The first scheduling gate
  failed because a new profile field lacked a DAG default; that failed receipt
  is retained alongside the corrected pass.

These follow-ups do not establish parity across all operation families or
justify legacy-path retirement. Broader family correctness and complete-query
performance evidence remain required, including window gates and top-k
measurements. Typed-storage, join-gather and top-k performance is not established
by the narrow join measurements below.

## Paired performance evidence recorded on 2026-10-20

[PR #889](https://github.com/randyzwitch/dataframe_mojo/pull/889) retains the raw
outputs, samples, source/compiler/SDK/binary receipts, qualification logs and
replay validators for these comparisons. Earlier archive links pin evidence
commit `2f9d29922c4ff913e664402df69259c7405c74a2`; the scheduling-correction
row pins `98e0c2d9fe06897075c22cba3b6ca89909b0ae26`. All comparisons measure complete
queries and retain CPU references, first-use measurements, warm samples and
requested-allocation counters. First use means a fresh query process, not a
cold machine; allocation counters do not measure device residency.

| Family and engine revision | Validated measurement scope | Observed result and limitations |
| --- | --- | --- |
| [Grouped integer reductions, `0a9297c3`](https://github.com/randyzwitch/dataframe_mojo/blob/2f9d29922c4ff913e664402df69259c7405c74a2/benchmarks/results/metal-group-integer/a52334e7-0a9297c3/README.md) | 234 workers; 18 uniform/high-cardinality/skewed cases at two sizes and three input variants; comparisons against frozen `a52334e7` and intermediate #865 `158285f7`. | Every warm case improves against the frozen baseline. Two large non-null uniform cases regress about 7% against #865; large skewed cases improve 4.75–4.89× against #865. Requested peak allocation increases; small first-use regressions remain in the report. Hash fallback frequency is not measured. |
| [Fused Float32 reductions, `004d73e7`](https://github.com/randyzwitch/dataframe_mojo/blob/2f9d29922c4ff913e664402df69259c7405c74a2/benchmarks/results/metal-float32/a52334e7-004d73e7/README.md) | 84 workers; affine SUM with/without filtering, two sizes and three variants against frozen `a52334e7`; correctness qualification logs retained. | Warm complete-query medians improve 3.610–6.517×; requested peak allocations fall approximately 95.0–97.9%. This is an unpublished engine candidate with narrow expression/type coverage, not a broad #872 replacement. |
| [Native Int32 join, `b426cc27`](https://github.com/randyzwitch/dataframe_mojo/blob/2f9d29922c4ff913e664402df69259c7405c74a2/benchmarks/results/metal-native-join/a52334e7-b426cc27/README.md) | 74 workers; bounded self-left-joins, two sizes and three variants against frozen `a52334e7`. | Warm medians improve 1.056–1.383× at 4,097 rows and 2.352–2.672× at 262,144 rows. Requested peak allocations fall 54.8–69.4%, but launches increase by 11 per case. Other key types, join modes and compositions need their own performance evidence. |
| [Grouped scheduling correction, `752eb233`](https://github.com/randyzwitch/dataframe_mojo/blob/98e0c2d9fe06897075c22cba3b6ca89909b0ae26/benchmarks/results/metal-group-integer/schedule-752eb233/README.md) | 333 primary workers against frozen `a52334e7`, #865 `158285f7` and prior `0a9297c3`; 20 follow-up workers for one small case. | The two large uniform cases improve 1.100×/1.070× over #865 and 1.152×/1.139× over the prior candidate, with four fewer launches. Large skewed gains remain. Requested peak allocations are unchanged versus the prior candidate. Small sorted skew is inconclusive: 1.7% slower initially, 1.1% faster in the longer follow-up; both results are retained. First-use results remain mixed. |

CPU is faster in every case in these archives. The comparisons measure
whole candidate stacks and do not isolate each preceding PR's contribution.
They support measured improvements in the stated cases and document the
large-uniform regression's correction without erasing the original results or
claiming uniform small-input/first-use improvement. Primitive measurements,
fallback-frequency reporting, remaining serial-state coverage and broader
family/whole-query validation remain outstanding under #873 and #877.

## Retirement and validation gates

Keep a family on its current route until its replacement has exact, build-identified Apple-hardware evidence and all required contracts have a replacement. Each family needs public-query parity against the CPU plus independent typed/bitwise expectations where available; null, NaN and signed-zero cases; composed and shared-branch execution; checked-error/fault-order checks; metadata and exact/under-budget behavior; and complete-query timing against a frozen current-Metal baseline. Retain slow and failed cases with the receipts and qualification logs. Kernel-only timing does not satisfy the performance gate.

The pinned PR #888 focused gate passed its reported 14 groups, and the paired #889 archive validates all 144 recorded Int64/UInt64 benchmark workers. Live CI status is tracked on [PR #888](https://github.com/randyzwitch/dataframe_mojo/pull/888/checks) and [PR #889](https://github.com/randyzwitch/dataframe_mojo/pull/889/checks); it does not replace the pinned hardware evidence above. The separate broader candidate gate was **not all green**: selector expansion, categorical grouped row count and qcut whole-DAG synchronization differences were reproduced on the frozen baseline; an ordered profile/resource-bound failure received focused fixes and passing tests, but the broad gate was not rerun as all-green. The reports do not establish full-suite parity, close #877 or justify retiring untouched routes. Automatic CPU/Metal placement remains deferred under #583.
