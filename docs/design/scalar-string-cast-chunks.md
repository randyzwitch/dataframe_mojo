# Primitive String casts over chunks

This #1003 subset reads cold host chunks for 22 nonidentity conversion pairs:
String to each numeric type or Bool, and each numeric type or Bool to String.
It removes the input merge while preserving the required owned output.
Examples include parsing textual measurements from concatenated Arrow batches
and formatting numeric or Boolean columns after reading Parquet row groups.
Admission uses physical storage properties, without new size or value thresholds.

String parsing retains the existing grammar and integer error priority. Except
for UInt64, integers parse through Int64 before target bounds. Float32 parses
through Float64 before narrowing. Numeric parsing keeps the existing worker
threshold, global worker ranges, and earliest observed strict error. Boolean
parsing remains serial and accepts only lowercase `true` and `false`.
Formatting uses each original numeric scalar width and lowercase Boolean text.
The contiguous numeric String parser is already typed and parallel; its input
merge is the avoidable step here. Boolean parsing also retires the generic
numeric intermediate, and formatting uses typed input traversal. Any measured
gains therefore need to distinguish those mechanisms from input-copy removal.

Direct masks apply only when their length equals the input length. Null,
inactive, and relaxed-failure numeric rows contain raw zero; Boolean rows
contain false. String outputs retain null validity and empty null spans.
Strict diagnostics preserve the original text, global row offset, and cause.

Cold String admission validates physical array types, host-readable backing
buffers, visible windows, offsets, descriptor bounds, and validity coverage.
Both conventional offsets and native views are supported, including mixed
chunks. Encoded String columns, published merge caches, and unsuitable storage
retain the adapter. Identity and String-to-Binary retagging remain earlier.
Conventional offset merges copy bytes and offsets; view merges can share byte
buffers, so their savings must be reported separately.

The production branch starts from main `78525c62`. The incremental comparison
includes the preceding primitive-cast change on both sides: BASE
`0809807ee39fbbc8c46107925c5b2874ea896a71` and HEAD
`96b990662839e31bed685e21a0b8fa3038fe0bb9`. Earlier gains remain controls.
Implementation `fc2c65d2` then incorporates merged #1097 from main `c9911860`
as `aa1361b0`. All 304 runtime files across dataframe, provider, Metal and
native directories match the actually validated HEAD byte for byte, as does
the complete new String contract blob. One pre-existing main Bool contract
has a formatting-only difference; full tree identity is not claimed.

## Compatibility checks

All six baseline-compatible String casting contracts pass on unchanged main.
The candidate passes all six with additional cold-cache assertions. Source
fingerprints were individually verified against the baseline and candidate
commits. All 13 noncompiler checks and changed-file formatting checks pass.

On the live M1, native compilation and execution both pass. Four complete
CPU/accelerator by flat/chunked fixtures check literal results, validity,
finite floating bits, and NaN classification. An actual Metal Int64 SUM
produces 130, followed by host formatting and parsing, including strict and
relaxed outcomes. Each context uses a fresh single-expression query.
These checks establish host-materialized compatibility; they make no GPU
cast offload, nonhost admission, or performance claim. Coded String native
coverage is omitted and remains a separate functional control.

All 87 broader public CPU cast checks and three kernel checks pass. Three
pre-existing binary/categorical expected failures remain declared; no
unexpected failure occurs. All 644 source fingerprints match the committed
candidate. All 84 functional cases pass per revision on Linux and live M1:
68 complete ordered/schema/validity/payload answers and 16 exact strict errors.
The two 65,573-row cases exercise four workers and error ordering, including
an earlier masked error and a later observed error. The catalog covers every
pair, masks, nonzero windows, offset/view/mixed storage, ownership, empty
input, identities, retags, encoded String fallback and the prior primitive
conversion control. An independent parser qualification covers 172 token
observations and 11 cast cases; 17 separately qualified original-width float
formatter witnesses cover SDK formatting without inventing expected numbers
from candidate cast results.

## Allocation and observed benefits

All 49 matched Linux allocation cases pass complete answer checks with zero
tracking errors. Calls, requested bytes, and freed bytes are lower in 28
cases and equal in 21; none increases. Peak newly owned bytes are lower in
26 and equal in 23. Live bytes on return are lower in 18 and equal in 31.
Every eligible cold case lowers requested bytes. All flat, published-cache,
and prior primitive-conversion allocation controls remain equal.

The heap scope begins after source fixture preparation and measures newly
owned allocations; it is not process RSS or total input ownership. Timing
uses the separate uninstrumented native build. Each table row compares the
same source configuration across revisions.

| Cold direct case | Actual chunks | Requested bytes BASE → HEAD | M1 median microseconds BASE → HEAD |
| --- | ---: | ---: | ---: |
| String to Int8, offsets, 100K rows, partial mask | 64 | 1,958,851 → 353,564 | 420 → 712 |
| String to UInt64, offsets, 100K rows, partial mask | 64 | 2,794,248 → 1,053,564 | 608 → 584 |
| String to Float64, offsets, 100K rows, partial mask | 64 | 2,570,312 → 1,053,564 | 1,008 → 982 |
| String to Bool, offsets, 100K rows, partial mask | 64 | 3,446,411 → 34,144 | 6,981 → 563 |
| Float32 to String, 100K rows, partial mask | 64 | 2,152,363 → 1,599,425 | 8,804 → 2,591 |
| Bool to String, 100K rows, partial mask | 64 | 1,622,587 → 1,468,351 | 7,167 → 818 |
| String to UInt64, 1K rows, partial mask | 983 | 468,629 → 144,549 | 343 → 59 |
| Bool to String, 1K rows, partial mask | 983 | 455,548 → 13,736 | 350 → 19 |

The tiny fixtures request 1,000 pieces but actually expose 983 nonempty
chunks after slicing. View and mixed-layout cases are retained separately:
String-to-Int8 view input requests 1,987,104 → 353,564 bytes and has medians
1,320 → 928 microseconds. Views may share UTF-8 backing storage during the
legacy merge, so this is not evidence that every avoided merge copied bytes.
The benefit includes retiring merge metadata and generic preparation where
applicable. Among all 28 eligible cold cases, 25 have lower timing medians
and three have higher medians; the latter remain tradeoffs, discussed below.

## Heldout validation and coverage limits

Both revisions independently match fresh DuckDB answers for all 43 unchanged
quick-tier ClickBench queries: 86 target verdicts and 774 verified snapshots.
Controller and collector both exit zero. Aggregate HEAD/BASE elapsed is
1.0007647543, effectively unchanged (+0.076%). This frozen ClickBench plan
has only the base variant, so its worst and only variant is that aggregate;
it does not establish development-suite or full-tier coverage. Polars'
35 wrong/eight correct outcomes remain retained without skipping queries.
No individual heldout query timings were inspected to choose or tune this
change. Benchmark, generator and suite-query code is unchanged.

Generic operator traces cannot certify use of these new helpers. Benchmark
rule 5 permits a path with absent suite coverage when its data property and
real-world examples justify it; those are stated above. No helper-specific
multi-query coverage, full external perturbation coverage, or engine-wide
speedup is claimed. Mechanism evidence separately retains flat and warm
controls, nulls/masks, nonzero slices, offset/view/mixed storage and tiny
fragmentation, including every higher timing median.


## Timing tradeoffs

All 49 frozen cost cases have three answer-verified samples per revision on
the live M1 under the calibrated known-process activity guard: 294 complete answers, with zero guard overlap or errors.
Medians are lower in 34 cases, equal in two, and higher in 13. This is a mixed
result, not a universal speedup. All 13 higher-median cases remain in evidence.
Overlapping three-sample ranges do not establish absence of regression.

| Case | BASE median [range], microseconds | HEAD median [range], microseconds |
| --- | ---: | ---: |
| Cold direct String to Int8, 100K rows, 64 offset chunks, mixed nulls, partial mask | 420 [390–751] | 712 [704–745] |
| Flat direct String to Float64 control, 100K rows, partial mask | 340 [335–830] | 768 [435–827] |
| Published-cache String to Float64 control, 100K rows, partial mask | 907 [472–1187] | 1194 [977–1206] |

The other ten higher-median cases range from +0.09% to +4.25%. The three
eligible cold cases with higher medians are the direct masked Int8 case,
whole-expression masked Int8 (+1.57%), and direct unmasked UInt64 (+4.25%).
The worst median increase is the flat Float64 control (+125.9%); the cold
Int8 case is +69.5%, and the published-cache Float64 control is +31.6%.
Inputs to the latter two controls retain the legacy adapter. No timing-driven
source change, threshold, retry, or sample exclusion was introduced.

The final profile uses a separately frozen fixture-v3. Two fixture setup
corrections used the proper categorical dictionary constructor and avoided
constructing an unused numeric conditional for the direct Binary retag
control. Original failed attempts and binaries are retained. The final
84/49 case catalog, independent oracle, and expected bytes are unchanged.
