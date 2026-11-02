# Decimal-to-Float64 casts over cold chunks

This #1003 subset avoids materializing a contiguous decimal source before
converting it to Float64. Decimal32, decimal64 and decimal128 host windows
write into one owned Float64 buffer and an offset-aware validity bitmap. The
existing flat conversion remains byte-identical.

## Preserved behavior

The conversion is exactly `Float64(raw) / Float64(pow10(scale))`: numerator and
divisor each round to Float64 before division. Rounding the exact decimal value
once can produce a different result. Every logical row's initialized payload
converts, including rows hidden by nulls. Unlike generic checked casts, this
specialized legacy path ignores masks, strictness and the diagnostic row
offset. Public conditional evaluation can still mask the resulting values
later.

The result retains the source name and Float64 dtype, with fresh default field
metadata and owned output buffers. Source field metadata is not propagated by
the existing converter. All-valid inputs without a bitmap retain the empty
bitmap sentinel without allocating a result bitmap.

The shortcut first declines actual published caches, whose values, validity,
physical type and length remain authoritative. It admits canonical decimal
metadata with matching integer physical width only. It validates all cold
arrays, host buffers, payload/bitmap bounds, cumulative ends and positive
window lengths before reading any payload. Physical empty arrays, mismatched or
heterogeneous storage, malformed layouts and nonhost buffers retain the
existing adapter. Public construction canonicalizes empty pieces separately. No
source merge cache is published by the new path. Unaligned validity spans reuse
the existing bitmap byte reader/writer; aligned and absent bitmaps retain the
existing bulk appender. Prefix bits, partial-byte tails and the absent bitmap
sentinel are preserved without an intermediate bitmap per chunk.

## Source and correctness evidence

Production starts independently from main `c9911860` on
`work/1003-decimal-float64-chunks`. Refined production revision is `e0658eb0`;
actual incremental BASE is `d49a645e` and refined HEAD is `de8b6a74`, retaining
preceding accepted work on both comparison sides. The four helper bodies and
dispatch hook are identical between contexts, and removing those insertions
restores each respective baseline cast file exactly. Both new contract blobs are
identical between production and incremental HEAD. The flat helper is
unchanged.

Both revisions pass all 226 functional configurations on Linux and M1, with
complete ordered Parquet schema/name/null/value checks and independently read
Float64 bitword files. Direct decimal checks include initialized null payloads;
public null payloads are not assigned the direct kernel's policy. Coverage
includes all 68 distinct width/scale combinations at maximum precision plus
precision-one controls, masks of several lengths, both strictness flags, bitmap
offsets zero through seven, empty/scalar inputs and public conditionals. This
is not an exhaustive test of all 1,022 legal precision/scale metadata
combinations. The primary oracle uses exact integer/rational binary64 rounding,
with independent IEEE cross-checks before compilation.

All seven focused production contracts pass on Linux and M1. They cover: exact
payloads, ignored masks, cold-cache nonmaterialization, ownership after input
destruction, published-cache authority and initialized wrong-physical-width
fallback. The bitmap contracts compare 11,264 offset/boundary/sentinel
configurations, including exact-minimum source buffers, against the legacy
appender and independent logical bits. Both independent production builds bind
all 947 committed source files, with identical contract stdout and empty
stderr. Each revision/host also passes 39 separate qualifiers: 21 complete
normal answers, nine complete constructor Error payloads and nine observed
private rejections. Thirty-three same-host stdout records match byte for byte.
Six private wrong-Bool/cache aborts differ only in the unchanged getter's
source line, 816 to 817 after an import was added. A policy frozen before
qualifier compilation binds those coordinates to each role's exact source, with
every other stdout byte and process outcome unchanged. Raw equality results
remain retained; user error-row offsets are never normalized. Private signals
are four on Linux and five on M1. Raw crash stderr is retained and
hash-verified; its ASLR-bearing bytes differ and are not normalized or claimed
equal. Normal/constructor stderr is empty.

## Scoped allocation results

All 44 Linux pairs verify complete schema/null/value answers and the declared
raw-bit checks with zero observer tracking errors. Direct decimal checks cover
every logical payload, including nulls; whole-expression raw-bit checks cover
valid output rows. Calls, requested/freed bytes and peak newly owned bytes
decrease in 30 and match in 14; none increase. Live newly owned bytes decrease
in 12 and match in 32. Flat and ordinary nondecimal controls match exactly.
These are scoped logical allocator requests, not process RSS, allocator
reservation or bytes retained before the observed call. Lower freed bytes alone
are not an improvement metric.

| Direct cold nullable input | Requested bytes BASE → HEAD |
| --- | ---: |
| Decimal32, 100K rows / 64 chunks | 1,265,918 → 812,980 |
| Decimal64, 100K rows / 64 chunks | 1,672,114 → 812,980 |
| Decimal128, 100K rows / 64 chunks | 2,484,506 → 812,980 |
| Decimal32, 1M rows / 64 chunks | 12,281,240 → 8,125,480 |
| Decimal64, 1M rows / 64 chunks | 16,281,240 → 8,125,480 |
| Decimal128, 1M rows / 64 chunks | 24,536,272 → 8,125,480 |
| Decimal32, 1K rows / 1,000 actual chunks | 540,991 → 8,605 |
| Decimal64, 1K rows / 1,000 actual chunks | 544,995 → 8,605 |
| Decimal128, 1K rows / 1,000 actual chunks | 553,003 → 8,605 |

Ready direct tiny controls already own their source merge before the call and
remain at 8,605 requested bytes. Public warm-parent expressions create new
physical evaluation windows and retain corresponding cold-window costs; parent
cache presence does not prove a callee consumed a ready cache. Required Float64
output allocation remains included. The warm-up fixture's lifetime is retained
outside the observed call. Results cover the fixed catalog, not every possible
bitmap representation or layout.

## Native and elapsed checks

The legacy baseline's source-bound M1 native proof is reused explicitly;
the refined HEAD independently passes the identical 28-query fixture. It
checks 24 CPU/accelerator-by-flat/chunked composition queries over all three
widths, actual Metal SUM130 and three subsequent host decimal conversions.
The driver and fourteen stdout markers are identical, with empty stderr.
Decimal conversion executes on the host; these checks establish compatibility,
not GPU decimal execution, helper-copy coverage or performance.

The original `7a254e16` timing window is retained in full: 34 lower, two equal
and eight higher medians over 44 cells. Direct cold 1M-row nullable decimal32
and decimal64 medians increased 21.25% and 16.49%, with broad overlapping
three-sample ranges. Source inspection identified serial per-bit work for
unaligned validity, motivating one bounded decimal-local byte-appender
experiment. These observations establish a mechanism hypothesis, not a causal
attribution. The unchanged payload loop and complete original results remain
preserved. The refined candidate uses a fresh full baseline/candidate window on
the identical catalog; historical elapsed samples are not reused.

The fresh refined M1 window verifies all 264 complete answers and raw-bit
files: 44 cases, three repetitions per revision, with zero compiler/worker
overlap, monitor errors or worker deadlines. Medians decrease in 36 cases,
match in two and increase in six. All samples and zero values remain retained.

| Direct cold 1M-row nullable input | BASE median [min–max] µs | Refined median [min–max] µs |
| --- | ---: | ---: |
| Decimal32 | 1,911 [958–3,790] | 742 [738–1,052] |
| Decimal64 | 2,201 [1,006–2,357] | 850 [739–1,278] |
| Decimal128 | 6,852 [6,685–7,713] | 4,766 [4,599–4,999] |

The six higher medians remain explicit: decimal32 flat whole 92→95 µs;
decimal32 ready direct 50→52; decimal64 flat direct 83→84 and ready direct
49→50; decimal128 flat whole 458→459; and decimal128 tiny ready direct 7
[6–13]→8 [6–12] µs. The last is +1 µs / 14.29%; the others are +0.22% to +4%.
No noise explanation, universal speedup or causal attribution from these
descriptive three-sample ranges is claimed.

Final broader CPU validation covers 169 cases across 23 modules: 166 ordinary
successes and three existing expected-failure wrappers, with no unexpected
failures or skips. Those three categorical defects are not fixed here. All 13
repository source checks pass on the runtime/test revision.

The unchanged quick-tier suite passes all 43 queries on both revisions against
fresh DuckDB references. All 774 target snapshots and 1,548 total engine
snapshots were copied, hash-verified and independently loaded. All 35 Polars
disagreements remain retained. Aggregate refined/BASE elapsed ratio is 1.00148;
this is a quick-tier aggregate observation only. No per-query timing was
inspected or used for tuning. Functional and instrumented allocation elapsed
fields support no timing claim.

## Remaining scope

This covers the specialized decimal-to-Float64 route only. Other decimal
targets, logical/temporal/Binary conversions, text/nested operations and
execution/storage/ pipeline consumers remain in the full #1003 inventory. It
does not complete an entire cast owner or justify unmeasured contiguous
adapters elsewhere.
