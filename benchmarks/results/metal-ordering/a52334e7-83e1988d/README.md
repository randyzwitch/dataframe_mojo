# Paired ordering and bounded top-k measurements

**CPU remains faster in every measured case.** In the frozen-baseline comparison, the current candidate's small top-k-31 queries take 4.579–4.742 ms versus CPU's 0.034–0.049 ms: **95.4–134.7× slower**. In the separately collected radix-intermediate comparison, they take 4.832–5.297 ms versus 0.033–0.042 ms: **126.1–159.6× slower**. These are matched per-case ratios, not ratios of the displayed range endpoints. The large top-k-31 cases remain 21.8–42.7× behind CPU across the two collections. Gains over old Metal do not establish competitiveness.

Candidate engine `83e1988db1815b4cc2a29c87d4310a94e68f34e1` (#900) is compared with frozen `a52334e758b7138c547883eb776d744442a1a0db`, and separately with radix intermediate `a03f7edfb797e02a7446aecb663790dbd96419ee` (#896). All **204 workers** validated: 24 smoke/build workers, 108 frozen-pair workers and 72 radix-pair workers. The matrix uses 4,097 and 1,048,576 input rows, base/sorted/null variants, three paired rounds, three timed repetitions and one warmup. Every timed and profiled output is checked against its complete CPU result. The focused engine correctness gate receipts and logs are retained separately.

## Complete-query measurements

The [frozen comparison](frozen-summary.md) contains full radix sort and top-k 31/1024. Every case's aggregate warm median improves versus old Metal. At 1,048,576 rows, radix takes 209.435–212.279 ms versus CPU's 5.361–14.557 ms, leaving a 14.6–39.1× CPU gap; its old-Metal improvement is 1.535–1.601×. Top-k 31 takes 42.315–43.018 ms versus CPU's 1.037–1.974 ms, while top-k 1024 takes 46.742–47.724 ms versus CPU's 1.599–3.374 ms. Their old-Metal improvements are respectively 3.786–4.080× and 3.635–3.674×. These are whole-stack comparisons, not isolated attribution to the top-k commit.

The [radix-intermediate comparison](radix-summary.md) isolates the top-k candidate stack against the preceding radix stack. At the large size, top-k 31 improves 1.077–1.105×, but **top-k 1024 is effectively flat**: ratios 0.999–1.005×, with individual rounds on both sides of parity. It must not be described as a demonstrated timing improvement over radix for that case. Small aggregate medians improve, but one small top-k-31 base round regresses by about 2.6%. In the frozen comparison, one small nullable radix round regresses by about 0.13%. The raw observations remain retained.

First-use results are mixed at small sizes. Large first-use results improve against frozen Metal, while large top-k-1024 first-use medians are slightly slower than the radix intermediate. First use is a fresh query process with runtime initialization, not a cold machine or cold filesystem.

## Allocation, route evidence and limits

Requested peak allocation for top-k falls about 45.6–62.7% against frozen Metal (depending on size and K), and similarly against the radix intermediate. Full radix peak allocation increases about 0.17–0.20% against frozen Metal. These are requested-allocation counters, not measured physical device residency; the plan's 1 GiB ceiling is a post-run audit, not an enforced runtime allocation budget.

The profile names only `mojo_gpu_ordered_rows`. Exact sorter attribution is therefore **inferred**, using pinned planner admission, workload/output shape and matching whole-query launch counts: radix 39/57 and top-k 17/24 at the small/large sizes. It is not a direct runtime sorter trace. Versus frozen Metal, radix launches rise from 16/23 to 39/57, and top-k launches rise to 17/24. Versus the radix intermediate, top-k reduces launches from 39/57 to 17/24; the flat large top-k-1024 timing demonstrates that fewer launches alone does not establish a useful query improvement.

This is narrow integer ordering evidence, not broad type/ordering coverage, GPU-over-CPU competitiveness, or completion of #874/#877. Automatic placement remains deferred under #583.

## Reproduce and audit

`plan.json` pins engine revisions, measurement commits, shared harness/lock hashes and gate identities. The archives retain raw process outputs, all samples, compiler/SDK/binary receipts, driver guards and focused gate logs. The three measurement trees have identical harness files and unchanged pinned engine trees.

Run `python3 validate_results.py` to reparse every raw worker and verify exact coverage, paired build identity, smoke-to-pair binary identity, source attribution, gate hashes and route-inference counters. Run `sha256sum -c SHA256SUMS.txt` for archive integrity. Reproduce summaries with:

```sh
python3 summarize_results.py frozen-pair/results.json frozen-summary
python3 summarize_results.py radix-pair/results.json radix-summary
```

No engine change is included in this archive. Hardware qualification applies to the recorded source pins, not an untested merged stack.
