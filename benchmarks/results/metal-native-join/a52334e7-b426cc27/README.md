# Paired native single-key Int32 join measurements

Candidate engine `b426cc27` improves warm complete-query time versus frozen baseline `a52334e7` in all 12 measured cases: **1.056–1.383× at 4,097 rows** and **2.352–2.672× at 262,144 rows**. Every paired round improves. This compares the full candidate stack with the frozen baseline, not the isolated contribution of each preceding PR.

All **74 workers** validate: two smoke workers and 72 paired workers covering two input sizes, `join_hash` and `join_fanout`, base/sorted/null inputs, three rounds, one warmup and three timed samples per worker. Every candidate paired worker reports `mojo_gpu_native_hash_join`. First-use medians also improve in all 12 cases.

Reported peak requested allocations fall **54.8–69.4%**, but kernel launches increase by 11 in each case (18 to 29 at the small size, 23 to 34 at the large size). **CPU remains faster in every case.** Allocation accounting is not measured device residency, and the 1 GiB audit ceiling is a post-run check, not an enforced runtime memory budget.

The workloads are single-key Int32 self-left-joins with bounded duplicate matches. `join_hash` permits at most three output rows per input row and `join_fanout` at most four. These results do not establish composite/wide/dictionary key performance, other join modes, ordered merge, build-side reuse, or broad replacement completion under #875.

See [summary.md](summary.md) for every warm, first-use, CPU, allocation and launch measurement. Per-round ranges show variation rather than confidence intervals. Build receipts pin both engine revisions and shared harness/compiler/SDK/flags. `plan.json` and `run_pairs.py` retain the execution guards; `validation/` retains the focused correctness receipt and logs; raw output, parsed samples and compiler logs remain archived.

Run `python3 validate_results.py` to replay every raw output, check matrix coverage, output bounds, native routing, source/build attribution and correctness hashes. Run `sha256sum -c SHA256SUMS.txt` for archive integrity. `python3 summarize_results.py paired/results.json summary` reproduces the report. The raw logs preserve machine-local paths.
