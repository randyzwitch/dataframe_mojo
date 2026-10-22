# Paired Float32 fused reduction measurements

On the Mac mini, candidate engine `004d73e7` improves warm complete-query time by **3.610–6.517×** versus frozen baseline `a52334e7` across all 12 measured cases. Every paired round improved. These results compare the complete candidate stack with the frozen baseline; they do not isolate each preceding PR's contribution.

The matrix covers affine Float32 sum, with and without filtering, at 4,097 and 1,048,576 rows, with base, sorted and null inputs. Three paired rounds each contain one warmup and three timed samples. All 72 paired workers and 12 smoke workers pass CPU-result validation. Every candidate paired worker reports `mojo_gpu_fused_float32_sum`.

Reported requested peak allocations fall by approximately **95.0–97.9%**. Kernel launches fall from four to two for unfiltered sums, and eight to two for filtered sums. First-use medians also improve in every case. **CPU remains faster in all 12 cases**; the improvement is over the previous GPU implementation. Requested allocation accounting is not measured device residency.

See [summary.md](summary.md) for per-case warm, first-use, CPU, memory and launch results. Per-round ranges describe observed variation, not confidence intervals. This narrow Float32 evidence does not establish broad expression, type or workload coverage, nor close the engine replacement issues.

Both binaries use identical benchmark sources, lockfile, compiler, SDK and flags, with clean source receipts. `attribution.json` records engine and measurement revisions; `plan.json` and `run_pairs.py` retain the original execution plan. `validation/` retains the passed correctness-gate receipt and logs. `paired/` and the smoke directories retain parsed JSON and raw worker logs; `buildlogs/` retains compiler and supervisor output. Raw outputs are preserved even where they contain machine-local paths.

Run `python3 validate_results.py` to reparse all raw outputs, check complete coverage, attribution, routes and gate hashes. Run `sha256sum -c SHA256SUMS.txt` to verify archive integrity. `summarize_results.py paired/results.json summary` reproduces the derived report.
