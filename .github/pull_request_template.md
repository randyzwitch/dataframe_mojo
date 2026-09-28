## What and why

<!-- The change, and the problem it solves. -->

## Performance claims (delete if none)

See docs/benchmarks.md.

- [ ] Measured on the external suites (`scripts/bench_suites.py`), not only on an in-repo micro-benchmark.
- [ ] Reported the worst data variant, not only the base case.
- [ ] Held-out suites (`--heldout`) do not regress, and were not used to choose this optimization.
- [ ] Any new fast path names the data property it detects and a real-world example of it, and `--trace` shows more than one suite query using it.
- [ ] No benchmark, data generator or suite query changed in this PR (or a `Benchmark-Change:` trailer explains why).

## Testing

<!-- Tests added or run, and results. -->
