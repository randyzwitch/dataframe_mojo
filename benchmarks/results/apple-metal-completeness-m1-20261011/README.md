# Metal completeness correctness snapshot, Apple M1, 2026-10-11

This is a forced-Metal correctness inventory of the existing CPU benchmark
queries at smoke scale. It is **not a passing full-suite result**, a speedup
claim, or automatic-placement calibration. It predates Float64 emulation.

## Results

| Suite | Exact result | Metal unsupported | CPU reference unsupported | Result mismatch |
| --- | ---: | ---: | ---: | ---: |
| H2O group-by | 15 | 35 | 0 | 0 |
| H2O join | 10 | 0 | 0 | 0 |
| PDS-H | 8 | 36 | 0 | 0 |
| TPC-DS | 8 | 170 | 20 | 0 |
| ClickBench | 24 | 13 | 1 | 5 |
| **Total** | **65** | **254** | **21** | **5** |

All 345 cases have raw logs. Each supported query is collected twice through an
explicit `MetalRuntime`, without automatic placement or a CPU retry. The driver
retains input dtypes and compares the complete result with CPU execution, allowing
row-order normalization only where the query's ordering contract permits it.
The strict driver exited 1 because five results differ.

The five mismatches are ClickBench q11, q17, q38, q39, and q40. Their diagnostics
show differences involving unordered output selection or ties at a nonunique
sort/limit boundary. They remain `wrong_answer` in the raw evidence; this report
does not reinterpret them as passing, alter query tie breakers, or weaken the
comparator. Repeated CPU outputs and result samples are preserved in their logs.

## Snapshot provenance

The four workers were built from clean commit
`55ea74517173fc161523788bbfce0755ec8a58ea`, whose Git tree is
`5fa4a565e3c6cac43543a1eebb3c60e34d2b6986`. That tree is identical to the
published implementation snapshot `d8966c9`, before the PR stack was rebased onto
CPU window change #592. This is historical evidence for that implementation,
not a claim that the complete tree equals the later rebased stack.

`results.json` records compiler/platform information, input SHA-256 hashes,
generated worker source hashes, worker binary hashes, and the native library
file hash. `sources/` preserves the actual generated workers as byte-identical
`.mojo.txt` files, so archived evidence is not treated as active repository
source by formatters and test discovery. Remove the final `.txt` suffix to
reconstruct a worker. The native library
file SHA-256 is
`b40037b318af5bc1311b37ca648780fd2eb8cf520f60cff0aef091405ba286bf`.
The compiler was Mojo `1.2.0.dev2026100905 (3b71af75)` on Apple Silicon macOS;
the run used eight threads and two collections per supported case.

The suite inventory and adaptation are provided by the separate correctness
runner PR #599, `scripts/bench_metal_suites.py`. The generated CPU expressions
and original input dtypes are retained. H2O eager operations are adapted to
lazy source scans for Metal execution; materialization boundaries are the only
intended query changes. Source and derivation hashes make those changes
inspectable.

## Integration checks

`public-tests-original-stack.log` records all 79 planning, original Metal, new
Metal, and time-window tests passing with the native library present, then
again with the library absent. Tests requiring a device return early in the
absent-library run; planning and rejection tests still execute.

After CPU window PR #592 merged and the stack was rebased, an independent
integration run at `be9dbfdf663956dc2aa0963a0a25490125f43b23` passed all 88
tests with the library present and again with it absent. This adds all nine CPU
window tests to the previous integration set. The complete log is
`public-tests-rebased-stack.log`. It does not substitute for a full suite rerun
on the rebased commit.

`native-tests.log` preserves the native ABI, join, and resident-runtime checks.
`summary.json` is derived directly from the 345 raw case records.
`snapshot-provenance.json` distinguishes the measured tree from the rebased
integration commit. `sha256.json` identifies the evidence files copied from the
Apple machine and the generated summaries.
