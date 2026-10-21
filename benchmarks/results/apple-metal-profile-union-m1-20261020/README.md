# Native JOIN → SUM profile schema regression

At exact revision `d241b736141b764f0bb9f7ec85aafeca7b5e590f`, the M1 Mac mini built and passed `tests/tooling/mojo_gpu_profile_phases.mojo`. This is correctness evidence, not a performance measurement.

An earlier benchmark smoke at engine `b426cc27` failed when DAG profile assembly requested the JOIN-only `native_hash_phase_launches` metric from a following reduction. Both report append sites now use diagonal concatenation: metrics introduced by either region survive, absent values become typed nulls, and first-seen column order remains stable. The patch and original failing worker log are retained.

The passing gate covers optional metrics in either region order, empty accumulation, common row values, native JOIN followed by SUM (CPU equality, scalar 150, six joined rows, resident byte continuity, and a nine-byte final host result), host phase reports, resident branches, exports, and budgets. Equal resident byte counts do not prove physical owner identity. All four completion markers are in `test.log`.

The receipt pins source, compiler, lockfile, runner, engine tree, binary and logs. The binary was copied from the Mac and independently hash-checked, but is not committed here. Source bundles and hardware binaries are omitted; `change.patch` records the change against its parent. The archived runner describes the original source bundle and Mac paths and is not a portable standalone test command.

Run `python3 validate_results.py` to verify the archived integrity and completion markers. Hardware reproduction requires the pinned source and compiler on Metal. The original failed smoke produced no accepted performance dataset. A new JOIN → SUM matrix is still pending.
