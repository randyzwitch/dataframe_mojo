# Validation provenance

`validation.log` was produced from the checked-in `results.json` using the
included `validate_results.py`. The validator checks run-key coverage,
completion, CPU-result validation flags, timed samples, first-use samples,
profile fields, receipt status, source cleanliness, and stable output counts.
It passed all 144 paired run keys (72 per engine), covering 24
size/workload/variant cells across three rounds.

No build, benchmark worker, Mojo test, or GPU test was run while assembling this
evidence package. The `validation/` directory contains the supplied pinned
candidate qualification and prior full-gate/reproduction logs. The prior
full-gate log records four failures: three were independently reproduced on
baseline (selector expansion, categorical grouped row count, and qcut
synchronization parity); an ordered profile/resource-estimate failure was
followed by focused resource-bound fixes with passing tests. The receipt and
logs do not claim an all-green full suite.
