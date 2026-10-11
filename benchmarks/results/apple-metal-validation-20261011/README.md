# Metal final source validation

Source execution changes are committed through `576ee81`; documentation follows
at `422d10e`. Validation used the existing isolated Mac source tree and native
build `ec16a3a0079af9a05a0ea1314d64a6ef96ad76f4123ec334ea55bd8e959328e6`.
Hashes of Metal execution, shared backend and cost/dictionary modules, provider
hooks, and the three Metal test files match the committed local source.

The 19 main Metal tests, four Metal dictionary tests, two Metal estimate tests,
three shared analytical-policy tests, and three shared dictionary tests passed:
31 Mojo tests in total. Native row, join and resident suites passed with this
native build; the metadata-probe fixture subsequently passed in the native row
suite. Eight Python calibration-driver tests also passed.

`metal-tests.log` contains the main suite invocation through its successful
summary. The subsequent dictionary invocation omitted the explicit native
library setting and failed its installation assertions; the corrected complete
dictionary/policy run is preserved in `dictionary-policy-tests.log`.

The repository bundle was subsequently transferred with approval and built as a
clean identified package in `/tmp/apple-final-validation` on the Mac. The
provider-neutral packaged consumer passed all 20 modes, including real Metal
automatic execution under synthetic test calibration, CPU fallback for stale or
invalid evidence, and device error propagation/recovery. The synthetic fixture
was never installed. `package-consumer.log` preserves the output.

Matched packaged cost collection is complete; see
[the cost evidence](../apple-metal-cost-m1-20261011/README.md). No calibration was
enabled. Hash/radix performance evidence is recorded separately from these
correctness and package checks.
