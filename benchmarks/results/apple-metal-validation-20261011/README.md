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

The clean package build and matched Metal cost collection remain pending the
specific repository-bundle transfer approval. Source tests do not establish
automatic-placement calibration or performance crossovers for hash and radix
dispatch. No new calibration has been enabled.
