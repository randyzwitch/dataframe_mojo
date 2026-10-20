# Validation provenance

The completed paired run produced 180 run records and 180 per-run text logs. The first post-run validator attempt exited with a Python syntax error caused by a missing closing bracket in its bad tuple list. That validator failure happened after all benchmark measurements had completed; it did not launch benchmark work. The completed results.json and raw logs were retained unchanged.

A corrected read-only validator was then run against the existing results.json. It confirmed all 180 unique round/runtime/row/workload/variant keys, status=ok, validated=true, three timed repetitions per run, CPU and Metal first-use values, stable output cardinalities, verified baseline/candidate build receipts, and clean engine/dataframe trees. The corrected validator output is in validation.log; the validator source is validate_results.py.

The original results JSON SHA-256 before packaging was 99484978b58a6cfb2380be64eba3daad7d0a76aa57080eed2194f5bd20960de1. It matched the local copy at validation time. No benchmark, compiler, or build command was run during this validation step.
