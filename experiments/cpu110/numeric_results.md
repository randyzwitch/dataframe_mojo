# Numeric parsing results

## Measured evidence

- The outlined exponent fallback reduced the exponent microbenchmark from the
  original fallback cost (about 6.5x the plain parser) to 30.1--30.7 ms; the
  prior exponent path measured about 53.2 ms.
- The integrated outlined helper kept plain CSV parsing near 22 ms, matching
  the baseline in the paired measurement.
- Generic integer parsing retained a bounded packed path only where its
  width/length gate proves the magnitude is in range. The measured 20M-field
  results are recorded in `integer_swar_results.md`.
- The bounded direct Float32 subset improved its matching short input by
  2--7%, but its intentional fallback was about 35% slower. It is rejected as
  the default path; see `float32_direct_audit.md`.

## Source inspection

The strict long-decimal route remains safe by construction: on the twentieth
digit, the scanner calls the exponent-specialized fallback, which in turn
uses the checked borrowed `_atof` conversion. It does not retain a partially
accumulated mantissa. This covers ambiguous and more-than-19-digit decimals
with the standard conversion's existing behavior.

The packed integer validator operates on raw `UInt8` words and rejects a byte
outside ASCII `0x30..0x39` through its high-bit range test before digit
reduction. Non-ASCII UTF-8 bytes therefore reject as non-decimal rather than
being interpreted as digits. The implementation deliberately has a
little-endian compile-time requirement.

No additional strict-grammar, rounding, or overflow defect was identified by
source inspection. This is inspection evidence only: exact grammar and
rounding confidence comes from the existing float reference tests (including
the exponent corpus) and exhaustive packed-integer byte tests, rather than
from a new benchmark or test run in this task.
