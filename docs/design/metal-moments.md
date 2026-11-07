# Metal variance and standard deviation

Direct Metal completion computes mathematical variance from exact first and
second moments. It preserves the statistical API's input conversion:
Float32 values are represented exactly, Int32 converts exactly to Float64,
and Int64 first rounds to Float64 with round-to-nearest-even. Integer
arithmetic and explicit casts are evaluated in their declared types before
that statistical conversion. Float64 input columns remain unsupported.

For a valid count `n` and nonnegative `ddof < n`, the returned variance and
standard deviation have relative error at most `4u`, where `u = 2**-53`.
Exact zero variance returns zero. The guarantee concerns the converted
per-row values, including the existing Float32 expression rounding rules.
It does not require reproducing the CPU's accumulation order. CPU and Metal
must both satisfy the shared accuracy contract and agree within its tolerance.
A material discrepancy is a correctness defect and blocks release; returning
an inaccurate CPU result through fallback does not satisfy that requirement.
For example, the population variance of `[2**60, 2**60 + 256]` is 16,384.
Both inputs are exactly representable in Float64. Conversely,
`[2**60 + 1, 2**60 + 2]` converts to two identical Float64 values, so its
statistical variance is zero under this API's conversion rule.

Empty inputs and `n <= ddof` return typed Float64 nulls. Negative `ddof`,
unsupported expressions and unsupported input types remain on CPU. Non-finite
values and checked expression faults use CPU adjudication, preserving the
existing null and error rules. Such completion is recorded as CPU adjudication
and is outside the direct-GPU numerical bound.

## Exact accumulation

A finite Float32 value is an integer coefficient times `2**-149`. Nine
radix-`2**32` bins represent its signed first moment; eighteen represent its
nonnegative square. Integer statistical inputs use integer units after their
Float64 conversion. First moments and squared moments merge exactly on the
host, independently of GPU partitioning, group order and input chunks.
Narrow windows are an optimization: device flags request a wider retry when
an input falls outside them.

The finisher forms `N = n * sum(x*x) - sum(x)**2` with twenty UInt32 limbs.
Both products and their subtraction are exact, so cancellation does not
remove significant bits. With at most `UInt32.MAX` contributions, the largest
Float32 coefficient needs at most 277 bits, its square at most 554 bits,
and either product at most 618 bits. Twenty limbs provide 640 bits.
Public execution also requires fewer than `2**31` source rows, matching the
existing row-ticket limit.

## Float64 finishing bound

Let `e` be zero for integer units and -298 for squared Float32 units, and
`D = n * (n - ddof)`. The exact positive variance is `v = N * 2**e / D`.
`D` is formed exactly in Int128 and is less than `2**64`.

The implementation rounds `N` once to a 53-bit significand using guard and
sticky bits and ties-to-even, then applies an exact power-of-two scale.
It rounds `D` once to Float64 and performs one Float64 division. All positive
results and these intermediate values are normal and finite: the smallest
possible positive variance is greater than `2**-362`, and the largest is
less than `2**288` across the supported nonnegative `ddof` range. The scaled
numerator is less than `2**320`. There is no overflow or gradual underflow in
these Float64 finishing operations.

For each rounded operation, write its relative error as a delta with magnitude
at most `u`. Then

```
computed_v / v = (1 + delta_N) * (1 + delta_div) / (1 + delta_D)
```

The upper relative error is at most `(3u + u*u) / (1-u)`, which is less than
`4u`; the lower relative error is smaller. A zero numerator stays exactly zero.
This bound assumes the ordinary round-to-nearest Float64 operations used by
the supported compiler configuration, without reassociation or approximate
reciprocal substitution.

STD applies one rounded Float64 square root. Its input has relative error
less than `4u`. For the upper error, `sqrt(1+t) <= 1+t/2` gives an error less
than `3u + 2u*u`, hence less than `4u`. For the lower error,
`1-sqrt(1-t) = t/(1+sqrt(1-t))`; here `t < 4u` and the denominator exceeds
`3/2`. Including the final square-root rounding gives a relative error less
than `8u/3 + u`, also less than `4u`.

## Validation and placement

The independent finishing oracle uses exact Python Fraction variances and
500-digit Decimal square roots. Its 8,704 fixtures cover 1,218 distributions:
all finite Float32 exponent classes, adjacent large values, cancellation,
subnormals, integer conversion, unequal weights, counter boundaries, and
`ddof` values at and beyond the valid count. It checks the relative bound,
typed nulls, and distance from independently rounded Float64 references.
Run the independent checker without an Apple device:

```sh
pixi run python scripts/check_metal_moment_accuracy.py
```

It retains its generator output, reference fractions, actual results,
compiler logs, strict Float64 IR and hashes under
`build/metal-moment-accuracy/`. GPU and public-execution contracts separately
validate accumulation, retries and provider routing.

Moment calibration keys include `moments:v1:exact:`, logical operations and
each output's `ddof`, as well as expression, predicate, grouping and physical
input properties. Layout matching uses normalized native inputs while the
logical plan retains VAR/STD. Unmeasured shapes remain on CPU. Calibration
must use independent mathematical references; incorrect CPU answers are not
valid placement-training measurements.

Release validation must check both engines against the independent reference
and check their agreement within the shared tolerance, including large-offset
and cancellation cases. The observed CPU result of 32,768 for the example
whose exact population variance is 16,384 is an unresolved correctness
blocker, not an accepted numerical policy. An independent Linux CPU
reproduction at main `12f1d93c` confirms the defect in all 72 combinations
of 2/128/65,536 rows, nullable/nonnullable inputs, contiguous/offset/chunked
storage, global/grouped aggregation and `ddof` 0/1. Variance relative errors
range from 0.999267578125 to 1.0; standard-deviation errors range from
0.41395458842389976 to 0.41421356237309515. These are correctness failures,
not timing measurements. Apple CPU confirmation and retesting after a CPU
fix remain required. The Metal work does not modify the CPU implementation.

DuckDB 2.0.0's built-in `var_pop` also returns 32,768 for the two-value
example, matching the CPU result. This is a shared numerical limitation,
not a CPU-only discrepancy against that oracle. DuckDB returns 16,384 after
subtracting the common `2**60` offset before aggregation. Independently,
forming `(n*sum(x*x)-sum(x)**2)/(n*n)` with DuckDB HUGEINT arithmetic gives
exactly 16,384, agreeing with a Python Fraction calculation. DuckDB
compatibility and mathematical accuracy therefore need separate checks for
these stress cases; built-in floating aggregates alone are insufficient as
the numerical oracle.

Run the CPU regression from a source checkout (it currently exits nonzero):

```sh
mojo run -O3 --target-accelerator=unknown -I providers/cpu -I . \
  scripts/check_metal_moment_cpu_accuracy.mojo
```
