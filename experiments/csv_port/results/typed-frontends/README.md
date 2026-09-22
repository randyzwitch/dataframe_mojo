# Typed integer frontend and trusted internal spans

Same fixture, environment and seven-sample sequential warm-cache methodology
as ../initial-source-port. Each variant is built before timing begins; no
compiler/test jobs run during measurements. Times below are medians in ms.

`builder-final` is 1fc4be0: builder alignment, original integer frontend,
checked spans. `integer-typed` adds compile-time width selection and typed
integer return values. `trusted-spans` instead adds only the internal iterator
span/projection accesses corresponding to Rust get_unchecked. `typed-trusted`
combines both. The combined correctness checks pass: decoder 9, splitter 9,
buffers 5, lazy validity 4, integers 3 default + 3 fallback, reader 7,
and actual Polars output comparisons 24.

| Variant | Full 1T | Projected 1T | Full 32T | Projected 32T |
|---|---:|---:|---:|---:|
| Builder baseline | 272.02 | 171.84 | 16.92 | 22.27 |
| Typed integer | 274.86 | 163.55 | 16.66 | 22.08 |
| Trusted spans alone | 293.04 | 174.50 | 17.23 | 22.89 |
| Both | 261.18 | 144.86 | 17.59 | 22.21 |

The effects are not additive: trusted spans alone regresses; the combined
code is faster on one thread. The projected 32-thread builder regression
is unresolved. Do not infer that deleting any individual bounds check will
improve performance. Checked externally-constructed field spans remain intact;
only iterator-proven ranges and length-proven buffer indices use unchecked
access. There is no CSV syntax-validation bypass.

A second round reproduced the combined single-thread projection result
(145.48 ms). Full reads varied to 253.77 ms / 19.07 ms at 1 / 32 threads.

An isolated float experiment inlined the scanner and outlined unchanged slow
fallbacks, shrinking the public Float64 stack frame from 1728 to 864 bytes.
It passed all 419 expected IEEE bit patterns and the 24 differential cases.
In a paired run it changed full 1T 253.77 to 246.30 ms, but projected 1T
145.48 to 159.66 ms, despite Float64 not being projected. Full 32T was 19.07
to 18.71 ms and projected 32T 22.30 to 22.38 ms. This is mixed evidence,
not justification for claiming a general float optimization.
Raw `recheck-float-inline` samples retain this experiment; the production
status of that candidate must be determined from the branch, not these samples.
