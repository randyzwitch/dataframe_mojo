# Polars-relative cached field boundaries

Paired source-alignment check, same initial one-million-row CSV fixture,
seven samples, warm cache, no simultaneous compiler/tests. `projection` is
64c78e7 decoder; `splitter` adds relative cached-end shifts and stores SIMD
separator/quote/EOL broadcasts in the iterator like Polars.

| Read | Threads | Projection baseline ms | Splitter candidate ms |
|---|---:|---:|---:|
| Full | 1 | 373.98 | 355.24 |
| Projected | 1 | 272.52 | 257.77 |
| Full | 32 | 20.53 | 19.54 |
| Projected | 32 | 16.58 | 17.30 |

These measurements indicate a small single-thread gain; projected 32-thread
reads did not improve. The old compiler output already hoisted broadcasts, so
storing SIMD vectors is source alignment, not a claimed broadcast reduction.
Relative-mask handling now follows Polars' trailing-zero/shift sequence.
Splitter tests 9/9 pass, including cached bit boundaries and EOF.
