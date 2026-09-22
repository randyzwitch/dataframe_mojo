# Compiled projection indices: paired check

Before: `19bbe5b` code (`/tmp/csv-bench-source-port`). After: decoder compiles
its projection mask into selected indices once per chunk and caches the
separator byte before the row loop, matching Polars parse_lines inputs.
Decoder tests: 9/9 pass. Same one-million-row input and harness as the initial
source-port report; seven samples, warm cache, sequential processes, no agent
builds/tests running. All raw samples retained here.

| Read | Threads | Before median ms | After median ms |
|---|---:|---:|---:|
| Full | 1 | 384.08 | 377.33 |
| Projected | 1 | 290.33 | 273.59 |
| Full | 32 | 20.81 | 20.49 |
| Projected | 32 | 16.90 | 16.95 |

The projected single-thread run improved about 5.8%; parallel results show no
material change. This is one workload and one paired run, not a general claim.
