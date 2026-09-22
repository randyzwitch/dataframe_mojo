# Diagnostic phase timings (instrumented, not benchmark scores)

Temporary snapshots based on f568d61 (old builder) and the active source-port
reader (new builder) added monotonic timings around CountLines, submit,
decode, drain/release, collection and concat. Each job adds its decode time
and count through atomics. Sources remained outside the production tree at
/tmp/csv-builder-{old,new}-profile. Both used the scoped shared-queue pool.
Same one-million-row fixture, seven timed iterations, 32 workers, no concurrent
compiler/test processes. The first phase row in each file is warmup and is
excluded. Each phase row repeats labels before its values.

All reads produced exactly 513 jobs. Medians in milliseconds:

| Projected read | Old builder | New builder |
|---|---:|---:|
| Scan | 7.83 | 8.12 |
| Submit | 0.85 | 8.62 |
| Aggregate worker decode (overlapping) | 312.69 | 181.11 |
| Drain/release | 2.61 | 0.87 |
| Collect | 2.58 | 2.55 |
| Concat | 0.90 | 0.88 |
| Total mapped read | 16.45 | 22.30 |

Faster decode coincides with much slower producer submission through the
single shared mutex. New projected submit samples range 8.36–10.24 ms;
the same case reproduces its uninstrumented ~22.4 ms wall time. This supports
queue contention as the projected-read regression, rather than a slower
string decoder. It is not proof that every remaining difference is scheduling.

Instrumentation perturbs scheduling: new full-read submit is bimodal and
instrumented full wall is ~22.2 ms versus ~17 ms uninstrumented. Per-job clocks
and contended atomics prohibit attributing exact percentage costs from these
numbers. Worker decode totals overlap and must not be added to wall phases.
Polars' persistent Rayon scope/work-stealing queues remain an explicit runtime
difference. The isolated Rayon benchmark had mixed results; see ../rayon-isolated.

Snapshot clarification: `new` includes builder alignment, typed integer
frontends, trusted internal spans, and the isolated scanner-inline/slow-outline
float candidate (before Array tables). `old` is f568d61. Thus these phases
compare cumulative parser changes, not the builder change in isolation.
Neither profile is the final static-table-only parser. The separate builder
benchmark in ../builder-and-splitter isolates builder state.

The later `profile-rayon` traces use the earlier builder-only parser, with
checked spans and original integer/float frontends. They must not be compared
against the new scoped profile as a scheduler-only A/B test. The original
uninstrumented ../rayon-isolated comparison did use the same builder-only
parser on both schedulers. These distinctions matter when assigning costs.
