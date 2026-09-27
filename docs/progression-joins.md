# Progression joins (#269)

> **Update (#308).** Equal-run support described below was removed. Its
> only consumer was the join matrix's `jk // 2` case; the sensor-panel model
> was written afterwards, and one missing reading rejects it. Repeated build
> keys now use the join index. Unit-step IDs and constant-step calendar grids
> keep the progression path, detected by one shared function,
> `int64_progression`. The measurements below are kept for the record.

The arithmetic join accepts matching logical types whose physical storage is
Int64: Int64, Date, Datetime, Duration and Time. Datetime/Duration units must
match; this is not a unit conversion. Null probes do not match, and any null
build key rejects the fast path. Right keys must be ascending with a constant
positive step and equal-length runs (the final run may be shorter). Detection
scans until the first violating key. Chunk boundaries do not change detection
or stable left/right row order.

Sequential-ID dimension tables and complete calendar grids serve the unique
progression case. Equal runs serve a complete timestamp-by-sensor panel:
every minute has one reading from each of four sensors, stored by timestamp.
A join on time alone expands each probe into those four sensor readings.
Missing readings in an interior timestamp or shuffled build rows reject the
path. This is a property of the input, not an assumption made by the join.

## Measurement

Measured 2026-09-26 on the same Threadripper 3970X and Mojo nightly recorded
in [join-cutoffs.md](join-cutoffs.md), with 32 workers. Two rounds reverse
variant order; each process warms once and records five repetitions. Values
below are the mean of per-round minima, in milliseconds. Construction and
output-height/type validation are untimed; the public join and gathering are
timed. [All samples](progression-joins.csv) are retained.

The build domain is constructed independently of random probes; 1/11 of
probes are outside that domain. Lookup uses unit-spaced IDs, calendar uses
minute-spaced microsecond timestamps, and panel uses four sensors/minute.
The missing panel removes one sensor every 101 timestamps; shuffled keeps
the complete data but randomizes row order.

| Rows per input (nominal) | Shape | Baseline | Candidate | Candidate without equal runs |
|---:|---|---:|---:|---:|
| 100k | ID lookup | 0.99 | 1.00 | 0.94 |
| 100k | Calendar | 12.21 | 1.37 | 1.32 |
| 100k | Sensor panel | 16.28 | 8.08 | 16.19 |
| 100k | Missing panel | 16.22 | 15.32 | 15.08 |
| 100k | Shuffled panel | 15.34 | 15.82 | 15.71 |
| 1M | ID lookup | 16.79 | 16.74 | 16.01 |
| 1M | Calendar | 36.37 | 21.21 | 20.24 |
| 1M | Sensor panel | 69.53 | 57.17 | 70.72 |
| 1M | Missing panel | 69.71 | 69.19 | 69.91 |
| 1M | Shuffled panel | 69.63 | 70.48 | 70.37 |

Keep equal-run support: it has a concrete data model and beats the alternative
on that model at both sizes. It is not a general claim about duplicate joins.
The missing/shuffled cases show the fallback cost, including small timing
variation between binaries with the same effective path.

Reproduce against baseline `a2b045c` (PR #288):

```bash
pixi run python3 scripts/bench_progression.py --build
pixi run python3 scripts/bench_progression.py --run --threads 32
```

The driver extracts isolated source copies into `build/progression`; it does
not patch production files. The `no_runs` variant rejects the first repeated
build key but keeps temporal progression support. Tests compare exact public
row order and logical types for contiguous/chunked grids, nulls, negative
values, unmatched probes and all timestamp units; direct helper tests ensure
those cases use the intended path and incompatible logical types reject it.
