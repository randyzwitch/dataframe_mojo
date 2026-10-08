# As-of joins

`DataFrame.join_asof` matches each left row to an ordered right key, optionally
within groups. It preserves left row order and cardinality. For example, attach
the most recent quote to each trade:

```mojo
var matched = trades.join_asof(
    quotes, on="time", by="symbol", strategy="backward", tolerance=500
)
```

`on` is the shared key name; use `left_on` and `right_on` when names differ.
`by` accepts a column name, a list of names, or `None` (the default is no
grouping). Both inputs must be sorted ascending by the ordered key within each
group. Groups may be interleaved. The check ignores null ordered keys and raises
an error naming the side, column and offending row when ordering is violated.

Keys must have identical integer, float, Date, Datetime or Duration dtypes.
Temporal units and time zones must agree. Key comparisons retain the full
integer precision, including UInt64 values above Int64.MAX. Distances cannot
overflow a signed 64-bit key.

| Strategy | Match |
| --- | --- |
| `backward` (default) | Last right key less than or equal to the left key |
| `forward` | First right key greater than or equal to the left key |
| `nearest` | Closest right key; ties prefer the later key and last adjacent duplicate |

`allow_exact_matches=False` excludes equal keys. Null ordered keys never match,
and a null in any grouping key prevents a match. Nearest follows Polars' scan
and duplicate-run behavior, including the effect of null positions in the
right input. Unmatched right values are null.

`tolerance` is inclusive and nonnegative. Plain integers and floats use the
ordered key's physical unit: days for Date and the declared unit for Datetime
or Duration. `None` imposes no limit. For temporal keys, a Duration scalar is
converted to the key's unit without floating-point rounding:

```mojo
from dataframe import AnyValue, DataType

var matched = trades.join_asof(
    quotes, on="time", by=["symbol"],
    tolerance=AnyValue.temporal(DataType.duration("ms"), 500),
)
```

Use `AnyValue(UInt64(...))` for an explicitly typed unsigned numeric tolerance.
A fractional temporal tolerance smaller than one key tick admits only an exact
match. Negative or NaN tolerances raise.

Output contains the left columns, then the right payload columns. Common
ordered/grouping keys appear once. Differently named right ordered keys remain
in the output. Colliding right names gain `suffix` (default `_right`); a suffix
that still produces duplicate names raises.

`LazyFrame.join_asof` accepts the same arguments. Sortedness is checked when
the plan executes, including when streaming collection is requested. This
operator materializes its inputs and performs a complete merge scan; filters
and slices are not pushed across it because that can change nearest matches
or hide unsorted input. Hash grouping and ordered scans take linear work in
the input row counts, with linear auxiliary storage.

## Validation and measurement

`pixi run -e oracle oracle-asof` compares eager Arrow interchange and lazy
Parquet plans against Polars. It covers all supported numeric key dtypes,
Date/Datetime/Duration units and a time zone, all strategies, exact-match
settings, grouped/null group keys, tolerances, duplicates, null ordered keys,
integer extremes and nonfinite floats. Build `libdfparquet` before running the
lazy oracle. `tests/test_asof.mojo` additionally verifies sortedness errors,
schema/name checks and optimization boundaries.

Run the issue's 10M-left/1M-right development measurement with:

```sh
pixi run -e oracle python3 scripts/bench_asof.py
```

The two engines load identical Parquet files outside the timer and materialize
the complete result. Every output cell is compared. The report includes raw
timings, thread count, revisions, seed and machine details for duplicate-heavy,
nullable and grouped variants. This is a mechanism benchmark, not an
external-suite performance claim.

### Recorded 10M / 1M run

Engine revision `ead59c8abefade3a4c787aec91a0bab0ce9569d4`, Polars 1.44.2, four threads, seed 224,
one warmup and three measured runs. All nine complete results matched Polars.
Times below are medians in milliseconds; ratios are Mojo / Polars.

| Variant | Strategy | Mojo ms | Polars ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| base | backward | 312.0 | 103.4 | 3.02× |
| base | forward | 313.1 | 73.6 | 4.25× |
| base | nearest | 442.2 | 96.1 | 4.60× |
| nulls | backward | 305.1 | 128.7 | 2.37× |
| nulls | forward | 306.8 | 120.0 | 2.56× |
| nulls | nearest | 434.9 | 198.3 | 2.19× |
| grouped | backward | 420.5 | 122.5 | 3.43× |
| grouped | forward | 416.7 | 97.3 | 4.28× |
| grouped | nearest | 548.5 | 138.0 | 3.97× |

[Raw measurements and machine information](asof-benchmark-results.json) include
all repetitions. This was a development workstation run with other compilation
work active, not an isolated performance baseline. The nullable variant checks
non-null key ordering outside Polars' timer and disables its sortedness check,
which otherwise rejects interspersed nulls. Mojo validates ordering during every
join. Polars cannot validate grouped sortedness; Mojo does. These differences
and auxiliary allocation are included in the measured operator times.
