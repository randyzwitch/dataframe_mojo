# Time windows

Row-count moments use a sliding Welford calculation with removal, translated
by a finite input value to preserve small variance around a large common
offset. Each value enters and leaves the state once. Null values do not count
toward `min_samples` or `ddof`.

```mojo
col("price").rolling_std(20, min_samples=5, ddof=1)
col("price").rolling_var(20, ddof=0)
```

`min_samples` defaults to the window size for row-count windows. A window with
at most `ddof` non-null samples produces null. The result is Float64.

For irregular sampling, the six `rolling_*_by` expressions use a sorted Date
or Datetime index. The `by` argument accepts a name or expression.

```mojo
col("price").rolling_mean_by("time", "1h30m")
col("price").rolling_std_by(col("time"), "7d", ddof=1)
col("price").rolling_sum_by("time", "1mo").over("symbol")
```

Available functions are sum, mean, min, max, std and var. `min_samples` defaults
to one; `closed="right"` means `(time - window_size, time]`. `left`, `both` and
`none` select the other endpoint combinations. All rows with the same timestamp
see the same window, including duplicate timestamps later in input order.
A null index or descending index within a partition raises. Lazy expressions
perform these checks when collected.

Durations use `ns`, `us`, `ms`, `s`, `m`, `h`, `d`, `w`, `mo`, `q` and `y`, including
combinations such as `1h30m`. Months and years use the Gregorian calendar and
clamp to the target month's last day. Calendar days preserve local wall-clock
time across daylight-saving changes; `24h` is always 24 elapsed hours. Duration
units finer than the index's physical resolution raise. This shares the
arithmetic and timezone rules of `dt.offset_by` and `dt.truncate`.

Dataframe time grouping returns a request with `.agg(...)`, accepting one or a
list of scalar aggregate expressions:

```mojo
var monthly = df.group_by_dynamic(
    "time", every="1mo", group_by="symbol"
).agg([col("price").mean(), col("volume").sum()])

var trailing = df.rolling("time", period="7d").agg(
    col("price").std().alias("volatility")
)
```

`group_by_dynamic` aligns the first window to `every`, applies `offset`
(default zero), then advances by `every`. `period` defaults to `every`; longer
periods overlap and shorter periods leave gaps. `closed` defaults to `left`.
`label` selects the left boundary, right boundary or first datapoint. Empty
windows are omitted. Monthly windows include all of February in leap years.

`rolling` produces one window per input timestamp. Its default offset is
negative `period`; an explicit offset changes the interval to
`(time + offset, time + offset + period]`. Empty windows remain, with each
aggregate's normal empty-input result. Both requests accept `group_by` as a
name, a list of names, or `None`; sortedness is checked within each group.

Time grouping gathers overlapping row membership once, then uses the existing
grouped expression reducers. Memory grows with total membership across windows.
Duration expression bounds use binary searches; sum, mean, min and max scan
each window, while std and var reuse sliding moments for monotone bounds.
These eager dataframe requests do not add lazy grouping operators.

Run the independent Polars comparison with `pixi run -e oracle oracle-time-windows`.
It covers irregular and duplicate timestamps, gaps, all endpoint options,
calendar periods, offsets, partitioned windows, time zones and large-offset
variance. Contract tests also check invalid requests and lazy execution.
