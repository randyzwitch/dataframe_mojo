"""Time windows and as-of joins run as rounds of the pipeline executor on
several workers, with the rows of independent brute-force references."""
# contract-group: planner.routing
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, col
from tests.support.asof_reference import reference_asof, assert_asof_rows
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime SECOND = 1_000_000


def _events(rows: Int) raises -> DataFrame:
    """Times out of order with repeats, in three groups, with values."""
    var times = List[Int64]()
    var groups = List[Int64]()
    var values = List[Int64]()
    for row in range(rows):
        times.append(Int64((row * 7919) % (rows // 2)) * SECOND)
        groups.append(Int64(row % 3))
        values.append(Int64(row % 17))
    return DataFrame(
        [
            Series("t", Column[Int64](times^)).with_dtype(
                DataType.datetime("us")
            ),
            Series("g", Column[Int64](groups^)),
            Series("x", Column[Int64](values^)),
        ]
    )


def test_eager_rolling_matches_brute_force_windows_on_several_workers() raises:
    var rows = 12000
    var frame = _events(rows)
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var result = frame.rolling("t", "20s", group_by=["g"]).agg(
        [col("x").sum().alias("s"), col("x").len().alias("n")]
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true("time_window.rounds" in paths)
    assert_equal(result.columns(), ["g", "t", "s", "n"])
    assert_equal(result.height(), rows)
    # Groups in order of first appearance, rows in input order within
    # each; a row's window is its group's rows in (t - 20s, t].
    var column = frame.column("t").cast(DataType.INT64)
    var times = List[Int64]()
    var values = List[Int64]()
    for row in range(rows):
        times.append(column.get(row).int64())
        values.append(frame.column("x").get(row).int64())
    var at = 0
    for g in range(3):
        for row in range(g, rows, 3):
            var end = times[row]
            var total = Int64(0)
            var count = Int64(0)
            for other in range(g, rows, 3):
                if times[other] > end - 20 * SECOND and times[other] <= end:
                    total += values[other]
                    count += 1
            assert_equal(result.item(at, "g").int64(), Int64(g))
            assert_equal(result.item(at, "s").int64(), total)
            assert_equal(result.item(at, "n").int64(), count)
            at += 1


def test_eager_dynamic_windows_match_brute_force_on_several_workers() raises:
    var rows = 12000
    var frame = _events(rows)
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var result = frame.group_by_dynamic("t", "10s").agg(
        [col("x").sum().alias("s")]
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true("time_window.rounds" in paths)
    # Every 10 s window [start, start + 10s) holding some row, in time
    # order.
    var span = rows // 2
    var totals = List[Int64](length=span // 10 + 1, fill=0)
    var seen = List[Bool](length=span // 10 + 1, fill=False)
    var times = frame.column("t").cast(DataType.INT64)
    for row in range(rows):
        var w = Int(times.get(row).int64() // (10 * SECOND))
        totals[w] += frame.column("x").get(row).int64()
        seen[w] = True
    var at = 0
    var labels = result.column("t").cast(DataType.INT64)
    for w in range(len(totals)):
        if not seen[w]:
            continue
        assert_equal(labels.get(at).int64(), Int64(w) * 10 * SECOND)
        assert_equal(result.item(at, "s").int64(), totals[w])
        at += 1
    assert_equal(result.height(), at)


def test_eager_asof_matches_brute_force_on_several_workers() raises:
    var rows = 12000
    var lk = List[Int64]()
    var lg = List[Int64]()
    var lv = List[Bool]()
    for row in range(rows):
        lk.append(Int64((row * 7919) % 5000))
        lg.append(Int64(row % 4))
        lv.append(row % 11 != 5)
    var rk = List[Int64]()
    var rg = List[Int64]()
    var rv = List[Bool]()
    var payload = List[Int64]()
    for row in range(900):
        rk.append(Int64((row * 37) % 5100))
        rg.append(Int64(row % 3))
        rv.append(row % 13 != 4)
        payload.append(Int64(row))
    var left = DataFrame(
        [
            Series("k", Column[Int64](lk.copy(), lv.copy())),
            Series("g", Column[Int64](lg.copy())),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int64](rk.copy(), rv.copy())),
            Series("g", Column[Int64](rg.copy())),
            Series("p", Column[Int64](payload^)),
        ]
    )
    var left_groups = List[Int]()
    for g in lg:
        left_groups.append(Int(g))
    var right_groups = List[Int]()
    for g in rg:
        right_groups.append(Int(g))
    for strategy in ["backward", "forward"]:
        for exact in [False, True]:
            var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
            var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
            var trace = StrategyTrace()
            var actual = left.join_asof(
                right,
                on="k",
                by=["g"],
                strategy=strategy,
                allow_exact_matches=exact,
            )
            var paths = trace.text()
            _ = trace^
            _ = setting^
            _ = threads^
            assert_true("asof.rounds" in paths)
            var selected = reference_asof[DType.int64](
                lk, rk, lv, rv, left_groups, right_groups, strategy, exact
            )
            assert_asof_rows(
                actual, left, right, selected, ["k", "g", "p"], ["p"]
            )
