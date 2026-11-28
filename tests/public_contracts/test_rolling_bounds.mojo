"""Duration windows of rolling_*_by hold exactly the rows a direct scan of
the times finds, for every closed side, repeated times and calendar
periods."""
# contract-group: expr.temporal
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col


def _days(n: Int) -> List[Int64]:
    """Days with repeats and gaps, out of order."""
    var out = List[Int64]()
    var state = UInt64(99)
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        out.append(Int64((state >> 33) % UInt64(n * 2)))
    return out^


def _shift_months(day: Int64, months: Int) -> Int64:
    """`day` (days since 1970-01-01) moved by whole months, clamped to the
    target month's last day."""
    var z = day + 719468
    var era = (z if z >= 0 else z - 146096) // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var y = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var d = doy - (153 * mp + 2) // 5 + 1
    var m = mp + 3 if mp < 10 else mp - 9
    if m <= 2:
        y += 1
    var total = y * 12 + (m - 1) + Int64(months)
    var ny = total // 12
    var nm = total % 12 + 1
    var leap = (ny % 4 == 0 and ny % 100 != 0) or ny % 400 == 0
    var lengths: List[Int64] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    var last = lengths[Int(nm - 1)] + Int64(1 if nm == 2 and leap else 0)
    var nd = min(d, last)
    var yy = ny - 1 if nm <= 2 else ny
    var era2 = (yy if yy >= 0 else yy - 399) // 400
    var yoe2 = yy - era2 * 400
    var mp2 = nm - 3 if nm > 2 else nm + 9
    var doy2 = (153 * mp2 + 2) // 5 + nd - 1
    var doe2 = yoe2 * 365 + yoe2 // 4 - yoe2 // 100 + doy2
    return era2 * 146097 + doe2 - 719468


def test_rolling_counts_by_duration_match_a_scan_of_the_times() raises:
    var n = 300
    var days = _days(n)
    var ones = List[Int64](length=n, fill=1)
    var frame = DataFrame(
        [
            Series("t", Column[Int64](days.copy())).with_dtype(DataType.DATE),
            Series("one", Column[Int64](ones^)),
        ]
    )
    var closeds: List[String] = ["right", "left", "both", "none"]
    for closed in closeds:
        for period in [1, 3, 40]:
            var exprs: List[Expr] = [
                col("one")
                .rolling_sum_by("t", String(period) + "d", closed, 0)
                .alias("days"),
                col("one").rolling_sum_by("t", "1mo", closed, 0).alias("month"),
            ]
            var eager = frame.select_exprs(exprs)
            var lazy = frame.lazy().select_exprs(exprs).collect()
            var left = closed == "left" or closed == "both"
            var right = closed == "right" or closed == "both"
            for result in [eager.copy(), lazy^]:
                for row in range(n):
                    var t = days[row]
                    for name in ["days", "month"]:
                        var start = t - Int64(period) if name == "days" else (
                            _shift_months(t, -1)
                        )
                        var count = 0
                        for k in range(n):
                            var x = days[k]
                            var lo_ok = x >= start if left else x > start
                            var hi_ok = x <= t if right else x < t
                            if lo_ok and hi_ok:
                                count += 1
                        assert_equal(
                            result.item(row, name).int64(),
                            Int64(count),
                            closed + " " + name + " row " + String(row),
                        )
