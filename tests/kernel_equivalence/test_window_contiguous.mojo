"""Window layouts: every partition's rows are contiguous and in input order (or in
order-key order, ties in input order), with partition and peer boundaries
marked; and `over()` on them gives each partition what the scalar window
models give it: running sums both ways, shift, average rank, rolling mean,
forward fill and broadcast sums and means, with string and integer keys,
null keys and null values, in many partitions and in one."""
# contract-group: expr.order_windows
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.window_sort import PARTITION_START, PEER_START, window_layout
from tests.support.group_strategy_reference import ScopedEnvironment
from tests.support.window_reference import (
    WindowCells,
    cumulative,
    filled,
    ranked,
    rolling,
    shifted,
)

comptime ROWS = 24_000


def test_layout_groups_rows_by_key_in_row_order_for_any_count() raises:
    for count in [1, 7, 256, 300, 5_000]:
        var ids = List[Int64]()
        for row in range(10_000):
            ids.append(Int64((row * 7919) % count))
        var key = Series("p", Column[Int64](ids.copy()))
        var layout = window_layout(
            [key^], List[Series](), List[Bool](), List[Bool](), 10_000
        )
        assert_equal(layout.count, count)
        assert_equal(layout.starts[0], 0)
        assert_equal(layout.starts[count], 10_000)
        var seen = List[Bool](length=count, fill=False)
        var row_partitions = layout.row_partitions()
        for g in range(count):
            var previous = -1
            var id = ids[layout.order[layout.starts[g]]]
            assert_true(not seen[Int(id)], "a key in two partitions")
            seen[Int(id)] = True
            for k in range(layout.starts[g], layout.starts[g + 1]):
                var row = layout.order[k]
                assert_equal(ids[row], id)
                assert_true(row > previous, "rows out of order")
                previous = row
                assert_equal(row_partitions[row], g)
                if not layout.in_place:
                    assert_equal(layout.inverse[row], k)
                assert_equal(
                    (layout.flags[k] & PARTITION_START) != 0,
                    k == layout.starts[g],
                )


def test_layout_orders_rows_by_two_keys_and_marks_peers() raises:
    var parts = List[Int64]()
    var a = List[Int64]()
    var b = List[String]()
    for row in range(6_000):
        parts.append(Int64(row % 13))
        a.append(Int64((row * 31) % 7))
        b.append(String((row * 17) % 5))
    var p = Series("p", Column[Int64](parts.copy()))
    var sa = Series("a", Column[Int64](a.copy()))
    var sb = Series("b", Column[String](b.copy()))
    var layout = window_layout(
        [p^], [sa^, sb^], [True, True], [False, False], 6_000
    )
    for g in range(layout.count):
        for k in range(layout.starts[g] + 1, layout.starts[g + 1]):
            var before = layout.order[k - 1]
            var row = layout.order[k]
            assert_equal(parts[row], parts[before])
            var same = a[row] == a[before] and b[row] == b[before]
            assert_true(
                a[row] < a[before]
                or (a[row] == a[before] and b[row] <= b[before]),
                "not sorted descending",
            )
            if same:
                assert_true(row > before, "ties out of input order")
            assert_equal((layout.flags[k] & PEER_START) != 0, not same)


def text_key(row: Int) -> String:
    return "partition " + String((row * 31) % 301)


def frame() raises -> DataFrame:
    var x = List[Int64]()
    var xv = List[Bool]()
    var k = List[String]()
    var kv = List[Bool]()
    var j = List[Int64]()
    var same = List[Int64]()
    for row in range(ROWS):
        x.append(Int64((row * 7 + 3) % 13 - 6))
        xv.append(row % 11 != 4)
        k.append(text_key(row))
        kv.append(row % 301 != 17)
        j.append(Int64(row % 2))
        same.append(0)
    return DataFrame(
        [
            Series("x", Column[Int64](x^, xv)),
            Series("k", Column[String](k^, kv)),
            Series("j", Column[Int64](j^)),
            Series("same", Column[Int64](same^)),
        ]
    )


def key_of(row: Int) -> String:
    var text = text_key(row) if row % 301 != 17 else String("null")
    return text + "|" + String(row % 2)


def test_over_on_contiguous_partitions_matches_the_scalar_models() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var input = frame()
    var keys: List[String] = ["k", "j"]
    var actual = input.select_exprs(
        [
            col("x").cum_sum().over(keys).alias("sum"),
            col("x").cum_sum(True).over(keys).alias("reverse"),
            col("x").shift().over(keys).alias("shift"),
            col("x").rank().over(keys).alias("rank"),
            col("x").rolling_mean(2, 1).over(keys).alias("rolling"),
            col("x").forward_fill().over(keys).alias("fill"),
            col("x").sum().over(keys).alias("total"),
            col("x").mean().over(keys).alias("mean"),
        ]
    )
    var members = Dict[String, List[Int]]()
    for row in range(ROWS):
        var key = key_of(row)
        if key not in members:
            members[key] = List[Int]()
        members[key].append(row)
    ref x = input.column("x")._data[Column[Int64]]
    for entry in members.items():
        ref rows = entry.value
        var values = List[Float64]()
        var valid = List[Bool]()
        var total = Int64(0)
        var counted = 0
        for row in rows:
            values.append(Float64(x._get(row)))
            valid.append(x._valid(row))
            if x._valid(row):
                total += x._get(row)
                counted += 1
        var cells = WindowCells(values, valid)
        var expected: List[WindowCells] = [
            cumulative(cells, "sum", False),
            cumulative(cells, "sum", True),
            shifted(cells, 1),
            ranked(cells, "average", False),
            rolling(cells, 2, 1, "mean"),
            filled(cells, False, -1),
        ]
        var names: List[String] = [
            "sum",
            "reverse",
            "shift",
            "rank",
            "rolling",
            "fill",
        ]
        for c in range(len(names)):
            var restored = actual.column(names[c]).take(rows)
            for r in range(len(rows)):
                var value = restored.get(r)
                assert_equal(value.is_null(), not expected[c].valid[r])
                if expected[c].valid[r]:
                    var got = value.float64() if (
                        c == 3 or c == 4
                    ) else Float64(value.int64())
                    assert_true(
                        abs(got - expected[c].values[r]) < 1e-9,
                        names[c] + " of " + entry.key,
                    )
        for row in rows:
            assert_equal(actual.item(row, "total").int64(), total)
            if counted > 0:
                assert_true(
                    abs(
                        actual.item(row, "mean").float64()
                        - Float64(total) / Float64(counted)
                    )
                    < 1e-9
                )
    _ = trace^
    _ = workers^


def test_one_partition_runs_over_every_row() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var input = frame()
    var actual = input.select_exprs([col("x").cum_sum().over("same")])
    ref x = input.column("x")._data[Column[Int64]]
    var running = Int64(0)
    for row in range(ROWS):
        var value = actual.item(row, "x")
        if not x._valid(row):
            assert_true(value.is_null())
            continue
        running += x._get(row)
        assert_equal(value.int64(), running)
    _ = trace^
    _ = workers^


def test_rows_already_in_partition_order_restart_at_each_key() raises:
    # A frame sorted by its key: running sums restart at each key.
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var x = List[Int64]()
    var key = List[Int64]()
    for row in range(ROWS):
        x.append(Int64(row % 9 - 4))
        key.append(Int64(row // 40))
    var input = DataFrame(
        [Series("x", Column[Int64](x^)), Series("key", Column[Int64](key^))]
    )
    var actual = input.select_exprs([col("x").cum_sum().over("key")])
    var running = Int64(0)
    for row in range(ROWS):
        if row % 40 == 0:
            running = 0
        running += Int64(row % 9 - 4)
        assert_equal(actual.item(row, "x").int64(), running)
    _ = trace^
    _ = workers^
