"""Window partitions laid out contiguously (DuckDB's hash groups) and
evaluated many at once: the layout is a stable sort of rows by partition id
for any number of partitions, and `over()` on it gives each partition what
the scalar window models give it: running sums both ways, shift, average
rank, rolling mean, forward fill and broadcast sums and means, with string
and integer keys, null keys and null values. Rows already in partition
order are evaluated where they are; rows in a single partition keep the
row-order route."""
# contract-group: expr.order_windows
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.window_layout import partition_layout
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)
from tests.support.window_reference import (
    WindowCells,
    cumulative,
    filled,
    ranked,
    rolling,
    shifted,
)

comptime ROWS = 24_000


def test_layout_groups_rows_by_id_in_row_order_for_any_count() raises:
    for count in [1, 7, 256, 300, 5_000]:
        var ids = List[Int]()
        for row in range(10_000):
            ids.append((row * 7919 + row // 3) % count)
        var layout = partition_layout(ids.copy(), count, 8)
        assert_equal(layout.starts[0], 0)
        assert_equal(layout.starts[count], 10_000)
        for g in range(count):
            var previous = -1
            for k in range(layout.starts[g], layout.starts[g + 1]):
                var row = layout.order[k]
                assert_equal(ids[row], g)
                assert_true(row > previous, "rows out of order")
                previous = row
                assert_equal(layout.inverse[row], k)


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
            Series("x", Column[Int64](x^, xv^)),
            Series("k", Column[String](k^, kv^)),
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
    var observed = StrategyTrace()
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
    assert_true(
        "dataframe-path: window.contiguous\n" in observed.text(),
        "over() did not lay out its partitions contiguously",
    )
    _ = observed^
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
        var cells = WindowCells(values^, valid^)
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


def test_one_partition_keeps_the_row_order_route() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var input = frame()
    var observed = StrategyTrace()
    var actual = input.select_exprs([col("x").cum_sum().over("same")])
    assert_true(
        "dataframe-path: window.contiguous\n" not in observed.text(),
        "a single partition was laid out",
    )
    _ = observed^
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


def test_rows_already_in_partition_order_are_evaluated_in_place() raises:
    # Ids that rise with the rows lay out as the rows themselves.
    var grouped = List[Int]()
    var scattered = List[Int]()
    for row in range(10_000):
        grouped.append(row // 40)
        scattered.append((row * 7) % 250)
    assert_true(partition_layout(grouped^, 250, 8).in_place)
    assert_true(not partition_layout(scattered^, 250, 8).in_place)
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
    var observed = StrategyTrace()
    var actual = input.select_exprs([col("x").cum_sum().over("key")])
    assert_true("dataframe-path: window.contiguous\n" in observed.text())
    _ = observed^
    var running = Int64(0)
    for row in range(ROWS):
        if row % 40 == 0:
            running = 0
        running += Int64(row % 9 - 4)
        assert_equal(actual.item(row, "x").int64(), running)
    _ = trace^
    _ = workers^
