"""The aggregate table's states by address against a scalar model of each
group: products and moments (a `Reducer` of every worker's groups, merged
through each group's chain of states at the finish), and the
order-sensitive reductions (first and last of lists and categoricals, the
position of a minimum or maximum), whose rows are sorted back into input
order before they reduce. Both routes that reach the table are checked
(the eager group-by, and a pipeline's grouped sink after a filter), with
tables that clear (eight workers), grow (two) and a single worker, with
and without maintain_order."""
# contract-group: reduce.grouped
# contract-group: kernel.reduce
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    ListColumn,
    Series,
    col,
    lit,
)
from dataframe.storage.column import _pack_bits
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def group_of(row: Int, groups: Int) -> Int:
    return (row * 7919) % groups


def int_valid(row: Int) -> Bool:
    return row % 7 != 5


def int_value(row: Int) -> Int32:
    # -1, 1 or 2: products stay within Int64 for a few dozen rows.
    var v = (row * 104729) % 3
    return Int32(-1 if v == 0 else v)


def list_valid(row: Int) -> Bool:
    return row % 9 != 4


def label_valid(row: Int) -> Bool:
    return row % 11 != 6


def label(row: Int) -> String:
    return "c" + String((row * 31) % 37)


def frame(rows: Int, groups: Int) raises -> DataFrame:
    var k = List[Int64]()
    var x = List[Int32]()
    var xv = List[Bool]()
    var offsets: List[Int64] = [0]
    var child = List[Int64]()
    var lv = List[Bool]()
    var c = List[String]()
    var cv = List[Bool]()
    var r = List[Int64]()
    for row in range(rows):
        k.append(Int64(group_of(row, groups)) * 1_000_003)
        x.append(int_value(row))
        xv.append(int_valid(row))
        # Each row's list is [row, -row] (null on some rows).
        if list_valid(row):
            child.append(Int64(row))
            child.append(-Int64(row))
        offsets.append(Int64(len(child)))
        lv.append(list_valid(row))
        c.append(label(row) if label_valid(row) else String(""))
        cv.append(label_valid(row))
        r.append(Int64(row))
    return DataFrame(
        [
            Series("k", Column[Int64](k^)),
            Series("x", Column[Int32](x^, xv)),
            Series(
                "l",
                ListColumn(
                    offsets^,
                    Series("child", Column[Int64](child^)),
                    _pack_bits(lv),
                ),
            ),
            Series("c", Column[String](c^, cv)).cast(DataType.CATEGORICAL),
            Series("r", Column[Int64](r^)),
        ]
    )


def reductions() -> List[Expr]:
    return [
        col("x").product().alias("x_product"),
        col("x").arg_min().alias("x_arg_min"),
        col("x").arg_max().alias("x_arg_max"),
        col("l").first().alias("l_first"),
        col("l").last().alias("l_last"),
        col("c").first().alias("c_first"),
        col("c").last().alias("c_last"),
        col("x").sum().alias("x_sum"),
    ]


def group_rows(rows: Int, groups: Int, start: Int) -> List[List[Int]]:
    var out = List[List[Int]](length=groups, fill=List[Int]())
    for row in range(start, rows):
        out[group_of(row, groups)].append(row)
    return out^


def check(given: DataFrame, rows: Int, groups: Int, start: Int) raises:
    # The lists' first elements name their rows; categoricals as strings.
    var actual = given.select_exprs(
        [
            col("k"),
            col("x_product").cast(DataType.INT64),
            col("x_arg_min").cast(DataType.INT64),
            col("x_arg_max").cast(DataType.INT64),
            col("l_first").list().first().alias("l_first"),
            col("l_last").list().first().alias("l_last"),
            col("c_first").cast(DataType.STRING).alias("c_first"),
            col("c_last").cast(DataType.STRING).alias("c_last"),
            col("x_sum").cast(DataType.INT64),
        ]
    ).rechunk()
    var expected = group_rows(rows, groups, start)
    var present = 0
    for g in range(groups):
        if len(expected[g]) > 0:
            present += 1
    assert_equal(actual.height(), present)
    for i in range(actual.height()):
        var g = Int(actual.item(i, "k").int64()) // 1_000_003
        ref members = expected[g]
        var what = " of group " + String(g)
        var product = Int64(1)
        var total = Int64(0)
        var low = -1
        var high = -1
        for p in range(len(members)):
            var row = members[p]
            if not int_valid(row):
                continue
            var v = Int64(int_value(row))
            product *= v
            total += v
            if low < 0 or v < Int64(int_value(members[low])):
                low = p
            if high < 0 or v > Int64(int_value(members[high])):
                high = p
        assert_equal(actual.item(i, "x_product").int64(), product, "product" + what)
        assert_equal(actual.item(i, "x_sum").int64(), total, "sum" + what)
        if low < 0:
            assert_true(actual.item(i, "x_arg_min").is_null(), "arg_min" + what)
            assert_true(actual.item(i, "x_arg_max").is_null(), "arg_max" + what)
        else:
            assert_equal(Int(actual.item(i, "x_arg_min").int64()), low, "arg_min" + what)
            assert_equal(Int(actual.item(i, "x_arg_max").int64()), high, "arg_max" + what)
        var first = members[0]
        var last = members[len(members) - 1]
        for pair in [(first, "l_first"), (last, "l_last")]:
            var value = actual.item(i, pair[1])
            if list_valid(pair[0]):
                assert_equal(value.int64(), Int64(pair[0]), pair[1] + what)
            else:
                assert_true(value.is_null(), pair[1] + what + " should be null")
        for pair in [(first, "c_first"), (last, "c_last")]:
            var value = actual.item(i, pair[1])
            if label_valid(pair[0]):
                assert_equal(value.string(), label(pair[0]), pair[1] + what)
            else:
                assert_true(value.is_null(), pair[1] + what + " should be null")


def eager(threads: Int, rows: Int, groups: Int) raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var source = frame(rows, groups)
    for ordered in [False, True]:
        var observed = StrategyTrace()
        var actual = source.group_by(["k"], maintain_order=ordered).agg(
            reductions()
        )
        var text = observed.text()
        assert_true(
            "dataframe-path: group_by.aggregate_table\n" in text,
            "states by address did not reach the aggregate table",
        )
        assert_true(
            "dataframe-path: aggregate_table.state_by_address\n" in text,
            "no state by address",
        )
        assert_true(
            "dataframe-path: aggregate_table.ordered_state\n" in text,
            "no ordered state",
        )
        _ = observed^
        check(actual, rows, groups, 0)
        if ordered:
            # First-occurrence order: rows 0 to groups - 1 each start a
            # group (7919 is prime to every group count here).
            var keys = actual.rechunk()
            for i in range(keys.height()):
                assert_equal(
                    Int(keys.item(i, "k").int64()) // 1_000_003,
                    group_of(i, groups),
                    "maintain_order",
                )
    _ = trace^
    _ = workers^


def test_eager_states_by_address_with_tables_that_clear() raises:
    eager(8, 60_000, 8_000)


def test_eager_states_by_address_with_tables_that_grow() raises:
    eager(2, 60_000, 3_000)


def test_eager_states_by_address_on_one_worker() raises:
    eager(1, 20_000, 700)


def test_pipeline_states_by_address_after_a_filter() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 120_000
    var groups = 9_000
    var start = 1_234
    var observed = StrategyTrace()
    var actual = (
        frame(rows, groups)
        .lazy()
        .filter(col("r") >= lit(Int64(start)))
        .group_by(["k"])
        .agg(reductions())
        .collect()
    )
    assert_true(
        "dataframe-path: pipeline.group_table\n" in observed.text(),
        "states by address did not reach the aggregate table sink",
    )
    _ = observed^
    check(actual, rows, groups, start)
    _ = trace^
    _ = workers^
