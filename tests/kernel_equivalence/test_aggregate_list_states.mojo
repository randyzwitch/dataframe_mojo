"""The aggregate table's list states against a scalar model of each group:
implode and mode, through the eager group-by and a
pipeline's grouped sink after a filter, on tables that clear (eight
workers) and on one worker. Expressions the table has no state for (a
filtered sum, a sorted head) are evaluated over the groups the table
finds, each group's rows coming out of it as an implode of row numbers."""
# contract-group: reduce.grouped
# contract-group: kernel.reduce
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    Expr,
    Series,
    col,
    lit,
)
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def group_of(row: Int, groups: Int) -> Int:
    return (row * 7919) % groups


def valid(row: Int) -> Bool:
    return row % 7 != 5


def value(row: Int) -> Int64:
    return Int64((row * 104729) % 5) - 2


def frame(rows: Int, groups: Int) raises -> DataFrame:
    var k = List[Int64]()
    var x = List[Int64]()
    var xv = List[Bool]()
    var r = List[Int64]()
    for row in range(rows):
        k.append(Int64(group_of(row, groups)) * 1_000_003)
        x.append(value(row))
        xv.append(valid(row))
        r.append(Int64(row))
    return DataFrame(
        [
            Series("k", Column[Int64](k^)),
            Series("x", Column[Int64](x^, xv^)),
            Series("r", Column[Int64](r^)),
        ]
    )


def group_rows(rows: Int, groups: Int, start: Int) -> List[List[Int]]:
    var out = List[List[Int]](length=groups, fill=List[Int]())
    for row in range(start, rows):
        out[group_of(row, groups)].append(row)
    return out^


def key_group(result: DataFrame, i: Int) raises -> Int:
    return Int(result.column("k").int64().value(i).value() // 1_000_003)


def check_lists(
    result: DataFrame, rows: Int, groups: Int, start: Int = 0
) raises:
    var model = group_rows(rows, groups, start)
    var present = 0
    for g in range(groups):
        if len(model[g]) > 0:
            present += 1
    assert_equal(result.height(), present)
    var lists = result.column("xs").rechunk().list_column()
    var child = lists.child().rechunk()
    var modes = result.column("m").rechunk().list_column()
    var mode_child = modes.child().rechunk()
    for i in range(result.height()):
        var g = key_group(result, i)
        ref want = model[g]
        assert_equal(lists._end(i) - lists._start(i), len(want))
        var counts = List[Int](length=6, fill=0)
        for e in range(len(want)):
            var row = want[e]
            var at = lists._start(i) + e
            assert_equal(child.int64().is_valid(at), valid(row))
            if valid(row):
                assert_equal(child.int64().value(at), value(row))
                counts[Int(value(row)) + 2] += 1
            else:
                counts[5] += 1
        var best = 0
        for c in counts:
            best = max(best, c)
        var expected = List[Int64]()
        for v in range(5):
            if counts[v] == best:
                expected.append(Int64(v - 2))
        var nulls = 1 if counts[5] == best else 0
        assert_equal(modes._end(i) - modes._start(i), len(expected) + nulls)
        for e in range(len(expected)):
            assert_equal(
                mode_child.int64().value(modes._start(i) + e).value(),
                expected[e],
            )


def list_reductions() -> List[Expr]:
    return [col("x").implode().alias("xs"), col("x").mode().alias("m")]


def eager_lists(threads: Int, rows: Int, groups: Int) raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    for ordered in [False, True]:
        var observed = StrategyTrace()
        var result = (
            frame(rows, groups)
            .group_by(["k"], maintain_order=ordered)
            .agg(list_reductions())
        )
        assert_true(
            "dataframe-path: group_by.aggregate_table\n" in observed.text(),
            "implode and mode did not reach the aggregate table",
        )
        _ = observed^
        check_lists(result, rows, groups)
        if ordered:
            for i in range(result.height()):
                assert_equal(key_group(result, i), group_of(i, groups))
    _ = trace^
    _ = workers^


def test_eager_list_states_with_tables_that_clear() raises:
    eager_lists(8, 120_000, 9_000)


def test_eager_list_states_on_one_worker() raises:
    eager_lists(1, 20_000, 700)


def test_pipeline_list_states_after_a_filter() raises:
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
        .agg(list_reductions())
        .collect()
    )
    assert_true(
        "dataframe-path: pipeline.group_table\n" in observed.text(),
        "list states did not reach the aggregate table sink",
    )
    _ = observed^
    check_lists(actual, rows, groups, start)
    _ = trace^
    _ = workers^


def row_list_expressions() -> List[Expr]:
    return [
        col("x").filter(col("x") > lit(Int64(0))).sum().alias("pos"),
        col("r").sort(descending=True).head(2).alias("top"),
    ]


def check_row_lists(
    result: DataFrame, rows: Int, groups: Int, start: Int = 0
) raises:
    var model = group_rows(rows, groups, start)
    var tops = result.column("top").rechunk().list_column()
    var child = tops.child().rechunk()
    for i in range(result.height()):
        var g = key_group(result, i)
        ref want = model[g]
        var total = Int64(0)
        for row in want:
            if valid(row) and value(row) > 0:
                total += value(row)
        assert_equal(result.column("pos").int64().value(i).value(), total)
        var n = min(2, len(want))
        assert_equal(tops._end(i) - tops._start(i), n)
        for e in range(n):
            assert_equal(
                child.int64().value(tops._start(i) + e).value(),
                Int64(want[len(want) - 1 - e]),
            )


def test_expressions_without_states_over_the_tables_groups() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 6_000
    var groups = 300
    for ordered in [False, True]:
        var observed = StrategyTrace()
        var eager = (
            frame(rows, groups)
            .group_by(["k"], maintain_order=ordered)
            .agg(row_list_expressions())
        )
        assert_true(
            "dataframe-path: group_by.row_lists\n" in observed.text(),
            "the groups' rows did not come from the aggregate table",
        )
        _ = observed^
        assert_equal(eager.height(), groups)
        check_row_lists(eager, rows, groups)
        if ordered:
            for i in range(eager.height()):
                assert_equal(key_group(eager, i), group_of(i, groups))
    var start = 77
    var lazy = (
        frame(rows, groups)
        .lazy()
        .filter(col("r") >= lit(Int64(start)))
        .group_by(["k"])
        .agg(row_list_expressions())
        .collect()
    )
    check_row_lists(lazy, rows, groups, start)
    _ = trace^
    _ = workers^


def test_expressions_without_states_over_no_rows() raises:
    var empty = frame(0, 1)
    var result = empty.group_by(["k"]).agg(row_list_expressions())
    assert_equal(result.height(), 0)
    assert_equal(result.width(), 3)
