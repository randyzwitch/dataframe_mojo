"""Distinct counts in the aggregate table's distinct tables (each worker sinks
every row's (keys, value) pair into a table of its own, and the finish sinks
those groups into the main table's counts) against a scalar model: string,
integer and float values with nulls counted once (Polars' n_unique), beside
other reductions, through the eager group-by and a pipeline's grouped sink
after a filter, on tables that clear (eight workers), grow (two) and one
worker; first-occurrence order is kept under maintain_order."""
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


def text_valid(row: Int) -> Bool:
    return row % 11 != 3


def text(row: Int) -> String:
    var v = (row * 31) % 23
    if v % 2 == 0:
        return "s" + String(v)
    return "a longer distinct value " + String(v)


def int_valid(row: Int) -> Bool:
    return row % 7 != 5


def int_value(row: Int) -> Int64:
    return Int64((row * 13) % 17)


def real(row: Int) -> Float64:
    var v = (row * 5) % 9
    if v == 0:
        return -0.0
    if v == 1:
        return 0.0
    return Float64(v) / 4


def frame(rows: Int, groups: Int) raises -> DataFrame:
    var k = List[Int64]()
    var t = List[String]()
    var tv = List[Bool]()
    var x = List[Int64]()
    var xv = List[Bool]()
    var y = List[Float64]()
    var r = List[Int64]()
    for row in range(rows):
        k.append(Int64(group_of(row, groups)) * 17 - 300)
        t.append(text(row) if text_valid(row) else String(""))
        tv.append(text_valid(row))
        x.append(int_value(row))
        xv.append(int_valid(row))
        y.append(real(row))
        r.append(Int64(row))
    return DataFrame(
        [
            Series("k", Column[Int64](k^)),
            Series("t", Column[String](t^, tv)),
            Series("x", Column[Int64](x^, xv)),
            Series("y", Column[Float64](y^)),
            Series("r", Column[Int64](r^)),
        ]
    )


def reductions() -> List[Expr]:
    return [
        col("t").n_unique().alias("t_distinct"),
        col("x").n_unique().alias("x_distinct"),
        col("y").n_unique().alias("y_distinct"),
        col("x").sum().alias("x_sum"),
        col("r").len().alias("rows"),
    ]


def check(
    given: DataFrame, rows: Int, groups: Int, start: Int, ordered: Bool
) raises:
    # One chunk, so reading a cell does not walk every chunk.
    var actual = given.rechunk()
    var texts = List[Dict[String, Bool]](
        length=groups, fill=Dict[String, Bool]()
    )
    var ints = List[Dict[Int, Bool]](length=groups, fill=Dict[Int, Bool]())
    var reals = List[Dict[Int, Bool]](length=groups, fill=Dict[Int, Bool]())
    var sums = List[Int64](length=groups, fill=0)
    var counts = List[Int](length=groups, fill=0)
    var firsts = List[Int](length=groups, fill=-1)
    for row in range(start, rows):
        var g = group_of(row, groups)
        if firsts[g] < 0:
            firsts[g] = row
        counts[g] += 1
        texts[g][text(row) if text_valid(row) else String("<null>")] = True
        ints[g][Int(int_value(row)) if int_valid(row) else -1] = True
        if int_valid(row):
            sums[g] += int_value(row)
        # -0.0 and 0.0 are one value.
        reals[g][Int(abs(real(row)) * 4)] = True
    var present = 0
    for g in range(groups):
        if counts[g] > 0:
            present += 1
    assert_equal(actual.height(), present)
    var last_first = -1
    for r in range(actual.height()):
        var g = Int((actual.item(r, "k").int64() + 300) // 17)
        assert_true(counts[g] > 0, "unexpected group " + String(g))
        var name = " of group " + String(g)
        assert_equal(
            actual.item(r, "t_distinct").int64(),
            Int64(len(texts[g])),
            "t" + name,
        )
        assert_equal(
            actual.item(r, "x_distinct").int64(),
            Int64(len(ints[g])),
            "x" + name,
        )
        assert_equal(
            actual.item(r, "y_distinct").int64(),
            Int64(len(reals[g])),
            "y" + name,
        )
        assert_equal(actual.item(r, "x_sum").int64(), sums[g], "sum" + name)
        assert_equal(
            actual.item(r, "rows").int64(), Int64(counts[g]), "rows" + name
        )
        if ordered:
            assert_true(firsts[g] > last_first, "order" + name)
            last_first = firsts[g]


def eager(threads: Int, rows: Int, groups: Int, ordered: Bool) raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var actual = (
        frame(rows, groups)
        .group_by(["k"], maintain_order=ordered)
        .agg(reductions())
    )
    assert_true(
        "dataframe-path: group_by.aggregate_table\n" in observed.text(),
        "distinct counts did not reach the aggregate table",
    )
    _ = observed^
    check(actual, rows, groups, 0, ordered)
    _ = trace^
    _ = workers^


def test_eager_distinct_counts_with_tables_that_clear() raises:
    eager(8, 60_000, 8_000, False)


def test_eager_distinct_counts_in_first_occurrence_order() raises:
    eager(8, 60_000, 8_000, True)


def test_eager_distinct_counts_with_tables_that_grow() raises:
    eager(2, 60_000, 4_000, False)


def test_eager_distinct_counts_on_one_worker() raises:
    eager(1, 20_000, 900, True)


def test_pipeline_distinct_counts_after_a_filter() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 120_000
    var groups = 9_000
    var start = 4_321
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
        "distinct counts did not reach the aggregate table sink",
    )
    _ = observed^
    check(actual, rows, groups, start, False)
    _ = trace^
    _ = workers^
