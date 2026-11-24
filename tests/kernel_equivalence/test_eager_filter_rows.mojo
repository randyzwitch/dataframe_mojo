"""Eager and lazy filters over in-memory frames keep each morsel's rows as
whole columns: the filters pick the rows, each column is gathered once."""
# contract-group: kernel.table_filters
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, col, lit
from dataframe.frame import concat
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def _part(first: Int, rows: Int) raises -> DataFrame:
    var ids = List[Int64](capacity=rows)
    var values = List[Float64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    var text = List[String](capacity=rows)
    var flags = List[Bool](capacity=rows)
    for k in range(rows):
        var row = first + k
        ids.append(Int64(row))
        values.append(Float64((row * 7919) % 1000) / 10.0)
        valid.append(row % 13 != 4)
        text.append("t" + String(row % 37))
        flags.append(row % 3 == 0)
    return DataFrame(
        [
            Series("id", Column[Int64](ids^)),
            Series("v", Column[Float64](values^, valid)),
            Series("s", Column[String](text^)),
            Series("b", Column[Bool](flags^)),
        ]
    )


def _frame() raises -> DataFrame:
    """Rows in chunks of different lengths, as a file's row groups are."""
    return concat([_part(0, 40_001), _part(40_001, 3), _part(40_004, 59_996)])


def test_eager_filters_gather_each_column_once() raises:
    var frame = _frame()
    for workers in ["1", "4"]:
        var threads = ScopedEnvironment("DATAFRAME_THREADS", workers)
        for cut in [99.0, 50.0, 100.0, -1.0]:
            var predicate = col("v") > lit(cut)
            var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
            var trace = StrategyTrace()
            var kept = frame.filter(predicate)
            var paths = trace.text()
            _ = trace^
            _ = setting^
            assert_true(
                "pipeline.filter_rows_whole" in paths,
                "the filter did not keep whole columns",
            )
            assert_frame(kept, frame._filter(predicate))
            var narrow = frame.filter(predicate, keep=["s", "id"])
            assert_frame(narrow, frame._filter(predicate, keep=["s", "id"]))
        _ = threads^


def test_chained_filters_and_projections_keep_whole_columns() raises:
    var frame = _frame()
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var first = col("v") > lit(20.0)
    var second = col("s") != lit("t5")
    var third = col("b")
    var lazy = (
        frame.lazy()
        .filter(first)
        .filter(second)
        .filter(third)
        .select(["s", "v"])
        .collect(optimize=False)
    )
    var expected = (
        frame._filter(first)._filter(second)._filter(third).select(["s", "v"])
    )
    assert_frame(lazy, expected)
    # A computed column is not a column passed on: it runs chunk by chunk.
    var computed = (
        frame.lazy()
        .filter(first)
        .with_columns([(col("id") * lit(Int64(2))).alias("twice")])
        .collect(optimize=False)
    )
    assert_frame(
        computed,
        frame._filter(first).with_columns(
            [(col("id") * lit(Int64(2))).alias("twice")]
        ),
    )
    _ = threads^


def test_a_filter_error_still_reaches_the_caller() raises:
    var frame = _frame()
    var failed = False
    try:
        _ = frame.filter(col("s").cast(DataType.INT64) > lit(Int64(0)))
    except:
        failed = True
    assert_true(failed, "a strict cast of text did not fail")
    assert_equal(frame.filter(col("id") < lit(Int64(0))).height(), 0)
