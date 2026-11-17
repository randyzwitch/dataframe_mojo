"""Head, limit and slice over pipeline steps keep the first rows in source
order on several workers."""
# contract-group: planner.routing
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col, lit
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def _source(rows: Int) raises -> DataFrame:
    var keys = List[Int64]()
    var values = List[Int64]()
    var text = List[String]()
    for row in range(rows):
        keys.append(Int64(row % 97))
        values.append(Int64(row))
        text.append("r" + String(row))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^)),
            Series("s", Column[String](text^)),
        ]
    )


def _dimension() raises -> DataFrame:
    # Every key but a few, once each: an inner join drops some rows and a
    # left join keeps them with nulls.
    var keys = List[Int64]()
    var names = List[String]()
    for key in range(97):
        if key % 10 == 3:
            continue
        keys.append(Int64(key))
        names.append("d" + String(key))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("d", Column[String](names^)),
        ]
    )


def _same(actual: DataFrame, expected: DataFrame) raises:
    assert_equal(actual.height(), expected.height())
    assert_equal(actual.columns(), expected.columns())
    assert_true(actual.equals(expected))


def _check(query: LazyFrame, expected: DataFrame) raises:
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    _same(query.collect(engine="cpu", streaming=True, optimize=False), expected)
    # The optimized plan gives the same rows.
    _same(query.collect(engine="cpu", streaming=True, optimize=True), expected)
    _ = threads^


def _assert_pipelined(query: LazyFrame) raises:
    """The slice runs in the morsel pipeline's limit sink, not over ordered
    batches."""
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    _ = query.collect(engine="cpu", streaming=True, optimize=False)
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        paths.find(
            "dataframe-strategy: morsel_pipeline: row_local_frame_pipeline"
        )
        >= 0,
        paths,
    )


def _windows() -> List[Tuple[Int, Int]]:
    # (offset, length); length -1 reads to the end.
    return [
        (0, 0),
        (0, 1),
        (0, 7),
        (0, 2051),
        (0, 15_000),
        (0, 1_000_000_000),
        (3, 5),
        (2049, 4096),
        (12_000, 9),
        (1_000_000_000, 4),
        (17, -1),
    ]


def test_slices_over_filters_keep_source_order() raises:
    for rows in [0, 1, 20_000]:
        var frame = _source(rows)
        var predicate = (col("v") % lit(Int64(3))) != lit(Int64(1))
        var eager = frame.filter(predicate)
        for window in _windows():
            var offset = window[0]
            var length = window[1]
            _check(
                frame.lazy().filter(predicate).slice(offset, length),
                eager.slice(offset, length),
            )
        _assert_pipelined(frame.lazy().filter(predicate).slice(5, 9))
        _check(frame.lazy().filter(predicate).head(11), eager.head(11))
        _check(frame.lazy().filter(predicate).limit(4097), eager.head(4097))


def test_slices_over_filters_that_keep_no_row() raises:
    var frame = _source(20_000)
    var predicate = col("v") < lit(Int64(0))
    var expected = frame.filter(predicate)
    _check(frame.lazy().filter(predicate).head(5), expected)
    _check(frame.lazy().filter(predicate).slice(2, 3), expected)


def test_slices_over_joins_keep_source_order() raises:
    var frame = _source(20_000)
    var dimension = _dimension()
    for how in ["inner", "left"]:
        var predicate = col("v") > lit(Int64(10))
        var eager = frame.filter(predicate).join(dimension, on="k", how=how)
        _assert_pipelined(
            frame.lazy()
            .filter(predicate)
            .join(dimension.lazy(), on="k", how=how)
            .head(9)
        )
        for window in _windows():
            var offset = window[0]
            var length = window[1]
            _check(
                frame.lazy()
                .filter(predicate)
                .join(dimension.lazy(), on="k", how=how)
                .slice(offset, length),
                eager.slice(offset, length),
            )
        # A filter after the join, then a projection, under the slice.
        var after = col("d").is_null() | (col("v") % lit(Int64(5)) == 0)
        _check(
            frame.lazy()
            .join(dimension.lazy(), on="k", how=how)
            .filter(after)
            .select(["s", "v"])
            .head(1234),
            frame.join(dimension, on="k", how=how)
            .filter(after)
            .select(["s", "v"])
            .head(1234),
        )
