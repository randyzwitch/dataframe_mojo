"""Streaming positional reductions retain source order across worker morsels."""
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series, col
from tests.support.group_strategy_reference import ScopedEnvironment


def _offset_chunks[D: DType](n: Int) raises -> DataFrame:
    var values = List[Scalar[D]]()
    var valid = List[Bool]()
    var rows = List[Int64]()
    for row in range(n + 6):
        values.append(Scalar[D](row % 11 + 1))
        valid.append(row % 7 != 0)
        rows.append(Int64(row))
    var source = Series("x", Column[Scalar[D]](values^, valid)).slice(3, n)
    var position = Series("row", Column[Int64](rows^)).slice(3, n)
    var columns = List[Series]()
    for column in [source^, position^]:
        columns.append(
            Series._from_chunks(
                [
                    column.slice(0, n // 2),
                    column.slice(n // 2, 0),
                    column.slice(n // 2, n - n // 2),
                ]
            )
        )
    return DataFrame(columns^)


def _assert_ordered[D: DType](actual: DataFrame, filtered: Bool) raises:
    assert_equal(actual.height(), 1)
    assert_equal(
        actual.item(0, "first").numeric[D](), Scalar[D](10 if filtered else 4)
    )
    assert_true(actual.item(0, "last").is_null())
    assert_equal(
        actual.item(0, "arg_min").numeric[DType.uint32](),
        UInt32(2 if filtered else 8),
    )
    assert_equal(
        actual.item(0, "arg_max").numeric[DType.uint32](),
        UInt32(12 if filtered else 7),
    )


def _check_chunked_last[D: DType](n: Int, filtered: Bool) raises:
    var frame = _offset_chunks[D](n)
    var query = frame.lazy()
    if filtered:
        query = query.filter(col("row") >= 20)
    query = query.select_exprs(
        [
            col("x").first().alias("first"),
            col("x").last().alias("last"),
            col("x").arg_min().alias("arg_min"),
            col("x").arg_max().alias("arg_max"),
        ]
    )
    var expected = query.collect(engine="cpu", streaming=False)
    _assert_ordered[D](expected, filtered)
    # Several worker assignments must retain the same source positions.
    for _ in range(16):
        var actual = query.collect(
            engine="cpu", streaming=True, batch_size=64 if filtered else 65_536
        )
        _assert_ordered[D](actual, filtered)
        assert_true(actual.equals(expected))


def test_chunked_last_streaming_preserves_terminal_null_and_extrema_positions() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    for n in [257, 1_027]:
        _check_chunked_last[DType.int8](n, False)
        _check_chunked_last[DType.uint8](n, False)
    _ = workers^


def test_chunked_last_streaming_preserves_filtered_source_order() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    _check_chunked_last[DType.int8](1_027, True)
    _check_chunked_last[DType.uint8](1_027, True)
    _ = workers^
