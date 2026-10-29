"""Metadata-only temporal operations retain tick buffers and logical schemas."""
# contract-group: expr.chunk_windows
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.expr import _node, DT_CONVERT_TZ, DT_WITH_TIME_UNIT
from dataframe.kernels.temporal import dt_op
from tests.support.coverage_compare import assert_frame


def _source(dtype: DataType) raises -> Series:
    var values: List[Int64] = [
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        -123,
        0,
        1,
        1577836800000000,
        1583650800000000,
        Int64.MIN,
        Int64.MAX,
        9,
    ]
    var valid: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    var parent = Series("x", Column[Int64](values^, valid)).with_dtype(dtype)
    var view = parent.slice(9, 7)
    var parts = List[Series]()
    parts.append(view.slice(0, 2))
    parts.append(view.slice(2, 0))
    parts.append(view.slice(2, 3))
    parts.append(view.slice(5, 2))
    return Series._from_chunks(parts)


def _expected(dtype: DataType) raises -> Series:
    return Series(
        "x",
        Column[Int64](
            [
                -123,
                0,
                1,
                1577836800000000,
                1583650800000000,
                Int64.MIN,
                Int64.MAX,
            ],
            [True, False, True, True, True, True, True],
        ),
    ).with_dtype(dtype)


def _assert_shared(input: Series, output: Series) raises:
    assert_true(input.is_chunked())
    assert_true(output.is_chunked())
    assert_equal(input.n_chunks(), output.n_chunks())
    var before = input.chunks()
    var after = output.chunks()
    for k in range(len(before)):
        assert_equal(
            Int(before[k]._data[Column[Int64]]._ptr()),
            Int(after[k]._data[Column[Int64]]._ptr()),
        )
    assert_equal(input._chunked.value()[].cache[].state.load(), Int64(0))
    assert_equal(output._chunked.value()[].cache[].state.load(), Int64(0))


def test_temporal_timezone_metadata_keeps_chunk_windows_and_tick_ownership() raises:
    var dtype = DataType.datetime("us", "UTC")
    var input = _source(dtype)
    var target = DataType.datetime("us", "America/New_York")
    var result = dt_op(
        _node(DT_CONVERT_TZ, text="America/New_York"), input, dtype
    )
    _assert_shared(input, result)
    assert_frame(DataFrame([result.copy()]), DataFrame([_expected(target)]))


def test_temporal_time_unit_metadata_preserves_datetime_and_duration_ticks() raises:
    for datetime in [False, True]:
        var dtype = DataType.datetime(
            "us", "UTC"
        ) if datetime else DataType.duration("us")
        var target = DataType.datetime(
            "ns", "UTC"
        ) if datetime else DataType.duration("ns")
        var input = _source(dtype)
        var result = dt_op(_node(DT_WITH_TIME_UNIT, text="ns"), input, dtype)
        _assert_shared(input, result)
        assert_frame(DataFrame([result.copy()]), DataFrame([_expected(target)]))


def _owned_result() raises -> Series:
    var dtype = DataType.datetime("us", "UTC")
    var input = _source(dtype)
    return dt_op(_node(DT_WITH_TIME_UNIT, text="ns"), input, dtype)


def test_temporal_metadata_result_outlives_source_and_handles_cached_input() raises:
    var result = _owned_result()
    assert_frame(
        DataFrame([result.copy()]),
        DataFrame([_expected(DataType.datetime("ns", "UTC"))]),
    )
    var dtype = DataType.duration("us")
    var input = _source(dtype)
    var cached = input.rechunk()
    assert_equal(input._chunked.value()[].cache[].state.load(), Int64(2))
    var tagged = dt_op(_node(DT_WITH_TIME_UNIT, text="ns"), input, dtype)
    assert_equal(input._chunked.value()[].cache[].state.load(), Int64(2))
    assert_frame(
        DataFrame([tagged.copy()]),
        DataFrame([_expected(DataType.duration("ns"))]),
    )
    _ = cached^


def test_temporal_metadata_public_results_empty_schemas_and_invalid_units() raises:
    var dtype = DataType.datetime("us", "UTC")
    var input = _source(dtype)
    var frame = DataFrame([input.copy()])
    assert_frame(
        frame.select(col("x").dt().with_time_unit("ns")),
        DataFrame([_expected(DataType.datetime("ns", "UTC"))]),
    )
    assert_frame(
        frame.select(col("x").dt().convert_time_zone("America/New_York")),
        DataFrame([_expected(DataType.datetime("us", "America/New_York"))]),
    )
    var empty = frame.clear()
    assert_frame(
        empty.select(col("x").dt().with_time_unit("ns")),
        DataFrame([_expected(DataType.datetime("ns", "UTC")).slice(0, 0)]),
    )
    var invalid_input = _source(dtype)
    with assert_raises():
        _ = dt_op(_node(DT_WITH_TIME_UNIT, text="weeks"), invalid_input, dtype)
    assert_equal(
        invalid_input._chunked.value()[].cache[].state.load(), Int64(0)
    )
