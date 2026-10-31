"""Temporal arithmetic keeps global masks, checked ticks and owned chunks."""
# contract-group: expr.temporal
# contract-group: expr.temporal_chunks
# contract-group: expr.chunk_windows
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.expr import ADD, SUB, MUL, FLOORDIV
from dataframe.kernels.temporal import temporal_binary
from tests.support.coverage_compare import assert_frame


def _source(
    name: String,
    dtype: DataType,
    values: List[Int64],
    valid: List[Bool] = List[Bool](),
    right: Bool = False,
) raises -> Series:
    var raw = List[Int64](length=13, fill=777)
    raw.extend(values)
    var flags = List[Bool](length=13, fill=True)
    flags.extend(valid)
    var column = Column[Int64](raw^) if len(valid) == 0 else Column[Int64](
        raw^, flags
    )
    var parent = Series(name, column^).with_dtype(dtype)
    var input = parent.slice(13, len(values))
    var parts = List[Series]()
    parts.append(input.slice(0, 0))
    var sizes: List[Int] = [2, 1, 1, len(values)] if right else [
        1,
        2,
        len(values),
    ]
    var at = 0
    for size in sizes:
        var count = min(size, len(values) - at)
        parts.append(input.slice(at, count))
        at += count
    return Series._from_chunks(parts)


def _scalar(dtype: DataType, value: Int64, valid: Bool = True) raises -> Series:
    return Series("scalar", Column[Int64]([value], [valid])).with_dtype(dtype)


def _assert(
    actual: Series,
    dtype: DataType,
    values: List[Int64],
    valid: List[Bool] = List[Bool](),
) raises:
    var expected = Series(
        "result",
        Column[Int64](values) if len(valid)
        == 0 else Column[Int64](values, valid),
    ).with_dtype(dtype)
    assert_frame(DataFrame([actual.renamed("result")]), DataFrame([expected^]))
    var output = actual.rechunk()
    ref column = output._data[Column[Int64]]
    for row in range(len(values)):
        # Null and inactive results retain the kernel's zero payload contract.
        assert_equal(column._get(row), values[row])


def _cold(input: Series) raises:
    if input.is_chunked():
        assert_equal(input._chunked.value()[].cache[].state.load(), Int64(0))


def _a(dtype: DataType) raises -> Series:
    return _source(
        "a",
        dtype,
        [-9, -5, -1, 0, 5, 9, 17, -8, 10],
        [True, True, True, False, True, True, True, True, True],
    )


def _b(dtype: DataType) raises -> Series:
    return _source(
        "b",
        dtype,
        [2, -2, 0, 3, 2, -3, 4, 2, -3],
        [True, True, True, True, True, False, True, True, True],
        True,
    )


def test_temporal_binary_unequal_chunks_keep_checked_operations_and_bitmap_tail() raises:
    var dtype = DataType.duration("ns")
    var outputs: List[List[Int64]] = [
        [-7, -7, -1, 0, 7, 0, 21, -6, 7],
        [-11, -3, -1, 0, 3, 0, 13, -10, 13],
        [-18, 10, 0, 0, 10, 0, 68, -16, -30],
        [-5, 2, 0, 0, 2, 0, 4, -4, -4],
    ]
    var operations: List[Int] = [ADD, SUB, MUL, FLOORDIV]
    for k in range(4):
        var right_type = DataType.INT64 if k >= 2 else dtype
        var a = _a(dtype)
        var b = _b(right_type)
        var valid: List[Bool] = [
            True,
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
        ]
        if k == 3:
            valid[2] = False
        var result = temporal_binary(
            operations[k], a, b, dtype, right_type, dtype
        )
        assert_equal(result.name(), "")
        _assert(result, dtype, outputs[k], valid)
        _cold(a)
        _cold(b)


def test_temporal_binary_date_scaling_and_public_logical_results_use_literal_ticks() raises:
    var units: List[String] = ["ms", "us", "ns"]
    var expected: List[List[Int64]] = [
        [86400002, -86400002, 3],
        [86400000002, -86400000002, 3],
        [86400000000002, -86400000000002, 3],
    ]
    for k in range(3):
        var d = _source("d", DataType.DATE, [1, -1, 0])
        var duration = DataType.duration(units[k])
        var delta = _source("u", duration, [2, -2, 3], right=True)
        var result_type = DataType.datetime(units[k])
        var result = temporal_binary(
            ADD, d, delta, DataType.DATE, duration, result_type
        )
        _assert(result, result_type, expected[k])
        _cold(d)
        _cold(delta)
        var public_result = DataFrame([d.copy(), delta.copy()]).select(
            (col("d") + col("u")).alias("result")
        )
        _assert(public_result.column("result"), result_type, expected[k])
    var left = _source("a", DataType.DATE, [-2, 0, 3])
    var right = _source("b", DataType.DATE, [1, 0, -1], right=True)
    _assert(
        temporal_binary(
            SUB,
            left,
            right,
            DataType.DATE,
            DataType.DATE,
            DataType.duration("ms"),
        ),
        DataType.duration("ms"),
        [-259200000, 0, 345600000],
    )
    _cold(left)
    _cold(right)


def test_temporal_binary_scalar_masks_empty_lengths_and_observed_overflow() raises:
    var dtype = DataType.duration("us")
    var a = _source("a", dtype, [Int64.MAX, 3, Int64.MIN, 7, 4, 5, 6])
    var two = _scalar(DataType.INT64, 2)
    var selected: List[Bool] = [False, True, False, True, False, True, False]
    _assert(
        temporal_binary(MUL, a, two, dtype, DataType.INT64, dtype, selected),
        dtype,
        [0, 6, 0, 14, 0, 10, 0],
        selected,
    )
    _assert(
        temporal_binary(MUL, a, two, dtype, DataType.INT64, dtype, [False]),
        dtype,
        [0, 0, 0, 0, 0, 0, 0],
        List[Bool](length=7, fill=False),
    )
    with assert_raises(contains="temporal value overflows Int64"):
        _ = temporal_binary(
            MUL, a, two, dtype, DataType.INT64, dtype, [True, False]
        )
    _cold(a)
    var vector = _b(dtype)
    _assert(
        temporal_binary(ADD, _scalar(dtype, 3), vector, dtype, dtype, dtype),
        dtype,
        [5, 1, 3, 6, 5, 0, 7, 5, 0],
        [True, True, True, True, True, False, True, True, True],
    )
    var null_scalar = _scalar(dtype, Int64.MAX, False)
    _assert(
        temporal_binary(ADD, vector, null_scalar, dtype, dtype, dtype),
        dtype,
        List[Int64](length=9, fill=0),
        List[Bool](length=9, fill=False),
    )
    var max_scalar = _scalar(dtype, Int64.MAX)
    var one = _scalar(dtype, 1)
    _assert(
        temporal_binary(
            ADD, max_scalar, one, dtype, dtype, dtype, [False, False]
        ),
        dtype,
        [0],
        [False],
    )
    with assert_raises(contains="temporal value overflows Int64"):
        _ = temporal_binary(
            ADD, max_scalar, one, dtype, dtype, dtype, [False, True]
        )
    var empty = _source("empty", dtype, List[Int64]())
    _assert(
        temporal_binary(ADD, empty, one, dtype, dtype, dtype),
        dtype,
        List[Int64](),
    )
    with assert_raises(contains="Incompatible expression lengths"):
        _ = temporal_binary(ADD, empty, vector, dtype, dtype, dtype, [False])
    with assert_raises(contains="Incompatible expression lengths"):
        _ = temporal_binary(ADD, vector, empty, dtype, dtype, dtype, [False])
    _cold(vector)


def _owned_result() raises -> Series:
    var left_type = DataType.datetime("us", "UTC")
    var right_type = DataType.duration("us")
    var a = _a(left_type)
    var b = _b(right_type)
    var result = temporal_binary(ADD, a, b, left_type, right_type, left_type)
    _cold(a)
    _cold(b)
    return result^


def test_temporal_binary_results_outlive_sources_and_keep_each_published_cache() raises:
    var dtype = DataType.datetime("us", "UTC")
    var duration = DataType.duration("us")
    var expected: List[Int64] = [-7, -7, -1, 0, 7, 0, 21, -6, 7]
    var valid: List[Bool] = [
        True,
        True,
        True,
        False,
        True,
        False,
        True,
        True,
        True,
    ]
    _assert(_owned_result(), dtype, expected, valid)
    for ready in [1, 2, 3]:
        var a = _a(dtype)
        var b = _b(duration)
        if ready & 1 != 0:
            var flat = a.rechunk()
            _ = flat^
        if ready & 2 != 0:
            var flat = b.rechunk()
            _ = flat^
        _assert(
            temporal_binary(ADD, a, b, dtype, duration, dtype),
            dtype,
            expected,
            valid,
        )
        assert_equal(
            a._chunked.value()[].cache[].state.load(),
            Int64(2 if ready & 1 != 0 else 0),
        )
        assert_equal(
            b._chunked.value()[].cache[].state.load(),
            Int64(2 if ready & 2 != 0 else 0),
        )
