"""Primitive casts retain checked observations, packed windows and IEEE rules."""
# contract-group: expr.primitive_cast_chunks
# contract-group: types.cast
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    DataType,
    Series,
    col,
    lit,
    when,
)
from dataframe.kernels.cast import cast_series
from tests.support.coverage_compare import assert_frame


def _source[
    D: DType
](values: List[Scalar[D]], valid: List[Bool]) raises -> Series:
    var raw = List[Scalar[D]](length=13, fill=Scalar[D](91))
    raw.extend(values.copy())
    var flags = List[Bool](length=13, fill=True)
    flags.extend(valid.copy())
    var parent = Series("x", Column[Scalar[D]](raw^, flags.copy()))
    var window = parent.slice(13, len(values))
    var parts = List[Series]()
    parts.append(window.slice(0, 0))
    var at = 0
    for size in [1, 2, 3, len(values)]:
        var count = min(size, len(values) - at)
        parts.append(window.slice(at, count))
        at += count
    return Series._from_chunks(parts)


def _bool_source() raises -> Series:
    var values = List[Bool](length=13, fill=False)
    values.extend(
        List[Bool]([True, False, True, True, False, True, False, True, True])
    )
    var valid = List[Bool](length=13, fill=True)
    valid.extend(
        List[Bool]([True, True, False, True, True, True, True, False, True])
    )
    var parent = Series("x", BoolColumn(values.copy(), valid.copy()))
    var window = parent.slice(13, 9)
    return Series._from_chunks(
        [window.slice(0, 1), window.slice(1, 2), window.slice(3, 6)]
    )


def _check[
    D: DType
](actual: Series, values: List[Scalar[D]], valid: List[Bool]) raises:
    var expected = Series("x", Column[Scalar[D]](values.copy(), valid.copy()))
    assert_frame(DataFrame([actual.copy()]), DataFrame([expected^]))
    var flat = actual.rechunk()
    ref column = flat._data[Column[Scalar[D]]]
    for row in range(len(values)):
        if not valid[row]:
            assert_equal(column._get(row), Scalar[D](0))


def _check_bool(actual: Series, values: List[Bool], valid: List[Bool]) raises:
    var expected = Series("x", BoolColumn(values.copy(), valid.copy()))
    assert_frame(DataFrame([actual.copy()]), DataFrame([expected^]))
    var flat = actual.rechunk()
    ref column = flat._data[BoolColumn]
    for row in range(len(values)):
        if not valid[row]:
            assert_equal(column._get(row), False)


def _error(
    source: Series,
    target: DataType,
    offset: Int,
    mask: List[Bool],
    row: Int,
    cause: String,
) raises:
    var rejected = False
    try:
        _ = cast_series(source, target, True, offset, mask)
    except error:
        var message = String(error)
        assert_true(
            message.find(
                "cast from " + source.dtype().name() + " to " + target.name()
            )
            >= 0
        )
        assert_true(message.find("failed at row " + String(row)) >= 0)
        assert_true(message.find(cause) >= 0)
        rejected = True
    assert_true(rejected, "observed invalid primitive conversion must reject")


def test_float_integer_cast_truncates_before_bounds_and_observes_global_rows() raises:
    var inf = Float64(1) / Float64(0)
    var nan = Float64(0) / Float64(0)
    var source = _source[DType.float64](
        [-0.75, inf, nan, 255.75, -1, -nan, 256],
        [True, True, False, True, True, False, True],
    )
    var mask: List[Bool] = [True, False, True, True, True, True, True]
    _error(source, DataType.UINT8, 701, mask, 705, "out of ")
    _error(source, DataType.UINT8, 701, [], 702, "out of ")
    _check[DType.uint8](
        cast_series(source, DataType.UINT8, False, 701, mask),
        [0, 0, 0, 255, 0, 0, 0],
        [True, False, False, True, False, False, False],
    )
    _check[DType.uint8](
        cast_series(
            source,
            DataType.UINT8,
            True,
            701,
            [True, False, True, True, False, True, False],
        ),
        [0, 0, 0, 255, 0, 0, 0],
        [True, False, False, True, False, False, False],
    )
    _check[DType.int8](
        cast_series(
            _source[DType.float32](
                [-128.75, 127.75, -0.75], [True, True, True]
            ),
            DataType.INT8,
            True,
            0,
            [],
        ),
        [-128, 127, 0],
        [True, True, True],
    )


def test_float_integer_64bit_bounds_are_half_open_powers_of_two() raises:
    var f64 = _source[DType.float64](
        [
            9223372036854774784.0,
            9223372036854775808.0,
            -9223372036854775808.0,
            -9223372036854777856.0,
            18446744073709549568.0,
            18446744073709551616.0,
        ],
        [True, True, True, True, True, True],
    )
    _check[DType.int64](
        cast_series(f64, DataType.INT64, False, 91, []),
        [Int64(9223372036854774784), 0, Int64.MIN, 0, 0, 0],
        [True, False, True, False, False, False],
    )
    _error(f64, DataType.INT64, 91, [], 92, "out of ")
    _check[DType.uint64](
        cast_series(f64, DataType.UINT64, False, 0, []),
        [
            UInt64(9223372036854774784),
            UInt64(9223372036854775808),
            0,
            0,
            UInt64(18446744073709549568),
            0,
        ],
        [True, True, False, False, True, False],
    )
    var f32 = _source[DType.float32](
        [
            9223371487098961920.0,
            9223372036854775808.0,
            -9223372036854775808.0,
            18446742974197923840.0,
            18446744073709551616.0,
        ],
        [True, True, True, True, True],
    )
    _check[DType.int64](
        cast_series(f32, DataType.INT64, False, 0, []),
        [Int64(9223371487098961920), 0, Int64.MIN, 0, 0],
        [True, False, True, False, False],
    )
    _check[DType.uint64](
        cast_series(f32, DataType.UINT64, False, 0, []),
        [
            UInt64(9223371487098961920),
            UInt64(9223372036854775808),
            0,
            UInt64(18446742974197923840),
            0,
        ],
        [True, True, False, True, False],
    )


def test_numeric_boolean_cast_nan_is_invalid_but_infinities_are_true() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var tiny = bitcast[DType.float64](UInt64(1))
    var source = _source[DType.float64](
        [-0.0, 0.0, inf, -inf, nan, tiny, nan, -tiny, 2],
        [True, True, True, True, True, True, False, True, True],
    )
    _error(source, DataType.BOOL, 303, [], 307, "NaN has no Boolean value")
    _check_bool(
        cast_series(source, DataType.BOOL, False, 303, []),
        [False, False, True, True, False, True, False, True, True],
        [True, True, True, True, False, True, False, True, True],
    )
    assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))
    _check_bool(
        cast_series(
            source,
            DataType.BOOL,
            True,
            303,
            [True, True, True, True, False, True, True, True, True],
        ),
        [False, False, True, True, False, True, False, True, True],
        [True, True, True, True, False, True, False, True, True],
    )
    _check_bool(
        cast_series(
            _source[DType.uint64]([UInt64.MAX, 0, 1], [True, True, False]),
            DataType.BOOL,
            True,
            0,
            [],
        ),
        [True, False, False],
        [True, True, False],
    )


def test_packed_boolean_numeric_cast_windows_and_malformed_masks_keep_observation() raises:
    var source = _bool_source()
    var values: List[Int8] = [1, 0, 0, 1, 0, 1, 0, 0, 1]
    var valid: List[Bool] = [
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
    ]
    _check[DType.int8](
        cast_series(source, DataType.INT8, True, 0, []), values, valid
    )
    assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))
    _check[DType.int8](
        cast_series(source, DataType.INT8, True, 0, [False]), values, valid
    )
    _check[DType.float64](
        cast_series(
            source,
            DataType.FLOAT64,
            True,
            0,
            [True, True, True, False, True, True, True, True, False],
        ),
        [1, 0, 0, 0, 0, 1, 0, 0, 0],
        [True, True, False, False, True, True, True, False, False],
    )
    var scalar = source.slice(0, 1)
    _check[DType.int8](
        cast_series(scalar, DataType.INT8, True, 0, [False, False]), [1], [True]
    )
    _check[DType.int8](
        cast_series(scalar, DataType.INT8, True, 0, [False]), [0], [False]
    )


def test_float_width_cast_ieee_rules_and_existing_unmasked_widening() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var source = _source[DType.float64](
        [
            -0.0,
            0.0,
            1e40,
            -1e40,
            1e-50,
            -1e-50,
            bitcast[DType.float64](UInt64(0x36A0000000000000)),
            nan,
            inf,
        ],
        [True, True, True, True, True, True, True, True, False],
    )
    var narrow = cast_series(source, DataType.FLOAT32, True, 0, [])
    assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))
    var expected: List[Float32] = [
        bitcast[DType.float32](UInt32(0x80000000)),
        0,
        bitcast[DType.float32](UInt32(0x7F800000)),
        bitcast[DType.float32](UInt32(0xFF800000)),
        0,
        bitcast[DType.float32](UInt32(0x80000000)),
        bitcast[DType.float32](UInt32(1)),
        Float32(nan),
        0,
    ]
    _check[DType.float32](
        narrow,
        expected,
        [True, True, True, True, True, True, True, True, False],
    )
    var flat = narrow.rechunk()
    ref column = flat._data[Column[Float32]]
    for row in range(7):
        assert_equal(
            bitcast[DType.uint32](column._get(row)),
            bitcast[DType.uint32](expected[row]),
        )
    assert_true(isnan(column._get(7)))
    var wide_source = _source[DType.float32](
        expected,
        [True, True, True, True, True, True, True, True, False],
    )
    var wide = cast_series(wide_source, DataType.FLOAT64, True, 0, [])
    assert_equal(wide_source._chunked.value()[].cache[].state.load(), Int64(0))
    _check[DType.float64](
        wide,
        [
            -0.0,
            0.0,
            inf,
            -inf,
            0.0,
            -0.0,
            bitcast[DType.float64](UInt64(0x36A0000000000000)),
            nan,
            0,
        ],
        [True, True, True, True, True, True, True, True, False],
    )
    var empty = source.slice(0, 0)
    _check[DType.float32](
        cast_series(empty, DataType.FLOAT32, True, 0, []), [], []
    )


def _owned_result(warm: Bool) raises -> Series:
    var nan = Float64(0) / Float64(0)
    var source = _source[DType.float64]([1.75, nan, -1.75], [True, True, False])
    if warm:
        var merged = source.rechunk()
        _ = merged^
    var result = cast_series(source, DataType.INT8, False, 0, [])
    assert_equal(
        source._chunked.value()[].cache[].state.load(), Int64(2 if warm else 0)
    )
    return result^


def test_public_primitive_conditionals_owners_caches_and_identity_keep_contracts() raises:
    for warm in [False, True]:
        _check[DType.int8](_owned_result(warm), [1, 0, 0], [True, False, False])
    var nan = Float64(0) / Float64(0)
    var frame = DataFrame(
        [
            _source[DType.float64]([7, nan, 9], [True, True, False]),
            Series("keep", BoolColumn([True, False, True])),
        ]
    )
    var selected = frame.select(
        when(col("keep"))
        .then(col("x").cast(DataType.BOOL))
        .otherwise(lit(False))
        .alias("x")
    )
    _check_bool(selected.column("x"), [True, False, False], [True, True, False])
    var source = _bool_source()
    var booleans = DataFrame(
        [
            source.copy(),
            Series(
                "keep",
                BoolColumn(
                    [True, False, True, True, True, True, True, True, True]
                ),
            ),
        ]
    )
    var numeric = booleans.select(
        when(col("keep"))
        .then(col("x").cast(DataType.FLOAT32))
        .otherwise(lit(Float32(0)))
        .alias("x")
    )
    _check[DType.float32](
        numeric.column("x"),
        [1, 0, 0, 1, 0, 1, 0, 0, 1],
        [True, True, False, True, True, True, True, False, True],
    )
    var identity = cast_series(
        source, DataType.BOOL, True, 999, List[Bool](length=9, fill=False)
    )
    var before = source.chunks()
    var after = identity.chunks()
    assert_equal(len(before), len(after))
    for index in range(len(before)):
        assert_true(
            before[index]
            ._data[BoolColumn]
            ._shares_buffers_with(after[index]._data[BoolColumn])
        )
    assert_equal(after[1]._data[BoolColumn]._get(1), True)
    assert_equal(after[1]._data[BoolColumn]._valid(1), False)
