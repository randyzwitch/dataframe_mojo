"""Checked integer casts preserve observation, exact bounds and rounding modes."""
# contract-group: expr.integer_cast_chunks
# contract-group: types.cast
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_false
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
](values: List[Scalar[D]], valid: List[Bool] = List[Bool]()) raises -> Series:
    var raw = List[Scalar[D]](length=13, fill=Scalar[D](91))
    raw.extend(values.copy())
    var flags = List[Bool](length=13, fill=True)
    flags.extend(valid.copy())
    var column = Column[Scalar[D]](raw^) if len(valid) == 0 else Column[
        Scalar[D]
    ](raw^, flags.copy())
    var parent = Series("x", column^)
    var window = parent.slice(13, len(values))
    var parts = List[Series]()
    parts.append(window.slice(0, 0))
    var at = 0
    for size in [1, 2, 3, len(values)]:
        var count = min(size, len(values) - at)
        parts.append(window.slice(at, count))
        at += count
    return Series._from_chunks(parts)


def _check[
    D: DType
](actual: Series, values: List[Scalar[D]], valid: List[Bool]) raises:
    var expected = Series("x", Column[Scalar[D]](values.copy(), valid.copy()))
    assert_equal(actual.name(), "x")
    assert_true(actual.dtype() == DataType.of(D))
    assert_frame(DataFrame([actual.copy()]), DataFrame([expected^]))
    var flat = actual.rechunk()
    ref column = flat._data[Column[Scalar[D]]]
    for row in range(len(values)):
        assert_equal(column._get(row), values[row])


def _error(
    source: Series,
    target: DataType,
    offset: Int,
    mask: List[Bool],
    row: Int,
    text: String,
) raises:
    var rejected = False
    try:
        _ = cast_series(source, target, True, offset, mask)
    except error:
        var message = String(error)
        assert_true(message.find("failed at row " + String(row)) >= 0)
        assert_true(message.find("for value '" + text + "'") >= 0)
        assert_true(message.find("out of ") >= 0)
        rejected = True
    assert_true(rejected, "an observed out-of-range integer must reject")


def _narrow_source() raises -> Series:
    return _source[DType.int16](
        [7, 128, 999, 8, -129, 127, -128, -999, 3],
        [True, True, False, True, True, True, True, False, True],
    )


def test_integer_checked_cast_global_error_offsets_and_relaxed_null_payloads() raises:
    var source = _narrow_source()
    var observed: List[Bool] = [
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    # 128 is hidden by the mask; 999/-999 are poisoned null payloads.
    # The first observed failure is row4, inside the third physical chunk.
    _error(source, DataType.INT8, 317, observed, 321, "-129")
    _error(source, DataType.INT8, 317, [], 318, "128")
    var result = cast_series(source, DataType.INT8, False, 317, observed)
    _check[DType.int8](
        result,
        [7, 0, 0, 8, 0, 127, -128, 0, 3],
        [True, False, False, True, False, True, True, False, True],
    )
    var fits = _source[DType.int16]([127, -128, 0])
    _check[DType.int8](
        cast_series(fits, DataType.INT8, True, 9, []),
        [127, -128, 0],
        [True, True, True],
    )


def test_unsigned_bounds_and_direct_malformed_masks_keep_legacy_observation() raises:
    var source = _source[DType.uint64](
        [
            UInt64.MAX,
            0,
            9007199254740993,
            UInt64.MAX,
            UInt64(9223372036854775807),
        ],
        [False, True, True, True, True],
    )
    _error(source, DataType.INT64, 700, [], 703, "18446744073709551615")
    # A nonempty wrong-length mask disables fast conversion but observes all rows.
    _error(source, DataType.INT64, 700, [False], 703, "18446744073709551615")
    var expected: List[Int64] = [0, 0, 9007199254740993, 0, Int64.MAX]
    var valid: List[Bool] = [False, True, True, False, True]
    _check[DType.int64](
        cast_series(source, DataType.INT64, False, 700, []), expected, valid
    )
    _check[DType.int64](
        cast_series(
            source, DataType.INT64, True, 700, [True, True, True, False, True]
        ),
        expected,
        valid,
    )
    var scalar = _source[DType.int16]([128])
    # Unlike expression fit_mask, this private cast API does not OR mismatches.
    _error(scalar, DataType.INT8, 19, [False, False], 19, "128")
    _check[DType.int8](
        cast_series(scalar, DataType.INT8, True, 19, [False]), [0], [False]
    )
    var empty = _source[DType.uint64](List[UInt64]())
    _check[DType.int64](
        cast_series(empty, DataType.INT64, True, 700, [False]),
        List[Int64](),
        List[Bool](),
    )


def _float_bits(
    actual: Series, expected: List[UInt32], valid: List[Bool], raw: Bool = True
) raises:
    assert_true(actual.dtype() == DataType.FLOAT32)
    assert_equal(len(actual), len(expected))
    var flat = actual.rechunk()
    ref column = flat._data[Column[Float32]]
    for row in range(len(expected)):
        assert_equal(column._valid(row), valid[row])
        if raw or valid[row]:
            assert_equal(bitcast[DType.uint32](column._get(row)), expected[row])


def test_integer_float_casts_keep_single_and_double_rounding_modes() raises:
    var signed = (Int64(1) << 62) + (Int64(1) << 38) + Int64(1)
    var unsigned = (UInt64(1) << 63) + (UInt64(1) << 39) + UInt64(1)
    var s = _source[DType.int64](
        [signed, -signed, Int64.MAX], [True, True, False]
    )
    var u = _source[DType.uint64](
        [unsigned, UInt64.MAX, UInt64.MAX], [True, True, False]
    )
    var present: List[Bool] = [True, True, False]
    for strict in [False, True]:
        _float_bits(
            cast_series(s, DataType.FLOAT32, strict, 0, []),
            [0x5E800001, 0xDE800001, 0],
            present,
        )
        _float_bits(
            cast_series(u, DataType.FLOAT32, strict, 0, []),
            [0x5F000001, 0x5F800000, 0],
            present,
        )
        for mask in [List[Bool](length=3, fill=True), List[Bool]([False])]:
            _float_bits(
                cast_series(s, DataType.FLOAT32, strict, 0, mask),
                [0x5E800000, 0xDE800000, 0],
                present,
            )
            _float_bits(
                cast_series(u, DataType.FLOAT32, strict, 0, mask),
                [0x5F000000, 0x5F800000, 0],
                present,
            )
    var precise = _source[DType.uint64](
        [UInt64.MAX, 9007199254740993, UInt64.MAX], present
    )
    var f64 = cast_series(precise, DataType.FLOAT64, True, 0, [])
    var flat = f64.rechunk()
    ref column = flat._data[Column[Float64]]
    var bits: List[UInt64] = [0x43F0000000000000, 0x4340000000000000, 0]
    for row in range(3):
        assert_equal(column._valid(row), present[row])
        assert_equal(bitcast[DType.uint64](column._get(row)), bits[row])


def test_public_cast_conditionals_preserve_actual_mask_and_total_cast_recipes() raises:
    var value = (Int64(1) << 62) + (Int64(1) << 38) + Int64(1)
    var source = _source[DType.int64](
        [value, -value, Int64.MAX], [True, True, False]
    )
    var frame = DataFrame(
        [source.copy(), Series("keep", BoolColumn([True, False, True]))]
    )
    # A nonnull otherwise passes a real observation mask to the cast.
    var masked = frame.select(
        when(col("keep"))
        .then(col("x").cast(DataType.FLOAT32))
        .otherwise(lit(Float32(0)))
        .alias("result")
    )
    assert_equal(masked.column("result").name(), "result")
    _float_bits(
        masked.column("result"), [0x5E800000, 0, 0], [True, True, False], False
    )
    # Null-otherwise total casts run unmasked, then overlay predicate validity.
    var total = frame.select(
        when(col("keep"))
        .then(col("x").cast(DataType.FLOAT32))
        .end()
        .alias("result")
    )
    _float_bits(
        total.column("result"), [0x5E800001, 0, 0], [True, False, False], False
    )


def _owned_checked(warm: Bool) raises -> Series:
    var source = _narrow_source()
    if warm:
        var flat = source.rechunk()
        _ = flat^
    var result = cast_series(
        source,
        DataType.INT8,
        False,
        317,
        [True, False, True, True, True, True, True, True, True],
    )
    assert_equal(
        source._chunked.value()[].cache[].state.load(), Int64(2 if warm else 0)
    )
    return result^


def test_integer_cast_results_own_buffers_and_preserve_identity_and_infallible_routes() raises:
    for warm in [False, True]:
        _check[DType.int8](
            _owned_checked(warm),
            [7, 0, 0, 8, 0, 127, -128, 0, 3],
            [True, False, False, True, False, True, True, False, True],
        )
    var source = _narrow_source()
    var identity = cast_series(
        source, DataType.INT16, False, 999, List[Bool](length=9, fill=False)
    )
    var before = source.chunks()
    var after = identity.chunks()
    assert_equal(len(before), len(after))
    for k in range(len(before)):
        assert_equal(
            Int(before[k]._data[Column[Int16]]._ptr()),
            Int(after[k]._data[Column[Int16]]._ptr()),
        )
    _check[DType.int16](
        identity,
        [7, 128, 999, 8, -129, 127, -128, -999, 3],
        [True, True, False, True, True, True, True, False, True],
    )
    var wide_source = _narrow_source()
    _check[DType.int64](
        cast_series(wide_source, DataType.INT64, True, 0, []),
        [7, 128, 0, 8, -129, 127, -128, 0, 3],
        [True, True, False, True, True, True, True, False, True],
    )
