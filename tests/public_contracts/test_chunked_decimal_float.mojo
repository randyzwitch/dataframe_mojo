"""Decimal float conversion preserves payloads, validity and adapter boundaries."""
# contract-group: types.cast
# contract-group: expr.chunk_windows
from std.memory import ArcPointer, bitcast
from std.testing import assert_equal, assert_false, assert_true
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.kernels.cast import cast_series
from tests.support.coverage_compare import assert_frame


def _decimal_source[D: DType](offset: Int, width: Int) raises -> Series:
    var raw = List[Scalar[D]](length=offset, fill=Scalar[D](87))
    var flags = List[Bool](length=offset, fill=True)
    var seeds: List[Int64] = [123, -456, 0, 999, -999, 50, -50]
    var valid: List[Bool] = [True, False, True, True, False, True, True]
    for i in range(21):
        raw.append(Scalar[D](seeds[i % 7]))
        flags.append(valid[i % 7])
    var physical = Series("amount", Column[Scalar[D]](raw^, flags.copy()))
    var typed = physical.with_dtype(DataType.decimal(9, 2, width))
    var window = typed.slice(offset, 21)
    var source = Series._from_chunks(
        [
            window.slice(0, 1),
            window.slice(1, 7),
            window.slice(8, 0),
            window.slice(8, 13),
        ]
    )
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    source._field_metadata = ArcPointer(metadata^)
    return source^


def _expected() raises -> Series:
    var seeds: List[Float64] = [1.23, -4.56, 0, 9.99, -9.99, 0.5, -0.5]
    var flags: List[Bool] = [True, False, True, True, False, True, True]
    var values = List[Float64]()
    var valid = List[Bool]()
    for i in range(21):
        values.append(seeds[i % 7])
        valid.append(flags[i % 7])
    return Series("amount", Column[Float64](values^, valid.copy()))


def _check(actual: Series, expected: Series) raises:
    assert_frame(DataFrame([actual.copy()]), DataFrame([expected.copy()]))
    assert_equal(actual.name(), String("amount"))
    assert_equal(actual.dtype(), DataType.FLOAT64)
    assert_false(actual.is_chunked())
    assert_false(Bool(actual._field_metadata))
    ref got = actual._data[Column[Float64]]
    ref want = expected._data[Column[Float64]]
    # Null payloads are initialized and observable in this specialized kernel.
    for row in range(len(expected)):
        assert_equal(
            bitcast[DType.uint64](got._get(row)),
            bitcast[DType.uint64](want._get(row)),
        )
        assert_equal(got._valid(row), want._valid(row))


def test_decimal_float_all_widths_offsets_masks_and_hidden_null_payloads() raises:
    comptime for k in range(3):
        comptime D = DType.int32 if k == 0 else DType.int64 if k == 1 else DType.int128
        comptime WIDTH = 32 if k == 0 else 64 if k == 1 else 128
        for offset in range(8):
            var source = _decimal_source[D](offset, WIDTH)
            var expected = _expected()
            for size in [0, 1, 20, 21, 22]:
                var mask = List[Bool](length=size, fill=False)
                for strict in [False, True]:
                    var result = cast_series(
                        source, DataType.FLOAT64, strict, -701, mask
                    )
                    _check(result, expected)
                    assert_equal(
                        source._chunked.value()[].cache[].state.load(), 0
                    )
            # Contiguous differential control is separate from literal oracle.
            var flat = source.rechunk()
            _check(
                cast_series(flat, DataType.FLOAT64, True, 999, List[Bool]()),
                expected,
            )

        var dense_values = List[Scalar[D]]()
        dense_values.append(Scalar[D](100))
        dense_values.append(Scalar[D](-200))
        var dense = Series(
            "amount", Column[Scalar[D]](dense_values^)
        ).with_dtype(DataType.decimal(9, 2, WIDTH))
        var dense_chunks = Series._from_chunks(
            [dense.slice(0, 1), dense.slice(1, 1)]
        )
        var dense_out = cast_series(
            dense_chunks, DataType.FLOAT64, True, 0, [False, False]
        )
        _check(dense_out, Series("amount", Column[Float64]([1.0, -2.0])))
        assert_equal(len(dense_out._data[Column[Float64]]._bits), 0)
        assert_equal(dense_chunks._chunked.value()[].cache[].state.load(), 0)


def test_decimal_float_int128_integer_rounding_and_public_full_schema() raises:
    var raw: List[Int128] = [
        Int128(9007199254740993),
        Int128(9007199254740995),
        -Int128(9007199254740993),
        -Int128(9007199254740995),
        0,
    ]
    var physical = Series(
        "amount", Column[Int128](raw^, [True, False, True, False, True])
    )
    var source = physical.with_dtype(DataType.decimal(38, 0, 128))
    var chunks = Series._from_chunks([source.slice(0, 1), source.slice(1, 4)])
    var values: List[Float64] = [
        9007199254740992.0,
        9007199254740996.0,
        -9007199254740992.0,
        -9007199254740996.0,
        0,
    ]
    var expected = Series(
        "amount", Column[Float64](values^, [True, False, True, False, True])
    )
    _check(cast_series(chunks, DataType.FLOAT64, False, 3, [False]), expected)
    var public_result = (
        DataFrame([chunks.copy()])
        .lazy()
        .select(col("amount").cast(DataType.FLOAT64))
        .collect()
    )
    assert_frame(public_result, DataFrame([expected.copy()]))


def _owned_result() raises -> Series:
    var source = _decimal_source[DType.int64](7, 64)
    return cast_series(source, DataType.FLOAT64, True, 0, List[Bool]())


def test_decimal_float_output_owns_buffers_after_all_input_owners_drop() raises:
    var result = _owned_result()
    var noise = List[Int64](length=4096, fill=987654321)
    _check(result, _expected())
    assert_equal(noise[0], 987654321)


def test_decimal_float_published_cache_values_validity_and_length_are_authoritative() raises:
    var source = _decimal_source[DType.int32](5, 32)
    _ = source.rechunk()
    var replacement = Series(
        "replacement", Column[Int32]([700, -800, 900], [False, True, True])
    )
    source._chunked.value()[].cache[].merged = replacement._data.copy()
    assert_equal(source._chunked.value()[].cache[].state.load(), 2)
    var expected = Series(
        "amount", Column[Float64]([7.0, -8.0, 9.0], [False, True, True])
    )
    _check(
        cast_series(source, DataType.FLOAT64, False, 42, [False, False, False]),
        expected,
    )
    assert_equal(source._chunked.value()[].cache[].state.load(), 2)


def test_decimal_float_initialized_wrong_physical_width_keeps_legacy_adapter() raises:
    var physical = Series(
        "amount", Column[Int64]([123, -456, 700], [True, False, True])
    )
    # Initialized private storage is legal to inspect; this bypasses with_dtype's
    # physical compatibility check and deliberately exercises the old adapter.
    var wrong_width = Series(
        "amount", physical._data.copy(), DataType.decimal(9, 2, 32)
    )
    var chunks = Series._from_chunks(
        [wrong_width.slice(0, 1), wrong_width.slice(1, 2)]
    )
    var expected = Series(
        "amount", Column[Float64]([1.23, -4.56, 7.0], [True, False, True])
    )
    _check(cast_series(chunks, DataType.FLOAT64, True, 0, [False]), expected)
    assert_equal(chunks._chunked.value()[].cache[].state.load(), 2)
    _check(
        cast_series(wrong_width, DataType.FLOAT64, False, 0, List[Bool]()),
        expected,
    )
