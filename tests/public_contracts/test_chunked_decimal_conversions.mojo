"""Remaining decimal conversions preserve values, masks, errors and cold caches."""
# contract-group: types.cast
# contract-group: expr.chunk_windows
# contract-group: expr.decimal_conversions
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true
from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.kernels.cast import cast_series
from tests.support.coverage_compare import assert_frame


def _pieces(input: Series) raises -> Series:
    var parts = List[Series]()
    parts.append(input.slice(0, 0))
    var at = 0
    for size in [1, 2, 0, 3, len(input)]:
        var count = min(size, len(input) - at)
        parts.append(input.slice(at, count))
        at += count
    assert_equal(at, len(input))
    return Series._from_chunks(parts)


def _numbers[
    D: DType
](values: List[Scalar[D]], valid: List[Bool]) raises -> Series:
    var raw = List[Scalar[D]](length=13, fill=Scalar[D](0))
    raw.extend(values.copy())
    var flags = List[Bool](length=13, fill=True)
    flags.extend(valid.copy())
    comptime if D == DType.bool:
        var bits = List[Bool]()
        for value in raw:
            bits.append(Bool(value))
        var parent = Series("x", BoolColumn(bits^, flags.copy()))
        return _pieces(parent.slice(13, len(values)))
    else:
        var parent = Series("x", Column[Scalar[D]](raw^, flags.copy()))
        return _pieces(parent.slice(13, len(values)))


def _texts(values: List[String], valid: List[Bool]) raises -> Series:
    var raw = List[String](length=13, fill="prefix")
    raw.extend(values.copy())
    var flags = List[Bool](length=13, fill=True)
    flags.extend(valid.copy())
    var parent = Series("x", StringColumn(raw.copy(), flags.copy()))
    return _pieces(parent.slice(13, len(values)))


def _decimal_typed[
    D: DType
](raw: List[Int128], valid: List[Bool], dtype: DataType) raises -> Series:
    var values = List[Scalar[D]]()
    for value in raw:
        values.append(value.cast[D]())
    return _numbers[D](values, valid).with_dtype(dtype)


def _decimal(
    width: Int, precision: Int, scale: Int, raw: List[Int128], valid: List[Bool]
) raises -> Series:
    var dtype = DataType.decimal(precision, scale, width)
    if width == 32:
        return _decimal_typed[DType.int32](raw, valid, dtype)
    if width == 64:
        return _decimal_typed[DType.int64](raw, valid, dtype)
    return _decimal_typed[DType.int128](raw, valid, dtype)


def _check_decimal(
    actual: Series, dtype: DataType, raw: List[Int128], valid: List[Bool]
) raises:
    assert_equal(actual.name(), "x")
    assert_true(actual.dtype() == dtype)
    assert_equal(len(actual), len(raw))
    assert_true(not Bool(actual._field_metadata))
    var flat = actual.rechunk()
    for row in range(len(raw)):
        assert_equal(flat._decimal_valid(row), valid[row])
        # Generic decimal destinations initialize ignored/invalid rows to0.
        assert_equal(flat._decimal_raw(row), raw[row])


def _error(input: Series, target: DataType, mask: List[Bool]) raises -> String:
    try:
        _ = cast_series(input, target, True, 37, mask)
    except error:
        return String(error)
    raise Error("Expected a strict cast rejection")


def _decimal_name(width: Int, precision: Int, scale: Int) -> String:
    return (
        ("decimal[" if width == 128 else "decimal" + String(width) + "[")
        + String(precision)
        + ","
        + String(scale)
        + "]"
    )


def test_decimal_rescale_all_width_pairs_ties_growth_and_precision() raises:
    var raw: List[Int128] = [
        1245,
        1255,
        -1245,
        -1255,
        995,
        1005,
        999995,
        7777777,
        7777,
    ]
    var valid: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
    ]
    for source_width in [32, 64, 128]:
        for target_width in [32, 64, 128]:
            var source = _decimal(source_width, 7, 3, raw, valid)
            var target = DataType.decimal(6, 2, target_width)
            _check_decimal(
                source.cast(target),
                target,
                [124, 126, -124, -126, 100, 100, 100000, 0, 778],
                valid,
            )
            assert_equal(
                source._chunked.value()[].cache[].state.load(), Int64(0)
            )
            var grown = DataType.decimal(8, 4, target_width)
            _check_decimal(
                source.cast(grown),
                grown,
                [12450, 12550, -12450, -12550, 9950, 10050, 9999950, 0, 77770],
                valid,
            )
            assert_equal(
                source._chunked.value()[].cache[].state.load(), Int64(0)
            )
        var tight = DataType.decimal(5, 2, source_width)
        _check_decimal(
            _decimal(source_width, 7, 3, raw, valid).cast(tight, strict=False),
            tight,
            [124, 126, -124, -126, 100, 100, 0, 0, 778],
            [True, True, True, True, True, True, False, False, True],
        )
        assert_equal(
            _error(_decimal(source_width, 7, 3, raw, valid), tight, []),
            "cast from "
            + _decimal_name(source_width, 7, 3)
            + " to "
            + _decimal_name(source_width, 5, 2)
            + " failed at row 43 for value '999.995': value exceeds decimal precision 5",
        )


def test_integer_boolean_and_uint64_sources_keep_decimal_exactness() raises:
    for width in [32, 64, 128]:
        var target = DataType.decimal(7, 2, width)
        var integers = _numbers[DType.int64](
            [0, 7, -3, 42, 99], [True, True, False, True, True]
        )
        _check_decimal(
            integers.cast(target),
            target,
            [0, 700, 0, 4200, 9900],
            [True, True, False, True, True],
        )
        assert_equal(integers._chunked.value()[].cache[].state.load(), Int64(0))
        var booleans = _numbers[DType.bool](
            [False, True, True, False, True],
            [True, True, False, True, True],
        )
        _check_decimal(
            booleans.cast(target),
            target,
            [0, 100, 0, 0, 100],
            [True, True, False, True, True],
        )
        assert_equal(booleans._chunked.value()[].cache[].state.load(), Int64(0))
        var whole = DataType.decimal(9, 0, width)
        _check_decimal(
            _numbers[DType.uint64]([0, 9007199, 3], [True, True, False]).cast(
                whole
            ),
            whole,
            [0, 9007199, 0],
            [True, True, False],
        )
    var large: List[UInt64] = [UInt64(9007199254740993), UInt64.MAX, UInt64(7)]
    var wide = DataType.decimal(20, 0, 128)
    var result = _numbers[DType.uint64](large, [True, True, False]).cast(wide)
    _check_decimal(
        result,
        wide,
        [Int128(9007199254740993), UInt64.MAX.cast[DType.int128](), 0],
        [True, True, False],
    )
    assert_frame(
        DataFrame([result.cast(DataType.UINT64)]),
        DataFrame(
            [Series("x", Column[UInt64](large.copy(), [True, True, False]))]
        ),
    )


def test_float_and_string_decimal_sources_keep_text_parser_rules() raises:
    for width in [32, 64, 128]:
        var target = DataType.decimal(6, 2, width)
        var valid: List[Bool] = [True, True, True, False, True]
        var wants: List[Int128] = [125, -250, 0, 0, 0]
        var accepted: List[Bool] = [True, True, False, False, True]
        var floats = _numbers[DType.float32](
            [1.25, -2.5, 0.125, 7.0, 0.0], valid
        )
        _check_decimal(
            floats.cast(target, strict=False),
            target,
            wants,
            accepted,
        )
        assert_equal(floats._chunked.value()[].cache[].state.load(), Int64(0))
        _check_decimal(
            _numbers[DType.float64]([1.25, -2.5, 0.125, 7.0, 0.0], valid).cast(
                target, strict=False
            ),
            target,
            wants,
            accepted,
        )
        var text = _texts(["+1.25", "-2.50", "1.250", "bad", "0"], valid)
        _check_decimal(
            text.cast(target, strict=False),
            target,
            wants,
            accepted,
        )
        assert_equal(text._chunked.value()[].cache[].state.load(), Int64(0))
        assert_equal(
            _error(
                _texts(["+1.25", "-2.50", "1.250", "bad", "0"], valid),
                target,
                [],
            ),
            "cast from string to "
            + _decimal_name(width, 6, 2)
            + " failed at row 39 for value '1.250': decimal has more than 2 fractional digits",
        )


def test_decimal_integer_boolean_float32_and_string_destinations() raises:
    for width in [32, 64, 128]:
        var source = _decimal(
            width,
            7,
            1,
            [5, 15, 25, -5, -15, -25, 19, 12345, 1275],
            [True, True, True, True, True, True, True, False, True],
        )
        assert_frame(
            DataFrame([source.cast(DataType.INT64)]),
            DataFrame(
                [
                    Series(
                        "x",
                        Column[Int64](
                            [0, 2, 2, 0, -2, -2, 2, 0, 128],
                            [
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                False,
                                True,
                            ],
                        ),
                    )
                ]
            ),
        )
        assert_frame(
            DataFrame([source.cast(DataType.INT8, strict=False)]),
            DataFrame(
                [
                    Series(
                        "x",
                        Column[Int8](
                            [0, 2, 2, 0, -2, -2, 2, 0, 0],
                            [
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                False,
                                False,
                            ],
                        ),
                    )
                ]
            ),
        )
        assert_frame(
            DataFrame([source.cast(DataType.UINT64, strict=False)]),
            DataFrame(
                [
                    Series(
                        "x",
                        Column[UInt64](
                            [0, 2, 2, 0, 0, 0, 2, 0, 128],
                            [
                                True,
                                True,
                                True,
                                True,
                                False,
                                False,
                                True,
                                False,
                                True,
                            ],
                        ),
                    )
                ]
            ),
        )
        assert_frame(
            DataFrame([source.cast(DataType.BOOL)]),
            DataFrame(
                [
                    Series(
                        "x",
                        BoolColumn(
                            [
                                False,
                                True,
                                True,
                                False,
                                True,
                                True,
                                True,
                                False,
                                True,
                            ],
                            [
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                False,
                                True,
                            ],
                        ),
                    )
                ]
            ),
        )
        var floats = source.cast(DataType.FLOAT32)
        assert_frame(
            DataFrame([floats.copy()]),
            DataFrame(
                [
                    Series(
                        "x",
                        Column[Float32](
                            [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 1.9, 0.0, 127.5],
                            [
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                False,
                                True,
                            ],
                        ),
                    )
                ]
            ),
        )
        var flat = floats.rechunk()
        assert_equal(flat._data[Column[Float32]]._get(7), Float32(0))
        var strings = source.cast(DataType.STRING)
        assert_frame(
            DataFrame([strings.copy()]),
            DataFrame(
                [
                    Series(
                        "x",
                        StringColumn(
                            [
                                "0.5",
                                "1.5",
                                "2.5",
                                "-0.5",
                                "-1.5",
                                "-2.5",
                                "1.9",
                                "",
                                "127.5",
                            ],
                            [
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                False,
                                True,
                            ],
                        ),
                    )
                ]
            ),
        )
        assert_equal(strings.rechunk()._data[StringColumn]._byte_length(7), 0)
        assert_equal(
            _error(source, DataType.INT8, []),
            "cast from "
            + _decimal_name(width, 7, 1)
            + " to int8 failed at row 45 for value '127.5': out of int8 range",
        )


def test_decimal_conversion_masks_and_first_global_observed_error() raises:
    for width in [32, 64, 128]:
        var source = _texts(
            ["1.25", "bad", "2.50", "3.00", "bad", "4.00", "bad"],
            [True, True, True, False, True, True, False],
        )
        var target = DataType.decimal(6, 2, width)
        _check_decimal(
            cast_series(
                source,
                target,
                True,
                37,
                [True, False, True, True, False, True, True],
            ),
            target,
            [125, 0, 250, 0, 0, 400, 0],
            [True, False, True, False, False, True, False],
        )
        _check_decimal(
            cast_series(
                source,
                target,
                True,
                37,
                [False, False, False, False, False, False, False],
            ),
            target,
            [0, 0, 0, 0, 0, 0, 0],
            [False, False, False, False, False, False, False],
        )
        for size in [0, 1, 6, 8]:
            assert_equal(
                _error(source, target, List[Bool](length=size, fill=False)),
                "cast from string to "
                + _decimal_name(width, 6, 2)
                + " failed at row 38 for value 'bad': invalid decimal value 'bad'",
            )
        assert_equal(
            _error(source, target, [True, False, True, True, True, True, True]),
            "cast from string to "
            + _decimal_name(width, 6, 2)
            + " failed at row 41 for value 'bad': invalid decimal value 'bad'",
        )


def _owned_result(width: Int, warm: Bool, text: Bool) raises -> Series:
    var source = _decimal(
        width,
        7,
        2,
        [150, 250, 7777777, -250, 350],
        [True, True, False, True, True],
    )
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    source._field_metadata = ArcPointer(metadata^)
    if warm:
        var prepared = source.rechunk()
        _ = prepared^
    var result = source.cast(
        DataType.STRING if text else DataType.decimal(7, 1, width)
    )
    assert_true(not Bool(result._field_metadata))
    if warm:
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(2))
    else:
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))
    return result^


def test_decimal_conversion_owns_outputs_and_keeps_ready_and_physical_fallbacks() raises:
    for width in [32, 64, 128]:
        for warm in [False, True]:
            var result = _owned_result(width, warm, False)
            var noise = List[Int64](length=4096, fill=999999)
            assert_equal(noise[0], Int64(999999))
            _check_decimal(
                result,
                DataType.decimal(7, 1, width),
                [15, 25, 0, -25, 35],
                [True, True, False, True, True],
            )
            var text = _owned_result(width, warm, True)
            assert_frame(
                DataFrame([text.copy()]),
                DataFrame(
                    [
                        Series(
                            "x",
                            StringColumn(
                                ["1.50", "2.50", "", "-2.50", "3.50"],
                                [True, True, False, True, True],
                            ),
                        )
                    ]
                ),
            )
        var source = _decimal(width, 7, 2, [150, 250, 350], [True, True, True])
        var cached = _decimal(width, 7, 2, [350, -450], [False, True]).rechunk()
        source._chunked.value()[].cache[].merged = cached._data.copy()
        source._chunked.value()[].cache[].state.store(2)
        assert_frame(
            DataFrame([source.cast(DataType.INT64)]),
            DataFrame([Series("x", Column[Int64]([0, -4], [False, True]))]),
        )
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(2))
    var physical = Series("x", Column[Int64]([150, 250, -250]))
    var tagged = Series("x", physical._data.copy(), DataType.decimal(5, 2, 32))
    assert_frame(
        DataFrame([_pieces(tagged).cast(DataType.INT64)]),
        DataFrame([Series("x", Column[Int64]([2, 2, -2]))]),
    )


def test_decimal_conversion_public_conditions_suppress_only_unobserved_rows() raises:
    for width in [32, 64, 128]:
        for batch in [1, 3, 64]:
            var target = DataType.decimal(6, 2, width)
            var text = DataFrame(
                [
                    _texts(
                        ["1.25", "bad", "2.50", "bad"], [True, True, True, True]
                    ),
                    Series(
                        "take",
                        BoolColumn(
                            [True, True, True, False], [True, False, True, True]
                        ),
                    ),
                ]
            )
            var parsed = (
                text.lazy()
                .select(
                    when(col("take"))
                    .then(col("x").cast(target))
                    .otherwise(lit("0.00").cast(target))
                    .alias("x")
                )
                .collect(engine="cpu", batch_size=batch)
            )
            _check_decimal(
                parsed.column("x"),
                target,
                [125, 0, 250, 0],
                [True, True, True, True],
            )
            var numbers = DataFrame(
                [
                    _decimal(
                        width,
                        5,
                        1,
                        [15, 1275, -25, 1285],
                        [True, True, True, True],
                    ),
                    Series(
                        "take",
                        BoolColumn(
                            [True, True, True, False], [True, False, True, True]
                        ),
                    ),
                ]
            )
            var narrow = (
                numbers.lazy()
                .select(
                    when(col("take"))
                    .then(col("x").cast(DataType.INT8))
                    .otherwise(lit(Int8(0)))
                    .alias("x")
                )
                .collect(engine="cpu", batch_size=batch)
            )
            assert_frame(
                narrow, DataFrame([Series("x", Column[Int8]([2, 0, -2, 0]))])
            )


def test_decimal_conversion_empty_scalar_identity_and_float64_control() raises:
    for width in [32, 64, 128]:
        var target = DataType.decimal(6, 2, width)
        _check_decimal(_texts([], []).cast(target), target, [], [])
        _check_decimal(
            _texts(["-1.25"], [True]).cast(target), target, [-125], [True]
        )
        var scalar = _decimal(width, 6, 2, [125], [False])
        var identity = scalar.cast(scalar.dtype())
        assert_true(identity.dtype() == scalar.dtype())
        assert_equal(identity.rechunk()._decimal_raw(0), Int128(125))
        assert_true(not identity.rechunk()._decimal_valid(0))
        var source = _decimal(width, 6, 1, [15, 25, 7777], [True, False, True])
        var special = cast_series(
            source, DataType.FLOAT64, False, 37, [False, False, False]
        )
        var flat = special.rechunk()
        assert_equal(flat._data[Column[Float64]]._get(1), Float64(2.5))
        assert_frame(
            DataFrame([special.copy()]),
            DataFrame(
                [
                    Series(
                        "x",
                        Column[Float64]([1.5, 2.5, 777.7], [True, False, True]),
                    )
                ]
            ),
        )
