"""String-adjacent casts preserve grammar, formatting, windows and ownership."""
# contract-group: expr.string_cast_chunks
# contract-group: types.cast
from std.memory import bitcast
from std.math import isnan
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
from dataframe.string_view import StringViewBuilder
from dataframe.kernels.cast import cast_series
from tests.support.coverage_compare import assert_frame


def _strings(
    values: List[String], valid: List[Bool], layout: Int = 0
) raises -> Series:
    var raw = List[String](length=13, fill="prefix")
    raw.extend(values.copy())
    var flags = List[Bool](length=13, fill=True)
    flags.extend(valid.copy())
    var offsets = Series("x", StringColumn(raw.copy(), flags.copy()))
    var builder = StringViewBuilder()
    for row in range(len(raw)):
        if flags[row]:
            builder.append(StringSlice(raw[row]))
        else:
            builder.append_null()
    var views = Series("x", StringColumn(builder^.finish()))
    var pieces = List[Series]()
    pieces.append(offsets.slice(13, 0))
    var at = 0
    var piece = 0
    for size in [1, 2, 3, len(values)]:
        var count = min(size, len(values) - at)
        if layout == 1 or (layout == 2 and piece % 2 == 1):
            pieces.append(views.slice(13 + at, count))
        else:
            pieces.append(offsets.slice(13 + at, count))
        at += count
        piece += 1
    return Series._from_chunks(pieces)


def _numbers[
    D: DType
](values: List[Scalar[D]], valid: List[Bool]) raises -> Series:
    var raw = List[Scalar[D]](length=13, fill=Scalar[D](91))
    raw.extend(values.copy())
    var flags = List[Bool](length=13, fill=True)
    flags.extend(valid.copy())
    var parent = Series("x", Column[Scalar[D]](raw^, flags.copy()))
    var window = parent.slice(13, len(values))
    var first = min(1, len(values))
    return Series._from_chunks(
        [window.slice(0, first), window.slice(first, len(values) - first)]
    )


def _check[
    D: DType
](actual: Series, values: List[Scalar[D]], valid: List[Bool]) raises:
    assert_frame(
        DataFrame([actual.copy()]),
        DataFrame(
            [Series("x", Column[Scalar[D]](values.copy(), valid.copy()))]
        ),
    )
    var flat = actual.rechunk()
    for row in range(len(values)):
        if not valid[row]:
            assert_equal(flat._data[Column[Scalar[D]]]._get(row), Scalar[D](0))


def _check_strings(
    actual: Series, values: List[String], valid: List[Bool]
) raises:
    assert_frame(
        DataFrame([actual.copy()]),
        DataFrame([Series("x", StringColumn(values.copy(), valid.copy()))]),
    )
    var flat = actual.rechunk()
    for row in range(len(values)):
        if not valid[row]:
            assert_equal(flat._data[StringColumn]._byte_length(row), 0)


def _check_bool(actual: Series, values: List[Bool], valid: List[Bool]) raises:
    assert_frame(
        DataFrame([actual.copy()]),
        DataFrame([Series("x", BoolColumn(values.copy(), valid.copy()))]),
    )
    var flat = actual.rechunk()
    for row in range(len(values)):
        if not valid[row]:
            assert_equal(flat._data[BoolColumn]._get(row), False)


def _error(
    source: Series,
    target: DataType,
    mask: List[Bool],
    row: Int,
    text: String,
    cause: String,
) raises:
    var rejected = False
    try:
        _ = cast_series(source, target, True, 300, mask)
    except error:
        var message = String(error)
        assert_true(message.find("failed at row " + String(row)) >= 0)
        assert_true(message.find("for value '" + text + "'") >= 0)
        assert_true(message.find(cause) >= 0)
        rejected = True
    assert_true(rejected, "an observed invalid String cast must reject")


def test_string_integer_parser_priority_exact_unsigned_and_signs() raises:
    var source = _strings(
        ["+7", "256x", "256", "-0", "999", "255"],
        [True, True, True, True, False, True],
    )
    _error(source, DataType.UINT8, [], 301, "256x", "non-decimal integer byte")
    _error(
        source,
        DataType.UINT8,
        [True, False, True, True, True, True],
        302,
        "256",
        "out of ",
    )
    _check[DType.uint8](
        cast_series(source, DataType.UINT8, False, 300, []),
        [7, 0, 0, 0, 0, 255],
        [True, False, False, True, False, True],
    )
    var huge = _strings(
        ["18446744073709551615", "-0", "+1", "18446744073709551616"],
        [True, True, True, True],
        1,
    )
    _check[DType.uint64](
        cast_series(huge, DataType.UINT64, False, 0, []),
        [UInt64.MAX, 0, 1, 0],
        [True, True, True, False],
    )
    _error(
        huge, DataType.INT64, [], 300, "18446744073709551615", "Int64 overflow"
    )
    _error(huge, DataType.UINT64, [], 303, "18446744073709551616", "overflow")


def test_string_float_grammar_specials_bridge_rounding_and_zero_sign() raises:
    var source = _strings(
        [
            "-0",
            "inf",
            "-Infinity",
            "nan",
            "1e400",
            " 1",
            "1.0000000596046447753906250000000000000001",
            "1e40",
        ],
        [True, True, True, True, True, True, True, True],
        2,
    )
    var inf = Float64(1) / Float64(0)
    var nan = Float64(0) / Float64(0)
    _check[DType.float64](
        cast_series(source, DataType.FLOAT64, False, 0, []),
        [-0.0, inf, -inf, nan, 0, 0, 1.000000059604644775390625, 1e40],
        [True, True, True, True, False, False, True, True],
    )
    _error(source, DataType.FLOAT64, [], 304, "1e400", "Float64 overflow")
    _error(
        source,
        DataType.FLOAT64,
        [False, False, False, False, False, True, False, False],
        305,
        " 1",
        "surrounding whitespace",
    )
    var f32 = cast_series(source, DataType.FLOAT32, False, 0, [])
    _check[DType.float32](
        f32,
        [
            -0.0,
            Float32(inf),
            Float32(-inf),
            Float32(nan),
            0,
            0,
            1,
            Float32(inf),
        ],
        [True, True, True, True, False, False, True, True],
    )
    var flat = f32.rechunk()
    assert_equal(
        bitcast[DType.uint32](flat._data[Column[Float32]]._get(0)),
        UInt32(0x80000000),
    )
    assert_equal(
        bitcast[DType.uint32](flat._data[Column[Float32]]._get(6)),
        UInt32(0x3F800000),
    )
    assert_true(isnan(flat._data[Column[Float32]]._get(3)))
    _error(
        _strings(["+nan"], [True]),
        DataType.FLOAT64,
        [],
        300,
        "+nan",
        "invalid Float64",
    )


def test_string_boolean_requires_lowercase_and_preserves_direct_mask_rules() raises:
    var source = _strings(
        ["true", "false", "TRUE", "1", "bad", "true"],
        [True, True, True, True, False, True],
        1,
    )
    _error(
        source,
        DataType.BOOL,
        [],
        302,
        "TRUE",
        "Boolean must be exactly 'true' or 'false'",
    )
    _check_bool(
        cast_series(source, DataType.BOOL, False, 0, []),
        [True, False, False, False, False, True],
        [True, True, False, False, False, True],
    )
    assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))
    _check_bool(
        cast_series(
            source,
            DataType.BOOL,
            True,
            0,
            [True, True, False, False, True, True],
        ),
        [True, False, False, False, False, True],
        [True, True, False, False, False, True],
    )
    _error(
        source,
        DataType.BOOL,
        List[Bool](length=9, fill=False),
        302,
        "TRUE",
        "Boolean must be exactly",
    )
    var scalar = _strings(["true"], [True])
    _check_bool(
        cast_series(scalar, DataType.BOOL, True, 0, [False, False]),
        [True],
        [True],
    )
    _check_bool(
        cast_series(scalar, DataType.BOOL, True, 0, [False]), [False], [False]
    )


def test_numeric_and_boolean_string_formatting_keeps_original_width_and_null_spans() raises:
    var f32 = Float32(1) / Float32(3)
    var inf = Float64(1) / Float64(0)
    var nan = Float64(0) / Float64(0)
    var source = _numbers[DType.float32](
        [f32, -0.0, Float32(inf), Float32(nan), 9],
        [True, True, True, True, False],
    )
    _check_strings(
        cast_series(source, DataType.STRING, True, 0, []),
        [
            String(f32),
            String(Float32(-0.0)),
            String(Float32(inf)),
            String(Float32(nan)),
            "",
        ],
        [True, True, True, True, False],
    )
    _check_strings(
        cast_series(
            _numbers[DType.float64]([1.0 / 3.0, -0.0, inf], [True, True, True]),
            DataType.STRING,
            True,
            0,
            [True, False, True],
        ),
        [String(Float64(1.0 / 3.0)), "", String(inf)],
        [True, False, True],
    )
    _check_strings(
        cast_series(
            _numbers[DType.uint64](
                [UInt64.MAX, 9007199254740993, 0], [True, True, False]
            ),
            DataType.STRING,
            True,
            0,
            [],
        ),
        ["18446744073709551615", "9007199254740993", ""],
        [True, True, False],
    )
    var parent = Series(
        "x", BoolColumn([False, True, False, True], [True, True, True, False])
    )
    var booleans = Series._from_chunks([parent.slice(1, 1), parent.slice(2, 2)])
    _check_strings(
        cast_series(booleans, DataType.STRING, True, 0, []),
        ["true", "false", ""],
        [True, True, False],
    )
    assert_equal(booleans._chunked.value()[].cache[].state.load(), Int64(0))


def test_string_cast_offset_view_mixed_layouts_null_poison_and_error_bytes() raises:
    for layout in range(3):
        var source = _strings(
            [
                "7",
                "long_invalid_null_payload_123456789",
                "-8",
                "9",
                "10",
                "11",
                "12",
            ],
            [True, False, True, True, True, True, True],
            layout,
        )
        _check[DType.int16](
            cast_series(source, DataType.INT16, True, 0, []),
            [7, 0, -8, 9, 10, 11, 12],
            [True, False, True, True, True, True, True],
        )
        _check[DType.int16](
            cast_series(
                source,
                DataType.INT16,
                True,
                0,
                [True, True, False, True, True, True, True],
            ),
            [7, 0, 0, 9, 10, 11, 12],
            [True, False, False, True, True, True, True],
        )
    for text in List[String](["1\0x", "1\n", "\t1", '"1"', "é"]):
        _error(
            _strings([text], [True], 1),
            DataType.INT64,
            [],
            300,
            text,
            "non-decimal integer byte",
        )
    _check[DType.int8](
        cast_series(_strings([], [], 2), DataType.INT8, True, 0, []), [], []
    )


def _owned_parse(warm: Bool) raises -> Series:
    var source = _strings(["7", "bad", "9", "11"], [True, True, False, True], 2)
    if warm:
        var flat = source.rechunk()
        _ = flat^
    var result = cast_series(source, DataType.INT8, False, 0, [])
    assert_equal(
        source._chunked.value()[].cache[].state.load(), Int64(2 if warm else 0)
    )
    return result^


def _owned_format() raises -> Series:
    var source = _numbers[DType.int64]([7, -8, 9], [True, True, False])
    var result = cast_series(source, DataType.STRING, True, 0, [])
    assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))
    return result^


def test_public_string_cast_conditionals_results_own_buffers_and_retags_alias() raises:
    for warm in [False, True]:
        _check[DType.int8](
            _owned_parse(warm), [7, 0, 0, 11], [True, False, False, True]
        )
    _check_strings(_owned_format(), ["7", "-8", ""], [True, True, False])
    var frame = DataFrame(
        [
            _strings(["true", "bad", "true"], [True, True, False], 2),
            Series("keep", BoolColumn([True, False, True])),
        ]
    )
    var booleans = frame.select(
        when(col("keep"))
        .then(col("x").cast(DataType.BOOL))
        .otherwise(lit(False))
        .alias("x")
    )
    _check_bool(booleans.column("x"), [True, False, False], [True, True, False])
    var numeric_frame = DataFrame(
        [
            _strings(["7", "bad", "9"], [True, True, False], 1),
            Series("keep", BoolColumn([True, False, True])),
        ]
    )
    var numeric = numeric_frame.select(
        when(col("keep"))
        .then(col("x").cast(DataType.INT8))
        .otherwise(lit(Int8(0)))
        .alias("x")
    )
    _check[DType.int8](numeric.column("x"), [7, 0, 0], [True, True, False])
    var source = _strings(["7", "null_poison", "9"], [True, False, True], 2)
    var identity = cast_series(
        source, DataType.STRING, True, 0, [False, False, False]
    )
    var binary = cast_series(
        source, DataType.BINARY, True, 0, [False, False, False]
    )
    assert_true(binary.dtype() == DataType.BINARY)
    var original = source.chunks()
    var same = identity.chunks()
    var retagged = binary.chunks()
    assert_equal(len(original), len(same))
    assert_equal(len(original), len(retagged))
    for index in range(len(original)):
        assert_true(
            original[index]
            ._data[StringColumn]
            ._shares_buffers_with(same[index]._data[StringColumn])
        )
        assert_true(
            original[index]
            ._data[StringColumn]
            ._shares_buffers_with(retagged[index]._data[StringColumn])
        )
