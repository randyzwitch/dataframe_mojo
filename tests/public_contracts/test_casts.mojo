"""Exact range checks and explicit lossy conversions at the public boundary."""
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    ListColumn,
    StructColumn,
    col,
)
from tests.support.numeric_model import (
    INTEGER_DTYPES,
    FLOAT_DTYPES,
    integer_bounds,
    integer_fits,
    truncated_integer,
)
from tests.support.coverage_compare import assert_float, assert_value
from tests.support.coverage_generators import chunk_layout, offset_layout


def integer_cast_matrix[S: DType, T: DType]() raises:
    var bounds = integer_bounds[S]()
    var raw: List[Int128] = [
        bounds[0],
        bounds[1],
        0,
        1,
        bounds[1] - 1,
        bounds[0],
        bounds[1],
    ]
    var payloads = List[Scalar[S]]()
    for value in raw:
        payloads.append(value.cast[S]())
    for mode in range(3):
        var valid = List[Bool]()
        for row in range(len(raw)):
            valid.append(mode == 0 or (mode == 1 and row % 3 != 0))
        var original = DataFrame(
            [Series("x", Column[Scalar[S]](payloads.copy(), valid.copy()))]
        )
        for layout in range(3):
            var frame = (
                original.copy() if layout
                == 0 else offset_layout(original) if layout
                == 1 else chunk_layout(original)
            )
            var relaxed = frame.column("x").cast(DataType.of(T), strict=False)
            assert_true(relaxed.dtype() == DataType.of(T))
            assert_equal(relaxed.name(), "x")
            assert_equal(len(relaxed), len(raw))
            var success = True
            for row in range(len(raw)):
                var representable = integer_fits[T](raw[row])
                var expected_valid = valid[row] and representable
                assert_equal(relaxed.get(row).is_null(), not expected_valid)
                if expected_valid:
                    assert_equal(
                        relaxed.get(row).numeric[T]().cast[DType.int128](),
                        raw[row],
                    )
                success = success and (not valid[row] or representable)
            if success:
                var strict = frame.select(col("x").cast(DataType.of(T)))
                for row in range(len(raw)):
                    assert_value(strict.item(row, "x"), relaxed.get(row))
            else:
                with assert_raises():
                    _ = frame.select(col("x").cast(DataType.of(T)))
            var empty = frame.clear().column("x").cast(DataType.of(T))
            assert_true(empty.dtype() == DataType.of(T))
            assert_equal(len(empty), 0)


def test_all_integer_width_pairs_preserve_exact_values_or_reject_range() raises:
    comptime for source in range(len(INTEGER_DTYPES)):
        comptime for target in range(len(INTEGER_DTYPES)):
            integer_cast_matrix[
                INTEGER_DTYPES[source], INTEGER_DTYPES[target]
            ]()


def float_integer_cast[S: DType, T: DType]() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var values: List[Float64] = [
        nan,
        inf,
        -inf,
        -0.0,
        0,
        1.75,
        -1.75,
        -128.75,
        127.75,
        256,
        Float64(Int128(1) << 63),
        Float64(Int128(1) << 64),
        1e30,
    ]
    var payloads = List[Scalar[S]]()
    for value in values:
        payloads.append(value.cast[S]())
    for mode in range(3):
        var valid = List[Bool]()
        for row in range(len(values)):
            valid.append(mode == 0 or (mode == 1 and row % 2 == 0))
        var source = Series(
            "x", Column[Scalar[S]](payloads.copy(), valid.copy())
        )
        var actual = source.cast(DataType.of(T), strict=False)
        var any_invalid = False
        for row in range(len(values)):
            var accepted = False
            var expected = Int128(0)
            if valid[row]:
                try:
                    expected = truncated_integer[T](Float64(payloads[row]))
                    accepted = True
                except:
                    any_invalid = True
            assert_equal(actual.get(row).is_null(), not accepted)
            if accepted:
                assert_equal(
                    actual.get(row).numeric[T]().cast[DType.int128](), expected
                )
        if any_invalid:
            with assert_raises():
                _ = source.cast(DataType.of(T))
        else:
            assert_equal(source.cast(DataType.of(T)).null_count(), len(values))


def test_float_to_integer_truncates_and_rejects_nonfinite_and_out_of_range() raises:
    comptime for source in range(len(FLOAT_DTYPES)):
        comptime for target in range(len(INTEGER_DTYPES)):
            float_integer_cast[FLOAT_DTYPES[source], INTEGER_DTYPES[target]]()


def integer_float_cast[S: DType, T: DType]() raises:
    var bounds = integer_bounds[S]()
    var raw: List[Int128] = [
        bounds[0],
        bounds[0] + 1,
        0,
        1,
        bounds[1] - 1,
        bounds[1],
    ]
    for value in [Int128(16777217), Int128(16777219), Int128(9007199254740993)]:
        if integer_fits[S](value):
            raw.append(value)
    var payloads = List[Scalar[S]]()
    for value in raw:
        payloads.append(value.cast[S]())
    for mode in range(3):
        var valid = List[Bool]()
        for row in range(len(raw)):
            valid.append(mode == 0 or (mode == 1 and row % 3 != 0))
        var original = DataFrame(
            [Series("x", Column[Scalar[S]](payloads.copy(), valid.copy()))]
        )
        for layout in range(3):
            var frame = (
                original.copy() if layout
                == 0 else offset_layout(original) if layout
                == 1 else chunk_layout(original)
            )
            var actual = frame.column("x").cast(DataType.of(T))
            assert_true(actual.dtype() == DataType.of(T))
            assert_equal(actual.name(), "x")
            assert_equal(len(actual), len(raw))
            for row in range(len(raw)):
                assert_equal(actual.get(row).is_null(), not valid[row])
                if valid[row]:
                    assert_float(
                        Float64(actual.get(row).numeric[T]()),
                        Float64(raw[row].cast[T]()),
                    )


def test_all_integer_to_float_pairs_follow_declared_precision_and_null_rules() raises:
    comptime for source in range(len(INTEGER_DTYPES)):
        comptime for target in range(len(FLOAT_DTYPES)):
            integer_float_cast[INTEGER_DTYPES[source], FLOAT_DTYPES[target]]()


def float_width_cast[S: DType, T: DType]() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var values: List[Float64] = [-0.0, 0.0, 0.1, 1.25, 1e40, -inf, inf, nan]
    var payloads = List[Scalar[S]]()
    for value in values:
        payloads.append(value.cast[S]())
    for mode in range(3):
        var valid = List[Bool]()
        for row in range(len(values)):
            valid.append(mode == 0 or (mode == 1 and row % 3 != 0))
        var original = DataFrame(
            [Series("x", Column[Scalar[S]](payloads.copy(), valid.copy()))]
        )
        for layout in range(3):
            var frame = (
                original.copy() if layout
                == 0 else offset_layout(original) if layout
                == 1 else chunk_layout(original)
            )
            var actual = frame.column("x").cast(DataType.of(T))
            assert_equal(actual.name(), "x")
            assert_true(actual.dtype() == DataType.of(T))
            assert_equal(len(actual), len(values))
            for row in range(len(values)):
                assert_equal(actual.get(row).is_null(), not valid[row])
                if valid[row]:
                    assert_float(
                        Float64(actual.get(row).numeric[T]()),
                        Float64(payloads[row].cast[T]()),
                    )


def test_all_float_width_pairs_preserve_nonfinites_signed_zero_and_validity() raises:
    comptime for source in range(len(FLOAT_DTYPES)):
        comptime for target in range(len(FLOAT_DTYPES)):
            float_width_cast[FLOAT_DTYPES[source], FLOAT_DTYPES[target]]()


def integer_text_cast[D: DType]() raises:
    var bounds = integer_bounds[D]()
    var expected: List[Int128] = [
        0,
        1,
        -1,
        0,
        7,
        bounds[0],
        bounds[1],
        bounds[0] - 1,
        bounds[1] + 1,
    ]
    var text: List[String] = [
        "0",
        "+1",
        "-1",
        "-0",
        "007",
        String(bounds[0]),
        String(bounds[1]),
        String(bounds[0] - 1),
        String(bounds[1] + 1),
    ]
    text.extend(
        [
            String(""),
            " ",
            " 1",
            "1 ",
            "1.0",
            "1e2",
            "1_000",
            "0x1",
            "+",
            "--1",
            "１",
            "nan",
        ]
    )
    var source = Series("x", Column[String](text.copy()))
    var actual = source.cast(DataType.of(D), strict=False)
    for row in range(len(text)):
        var valid = row < len(expected)
        if valid:
            valid = integer_fits[D](expected[row])
        assert_equal(actual.get(row).is_null(), not valid)
        if valid:
            assert_equal(
                actual.get(row).numeric[D]().cast[DType.int128](), expected[row]
            )
    with assert_raises():
        _ = source.cast(DataType.of(D))
    var nulls = Series(
        "x",
        Column[String](text.copy(), List[Bool](length=len(text), fill=False)),
    )
    assert_equal(nulls.cast(DataType.of(D)).null_count(), len(text))


def test_string_integer_cast_grammar_and_extrema_are_exact() raises:
    comptime for target in range(len(INTEGER_DTYPES)):
        integer_text_cast[INTEGER_DTYPES[target]]()


def test_float_widths_boolean_and_typed_extraction_have_explicit_rules() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var values: List[Float64] = [-0.0, 0.0, 0.1, 1.25, 1e40, -inf, inf, nan]
    var source = Series("x", Column[Float64](values.copy()))
    var narrow = source.cast(DataType.FLOAT32)
    for row in range(len(values)):
        assert_float(
            Float64(narrow.get(row).float32()), Float64(Float32(values[row]))
        )
        with assert_raises():
            _ = narrow.get(row).float64()
        with assert_raises():
            _ = source.get(row).int64()
    var bools = source.cast(DataType.BOOL, strict=False)
    for row in range(len(values)):
        assert_equal(bools.get(row).is_null(), isnan(values[row]))
        if not isnan(values[row]):
            assert_equal(bools.get(row).bool(), values[row] != 0)
    with assert_raises():
        _ = source.cast(DataType.BOOL)
    var integer = Series("u", Column[UInt64]([UInt64.MAX, 9007199254740993, 0]))
    var as_float = integer.cast(DataType.FLOAT64)
    assert_equal(as_float.get(0).float64(), Float64(Int128(1) << 64))
    assert_equal(as_float.get(1).float64(), Float64(9007199254740992))
    assert_equal(as_float.get(2).float64(), Float64(0))
    var bool_text = Series(
        "b", Column[String](["true", "false", "TRUE", "1", " false", ""])
    )
    var parsed = bool_text.cast(DataType.BOOL, strict=False)
    assert_equal(parsed.get(0).bool(), True)
    assert_equal(parsed.get(1).bool(), False)
    for row in range(2, 6):
        assert_true(parsed.get(row).is_null())


def test_decimal_scale_precision_and_half_even_rounding_are_exact() raises:
    for width in [32, 64, 128]:
        var target = DataType.decimal(6, 2, width)
        var source = Series(
            "x",
            Column[String](
                [
                    "1.245",
                    "1.255",
                    "-1.245",
                    "-1.255",
                    "9999.99",
                    "10000.00",
                    "bad",
                ]
            ),
        )
        # Text parsing is exact at the declared scale. Rounding applies when
        # converting an existing decimal value to a smaller scale.
        var parsed = source.cast(DataType.decimal(7, 3, width), strict=False)
        var actual = parsed.cast(target, strict=False)
        assert_true(actual.dtype() == target)
        var expected: List[String] = [
            "1.24",
            "1.26",
            "-1.24",
            "-1.26",
            "9999.99",
        ]
        for row in range(7):
            assert_equal(actual.get(row).is_null(), row >= len(expected))
            if row < len(expected):
                assert_equal(String(actual.get(row)), expected[row])
        with assert_raises():
            _ = source.cast(target)
        var strict = (
            source.slice(0, 5).cast(DataType.decimal(7, 3, width)).cast(target)
        )
        for row in range(5):
            assert_equal(String(strict.get(row)), expected[row])
        var rescaled = actual.cast(DataType.decimal(7, 1, width), strict=False)
        var rounded: List[String] = ["1.2", "1.3", "-1.2", "-1.3", "10000.0"]
        for row in range(5):
            assert_equal(String(rescaled.get(row)), rounded[row])
    var precise = Series(
        "x", Column[String](["9007199254740993", "18446744073709551615"])
    )
    var decimal = precise.cast(DataType.decimal(30, 2))
    assert_equal(String(decimal.get(0)), "9007199254740993.00")
    assert_equal(String(decimal.get(1)), "18446744073709551615.00")
    var wide = decimal.cast(DataType.UINT64)
    assert_equal(wide.get(0).uint64(), UInt64(9007199254740993))
    assert_equal(wide.get(1).uint64(), UInt64.MAX)


def test_temporal_units_and_nested_cast_rejections_preserve_contracts() raises:
    var source = Series(
        "t", Column[Int64]([-1001, -1, 0, 1001], [True, True, False, True])
    ).with_dtype(DataType.datetime("us", "UTC"))
    var milliseconds = source.cast(DataType.datetime("ms", "UTC"))
    var nanos = source.cast(DataType.datetime("ns", "UTC"))
    var expected: List[Int64] = [-2, -1, 0, 1]
    for row in range(4):
        assert_equal(milliseconds.get(row).is_null(), row == 2)
        assert_equal(nanos.get(row).is_null(), row == 2)
        if row != 2:
            assert_equal(milliseconds.get(row).to_physical(), expected[row])
            assert_equal(
                nanos.get(row).to_physical(),
                Int64([-1001000, -1000, 0, 1001000][row]),
            )
    assert_true(milliseconds.dtype() == DataType.datetime("ms", "UTC"))
    var zoned = source.cast(DataType.datetime("us", "America/New_York"))
    for row in [0, 1, 3]:
        assert_equal(
            zoned.get(row).to_physical(), source.get(row).to_physical()
        )
    var nested: List[Series] = [
        Series("x", ListColumn([0, 1], Series("child", Column[Int64]([1])))),
        Series("x", StructColumn([Series("child", Column[Int64]([1]))])),
    ]
    for value in nested:
        for candidate in [value.copy(), value.slice(0, 0)]:
            with assert_raises():
                _ = candidate.cast(DataType.STRING)
            with assert_raises():
                _ = candidate.cast(candidate.dtype(), strict=False)
    with assert_raises():
        _ = Series("x", Column[Int64]([1])).cast(DataType.list(DataType.INT64))
