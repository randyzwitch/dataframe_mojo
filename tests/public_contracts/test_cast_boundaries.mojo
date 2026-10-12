"""Categorical value conversions and strict lexical boundaries use public APIs."""
from std.collections import Dict
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.dtype import CategoricalDictionary
from tests.support.numeric_model import (
    INTEGER_DTYPES,
    FLOAT_DTYPES,
    integer_bounds,
)
from tests.support.coverage_compare import assert_float, assert_value
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.cast_boundaries_reference import (
    first_observed_dictionary,
    float_lexical_examples,
)


def _layout(frame: DataFrame, layout: Int) raises -> DataFrame:
    return (
        frame.copy() if layout
        == 0 else offset_layout(frame) if layout
        == 1 else chunk_layout(frame)
    )


def _dictionary(entries: List[String]) -> DataType:
    var dictionary = CategoricalDictionary()
    for entry in entries:
        dictionary.append(entry)
    return DataType.categorical(dictionary^)


def _assert_text(
    actual: Series, expected: List[String], valid: List[Bool]
) raises:
    assert_equal(actual.name(), "x")
    assert_true(actual.dtype() == DataType.STRING)
    assert_equal(len(actual), len(expected))
    for row in range(len(expected)):
        assert_equal(actual.get(row).is_null(), not valid[row])
        if valid[row]:
            assert_equal(actual.get(row).string(), expected[row])


def test_categorical_encoding_and_decoding_preserve_unicode_empty_null_and_dictionary_order() raises:
    var text: List[String] = [
        "z",
        "",
        "é",
        "z",
        "😀abcdefghijklmnop",
        "poison",
        "é",
    ]
    for mode in range(3):
        var valid = List[Bool](length=len(text), fill=mode != 2)
        if mode == 1:
            valid[0] = False
            valid[5] = False
        var reference = first_observed_dictionary(text, valid)
        var raw = DataFrame(
            [Series("x", Column[String](text.copy(), valid.copy()))]
        )
        for layout in range(3):
            var frame = _layout(raw, layout)
            for expression in [False, True]:
                var actual = (
                    frame.select(col("x").cast(DataType.CATEGORICAL))
                    .column("x") if expression else frame.column("x")
                    .cast(DataType.CATEGORICAL)
                )
                assert_true(actual.dtype().is_categorical())
                assert_equal(actual.name(), "x")
                assert_equal(len(actual), len(text))
                var dictionary = actual.dtype().dictionary()
                assert_equal(len(dictionary[]), len(reference[0]))
                for code in range(len(reference[0])):
                    assert_equal(
                        String(dictionary[].get(code)), reference[0][code]
                    )
                var physical = actual.with_dtype(DataType.UINT32)
                for row in range(len(text)):
                    assert_equal(actual.get(row).is_null(), not valid[row])
                    if valid[row]:
                        assert_equal(actual.get(row).string(), text[row])
                        assert_equal(
                            physical.get(row).numeric[DType.uint32](),
                            reference[1][row],
                        )
                _assert_text(actual.cast(DataType.STRING), text, valid)
                var identity = actual.cast(DataType.CATEGORICAL)
                assert_true(identity.dtype() == actual.dtype())
                for row in range(len(text)):
                    assert_value(identity.get(row), actual.get(row))
            var empty = frame.clear().column("x").cast(DataType.CATEGORICAL)
            assert_equal(len(empty), 0)
            assert_true(empty.dtype().is_categorical())
            assert_equal(len(empty.dtype().dictionary()[]), 0)


def test_categorical_recode_maps_values_to_supplied_dictionary_codes() raises:
    var text: List[String] = ["z", "", "é", "z", "long external value", "bad"]
    var valid: List[Bool] = [True, True, True, True, True, False]
    var target = _dictionary(["unused", "é", "long external value", "", "z"])
    var expected: List[UInt32] = [4, 3, 1, 4, 2, 0]
    var source = Series("x", Column[String](text.copy(), valid.copy()))
    for encoded in [False, True]:
        var original = source.cast(
            DataType.CATEGORICAL
        ) if encoded else source.copy()
        for layout in range(3):
            var frame = _layout(DataFrame([original.copy()]), layout)
            var actual = frame.column("x").cast(target)
            assert_true(actual.dtype() == target)
            var physical = actual.with_dtype(DataType.UINT32)
            for row in range(len(text)):
                assert_equal(actual.get(row).is_null(), not valid[row])
                if valid[row]:
                    assert_equal(
                        physical.get(row).numeric[DType.uint32](), expected[row]
                    )
                    assert_equal(actual.get(row).string(), text[row])
            _assert_text(actual.cast(DataType.STRING), text, valid)
            assert_true(
                frame.clear().column("x").cast(target).dtype() == target
            )


def test_categorical_decoding_ignores_unused_members_and_hidden_invalid_codes() raises:
    var dtype = _dictionary(["bad", "7", "8", "unused😀abcdefghijklmnop"])
    var source = Series(
        "x", Column[UInt32]([1, UInt32.MAX, 2, 1], [True, False, True, True])
    ).with_dtype(dtype)
    for layout in range(3):
        var frame = _layout(DataFrame([source.copy()]), layout)
        var numeric = frame.column("x").cast(DataType.INT64)
        var expression = frame.select(col("x").cast(DataType.INT64)).column("x")
        var expected: List[Int64] = [7, 0, 8, 7]
        for row in range(4):
            assert_equal(numeric.get(row).is_null(), row == 1)
            assert_value(numeric.get(row), expression.get(row))
            if row != 1:
                assert_equal(numeric.get(row).int64(), expected[row])
        var text: List[String] = ["7", "", "8", "7"]
        _assert_text(
            frame.column("x").cast(DataType.STRING),
            text,
            [True, False, True, True],
        )


def test_categorical_recode_does_not_require_unused_source_members_in_target() raises:
    var source = Series("x", Column[UInt32]([1, 1])).with_dtype(
        _dictionary(["unused source", "present"])
    )
    var target = _dictionary(["present"])
    var actual = source.cast(target)
    assert_true(actual.dtype() == target)
    assert_equal(len(actual), 2)
    for row in range(2):
        assert_equal(actual.get(row).string(), "present")
        assert_equal(
            actual.with_dtype(DataType.UINT32).get(row).numeric[DType.uint32](),
            UInt32(0),
        )


def test_categorical_recode_relaxed_conversion_nulls_missing_observed_values() raises:
    var source = Series(
        "x", Column[String](["present", "missing", "present"])
    ).cast(DataType.CATEGORICAL)
    var target = _dictionary(["present"])
    var actual = source.cast(target, strict=False)
    assert_true(actual.dtype() == target)
    assert_equal(len(actual), 3)
    for row in range(3):
        assert_equal(actual.get(row).is_null(), row == 1)
        if row != 1:
            assert_equal(actual.get(row).string(), "present")


def _numeric_category_roundtrip[D: DType]() raises:
    var values = List[Scalar[D]]()
    comptime if D.is_floating_point():
        var nan = Float64(0) / Float64(0)
        var inf = Float64(1) / Float64(0)
        for value in [-0.0, 0.0, 0.1, 1.25, inf, -inf, nan]:
            values.append(value.cast[D]())
        values.append(Scalar[D].MAX)
        values.append(-Scalar[D].MAX)
    else:
        var bounds = integer_bounds[D]()
        for value in [bounds[0], bounds[1], Int128(0), Int128(1)]:
            values.append(value.cast[D]())
    for mode in range(3):
        var valid = List[Bool](length=len(values), fill=mode != 2)
        if mode == 1:
            valid[0] = False
        var original = Series(
            "x", Column[Scalar[D]](values.copy(), valid.copy())
        )
        for layout in range(3):
            var frame = _layout(DataFrame([original.copy()]), layout)
            var category = frame.column("x").cast(DataType.CATEGORICAL)
            assert_true(category.dtype().is_categorical())
            assert_equal(category.name(), "x")
            var actual = category.cast(DataType.of(D))
            assert_true(actual.dtype() == DataType.of(D))
            assert_equal(len(actual), len(values))
            for row in range(len(values)):
                assert_value(actual.get(row), original.get(row))
            var expression = frame.select(
                col("x").cast(DataType.CATEGORICAL).cast(DataType.of(D))
            ).column("x")
            for row in range(len(values)):
                assert_value(expression.get(row), original.get(row))


def test_all_numeric_widths_roundtrip_through_categorical_values_with_exact_bits() raises:
    comptime for i in range(len(INTEGER_DTYPES)):
        _numeric_category_roundtrip[INTEGER_DTYPES[i]]()
    comptime for i in range(len(FLOAT_DTYPES)):
        _numeric_category_roundtrip[FLOAT_DTYPES[i]]()


def test_boolean_decimal_and_temporal_categorical_roundtrips_keep_logical_values() raises:
    var valid: List[Bool] = [True, False, True]
    var sources = List[Series]()
    sources.append(Series("x", Column[Bool]([False, True, True], valid.copy())))
    for width in [32, 64, 128]:
        sources.append(
            Series("x", Column[Int128]([125, -250, 0], valid.copy()))
            .with_dtype(DataType.decimal(8, 2))
            .cast(DataType.decimal(8, 2, width))
        )
    for dtype in [
        DataType.DATE,
        DataType.TIME,
        DataType.datetime("ns", "UTC"),
        DataType.datetime("us"),
        DataType.datetime("ms", "America/New_York"),
    ]:
        sources.append(
            Series("x", Column[Int64]([1, -1, 0], valid.copy())).with_dtype(
                dtype
            )
        )
    for source in sources:
        for layout in range(3):
            var frame = _layout(DataFrame([source.copy()]), layout)
            var category = frame.column("x").cast(DataType.CATEGORICAL)
            assert_true(category.dtype().is_categorical())
            var actual = category.cast(source.dtype())
            assert_true(actual.dtype() == source.dtype())
            assert_equal(actual.name(), "x")
            for row in range(3):
                assert_value(actual.get(row), source.get(row))


def test_string_and_categorical_float_parsing_follow_explicit_literal_grammar() raises:
    var model = float_lexical_examples()
    comptime for i in range(len(FLOAT_DTYPES)):
        comptime D = FLOAT_DTYPES[i]
        for encoded in [False, True]:
            for mode in range(3):
                var valid = List[Bool](length=len(model[0]), fill=mode != 2)
                if mode == 1:
                    for row in range(len(valid)):
                        valid[row] = row % 3 != 0
                var source = Series(
                    "x", Column[String](model[0].copy(), valid.copy())
                )
                if encoded:
                    source = source.cast(DataType.CATEGORICAL)
                for layout in range(3):
                    var frame = _layout(DataFrame([source.copy()]), layout)
                    var actual = frame.column("x").cast(
                        DataType.of(D), strict=False
                    )
                    assert_true(actual.dtype() == DataType.of(D))
                    assert_equal(actual.name(), "x")
                    assert_equal(len(actual), len(valid))
                    for row in range(len(valid)):
                        var accepted = valid[row] and model[2][row]
                        assert_equal(actual.get(row).is_null(), not accepted)
                        if accepted:
                            assert_float(
                                Float64(actual.get(row).numeric[D]()),
                                Float64(model[1][row].cast[D]()),
                            )
                    if mode == 2:
                        assert_equal(
                            frame.column("x").cast(DataType.of(D)).null_count(),
                            len(valid),
                        )
                    else:
                        with assert_raises():
                            _ = frame.column("x").cast(DataType.of(D))
                    var expression = frame.select(
                        col("x").cast(DataType.of(D), strict=False)
                    ).column("x")
                    for row in range(len(valid)):
                        assert_value(expression.get(row), actual.get(row))
                    assert_equal(
                        len(frame.clear().column("x").cast(DataType.of(D))), 0
                    )


def test_categorical_integer_boolean_decimal_casts_use_values_and_strict_row_rules() raises:
    var source = Series(
        "x",
        Column[String](
            ["7", "bad", "-2", "999", "bad"], [True, True, True, True, False]
        ),
    ).cast(DataType.CATEGORICAL)
    var raw = DataFrame(
        [
            source.copy(),
            Series("pick", Column[Bool]([True, False, True, True, True])),
        ]
    )
    for layout in range(3):
        var frame = _layout(raw, layout)
        var actual = frame.column("x").cast(DataType.INT8, strict=False)
        var expected: List[Int8] = [7, 0, -2, 0, 0]
        for row in range(5):
            assert_equal(actual.get(row).is_null(), row == 1 or row >= 3)
            if row == 0 or row == 2:
                assert_equal(
                    actual.get(row).numeric[DType.int8](), expected[row]
                )
        var failed = False
        try:
            _ = frame.column("x").cast(DataType.INT8)
        except error:
            assert_true(String(error).find("failed at row 1") >= 0)
            assert_true(String(error).find("value 'bad'") >= 0)
            failed = True
        assert_true(failed)
        var selected = frame.select(
            when(col("pick"))
            .then(col("x").cast(DataType.INT64))
            .otherwise(lit(Int64(0)))
            .alias("x")
        ).column("x")
        var wanted: List[Int64] = [7, 0, -2, 999, 0]
        for row in range(5):
            assert_equal(selected.get(row).is_null(), row == 4)
            if row != 4:
                assert_equal(selected.get(row).int64(), wanted[row])
    var boolean = (
        Series(
            "x",
            Column[String](
                ["true", "false", "TRUE", "1", "bad"],
                [True, True, True, True, False],
            ),
        )
        .cast(DataType.CATEGORICAL)
        .cast(DataType.BOOL, strict=False)
    )
    for row in range(5):
        assert_equal(boolean.get(row).is_null(), row >= 2)
        if row < 2:
            assert_equal(boolean.get(row).bool(), row == 0)
    var decimal = (
        Series("x", Column[String](["1.25", "-2.50", "bad"]))
        .cast(DataType.CATEGORICAL)
        .cast(DataType.decimal(6, 2), strict=False)
    )
    assert_equal(String(decimal.get(0)), "1.25")
    assert_equal(String(decimal.get(1)), "-2.50")
    assert_true(decimal.get(2).is_null())


def test_duration_categorical_conversion_preserves_formatted_unit_text() raises:
    for unit in ["ns", "us", "ms"]:
        var source = Series(
            "x", Column[Int64]([1, -1, 0], [True, True, False])
        ).with_dtype(DataType.duration(unit))
        var actual = source.cast(DataType.CATEGORICAL)
        assert_true(actual.dtype().is_categorical())
        var expected = (
            "0.000000001s" if unit
            == "ns" else "0.000001s" if unit
            == "us" else "0.001s"
        )
        assert_equal(actual.get(0).string(), expected)
        assert_equal(actual.get(1).string(), "-" + expected)
        assert_true(actual.get(2).is_null())


def test_binary_to_categorical_rejects_unsupported_target_even_for_valid_utf8() raises:
    # docs/expressions.md explicitly restricts binary casts to String. An
    # unsupported target must reject even valid UTF-8 and strict=False; row
    # conversion policy cannot make a new source/target pair supported.
    for byte in [UInt8(97), UInt8(255)]:
        for valid in [True, False]:
            var source = Series.binary("x", [[byte]], [valid])
            for strict in [False, True]:
                var rejected = False
                try:
                    _ = source.cast(DataType.CATEGORICAL, strict=strict)
                except error:
                    assert_true(
                        String(error).find("unsupported cast") >= 0
                        or String(error).find(
                            "cannot cast binary to categorical"
                        )
                        >= 0
                    )
                    rejected = True
                assert_true(
                    rejected,
                    "binary to categorical must reject unsupported target",
                )


def test_dataframe_cast_replaces_named_columns_in_schema_order_and_keeps_other_values() raises:
    var raw = DataFrame(
        [
            Series("x", Column[String](["z", "bad", ""], [True, False, True])),
            Series("y", Column[Int64]([1, 300, -2])),
            Series("flag", Column[Bool]([False, True, False])),
        ]
    )
    var changes = Dict[String, String]()
    changes["y"] = "int8"
    changes["x"] = "categorical"
    for layout in range(3):
        var frame = _layout(raw, layout)
        var actual = frame.cast(changes, strict=False)
        assert_equal(actual.columns(), ["x", "y", "flag"])
        assert_equal(actual.height(), 3)
        assert_true(actual.column("x").dtype().is_categorical())
        assert_true(actual.column("y").dtype() == DataType.INT8)
        assert_true(actual.column("flag").dtype() == DataType.BOOL)
        for row in range(3):
            assert_equal(actual.item(row, "x").is_null(), row == 1)
            assert_equal(actual.item(row, "y").is_null(), row == 1)
            assert_value(actual.item(row, "flag"), raw.item(row, "flag"))
            if row != 1:
                assert_equal(
                    actual.item(row, "x").string(), "z" if row == 0 else ""
                )
                assert_equal(
                    actual.item(row, "y").numeric[DType.int8](),
                    Int8(1 if row == 0 else -2),
                )
        with assert_raises():
            _ = frame.cast(changes)
        var empty = frame.clear().cast(changes)
        assert_equal(empty.columns(), actual.columns())
        assert_equal(empty.height(), 0)
        assert_true(empty.column("x").dtype().is_categorical())
        assert_true(empty.column("y").dtype() == DataType.INT8)
        for invalid_name in [False, True]:
            var invalid = Dict[String, String]()
            invalid[
                "missing" if invalid_name else "x"
            ] = "int8" if invalid_name else "unknown_dtype"
            with assert_raises():
                _ = frame.cast(invalid, strict=False)
            with assert_raises():
                _ = frame.clear().cast(invalid, strict=False)


def test_expression_categorical_cast_retains_explicit_target_dictionary() raises:
    var target = _dictionary(["unused", "a", "z"])
    var source = Series("x", Column[String](["z", "a", "z"]))
    var actual = (
        DataFrame([source.copy()]).select(col("x").cast(target)).column("x")
    )
    assert_true(
        actual.dtype() == target,
        "expression categorical cast must retain supplied target dictionary",
    )
    var expected: List[UInt32] = [2, 1, 2]
    for row in range(3):
        assert_equal(actual.get(row).string(), "a" if row == 1 else "z")
        assert_equal(
            actual.with_dtype(DataType.UINT32).get(row).numeric[DType.uint32](),
            expected[row],
        )
