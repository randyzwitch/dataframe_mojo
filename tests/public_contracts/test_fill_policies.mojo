"""Filling and predicates choose source validity without conflating null and NaN."""
# contract-group: expr.null_logic
from std.math import isnan, isinf
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, null
from tests.support.numeric_model import FLOAT_DTYPES
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_float


def test_dataframe_fill_null_resolves_untyped_values_per_column_and_subset() raises:
    var raw = DataFrame(
        [
            Series("i", Column[Int64]([1, Int64.MAX], [True, False])),
            Series("u", Column[UInt8]([1, 255], [True, False])),
            Series("f", Column[Float32]([1, 0], [True, False])),
            Series("s", Column[String](["a", "hidden"], [True, False])),
            Series("b", Column[Bool]([True, True], [True, False])),
        ]
    )
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        var outputs = [
            frame.fill_null(7),
            frame.fill_null(0.5),
            frame.fill_null(300),
            frame.fill_null(lit(Int64(5))),
            frame.fill_null("x"),
        ]
        for index in range(len(outputs)):
            var actual = outputs[index].copy()
            assert_equal(actual.columns(), ["i", "u", "f", "s", "b"])
            assert_equal(actual.dtypes(), frame.dtypes())
            assert_equal(actual.height(), 2)
            for name in actual.columns():
                var eligible = (
                    (index == 0 and (name == "i" or name == "u" or name == "f"))
                    or (index == 1 and name == "f")
                    or (index == 2 and (name == "i" or name == "f"))
                    or (index == 3 and name == "i")
                    or (index == 4 and name == "s")
                )
                assert_equal(actual.item(1, name).is_null(), not eligible)
            if index == 0:
                assert_equal(actual.item(1, "i").int64(), Int64(7))
                assert_equal(
                    actual.item(1, "u").numeric[DType.uint8](), UInt8(7)
                )
                assert_equal(
                    actual.item(1, "f").numeric[DType.float32](), Float32(7)
                )
            elif index == 1:
                assert_equal(
                    actual.item(1, "f").numeric[DType.float32](), Float32(0.5)
                )
            elif index == 2:
                assert_equal(actual.item(1, "i").int64(), Int64(300))
                assert_equal(
                    actual.item(1, "f").numeric[DType.float32](), Float32(300)
                )
            elif index == 3:
                assert_equal(actual.item(1, "i").int64(), Int64(5))
            else:
                assert_equal(actual.item(1, "s").string(), "x")
            assert_equal(actual.item(0, "i").int64(), Int64(1))
            assert_equal(actual.item(0, "u").numeric[DType.uint8](), UInt8(1))
            assert_equal(
                actual.item(0, "f").numeric[DType.float32](), Float32(1)
            )
            assert_equal(actual.item(0, "s").string(), "a")
            assert_true(actual.item(0, "b").bool())
        with assert_raises(contains="integer literal 300 does not fit uint8"):
            _ = frame.fill_null(300, subset=["u"])


def test_float_class_predicates_and_fill_nan_distinguish_all_ieee_classes_from_null() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var source: List[Float64] = [nan, inf, -inf, 0.0, -0.0, 1.5, nan, inf]
    var valid: List[Bool] = [True, True, True, True, True, True, False, False]
    comptime for index in range(len(FLOAT_DTYPES)):
        comptime D = FLOAT_DTYPES[index]
        var values = List[Scalar[D]]()
        for value in source:
            values.append(value.cast[D]())
        var raw = DataFrame(
            [Series("x", Column[Scalar[D]](values^, valid.copy()))]
        )
        for layout in range(3):
            var frame = (
                raw.copy() if layout
                == 0 else offset_layout(raw) if layout
                == 1 else chunk_layout(raw)
            )
            var actual = frame.select_exprs(
                [
                    col("x").is_nan().alias("nan"),
                    col("x").is_not_nan().alias("not_nan"),
                    col("x").is_finite().alias("finite"),
                    col("x").is_infinite().alias("infinite"),
                    col("x").fill_nan(lit(Scalar[D](7))).alias("filled"),
                    col("x").fill_nan(null(DataType.of(D))).alias("nulled"),
                ]
            )
            for name in ["nan", "not_nan", "finite", "infinite"]:
                assert_true(actual.column(name).dtype() == DataType.BOOL)
                for row in range(len(source)):
                    assert_equal(
                        actual.item(row, name).is_null(), not valid[row]
                    )
                    if valid[row]:
                        var expected = Bool(
                            isnan(source[row])
                        ) if name == "nan" else not isnan(
                            source[row]
                        ) if name == "not_nan" else not isnan(
                            source[row]
                        ) and not isinf(
                            source[row]
                        ) if name == "finite" else Bool(
                            isinf(source[row])
                        )
                        assert_equal(actual.item(row, name).bool(), expected)
            for name in ["filled", "nulled"]:
                assert_true(actual.column(name).dtype() == DataType.of(D))
                for row in range(len(source)):
                    var absent = not valid[row] or (
                        name == "nulled" and isnan(source[row])
                    )
                    assert_equal(actual.item(row, name).is_null(), absent)
                    if not absent:
                        assert_float(
                            Float64(actual.item(row, name).numeric[D]()),
                            Float64(7) if isnan(source[row]) else source[row],
                        )
    var integers = DataFrame([Series("x", Column[Int64]([1]))])
    for frame in [integers.copy(), integers.clear()]:
        for expression in [
            col("x").fill_nan(0),
            col("x").is_nan(),
            col("x").is_finite(),
            col("x").is_infinite(),
        ]:
            with assert_raises():
                _ = frame.select(expression)
