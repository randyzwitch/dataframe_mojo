"""cast: full source/target matrix, strict vs non-strict, and round trips."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import Column, DataFrame, Expr, Series, col, lit, when

comptime MIN = Int64(-9223372036854775807) - 1
comptime MAX = Int64(9223372036854775807)


def nan() -> Float64:
    return Float64(0) / Float64(0)


def inf() -> Float64:
    return Float64(1) / Float64(0)


def values(series: Series) raises -> List[String]:
    var out = List[String]()
    for i in range(len(series)):
        out.append(String(series.get(i)))
    return out^


def one(df: DataFrame, expr: Expr) raises -> List[String]:
    return values(df.select(expr.alias("r")).column("r"))


def test_numeric_and_bool_conversions() raises:
    var ints = DataFrame(
        [
            Series(
                "i",
                Column[Int64](
                    [0, -3, MAX, MIN, 7], [True, True, True, True, False]
                ),
            )
        ]
    )
    assert_equal(
        one(ints, col("i").cast("float64")),
        [
            String("0.0"),
            "-3.0",
            "9.223372036854776e+18",
            "-9.223372036854776e+18",
            "null",
        ],
    )
    assert_equal(
        one(ints, col("i").cast("bool")),
        [String("false"), "true", "true", "true", "null"],
    )
    assert_equal(
        one(ints, col("i").cast("string")),
        [
            String("0"),
            "-3",
            "9223372036854775807",
            "-9223372036854775808",
            "null",
        ],
    )
    assert_equal(one(ints, col("i").cast("int64")), one(ints, col("i")))
    var floats = DataFrame(
        [
            Series(
                "f",
                Column[Float64](
                    [2.9, -2.9, -0.0, 9.2e18, -9223372036854775808.0, 1.5],
                    [True, True, True, True, True, False],
                ),
            )
        ]
    )
    assert_equal(
        one(floats, col("f").cast("int64")),
        [String("2"), "-2", "0", "9200000000000000000", String(MIN), "null"],
    )
    assert_equal(one(floats, col("f").cast("bool"))[2], "false")
    assert_equal(one(floats, col("f").cast("string"))[2], "-0.0")
    var bools = DataFrame(
        [Series("b", Column[Bool]([True, False, True], [True, True, False]))]
    )
    assert_equal(one(bools, col("b").cast("int64")), [String("1"), "0", "null"])
    assert_equal(
        one(bools, col("b").cast("float64")), [String("1.0"), "0.0", "null"]
    )
    assert_equal(
        one(bools, col("b").cast("string")), [String("true"), "false", "null"]
    )


def test_float_to_int_failures() raises:
    var bad = DataFrame(
        [
            Series(
                "f",
                Column[Float64](
                    [1, nan(), inf(), 9223372036854775808.0, -1e19]
                ),
            )
        ]
    )
    assert_equal(
        one(bad, col("f").cast("int64", strict=False)),
        [String("1"), "null", "null", "null", "null"],
    )
    with assert_raises(
        contains="cast from float64 to int64 failed at row 1 for value 'nan'"
    ):
        _ = bad.select(col("f").cast("int64"))
    assert_equal(one(bad, col("f").cast("bool", strict=False))[1], "null")
    with assert_raises(contains="NaN has no Boolean value"):
        _ = bad.select(col("f").cast("bool"))


def test_string_parsing_matches_csv_rules() raises:
    var text = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    [
                        "42",
                        "+7",
                        "-0",
                        " 1",
                        "1.5",
                        "9223372036854775808",
                        "",
                        "x",
                    ],
                    [True, True, True, True, True, True, True, True],
                ),
            )
        ]
    )
    assert_equal(
        one(text, col("s").cast("int64", strict=False)),
        [String("42"), "7", "0", "null", "null", "null", "null", "null"],
    )
    with assert_raises(contains="failed at row 3 for value ' 1'"):
        _ = text.select(col("s").cast("int64"))
    var floats = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ["1.5", "nan", "-inf", "1e999", "Infinity", " 2"]
                ),
            )
        ]
    )
    assert_equal(
        one(floats, col("s").cast("float64", strict=False)),
        [String("1.5"), "nan", "-inf", "null", "inf", "null"],
    )
    var bools = DataFrame(
        [Series("s", Column[String](["true", "false", "True", "1"]))]
    )
    assert_equal(
        one(bools, col("s").cast("bool", strict=False)),
        [String("true"), "false", "null", "null"],
    )


def test_round_trips_through_string() raises:
    var frame = DataFrame(
        [
            Series("i", Column[Int64]([MIN, MAX, 0, -1, 123456789])),
            Series("f", Column[Float64]([0.1, -0.0, 1e300, nan(), -inf()])),
            Series("b", Column[Bool]([True, False, True, False, True])),
        ]
    )
    var text = frame.cast({"i": "string", "f": "string", "b": "string"})
    assert_equal(text.dtypes(), [String("string"), "string", "string"])
    var back = text.cast({"i": "int64", "f": "float64", "b": "bool"})
    assert_true(back.equals(frame))
    assert_equal(
        frame.cast({"f": "int64"}, strict=False).columns(), frame.columns()
    )


def test_series_cast_masks_and_validation() raises:
    var series = Series("s", Column[String](["5", "oops"]))
    assert_equal(
        values(series.cast("int64", strict=False)), [String("5"), "null"]
    )
    with assert_raises(contains="Unknown cast dtype"):
        _ = series.cast("int32")
    var frame = DataFrame([series.copy()])
    with assert_raises(contains="Unknown cast dtype: date"):
        _ = frame.select(col("s").cast("date"))
    with assert_raises(contains="Unknown column"):
        _ = frame.cast({"zzz": "int64"})
    # A strict cast only fails for rows its branch can select.
    var guarded = frame.select(
        when(col("s").ne(lit(String("oops"))))
        .then(col("s").cast("int64"))
        .otherwise(lit(Int64(-1)))
        .alias("v")
    )
    assert_equal(values(guarded.column("v")), [String("5"), "-1"])
    assert_equal(
        frame.select(lit(Int64(3)).cast("string")).item().string(), "3"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
