"""Bare scalars in expressions and untyped numeric literals."""
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.binding import bind
from dataframe.dtype import NUMERIC_DTYPES


def frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("i", Column[Int64]([1, 2, 3, 4], [True, True, False, True])),
            Series("f", Column[Float64]([0.5, 1.5, 2.5, 3.5])),
            Series("s", StringColumn(["a", "b", "c", "a"])),
            Series("b", Column[Bool]([True, False, True, False])),
            Series("u8", Column[UInt8]([0, 10, 200, 255])),
            Series("f32", Column[Float32]([1.0, 2.0, 3.0, 4.0])),
        ]
    )


def dtype_of(df: DataFrame, e: Expr) raises -> DataType:
    return df.select(e)._columns[0].dtype()


def error_of(df: DataFrame, e: Expr) raises -> String:
    try:
        _ = df.select(e)
    except err:
        return String(err)
    return "no error"


def test_operators_accept_bare_numbers() raises:
    var df = frame()
    var r = df.select_exprs(
        [
            (col("i") > 1).alias("gt"),
            (col("i") * 10).alias("times"),
            (col("f") * 2.5).alias("scaled"),
            (col("f") + 1).alias("f_plus_int"),
            (col("i") // 2).alias("half"),
            (col("i") % 3).alias("mod"),
            col("i").pow(2).alias("sq"),
            (-col("i")).alias("neg"),
        ]
    )
    assert_true(r.item(1, "gt").bool())
    assert_equal(r.item(3, "times").int64(), 40)
    assert_true(r.item(2, "times").is_null())
    assert_equal(r.item(0, "scaled").float64(), 1.25)
    assert_equal(r.item(0, "f_plus_int").float64(), 1.5)
    assert_equal(r.item(3, "half").int64(), 2)
    assert_equal(r.item(3, "mod").int64(), 1)
    assert_equal(r.item(1, "sq").int64(), 4)
    assert_equal(r.item(0, "neg").int64(), -1)
    # Runtime variables convert the same way as literals.
    var threshold = 2
    assert_equal(df.filter(col("i") >= threshold).height(), 2)


def test_reflected_operators() raises:
    var df = frame()
    var r = df.select_exprs(
        [
            (1 + col("i")).alias("a"),
            (10 - col("i")).alias("b"),
            (2 * col("f")).alias("c"),
            (1.0 / col("f")).alias("d"),
        ]
    )
    assert_equal(r.item(0, "a").int64(), 2)
    assert_equal(r.item(3, "b").int64(), 6)
    assert_equal(r.item(1, "c").float64(), 3.0)
    assert_equal(r.item(0, "d").float64(), 2.0)
    # A literal on the left names the output "literal", as in Polars.
    assert_equal((1 + col("i"))._name, "literal")


def test_untyped_literals_adopt_every_numeric_dtype() raises:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        var df = DataFrame([Series("x", Column[Scalar[D]]([1, 2, 3]))])
        var dtype = DataType.of(D)
        assert_true(dtype_of(df, col("x") + 1) == dtype, dtype.name())
        assert_true(dtype_of(df, 2 * col("x")) == dtype, dtype.name())
        assert_true(dtype_of(df, col("x") * (1 + 1)) == dtype, dtype.name())
        assert_equal(df.filter(col("x") > 1).height(), 2)
        assert_true(dtype_of(df, col("x").fill_null(0)) == dtype, dtype.name())
        comptime if D.is_floating_point():
            assert_true(dtype_of(df, col("x") * 0.5) == dtype, dtype.name())


def test_adoption_errors_explain_themselves() raises:
    var df = frame()
    var message = error_of(df, col("u8") + 300)
    assert_true("integer literal 300 does not fit uint8" in message, message)
    assert_true("lit(...)" in message, message)
    message = error_of(df, col("u8") > -1)
    assert_true("integer literal -1 does not fit uint8" in message, message)
    message = error_of(df, col("i") * 2.5)
    assert_true(
        "float literal 2.5 cannot adopt int64 (no implicit float-to-integer"
        in message,
        message,
    )
    message = error_of(df, col("s") + 1)
    assert_true(
        "integer literal 1 cannot be used with string" in message, message
    )
    message = error_of(df, col("b") & 1)
    assert_true(
        "integer literal 1 cannot be used with bool" in message, message
    )
    # Typed literals keep their explicit type (no adoption).
    message = error_of(df, col("u8") + lit(Int64(1)))
    assert_true(
        "requires matching dtypes, found uint8 and int64" in message, message
    )


def test_standalone_literals_default() raises:
    var df = frame()
    assert_true(dtype_of(df, lit(Int64(1)) + 2) == DataType.INT64)
    var alone = DataFrame([], height=3).select_exprs(
        [
            Expr(7).alias("i"),
            Expr(0.5).alias("f"),
            (Expr(1) + 2.5).alias("mixed"),
            (Expr(3) > 2).alias("cmp"),
        ]
    )
    assert_equal(alone.height(), 1)
    assert_true(alone._columns[0].dtype() == DataType.INT64)
    assert_true(alone._columns[1].dtype() == DataType.FLOAT64)
    assert_equal(alone.item(0, "mixed").float64(), 3.5)
    assert_true(alone.item(0, "cmp").bool())
    # Broadcasting a bare literal against rows.
    var rows = df.select_exprs([col("i"), Expr(9).alias("nine")])
    assert_equal(rows.height(), 4)
    assert_equal(rows.item(3, "nine").int64(), 9)


def test_strings_bools_and_methods() raises:
    var df = frame()
    assert_equal(df.filter(col("s") == "a").height(), 2)
    assert_equal(df.filter(col("s") != "a").height(), 2)
    assert_equal(df.filter(col("s") >= "b").height(), 2)
    assert_equal(df.filter(col("s").is_in(["a", "c"])).height(), 3)
    assert_equal(df.filter(col("i").is_in([1, 4])).height(), 2)
    assert_equal(df.filter(col("u8").is_in([10, 255])).height(), 2)
    assert_equal(df.filter(col("b") == True).height(), 2)
    var r = df.select_exprs(
        [
            col("i").fill_null(0).alias("filled"),
            when(col("i") > 2).then(1).otherwise(0).alias("flag"),
            when(col("s") == "a").then("yes").otherwise("no").alias("label"),
            when(col("f") > 1).then(col("f")).otherwise(0).alias("f_or_0"),
            col("f").clip(1, 3).alias("clipped"),
            when(col("u8") > 100).then(col("u8")).otherwise(1).alias("u"),
        ]
    )
    assert_equal(r.item(2, "filled").int64(), 0)
    assert_equal(r.item(3, "flag").int64(), 1)
    assert_equal(r.item(0, "label").string(), "yes")
    assert_equal(r.item(1, "label").string(), "no")
    assert_equal(r.item(0, "f_or_0").float64(), 0.0)
    assert_true(r._columns[3].dtype() == DataType.FLOAT64)
    assert_equal(r.item(0, "clipped").float64(), 1.0)
    assert_true(r._columns[5].dtype() == DataType.UINT8)
    assert_equal(r.item(0, "u").uint8(), 1)


def test_float_literal_adopts_float32_and_fusion_is_kept() raises:
    var df = frame()
    var r = df.select((col("f32") * 0.5 + 1).alias("y"))
    assert_true(r._columns[0].dtype() == DataType.FLOAT32)
    assert_equal(r.item(3, "y").float32(), 3.0)
    # Integer literals next to Float64 become float literals, so the whole
    # expression stays eligible for the fused SIMD path.
    var bound = bind((col("f") * 2 + 1) > 3, df._columns)
    assert_true(bound.fusible[len(bound.fusible) - 1])


def test_lists_of_names_are_not_ambiguous() raises:
    var df = frame()
    assert_equal(df.select(["i", "f"]).width(), 2)
    assert_equal(df.group_by(["s"]).agg(col("i").sum()).height(), 3)
    assert_equal(df.sort(["s", "i"]).item(0, "s").string(), "a")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
