"""Literal temporal parsing and safe rejection before dictionary encoding."""
from std.testing import TestSuite, assert_true, assert_equal, assert_raises
from dataframe import (
    DataFrame,
    Series,
    Column,
    StringColumn,
    Expr,
    col,
    lit,
    when,
)
from dataframe.dtype import DataType
from dataframe.metal import MetalRuntime, metal_installed
from dataframe.accelerator.constants import (
    fold_literal_text_parsing,
    has_literal_text_parsing,
)
from dataframe.accelerator.dictionaries import DictionaryCodes, encode_query


def test_literal_temporal_parsing() raises:
    if not metal_installed():
        return
    var frame = DataFrame(
        [
            Series("d", Column[Int64]([0, 1, 2])).with_dtype(DataType.DATE),
            Series("s", StringColumn(List[String](["x", "y", "z"]))),
        ]
    )
    var parsed = lit("1970-01-02").str().strptime("date")
    var query = frame.lazy().filter(col("d") >= parsed).select(["s", "d"])
    assert_true(has_literal_text_parsing(query))
    var folded = fold_literal_text_parsing(query)
    assert_true(not has_literal_text_parsing(folded))
    assert_true(
        folded.collect(engine="cpu").equals(query.collect(engine="cpu"))
    )
    assert_true(
        query.collect(accelerator=MetalRuntime()).equals(
            query.collect(engine="cpu")
        )
    )
    var empty = frame.slice(0, 0).lazy().filter(col("d") >= parsed)
    assert_true(
        empty.collect(accelerator=MetalRuntime()).equals(
            empty.collect(engine="cpu")
        )
    )
    var nonstrict = frame.lazy().filter(
        col("d") == lit("bad").str().strptime("date", strict=False)
    )
    assert_true(
        nonstrict.collect(accelerator=MetalRuntime()).equals(
            nonstrict.collect(engine="cpu")
        )
    )


def test_literal_datetime_and_time_dtypes() raises:
    if not metal_installed():
        return
    var frame = DataFrame([Series("x", Column[Int32]([1, 2]))])
    var expressions = List[Expr](
        [
            lit("2001-02-03T04:05:06").str().strptime("datetime[us]"),
            lit("04:05:06").str().strptime("time"),
        ]
    )
    for expression in expressions:
        var query = frame.lazy().select_exprs(
            [col("x"), expression.alias("parsed")]
        )
        var folded = fold_literal_text_parsing(query)
        assert_true(
            folded.collect(engine="cpu").equals(query.collect(engine="cpu"))
        )
        assert_true(
            query.collect(accelerator=MetalRuntime()).equals(
                query.collect(engine="cpu")
            )
        )


def test_unlowered_text_rejected_before_encoding() raises:
    var frame = DataFrame(
        [Series("s", StringColumn(List[String](["1970-01-02"])))]
    )
    var query = frame.lazy().select(col("s").str().strptime("date"))
    assert_true(not has_literal_text_parsing(query))
    var dictionary = DictionaryCodes(query)
    with assert_raises(contains="unsupported [dictionary]"):
        _ = encode_query(query, dictionary)
    var invalid = frame.lazy().select(lit("bad").str().strptime("date"))
    with assert_raises(contains="unsupported [literal text parsing]"):
        _ = fold_literal_text_parsing(invalid)


def test_literal_temporal_folding_without_device() raises:
    var frame = DataFrame([Series("x", Column[Int32]([1, 2, 3]))])
    var expressions = List[Expr](
        [
            lit("2000-02-29").str().strptime("date"),
            lit("2000-02-29T12:34:56.123456").str().strptime("datetime[us]"),
            lit("12:34:56.123456789").str().strptime("time"),
            lit("bad").str().strptime("date", strict=False),
        ]
    )
    for expression in expressions:
        var query = frame.lazy().select_exprs(
            [col("x"), expression.alias("parsed")]
        )
        assert_true(
            fold_literal_text_parsing(query)
            .collect(engine="cpu")
            .equals(query.collect(engine="cpu"))
        )


def test_literal_numeric_and_decimal_casts_without_device() raises:
    var frame = DataFrame([Series("x", Column[Int32]([1, 2, 3]))])
    var expressions = List[Expr](
        [
            lit("-128").cast(DataType.INT8),
            lit(True).cast(DataType.INT8),
            lit("2024-01-02").cast(DataType.DATE),
            lit(Int64(86400)).cast(DataType.DATE),
            lit("255").cast(DataType.UINT8),
            lit("-9223372036854775808").cast(DataType.INT64),
            lit("18446744073709551615").cast(DataType.UINT64),
            lit("1.401298464324817e-45").cast(DataType.FLOAT32),
            lit("-0.0").cast(DataType.FLOAT64),
            lit("1234567.89").cast(DataType.decimal(9, 2, 32)),
            lit("1234567890123456.78").cast(DataType.decimal(18, 2, 64)),
            lit("bad").cast(DataType.INT32, strict=False),
            lit("123").cast(DataType.INT32).cast(DataType.FLOAT64),
        ]
    )
    for expression in expressions:
        var query = frame.lazy().select_exprs(
            [col("x"), expression.alias("parsed")]
        )
        assert_true(has_literal_text_parsing(query))
        var folded = fold_literal_text_parsing(query)
        assert_true(not has_literal_text_parsing(folded))
        assert_true(
            folded.collect(engine="cpu").equals(query.collect(engine="cpu"))
        )


def test_literal_cast_rejections_preserve_unobserved_errors() raises:
    var frame = DataFrame([Series("x", Column[Int32]([1, 2]))])
    var invalid = frame.lazy().select(
        when(lit(True))
        .then(lit(Int32(1)))
        .otherwise(lit("bad").cast(DataType.INT32))
    )
    with assert_raises(contains="unsupported [literal text parsing]"):
        _ = fold_literal_text_parsing(invalid)
    var wide = frame.lazy().select(lit("123.45").cast(DataType.decimal(20, 2)))
    with assert_raises(contains="unsupported parsed literal dtype"):
        _ = fold_literal_text_parsing(wide)
    var dynamic = DataFrame(
        [Series("s", StringColumn(List[String](["1", "2"])))]
    )
    var query = dynamic.lazy().select(col("s").cast(DataType.INT32))
    assert_true(not has_literal_text_parsing(query))
    var dictionary = DictionaryCodes(query)
    with assert_raises(contains="String/categorical casts"):
        _ = encode_query(query, dictionary)


def test_literal_numeric_and_decimal_casts_on_metal() raises:
    if not metal_installed():
        return
    var dtype = DataType.decimal(18, 2, 64)
    var frame = DataFrame(
        [
            Series("x", Column[Int32]([1, 2, 3, 4])),
            Series(
                "price",
                Column[Int64]([125, 250, 375, 500], [True, False, True, True]),
            ).with_dtype(dtype),
        ]
    )
    var expressions = List[Expr](
        [
            col("x"),
            lit("18446744073709551615").cast(DataType.UINT64).alias("unsigned"),
            lit("-0.0").cast(DataType.FLOAT64).alias("zero"),
            lit("2024-01-02").cast(DataType.DATE).alias("date"),
            lit(Int64(86400)).cast(DataType.DATE).alias("integer_date"),
            lit("1234567.89").cast(DataType.decimal(9, 2, 32)).alias("small"),
            lit("1234567890123456.78").cast(dtype).alias("large"),
            lit("bad").cast(DataType.INT32, strict=False).alias("null"),
        ]
    )
    var query = frame.lazy().select_exprs(expressions)
    assert_true(
        query.collect(accelerator=MetalRuntime()).equals(
            query.collect(engine="cpu")
        )
    )
    var filtered = (
        frame.lazy().filter(col("price") >= lit("3.75").cast(dtype)).head(1)
    )
    assert_true(
        filtered.collect(accelerator=MetalRuntime()).equals(
            filtered.collect(engine="cpu")
        )
    )
    var child = (
        frame.lazy()
        .with_columns(lit("42").cast(DataType.INT32).alias("parsed"))
        .slice(1, 2)
    )
    assert_true(
        child.collect(accelerator=MetalRuntime()).equals(
            child.collect(engine="cpu")
        )
    )
    var empty = frame.slice(0, 0).lazy().select_exprs(expressions)
    assert_true(
        empty.collect(accelerator=MetalRuntime()).equals(
            empty.collect(engine="cpu")
        )
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
