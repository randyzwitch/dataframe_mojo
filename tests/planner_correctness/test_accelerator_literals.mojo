"""Literal temporal parsing and safe rejection before dictionary encoding."""
# contract-group: planner.routing
from std.testing import assert_true, assert_equal, assert_raises
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
from dataframe_metal.metal import MetalRuntime, metal_installed
from dataframe_metal.accelerator.constants import (
    fold_literal_text_parsing,
    has_literal_text_parsing,
)
from dataframe_metal.accelerator.dictionaries import (
    DictionaryCodes,
    encode_query,
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
    assert_true(
        fold_literal_text_parsing(wide)
        .collect(engine="cpu")
        .equals(wide.collect(engine="cpu"))
    )
    var dynamic = DataFrame(
        [Series("s", StringColumn(List[String](["1", "2"])))]
    )
    var query = dynamic.lazy().select(col("s").cast(DataType.INT32))
    assert_true(not has_literal_text_parsing(query))
    var dictionary = DictionaryCodes(query)
    with assert_raises(contains="String/categorical casts"):
        _ = encode_query(query, dictionary)
