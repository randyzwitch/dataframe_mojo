"""Literal temporal parsing and safe rejection before dictionary encoding."""
# contract-group: execution.backends
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
from dataframe.metal import MetalRuntime
from dataframe.accelerator.constants import (
    fold_literal_text_parsing,
    has_literal_text_parsing,
)
from dataframe.accelerator.dictionaries import DictionaryCodes, encode_query


from tests.execution_modes.test_backend_metal import require_metal


# requires: metal
def test_native_metal_literal_numeric_and_decimal_casts_on_metal() raises:
    _ = require_metal()
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
