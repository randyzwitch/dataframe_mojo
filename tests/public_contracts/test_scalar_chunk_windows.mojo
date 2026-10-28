"""Scalar consumers preserve values and observation across unequal chunks."""
# contract-group: expr.chunk_windows
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
    sum_horizontal,
    when,
)
from tests.support.coverage_compare import assert_frame


def pieces(series: Series, sizes: List[Int]) raises -> Series:
    var chunks = List[Series]()
    var first = 0
    for size in sizes:
        chunks.append(series.slice(first, size))
        first += size
    assert_equal(first, len(series))
    return Series._from_chunks(chunks)


def source() raises -> DataFrame:
    return DataFrame(
        [
            pieces(
                Series(
                    "a",
                    Column[Int64](
                        [-1, 4, 9, 16, 25], [True, False, True, True, True]
                    ),
                ),
                [2, 0, 3],
            ),
            pieces(
                Series(
                    "b",
                    Column[Int64](
                        [10, 20, 30, 40, 50], [True, True, True, False, True]
                    ),
                ),
                [1, 3, 1],
            ),
        ]
    )


def test_chunk_arithmetic_unary_casts_and_scalar_broadcast_have_complete_outputs() raises:
    var frame = source()
    var result = frame.select_exprs(
        [
            (col("a") + col("b")).alias("sum"),
            col("a").abs().alias("abs"),
            (col("a") + 7).alias("scalar"),
            col("a").cast(DataType.FLOAT32).alias("cast"),
        ]
    )
    var valid: List[Bool] = [True, False, True, True, True]
    var expected = DataFrame(
        [
            Series(
                "sum",
                Column[Int64](
                    [9, 0, 39, 0, 75], [True, False, True, False, True]
                ),
            ),
            Series("abs", Column[Int64]([1, 0, 9, 16, 25], valid)),
            Series("scalar", Column[Int64]([6, 0, 16, 23, 32], valid)),
            Series("cast", Column[Float32]([-1, 0, 9, 16, 25], valid)),
        ]
    )
    assert_frame(result, expected)


def test_chunk_math_conditionals_and_horizontal_null_rules() raises:
    var result = source().select_exprs(
        [
            col("a").abs().sqrt().alias("root"),
            when(col("a") % 2 == 0)
            .then(col("a"))
            .otherwise(col("b"))
            .alias("chosen"),
            sum_horizontal([col("a"), col("b")]).alias("horizontal"),
            col("a").is_null().alias("null"),
        ]
    )
    assert_frame(
        result,
        DataFrame(
            [
                Series(
                    "root",
                    Column[Float64](
                        [1, 0, 3, 4, 5], [True, False, True, True, True]
                    ),
                ),
                Series("chosen", Column[Int64]([10, 20, 30, 16, 50])),
                Series("horizontal", Column[Int64]([9, 20, 39, 16, 75])),
                Series(
                    "null", Column[Bool]([False, True, False, False, False])
                ),
            ]
        ),
    )


def test_chunk_cast_errors_keep_global_rows_and_inactive_rows_unobserved() raises:
    var frame = DataFrame(
        [
            pieces(
                Series("s", StringColumn(["10", "20", "30", "bad", "50"])),
                [2, 2, 1],
            ),
            pieces(
                Series("take", Column[Bool]([True, True, True, False, True])),
                [1, 3, 1],
            ),
        ]
    )
    var rejected = False
    try:
        _ = frame.select(col("s").cast(DataType.INT64))
    except error:
        rejected = True
        assert_true(String(error).find("row 3") >= 0)
    assert_true(rejected)
    var result = frame.select(
        when(col("take"))
        .then(col("s").cast(DataType.INT64))
        .otherwise(lit(Int64(0)))
        .alias("value")
    )
    assert_frame(
        result, DataFrame([Series("value", Column[Int64]([10, 20, 30, 0, 50]))])
    )
    rejected = False
    try:
        _ = frame.slice(2, 3).select(col("s").cast(DataType.INT64))
    except error:
        rejected = True
        assert_true(String(error).find("row 1") >= 0)
    assert_true(rejected)


def test_empty_chunked_slices_retain_result_types_and_names() raises:
    var result = (
        source()
        .slice(1, 0)
        .select_exprs(
            [
                col("a").cast(DataType.FLOAT32).alias("a32"),
                (col("a") + col("b")).alias("sum"),
            ]
        )
    )
    assert_equal(result.height(), 0)
    assert_equal(result.columns(), ["a32", "sum"])
    assert_true(result.column("a32").dtype() == DataType.FLOAT32)
    assert_true(result.column("sum").dtype() == DataType.INT64)
