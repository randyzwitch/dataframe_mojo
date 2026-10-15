"""Exact products have one logical value across dense and nullable storage."""
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col, lit, when
from tests.support.coverage_generators import offset_layout, chunk_layout


def exact_scale_fixture(nullable: Bool) raises -> DataFrame:
    var valid: List[Bool] = [True, True, True, not nullable]
    return DataFrame(
        [
            Series("a", Column[Int128]([2, -2, 0, 2], valid.copy())).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("b", Column[Int128]([3, 3, 3, 3], valid.copy())).with_dtype(
                DataType.decimal(38, 38)
            ),
            Series("p", Column[Bool]([True, True, True, False])),
        ]
    )


def test_decimal_exact_product_layout_and_nulls() raises:
    for nullable in [False, True]:
        var source = exact_scale_fixture(nullable)
        for frame in [
            source.copy(),
            offset_layout(source),
            chunk_layout(source),
        ]:
            var result = (
                frame.lazy().select(col("a") * col("b")).collect(engine="cpu")
            )
            assert_equal(result.column("a").dtype(), DataType.decimal(38, 38))
            for row in range(3):
                assert_equal(
                    result.column("a")._decimal_raw(row),
                    Int128(6) if row
                    == 0 else Int128(-6) if row
                    == 1 else Int128(0),
                )
            assert_equal(result.column("a").null_count(), Int(nullable))
            var masked = (
                frame.lazy()
                .select(
                    when(col("p"))
                    .then(col("a") * col("b"))
                    .otherwise(lit("0").cast(DataType.decimal(38, 38)))
                )
                .collect(engine="cpu")
            )
            assert_equal(masked.column("a")._decimal_raw(3), Int128(0))


def capped_precision_fixture(nullable: Bool) raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "a", Column[Int64]([1_000_000_000, 1], [True, not nullable])
            ).with_dtype(DataType.decimal(18, 0, 64)),
            Series("b", Column[Int64]([1_000_000_000, 1])).with_dtype(
                DataType.decimal(18, 0, 64)
            ),
            Series("p", Column[Bool]([False, True])),
        ]
    )


def test_decimal_exact_product_precision_and_mask() raises:
    for nullable in [False, True]:
        var source = capped_precision_fixture(nullable)
        for frame in [
            source.copy(),
            offset_layout(source),
            chunk_layout(source),
        ]:
            with assert_raises(contains="decimal precision"):
                _ = (
                    frame.lazy()
                    .select(col("a") * col("b"))
                    .collect(engine="cpu")
                )
            var masked = (
                frame.lazy()
                .select(
                    when(col("p"))
                    .then(col("a") * col("b"))
                    .otherwise(lit("0").cast(DataType.decimal(18, 0, 64)))
                )
                .collect(engine="cpu")
            )
            assert_equal(masked.column("a")._decimal_raw(0), Int128(0))
            assert_equal(masked.column("a").null_count(), Int(nullable))


def test_decimal_product_physical_values_check_result_before_narrowing() raises:
    for nullable in [False, True]:
        for raw in [Int64(99), Int64.MAX]:
            var source = DataFrame(
                [
                    Series(
                        "a", Column[Int64]([raw, 1], [True, not nullable])
                    ).with_dtype(DataType.decimal(1, 0, 64)),
                    Series("b", Column[Int64]([2, 1])).with_dtype(
                        DataType.decimal(1, 0, 64)
                    ),
                ]
            )
            for frame in [
                source.copy(),
                offset_layout(source),
                chunk_layout(source),
            ]:
                with assert_raises(contains="decimal precision"):
                    _ = (
                        frame.lazy()
                        .select(col("a") * col("b"))
                        .collect(engine="cpu")
                    )
        var fitting = DataFrame(
            [
                Series(
                    "a", Column[Int64]([39, 1], [True, not nullable])
                ).with_dtype(DataType.decimal(1, 0, 64)),
                Series("b", Column[Int64]([2, 1])).with_dtype(
                    DataType.decimal(1, 0, 64)
                ),
            ]
        )
        var result = (
            fitting.lazy().select(col("a") * col("b")).collect(engine="cpu")
        )
        assert_equal(result.column("a").dtype(), DataType.decimal(2, 0, 32))
        assert_equal(result.column("a")._decimal_raw(0), Int128(78))
        assert_equal(result.column("a").null_count(), Int(nullable))
