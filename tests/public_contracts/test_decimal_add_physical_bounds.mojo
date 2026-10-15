"""Dense Decimal add/subtract obey physical bounds and result precision."""
from std.testing import assert_equal, assert_raises
from dataframe import DataFrame, Series, Column, DataType, col, lit, when
from dataframe.expr import Expr


def arithmetic() -> List[Expr]:
    return [col("a") + col("b"), col("a") - col("b")]


def test_decimal_add_subtract_dense_and_nullable_precision_and_alignment() raises:
    # Raw physical values can exceed the logical precision. Dense CPU kernels
    # must give the same precision failures as their nullable counterpart.
    for valid in List[List[Bool]]([List[Bool](), List[Bool]([True, False])]):
        var values: List[Int64] = [Int64.MAX, 0]
        var zeros: List[Int64] = [1, 0]
        var a = Series("a", Column[Int64](values.copy())) if len(
            valid
        ) == 0 else Series("a", Column[Int64](values.copy(), valid.copy()))
        var b = Series("b", Column[Int64](zeros.copy())) if len(
            valid
        ) == 0 else Series("b", Column[Int64](zeros.copy(), valid.copy()))
        for scale in List[Int]([0, 1]):
            var frame = DataFrame(
                [
                    a.with_dtype(DataType.decimal(18, 0, 64)),
                    b.with_dtype(DataType.decimal(18, scale, 64)),
                ]
            )
            for expression in arithmetic():
                with assert_raises(contains="decimal precision"):
                    _ = frame.lazy().select(expression).collect(engine="cpu")


def test_decimal_add_subtract_dense_precision_and_mask_literals() raises:
    for valid in List[List[Bool]]([List[Bool](), List[Bool]([True, False])]):
        var a = Series("a", Column[Int32]([10002, 0])) if len(
            valid
        ) == 0 else Series("a", Column[Int32]([10002, 0], valid.copy()))
        var frame = DataFrame(
            [
                a.with_dtype(DataType.decimal(3, 0, 32)),
                Series("b", Column[Int32]([1, 0])).with_dtype(
                    DataType.decimal(3, 0, 32)
                ),
                Series("p", Column[Bool]([False, True])),
            ]
        )
        for expression in arithmetic():
            with assert_raises(contains="decimal precision"):
                _ = frame.lazy().select(expression).collect(engine="cpu")
            var result = (
                frame.lazy()
                .select(
                    when(col("p"))
                    .then(expression)
                    .otherwise(lit("0").cast(DataType.decimal(3, 0, 32)))
                    .alias("v")
                )
                .collect(engine="cpu")
            )
            assert_equal(result.column("v").dtype(), DataType.decimal(4, 0, 32))
            assert_equal(result.column("v")._decimal_raw(0), Int128(0))
    var good = DataFrame(
        [
            Series("a", Column[Int64]([125, -125])).with_dtype(
                DataType.decimal(3, 2, 64)
            ),
            Series("b", Column[Int64]([75, 75])).with_dtype(
                DataType.decimal(3, 1, 64)
            ),
        ]
    )
    var result = (
        good.lazy()
        .select((col("a") + col("b")).alias("v"))
        .collect(engine="cpu")
    )
    assert_equal(result.column("v").dtype(), DataType.decimal(5, 2, 32))
    assert_equal(result.column("v")._decimal_raw(0), Int128(875))
    assert_equal(result.column("v")._decimal_raw(1), Int128(625))
