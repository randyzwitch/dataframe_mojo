"""Exact eager checked-rank operand phases; no backend invocation."""
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, Expr, Series, col, lit, when
from tests.support.lifecycle_jobs import WorkerSetting


def ordered_source(n: Int = 2050, overflow_row: Int = 1500) raises -> DataFrame:
    var x = List[Int64](length=n, fill=1)
    var y = List[Int64](length=n, fill=0)
    var exponent = List[Int64](length=n, fill=1)
    x[overflow_row] = Int64.MAX
    y[overflow_row] = 1
    exponent[0] = -1
    var columns = List[Series]()
    columns.append(Series("x", Column[Int64](x^)))
    columns.append(Series("y", Column[Int64](y^)))
    columns.append(Series("p", Column[Int64](exponent^)))
    columns.append(Series("k", Column[Int32](List[Int32](length=n, fill=0))))
    return DataFrame(columns^)


def test_cpu_checked_rank_input_batch_and_outer_when_phase() raises:
    var setting = WorkerSetting(1)
    for fixture_index in range(2):
        var frame = ordered_source(
            2050 if fixture_index == 0 else 10000,
            1500 if fixture_index == 0 else 9000,
        )
        var operand = (col("x") + col("y")).pow(col("p"))
        var expected = (
            "overflow" if fixture_index == 0 else "nonnegative exponent"
        )
        for partitioned in [False, True]:
            var ranked = operand.rank("min").over(
                "k"
            ) if partitioned else operand.rank("min")
            for hidden in [False, True]:
                var expression = (
                    when(lit(False))
                    .then(ranked)
                    .otherwise(lit(Int64(7))) if hidden else ranked.copy()
                )
                var query = frame.lazy().select(expression.alias("r"))
                # Actual M1 route80: public eager Frame methods pass8192.
                # ADDrow1500 wins in the first batch; ADDrow9000 is later,
                # so POWrow0 wins first. A streaming option is not this batch.
                for requested in [1, 65536]:
                    with assert_raises(contains=expected):
                        _ = query.collect(
                            engine="cpu",
                            streaming=False,
                            optimize=False,
                            batch_size=requested,
                        )
                    with assert_raises(contains=expected):
                        _ = query.head(0).collect(
                            engine="cpu",
                            streaming=False,
                            optimize=False,
                            batch_size=requested,
                        )
                # Direct Frame callers may explicitly choose1024. This is a
                # distinct CPU API, not the LazyFrame accelerator contract.
                with assert_raises(contains="nonnegative exponent"):
                    _ = frame.select(expression, batch_size=1024)
        var masked = (
            when(col("p") >= lit(Int64(0)))
            .then(operand)
            .otherwise(lit(Int64(0)))
        )
        with assert_raises(contains="overflow"):
            _ = (
                frame.lazy()
                .select(masked.rank("dense"))
                .collect(engine="cpu", streaming=False, optimize=False)
            )
    _ = setting^


def test_cpu_checked_rank_literal_methods_ties_nulls_partitions() raises:
    var setting = WorkerSetting(1)
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Int64]([3, 1, 1, 9, 2], [True, True, True, False, True]),
            ),
            Series("k", Column[Int32]([0, 0, 0, 0, 1])),
        ]
    )
    var value = col("x") + lit(Int64(1))
    var expressions: List[Expr] = [
        value.rank("min").alias("minimum"),
        value.rank("max").alias("maximum"),
        value.rank("dense").alias("dense"),
        value.rank("ordinal").alias("ordinal"),
        value.rank("average").alias("average"),
        value.rank("min").over("k").alias("partitioned"),
    ]
    var actual = (
        frame.lazy()
        .select_exprs(expressions)
        .collect(engine="cpu", streaming=False, optimize=False)
    )
    var minimum: List[Int64] = [4, 1, 1, 0, 3]
    var maximum: List[Int64] = [4, 2, 2, 0, 3]
    var dense: List[Int64] = [3, 1, 1, 0, 2]
    var ordinal: List[Int64] = [4, 1, 2, 0, 3]
    var partitioned: List[Int64] = [3, 1, 1, 0, 1]
    var average: List[Float64] = [4, 1.5, 1.5, 0, 3]
    for row in range(5):
        if row == 3:
            for name in [
                "minimum",
                "maximum",
                "dense",
                "ordinal",
                "average",
                "partitioned",
            ]:
                assert_true(actual.item(row, name).is_null())
            continue
        assert_equal(actual.item(row, "minimum").int64(), minimum[row])
        assert_equal(actual.item(row, "maximum").int64(), maximum[row])
        assert_equal(actual.item(row, "dense").int64(), dense[row])
        assert_equal(actual.item(row, "ordinal").int64(), ordinal[row])
        assert_equal(actual.item(row, "partitioned").int64(), partitioned[row])
        assert_equal(actual.item(row, "average").float64(), average[row])
    _ = setting^
