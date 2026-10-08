"""Contracts for row moments, duration expressions and time grouping."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    Column,
    Series,
    DataType,
    Expr,
    col,
    lit,
    LazyFrame,
    concat,
)


def frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("t", Column[Int64]([0, 1, 1, 3, 10])).with_dtype(
                DataType.DATE
            ),
            Series(
                "v",
                Column[Float64](
                    [
                        1000000001.0,
                        1000000002.0,
                        1000000003.0,
                        1000000004.0,
                        1000000005.0,
                    ]
                ),
            ),
            Series("g", Column[String](["a", "b", "a", "b", "a"])),
        ]
    )


def texts(series: Series) raises -> List[String]:
    var result = List[String]()
    for i in range(len(series)):
        result.append(String(series.get(i)))
    return result^


def test_stable_row_variance() raises:
    var df = frame()
    var result = df.select(col("v").rolling_var(3)).column("v")
    assert_equal(texts(result), [String("null"), "null", "1.0", "1.0", "1.0"])
    result = df.select(col("v").rolling_std(3, min_samples=1, ddof=0)).column(
        "v"
    )
    assert_equal(result.get(0).float64(), Float64(0))
    assert_equal(result.get(1).float64(), Float64(0.5))
    var many = List[Float64]()
    for i in range(10000):
        many.append(1e9 + Float64(i % 2))
    var large = DataFrame([Series("v", Column[Float64](many^))])
    result = large.select(col("v").rolling_var(100, ddof=0)).column("v")
    for i in range(99, len(result)):
        assert_true(abs(result.get(i).float64() - 0.25) < 1e-12)


def test_duplicate_boundaries_and_empty_windows() raises:
    var df = frame()
    var sum = df.select(col("v").rolling_sum_by("t", "2d")).column("v")
    assert_equal(sum.get(1).float64(), Float64(3000000006))
    assert_equal(sum.get(2).float64(), Float64(3000000006))
    var empty = df.rolling("t", "1d", closed="none").agg(
        [
            col("v").sum().alias("sum"),
            col("v").count().alias("count"),
            col("v").mean().alias("mean"),
        ]
    )
    assert_equal(empty.height(), df.height())
    assert_equal(empty.column("count").get(0).int64(), Int64(0))
    assert_true(empty.column("mean").get(0).is_null())
    var none = df.head(0).group_by_dynamic("t", "1mo").agg(col("v").sum())
    assert_equal(none.height(), 0)
    assert_equal(none.column("t").dtype(), DataType.DATE)


def test_grouped_sorting() raises:
    var df = frame().take([0, 2, 4, 1, 3])
    _ = df.group_by_dynamic("t", "2d", group_by="g").agg(col("v").sum())
    _ = df.rolling("t", "2d", group_by=["g"]).agg(col("v").sum())
    _ = df.select(col("v").rolling_mean_by("t", "2d").over("g"))
    with assert_raises(contains="not sorted"):
        _ = df.select(col("v").rolling_mean_by("t", "2d"))
    with assert_raises(contains="not sorted"):
        _ = df.rolling("t", "2d").agg(col("v").sum())
    with assert_raises(contains="not sorted"):
        _ = df.group_by_dynamic("t", "2d").agg(col("v").sum())


def test_unsigned_and_nonfinite_values() raises:
    var big = DataFrame(
        [
            Series("t", Column[Int64]([0, 1])).with_dtype(DataType.DATE),
            Series("v", Column[UInt64]([UInt64.MAX - 1, 1])),
        ]
    )
    var sums = big.select(col("v").rolling_sum_by("t", "2d")).column("v")
    assert_equal(sums.get(1).uint64(), UInt64.MAX)
    var bad = Float64(0) / Float64(0)
    var df = DataFrame([Series("v", Column[Float64]([bad, 1, 2, 3, 4]))])
    var variances = df.select(col("v").rolling_var(2)).column("v")
    assert_equal(variances.get(2).float64(), Float64(0.5))
    assert_equal(variances.get(4).float64(), Float64(0.5))


def test_invalid_requests() raises:
    var df = frame()
    with assert_raises(contains="positive"):
        _ = df.select(col("v").rolling_sum_by("t", "0d"))
    with assert_raises(contains="closed"):
        _ = df.rolling("t", "1d", closed="bad").agg(col("v").sum())
    with assert_raises(contains="label"):
        _ = df.group_by_dynamic("t", "1d", label="bad").agg(col("v").sum())
    with assert_raises(contains="Date or Datetime"):
        _ = df.select(col("v").rolling_sum_by("v", "1d"))
    with assert_raises(contains="ddof"):
        _ = df.select(col("v").rolling_std(3, ddof=-1))
    with assert_raises(contains="window_size"):
        _ = df.select(col("v").rolling_var(0))
    with assert_raises(contains="collides"):
        _ = df.rolling("t", "1d").agg(col("v").sum().alias("t"))
    with assert_raises(contains="scalar aggregate"):
        _ = df.rolling("t", "1d").agg(col("v"))
    var nulls = DataFrame(
        [
            Series("t", Column[Int64]([0], [False])).with_dtype(DataType.DATE),
            Series("v", Column[Int64]([1])),
        ]
    )
    with assert_raises(contains="nulls"):
        _ = nulls.rolling("t", "1d").agg(col("v").sum())


def test_lazy_and_chunked_windows() raises:
    var df = frame()
    var exprs = List[Expr](
        [
            col("v").rolling_std(3, min_samples=1).alias("std"),
            col("v").rolling_mean_by(col("t"), "2d").alias("mean"),
        ]
    )
    var expected = df.select_exprs(exprs)
    var chunks = concat([df.head(2), df.slice(2)])
    var chunked = chunks.select_exprs(exprs)
    for name in [String("std"), "mean"]:
        assert_equal(texts(chunked.column(name)), texts(expected.column(name)))
    var plan = LazyFrame(df).select_exprs(exprs)
    for optimized in [True, False]:
        var actual = plan.collect(optimize=optimized)
        for name in [String("std"), "mean"]:
            assert_equal(
                texts(actual.column(name)), texts(expected.column(name))
            )
    var unsorted = LazyFrame(df.take([4, 0])).select(
        col("v").rolling_sum_by("t", "2d")
    )
    with assert_raises(contains="not sorted"):
        _ = unsorted.collect()


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
