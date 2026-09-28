"""DataFrame.describe, DataFrame/Series.sample and Expr.value_counts (#220).

describe's layout and values follow Polars 1.44; the scripts/oracle.py
`describe` operation compares numeric columns against Polars on random
inputs. sample uses its own generator, so these tests check its contract
rather than particular rows: seeded runs repeat, rows never repeat without
replacement, and every row is drawn about equally often.
"""
from std.math import isnan
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from dataframe import Column, DataFrame, DataType, Series, col


def nan() -> Float64:
    return Float64(0) / Float64(0)


def mixed() raises -> DataFrame:
    return DataFrame(
        [
            Series("a", Column[Float64]([1.0, 2, 3, nan()])),
            Series("i", Column[Int64]([4, 0, 6, 1], [True, False, True, True])),
            Series("b", Column[Bool]([True, False, True, True])),
            Series(
                "s",
                Column[String](["x", "y", "", "x"], [True, True, False, True]),
            ),
            Series(
                "d",
                Column[Int64](
                    [18262, 0, 18265, 18263], [True, False, True, True]
                ),
            ).with_dtype(DataType.DATE),
            Series(
                "t",
                Column[Int64](
                    [1577836800000000, 1577923200000001, 0, 1578009600000000],
                    [True, True, False, True],
                ),
            ).with_dtype(DataType.datetime("us")),
        ]
    )


def text(frame: DataFrame, row: Int, name: String) raises -> String:
    var value = frame.item(row, name)
    return "null" if value.is_null() else value.string()


def test_describe_layout() raises:
    var d = mixed().describe(percentiles=[0.333, 0.1])
    assert_equal(d.height(), 8)
    var labels: List[String] = [
        "count",
        "null_count",
        "mean",
        "std",
        "min",
        "10%",
        "33.3%",
        "max",
    ]
    for row in range(8):
        assert_equal(text(d, row, "statistic"), labels[row])
    assert_true(d.column("a").dtype() == DataType.FLOAT64)
    assert_true(d.column("b").dtype() == DataType.FLOAT64)
    assert_true(d.column("s").dtype() == DataType.STRING)
    assert_true(d.column("d").dtype() == DataType.STRING)
    var default = mixed().describe()
    assert_equal(text(default, 5, "statistic"), "25%")
    assert_equal(text(default, 6, "statistic"), "50%")
    assert_equal(text(default, 7, "statistic"), "75%")
    assert_equal(mixed().describe(percentiles=[]).height(), 6)
    with assert_raises(contains="percentiles"):
        _ = mixed().describe(percentiles=[1.5])


def test_describe_numeric() raises:
    var d = mixed().describe(percentiles=[0.333, 0.1])
    # a: NaN propagates into mean and std; min, max and percentiles skip or
    # sort it above numbers.
    assert_equal(d.item(0, "a").float64(), 4.0)
    assert_true(isnan(d.item(2, "a").float64()))
    assert_true(isnan(d.item(3, "a").float64()))
    assert_equal(d.item(4, "a").float64(), 1.0)
    assert_equal(d.item(6, "a").float64(), 2.0)
    assert_equal(d.item(7, "a").float64(), 3.0)
    # i: nulls counted separately and skipped by every statistic.
    assert_equal(d.item(0, "i").float64(), 3.0)
    assert_equal(d.item(1, "i").float64(), 1.0)
    assert_almost_equal(d.item(2, "i").float64(), 11.0 / 3.0)
    assert_almost_equal(d.item(3, "i").float64(), 2.516611478423583)
    assert_equal(d.item(5, "i").float64(), 1.0)
    assert_equal(d.item(6, "i").float64(), 4.0)
    # b: the share of true values; no std or percentiles.
    assert_equal(d.item(2, "b").float64(), 0.75)
    assert_true(d.item(3, "b").is_null())
    assert_true(d.item(5, "b").is_null())
    assert_equal(d.item(4, "b").float64(), 0.0)
    assert_equal(d.item(7, "b").float64(), 1.0)


def test_describe_strings_and_temporal() raises:
    var d = mixed().describe(percentiles=[0.333, 0.1])
    assert_equal(text(d, 0, "s"), "3")
    assert_equal(text(d, 1, "s"), "1")
    assert_equal(text(d, 2, "s"), "null")
    assert_equal(text(d, 4, "s"), "x")
    assert_equal(text(d, 5, "s"), "null")
    assert_equal(text(d, 7, "s"), "y")
    # A Date column's mean is a datetime, as in Polars.
    assert_equal(text(d, 2, "d"), "2020-01-02 08:00:00")
    assert_equal(text(d, 3, "d"), "null")
    assert_equal(text(d, 4, "d"), "2020-01-01")
    assert_equal(text(d, 6, "d"), "2020-01-02")
    assert_equal(text(d, 7, "d"), "2020-01-04")
    assert_equal(text(d, 2, "t"), "2020-01-02 00:00:00")
    assert_equal(text(d, 6, "t"), "2020-01-02 00:00:00.000001")
    assert_equal(text(d, 7, "t"), "2020-01-03 00:00:00")


def test_describe_edges() raises:
    var empty = mixed().head(0).describe()
    assert_equal(empty.height(), 9)
    assert_equal(empty.item(0, "a").float64(), 0.0)
    assert_equal(text(empty, 0, "s"), "0")
    for row in range(2, 9):
        assert_true(empty.item(row, "a").is_null())
        assert_true(empty.item(row, "d").is_null())
    var nans = DataFrame(
        [Series("x", Column[Float64]([nan(), nan(), 0], [True, True, False]))]
    ).describe()
    assert_true(isnan(nans.item(4, "x").float64()))
    assert_true(isnan(nans.item(8, "x").float64()))
    var none = DataFrame(List[Series](), height=3).describe()
    assert_equal(none.width(), 1)
    assert_equal(none.height(), 9)


def numbered(n: Int) raises -> DataFrame:
    var values = List[Int64](capacity=n)
    for i in range(n):
        values.append(Int64(i))
    return DataFrame([Series("i", Column[Int64](values^))])


def values(frame: DataFrame) raises -> List[Int]:
    var out = List[Int]()
    for row in range(frame.height()):
        out.append(Int(frame.item(row, "i").int64()))
    return out^


def test_sample_contract() raises:
    var frame = numbered(1000)
    var a = values(frame.sample(100, seed=42))
    var b = values(frame.sample(100, seed=42))
    assert_equal(len(a), 100)
    for k in range(100):
        assert_equal(a[k], b[k])
    for k in range(1, 100):
        assert_true(a[k - 1] < a[k])  # distinct and in original order
    var c = values(frame.sample(100, seed=43))
    var same = True
    for k in range(100):
        same = same and a[k] == c[k]
    assert_false(same)
    # Most of the rows: the partial Fisher-Yates path.
    var most = values(frame.sample(fraction=0.9, seed=5))
    assert_equal(len(most), 900)
    for k in range(1, 900):
        assert_true(most[k - 1] < most[k])
    assert_equal(frame.sample().height(), 1)
    assert_equal(frame.sample(0).height(), 0)
    assert_equal(frame.sample(fraction=0.0015).height(), 1)
    with assert_raises(contains="larger sample"):
        _ = frame.sample(1001)
    with assert_raises(contains="not both"):
        _ = frame.sample(5, fraction=0.5)
    with assert_raises(contains="empty population"):
        _ = frame.head(0).sample(1, with_replacement=True)
    assert_equal(frame.sample(5000, with_replacement=True).height(), 5000)


def test_sample_shuffle() raises:
    var frame = numbered(500)
    var all = values(frame.sample(fraction=1.0, shuffle=True, seed=9))
    var seen = List[Bool](length=500, fill=False)
    var moved = 0
    for k in range(500):
        assert_false(seen[all[k]])
        seen[all[k]] = True
        if all[k] != k:
            moved += 1
    assert_true(moved > 400)
    var few = values(frame.sample(20, shuffle=True, seed=9))
    var ascending = True
    for k in range(1, 20):
        ascending = ascending and few[k - 1] < few[k]
    assert_false(ascending)


def test_sample_uniform() raises:
    # 10 rows, 3 drawn per seed: each row should appear 30% of the time.
    # Over 2000 seeds that is 600 per row with a standard deviation near 20.
    var frame = numbered(10)
    var counts = List[Int](length=10, fill=0)
    var replaced = List[Int](length=10, fill=0)
    for seed in range(2000):
        for i in values(frame.sample(3, seed=seed)):
            counts[i] += 1
        for i in values(frame.sample(3, with_replacement=True, seed=seed)):
            replaced[i] += 1
    for i in range(10):
        assert_true(counts[i] > 500 and counts[i] < 700)
        assert_true(replaced[i] > 500 and replaced[i] < 700)


def test_series_sample() raises:
    var s = Series("v", Column[Int64]([5, 6, 7, 8]))
    var picked = s.sample(2, seed=1)
    assert_equal(len(picked), 2)
    assert_equal(picked.name(), "v")
    assert_equal(len(s.sample(fraction=1.0)), 4)


def test_value_counts() raises:
    var df = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [2, 1, 2, 0, 0, 7, 2],
                    [True, True, True, False, False, True, True],
                ),
            ),
            Series("g", Column[String](["a", "a", "b", "b", "a", "b", "a"])),
        ]
    )
    var plain = df.select_exprs([col("x").value_counts()])
    assert_equal(plain.height(), 4)
    var first = plain.item(0, "x")
    assert_equal(first.struct_field("x").int64(), 1)
    assert_equal(Int(first.struct_field("count").uint32()), 1)
    assert_true(plain.item(3, "x").struct_field("x").is_null())
    assert_equal(Int(plain.item(3, "x").struct_field("count").uint32()), 2)

    var ranked = df.select_exprs(
        [col("x").value_counts(sort=True, normalize=True)]
    )
    assert_equal(ranked.item(0, "x").struct_field("x").int64(), 2)
    assert_almost_equal(
        ranked.item(0, "x").struct_field("proportion").float64(), 3.0 / 7.0
    )
    # Ties on count keep value order: 1 before 7.
    assert_equal(ranked.item(2, "x").struct_field("x").int64(), 1)
    assert_equal(ranked.item(3, "x").struct_field("x").int64(), 7)

    var named = df.select_exprs([col("g").value_counts(name="n")])
    assert_equal(named.item(0, "g").struct_field("g").string(), "a")
    assert_equal(Int(named.item(0, "g").struct_field("n").uint32()), 4)

    var grouped = df.group_by("g").agg([col("x").value_counts(sort=True)])
    assert_true(
        grouped.column("x").dtype()
        == DataType.list(
            DataType.struct(["x", "count"], [DataType.INT64, DataType.UINT32])
        )
    )
    assert_equal(df.head(0).select_exprs([col("x").value_counts()]).height(), 0)


def test_value_counts_types_and_errors() raises:
    var days = DataFrame(
        [
            Series("d", Column[Int64]([3, 1, 3])).with_dtype(DataType.DATE),
            Series("b", Column[Bool]([True, False, True])),
        ]
    )
    var counted = days.select_exprs([col("d").value_counts()])
    assert_true(
        counted.column("d").dtype()
        == DataType.struct(["d", "count"], [DataType.DATE, DataType.UINT32])
    )
    var flags = days.select_exprs([col("b").value_counts(sort=True)])
    assert_true(flags.item(0, "b").struct_field("b").bool())
    with assert_raises(contains="over()"):
        _ = days.select_exprs([col("d").value_counts().over("b")])
    with assert_raises(contains="must differ"):
        _ = days.select_exprs([col("d").value_counts(name="d")])
    with assert_raises(contains="row-valued"):
        _ = days.select_exprs([col("d").value_counts(), col("b")])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
