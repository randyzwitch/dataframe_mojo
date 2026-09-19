"""Deduplication: unique, n_unique, is_duplicated/is_unique, drop_nulls, and fill_null."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import Column, DataFrame, Series, col, lit, null


def nan() -> Float64:
    return Float64(0) / Float64(0)


def frame() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "k",
                Column[String](
                    ["a", "b", "a", "", "b", "a", "x"],
                    [True, True, True, True, True, True, False],
                ),
            ),
            Series(
                "f",
                Column[Float64](
                    [nan(), 1, nan(), -0.0, 1, 0.0, 3],
                    [True, True, True, True, True, True, True],
                ),
            ),
            Series("id", Column[Int64]([0, 1, 2, 3, 4, 5, 6])),
        ]
    )


def ids(df: DataFrame) raises -> List[Int64]:
    return df.column("id").int64()._to_list()


def test_unique_keep_policies() raises:
    var df = frame()
    var subset: List[String] = ["k", "f"]
    # Keys: (a,NaN) x2, (b,1) x2, ("",0) and (a,0) distinct, (null,3).
    assert_equal(
        ids(df.unique(subset, maintain_order=True)), [Int64(0), 1, 3, 5, 6]
    )
    assert_equal(
        ids(df.unique(subset, keep="first", maintain_order=True)),
        [Int64(0), 1, 3, 5, 6],
    )
    assert_equal(
        ids(df.unique(subset, keep="last", maintain_order=True)),
        [Int64(2), 3, 4, 5, 6],
    )
    assert_equal(
        ids(df.unique(subset, keep="none", maintain_order=True)),
        [Int64(3), 5, 6],
    )
    assert_equal(
        ids(df.unique(["k"], maintain_order=True)), [Int64(0), 1, 3, 6]
    )
    # Every column is distinct because of the id column.
    assert_equal(df.unique().height(), 7)
    var unordered = df.unique(subset, keep="last")
    assert_true(
        unordered.sort(["id"]).equals(
            df.unique(subset, keep="last", maintain_order=True)
        )
    )
    with assert_raises(contains="keep must be"):
        _ = df.unique(subset, keep="middle")
    with assert_raises(contains="listed twice"):
        _ = df.unique(["k", "k"])
    with assert_raises(contains="at least one column"):
        _ = DataFrame([], height=2).unique()
    assert_equal(df.clear().unique(subset).height(), 0)


def test_n_unique_and_duplicate_masks() raises:
    var df = frame()
    assert_equal(df.n_unique(["k", "f"]), 5)
    assert_equal(df.n_unique(["k"]), 4)
    assert_equal(df.n_unique(["f"]), 4)
    assert_equal(df.n_unique(), 7)
    assert_equal(df.clear().n_unique(), 0)
    var dup = df.is_duplicated(["k", "f"]).bool()._to_list()
    assert_equal(dup, [True, True, True, False, True, False, False])
    var uniq = df.is_unique(["k", "f"]).bool()._to_list()
    for i in range(len(dup)):
        assert_equal(uniq[i], not dup[i])
    assert_equal(df.is_duplicated().name(), "is_duplicated")
    assert_equal(df.is_unique().null_count(), 0)


def test_drop_nulls() raises:
    var df = DataFrame(
        [
            Series("a", Column[Int64]([1, 2, 3, 4], [True, False, True, True])),
            Series(
                "b",
                Column[String](["p", "q", "r", ""], [True, True, False, True]),
            ),
        ]
    )
    assert_equal(df.drop_nulls().column("a").int64()._to_list(), [Int64(1), 4])
    assert_equal(df.drop_nulls(["a"]).height(), 3)
    assert_equal(df.drop_nulls(["b"]).height(), 3)
    with assert_raises(contains="Unknown column"):
        _ = df.drop_nulls(["zzz"])
    assert_equal(DataFrame([], height=3).drop_nulls().height(), 3)


def test_frame_fill_null() raises:
    var df = DataFrame(
        [
            Series("a", Column[Int64]([1, 0, 3], [True, False, True])),
            Series("b", Column[Int64]([0, 5, 0], [False, True, False])),
            Series("s", Column[String](["x", "", "z"], [True, False, True])),
            Series("f", Column[Float64]([nan(), 0, 1], [True, False, True])),
        ]
    )
    var filled = df.fill_null(lit(Int64(-1)))
    assert_equal(filled.column("a").int64()._to_list(), [Int64(1), -1, 3])
    assert_equal(filled.column("b").null_count(), 0)
    assert_equal(filled.column("s").null_count(), 1)
    assert_equal(filled.columns(), df.columns())
    var only_a = df.fill_null(lit(Int64(9)), ["a"])
    assert_equal(only_a.column("b").null_count(), 2)
    var strings = df.fill_null(lit(String("?")))
    assert_equal(strings.column("s").string().value(1), "?")
    # NaN is a value, not a null.
    var floats = df.fill_null(lit(Float64(0)))
    assert_true(
        floats.column("f").float64().value(0)
        != floats.column("f").float64().value(0)
    )
    assert_equal(floats.column("f").null_count(), 0)
    assert_true(df.fill_null(lit(True)).equals(df))
    with assert_raises(
        contains="fill_null value is int64 but column s is string"
    ):
        _ = df.fill_null(lit(Int64(0)), ["a", "s"])
    with assert_raises(contains="requires a scalar value"):
        _ = df.fill_null(col("a"))
    assert_equal(df.fill_null(null("int64")).column("a").null_count(), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
