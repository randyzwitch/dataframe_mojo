"""pivot and unpivot reshaping."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import Column, DataFrame, Series, col, lit


def long() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "city", Column[String](["nyc", "nyc", "sf", "sf", "la", "nyc"])
            ),
            Series(
                "q",
                Column[String](
                    ["q1", "q2", "q2", "q1", "q2", "q1"],
                    [True, True, True, True, True, False],
                ),
            ),
            Series("year", Column[Int64]([2023, 2024, 2023, 2024, 2024, 2024])),
            Series("sales", Column[Float64]([1, 2, 3, 4, 5, 6])),
        ]
    )


def test_pivot_layout_and_missing_cells() raises:
    var wide = long().head(5).pivot("year", index=["city"], values="sales")
    assert_equal(wide.columns(), [String("city"), "2023", "2024"])
    assert_equal(wide.height(), 3)
    assert_equal(wide.item(0, "2023").float64(), 1.0)
    assert_equal(wide.item(1, "2024").float64(), 4.0)
    assert_true(wide.item(2, "2023").is_null())
    # Two rows (nyc and la) share the (2024, q2) cell.
    with assert_raises(contains="several rows for one cell"):
        _ = long().pivot("q", index=["year"], values="sales")


def test_pivot_aggregates_and_null_labels() raises:
    var summed = long().pivot(
        "q", index=["city"], values="sales", aggregate_function="sum"
    )
    assert_equal(summed.columns(), [String("city"), "q1", "q2", "null"])
    assert_equal(summed.item(0, "null").float64(), 6.0)
    assert_equal(summed.item(1, "q1").float64(), 4.0)
    var counted = long().pivot(
        "year", index=["city"], values="sales", aggregate_function="len"
    )
    assert_equal(counted.item(0, "2024").int64(), Int64(2))
    assert_true(counted.item(2, "2023").is_null())
    var multi = long().pivot(
        "q", index=["city", "year"], values="sales", aggregate_function="max"
    )
    assert_equal(multi.height(), 5)
    assert_equal(multi.columns()[0], "city")
    assert_equal(multi.columns()[1], "year")
    var sorted = long().pivot(
        "city",
        index=["year"],
        values="sales",
        aggregate_function="mean",
        sort_columns=True,
    )
    assert_equal(sorted.columns(), [String("year"), "la", "nyc", "sf"])
    assert_equal(sorted.item(1, "nyc").float64(), 4.0)
    var no_index = long().pivot(
        "city", index=List[String](), values="sales", aggregate_function="sum"
    )
    assert_equal(no_index.height(), 1)
    assert_equal(no_index.item(0, "nyc").float64(), 9.0)
    var empty = long().clear().pivot("year", index=["city"], values="sales")
    assert_equal(empty.columns(), [String("city")])
    assert_equal(empty.height(), 0)


def test_pivot_validation() raises:
    var df = long()
    with assert_raises(contains="aggregate_function must be"):
        _ = df.pivot(
            "q", index=["city"], values="sales", aggregate_function="mode"
        )
    with assert_raises(contains="cannot also be index"):
        _ = df.pivot("city", index=["city"], values="sales")
    with assert_raises(contains="Unknown column"):
        _ = df.pivot("q", index=["zzz"], values="sales")
    with assert_raises(contains="pivot column name collides: city"):
        _ = df.with_columns(lit(String("city")).alias("label")).pivot(
            "label", index=["city"], values="sales", aggregate_function="sum"
        )


def test_unpivot_and_round_trip() raises:
    var wide = DataFrame(
        [
            Series("id", Column[Int64]([1, 2])),
            Series("a", Column[Float64]([1.5, 0], [True, False])),
            Series("b", Column[Float64]([3, 4])),
        ]
    )
    var tall = wide.unpivot(index=["id"])
    assert_equal(tall.columns(), [String("id"), "variable", "value"])
    assert_equal(tall.height(), 4)
    assert_equal(tall.item(2, "variable").string(), "b")
    assert_equal(tall.item(2, "id").int64(), Int64(1))
    assert_true(tall.item(1, "value").is_null())
    var named = wide.unpivot(["b"], ["id"], variable_name="k", value_name="v")
    assert_equal(named.columns(), [String("id"), "k", "v"])
    assert_equal(named.height(), 2)
    var no_index = wide.unpivot(["a", "b"])
    assert_equal(no_index.columns(), [String("variable"), "value"])
    # pivot(unpivot(x)) restores x.
    var back = tall.pivot("variable", index=["id"], values="value")
    assert_true(back.equals(wide))
    with assert_raises(contains="must share one dtype; id is int64"):
        _ = wide.unpivot(["a", "id"])
    with assert_raises(contains="both index and on"):
        _ = wide.unpivot(["a"], ["a"])
    with assert_raises(contains="distinct"):
        _ = wide.unpivot(index=["id"], variable_name="id")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
