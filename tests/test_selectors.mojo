"""Column selectors, expansion order, name transforms, and validation."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import (
    Column,
    DataFrame,
    Expr,
    Series,
    all,
    by_dtype,
    col,
    exclude,
    first,
    last,
    lit,
    nth,
)
from dataframe.binding import bind


def frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("k", Column[String](["a", "b", "a"])),
            Series("x", Column[Int64]([1, 2, 3], [True, False, True])),
            Series("f", Column[Float64]([0.5, 1, 2])),
            Series("y", Column[Int64]([4, 5, 6])),
            Series("ok", Column[Bool]([True, False, True])),
        ]
    )


def test_expansion_order_and_kinds() raises:
    var df = frame()
    assert_equal(df.select(all()).columns(), df.columns())
    assert_true(df.select(all()).equals(df))
    assert_equal(df.select(col(["y", "k"])).columns(), [String("y"), "k"])
    assert_equal(
        df.select(exclude(["k", "ok"])).columns(), [String("x"), "f", "y"]
    )
    assert_equal(
        df.select(by_dtype(["int64", "bool"])).columns(),
        [String("x"), "y", "ok"],
    )
    assert_equal(df.select(nth(1)).columns(), [String("x")])
    assert_equal(df.select(nth(-2)).columns(), [String("y")])
    assert_equal(df.select(first()).columns(), [String("k")])
    assert_equal(df.select(last()).columns(), [String("ok")])
    assert_equal(
        df.select(by_dtype(["string"]).str().to_uppercase())
        .item(1, "k")
        .string(),
        "B",
    )
    # Empty expansions contribute no columns and keep the height.
    var none = df.select_exprs([by_dtype(["float64"]), exclude(df.columns())])
    assert_equal(none.columns(), [String("f")])


def test_operations_apply_to_every_column() raises:
    var df = frame()
    var counts = df.select(all().count())
    assert_equal(counts.height(), 1)
    assert_equal(counts.item(0, "x").int64(), Int64(2))
    assert_equal(counts.item(0, "ok").int64(), Int64(3))
    var scaled = df.with_columns(by_dtype(["int64"]) * lit(Int64(10)))
    assert_equal(scaled.columns(), df.columns())
    assert_equal(scaled.item(2, "y").int64(), Int64(60))
    assert_true(scaled.item(1, "x").is_null())
    var named = df.select_exprs(
        [
            by_dtype(["int64"]).sum().name_suffix("_sum"),
            by_dtype(["int64"]).max().name_prefix("max_"),
            col("x").mean().name_suffix("_avg"),
        ]
    )
    assert_equal(
        named.columns(),
        [String("x_sum"), "y_sum", "max_x", "max_y", "x_avg"],
    )
    # A binary expression with a selector on the right keeps the left name.
    var shifted = df.select((lit(Int64(1)) + col(["x"])).alias("plus"))
    assert_equal(shifted.columns(), [String("plus")])
    var grouped = df.group_by("k", maintain_order=True).agg(
        exclude(["k", "ok"]).sum()
    )
    assert_equal(grouped.columns(), [String("k"), "x", "f", "y"])
    assert_equal(grouped.item(0, "y").int64(), Int64(10))
    var keyed = df.group_by([col(["k", "ok"])], maintain_order=True).len()
    assert_equal(keyed.columns(), [String("k"), "ok", "len"])
    assert_equal(keyed.height(), 2)
    assert_equal(df.filter(nth(-1)).height(), 2)


def test_validation() raises:
    var df = frame()
    with assert_raises(contains="Unknown selector column: zzz"):
        _ = df.select(col(["x", "zzz"]))
    with assert_raises(contains="Unknown selector column: zzz"):
        _ = df.select(exclude(["zzz"]))
    with assert_raises(contains="Unknown selector dtype: int128"):
        _ = df.select(by_dtype(["int128"]))
    with assert_raises(contains="nth selector index 9 is out of range"):
        _ = df.select(nth(9))
    with assert_raises(contains="at most one selector"):
        _ = df.select(col(["x"]) + col(["y"]))
    with assert_raises(contains="Duplicate expression output name: total"):
        _ = df.select(by_dtype(["int64"]).sum().alias("total"))
    with assert_raises(contains="Duplicate expression output name: x"):
        _ = df.select_exprs([all(), col("x")])
    with assert_raises(contains="exactly one column"):
        _ = df.filter(by_dtype(["bool", "string"]))
    with assert_raises(contains="numeric operand"):
        _ = df.select(all().abs())
    with assert_raises(contains="Selectors must be expanded"):
        _ = bind(all(), df._columns)
    with assert_raises(contains="collides with grouping key"):
        _ = df.group_by("k").agg(all().first())
    # Validation happens even on empty frames.
    with assert_raises(contains="Unknown selector column"):
        _ = df.clear().select(col(["nope"]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
