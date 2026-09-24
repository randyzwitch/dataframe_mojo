"""LazyFrame: optimized plans agree with eager execution; pushdown rules."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import (
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    LazyFrame,
    Series,
    col,
    lit,
    scan_csv,
    write_csv,
)

comptime PATH = "/tmp/dataframe_mojo_lazy_test.csv"


def frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("k", Column[String](["a", "b", "a", "c", "b", "a"])),
            Series(
                "v",
                Column[Int64](
                    [1, 2, 3, 4, 5, 6], [True, True, False, True, True, True]
                ),
            ),
            Series("w", Column[Float64]([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])),
            Series("note", Column[String](["x", "y", "z", "w", "v", "u"])),
        ]
    )


def other() raises -> DataFrame:
    return DataFrame(
        [
            Series("k", Column[String](["a", "b", "d"])),
            Series("label", Column[String](["A", "B", "D"])),
        ]
    )


def same(lazy: LazyFrame, eager: DataFrame) raises:
    """Optimized and unoptimized plans both match the eager result."""
    assert_true(lazy.collect().equals(eager))
    assert_true(lazy.collect(optimize=False).equals(eager))


def test_matches_eager_chains() raises:
    var df = frame()
    same(
        df.lazy().filter(col("v") > lit(Int64(1))).select(["k", "v"]),
        df.filter(col("v") > lit(Int64(1))).select(["k", "v"]),
    )
    same(
        df.lazy()
        .with_columns((col("v") * lit(Int64(2))).alias("v2"))
        .filter(col("k").ne(lit(String("b"))))
        .group_by("k", maintain_order=True)
        .agg([col("v2").sum(), col("w").mean().alias("mw")]),
        df.with_columns((col("v") * lit(Int64(2))).alias("v2"))
        .filter(col("k").ne(lit(String("b"))))
        .group_by("k", maintain_order=True)
        .agg([col("v2").sum(), col("w").mean().alias("mw")]),
    )
    same(
        df.lazy().sort(["w"], descending=True).head(3).drop(["note"]),
        df.sort(["w"], descending=True).head(3).drop(["note"]),
    )
    same(
        df.lazy().unique(["k"], keep="last", maintain_order=True),
        df.unique(["k"], keep="last", maintain_order=True),
    )
    same(
        df.lazy()
        .join(other().lazy(), "k", "left")
        .filter(col("v") > lit(Int64(2)))
        .select(["k", "label", "v"]),
        df.join(other(), "k", "left")
        .filter(col("v") > lit(Int64(2)))
        .select(["k", "label", "v"]),
    )
    same(
        df.lazy()
        .join(other().lazy(), "k")
        .filter(col("label").ne(lit(String("A")))),
        df.join(other(), "k").filter(col("label").ne(lit(String("A")))),
    )
    assert_true(df.lazy().fetch(2).equals(df.head(2)))


def test_predicate_pushdown_rules() raises:
    var df = frame()
    var pushed = (
        df.lazy()
        .with_columns((col("v") * lit(Int64(2))).alias("v2"))
        .filter(col("k").eq(lit(String("a"))))
        .explain()
    )
    assert_equal(pushed, "WITH_COLUMNS v2\n  FILTER\n    SCAN frame\n")
    # A filter on a produced column must stay above its producer.
    var kept = (
        df.lazy()
        .with_columns((col("v") * lit(Int64(2))).alias("v2"))
        .filter(col("v2") > lit(Int64(4)))
        .explain()
    )
    assert_equal(kept, "FILTER\n  WITH_COLUMNS v2\n    SCAN frame\n")
    # Never past a slice: that would change which rows survive.
    var sliced = df.lazy().head(3).filter(col("v") > lit(Int64(1)))
    assert_true(sliced.explain().startswith("FILTER\n  SLICE"))
    assert_true(
        sliced.collect().equals(df.head(3).filter(col("v") > lit(Int64(1))))
    )
    # Into the owning side of an inner join; only the left side of a left join.
    var inner = (
        df.lazy()
        .join(other().lazy(), "k")
        .filter(col("label").eq(lit(String("A"))))
    )
    assert_equal(
        inner.explain(),
        "JOIN inner on k\n  SCAN frame\n  FILTER\n    SCAN frame\n",
    )
    var left = (
        df.lazy()
        .join(other().lazy(), "k", "left")
        .filter(col("label").eq(lit(String("A"))))
    )
    assert_true(left.explain().startswith("FILTER\n  JOIN left"))


def test_projection_and_slice_pushdown_into_csv() raises:
    write_csv(frame(), PATH)
    var q = scan_csv(PATH).select(["k", "v"]).head(2)
    assert_equal(
        q.explain(),
        "SELECT k, v\n  SLICE 0 2\n    SCAN CSV "
        + PATH
        + " [project k, v] [n_rows 2]\n",
    )
    assert_true(q.collect().equals(frame().select(["k", "v"]).head(2)))
    # A slice is not pushed below an aggregate projection.
    var agg = scan_csv(PATH).select(col("v").sum()).head(1)
    assert_true(agg.explain().startswith("SLICE"))
    assert_equal(agg.collect().item().int64(), Int64(18))
    var schema = CsvSchema(
        [
            CsvField.string("k"),
            CsvField.int64("v"),
            CsvField.float64("w"),
            CsvField.string("note"),
        ]
    )
    var typed = (
        scan_csv(PATH, schema)
        .filter(col("w") > lit(Float64(2)))
        .select(["note"])
    )
    assert_true(typed.explain().endswith("[project w, note]\n"))
    assert_equal(typed.collect().height(), 4)
    assert_true(
        typed.collect().equals(
            frame().filter(col("w") > lit(Float64(2))).select(["note"])
        )
    )
    # Both inferred and explicit scans filter before joining decoded ranges.
    var inferred = (
        scan_csv(PATH).filter(col("w") > lit(Float64(2))).select(["note"])
    )
    assert_true(
        inferred.collect().equals(
            frame().filter(col("w") > lit(Float64(2))).select(["note"])
        )
    )
    var empty = scan_csv(PATH, schema).filter(col("w") > lit(Float64(99)))
    assert_equal(empty.collect().height(), 0)


def test_laziness_and_schema() raises:
    # Building a plan over a missing file reads nothing.
    var missing = scan_csv("/tmp/dataframe_mojo_does_not_exist.csv").filter(
        col("x") > lit(Int64(0))
    )
    with assert_raises():
        _ = missing.collect()
    var q = (
        frame()
        .lazy()
        .with_columns((col("v") / lit(Int64(2))).alias("half"))
        .group_by("k")
        .agg(col("half").max())
    )
    assert_equal(q.collect_schema(), [String("k: string"), "half: float64"])
    # Validation errors surface at collect_schema/collect, not at build time.
    var bad = frame().lazy().select(col("k") + lit(Int64(1)))
    with assert_raises(contains="requires matching dtypes"):
        _ = bad.collect_schema()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
