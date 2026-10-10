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
    # These assertions inspect the CPU logical plan. Default explain also
    # reports automatic placement; backend diagnostics have their own tests.
    var df = frame()
    var pushed = (
        df.lazy()
        .with_columns((col("v") * lit(Int64(2))).alias("v2"))
        .filter(col("k").eq(lit(String("a"))))
        .explain(streaming=False, engine="cpu")
    )
    assert_equal(pushed, "WITH_COLUMNS v2\n  FILTER\n    SCAN frame\n")
    # A filter on a produced column must stay above its producer.
    var kept = (
        df.lazy()
        .with_columns((col("v") * lit(Int64(2))).alias("v2"))
        .filter(col("v2") > lit(Int64(4)))
        .explain(streaming=False, engine="cpu")
    )
    assert_equal(kept, "FILTER\n  WITH_COLUMNS v2\n    SCAN frame\n")
    # Never past a slice: that would change which rows survive.
    var sliced = df.lazy().head(3).filter(col("v") > lit(Int64(1)))
    assert_true(
        sliced.explain(streaming=False, engine="cpu").startswith(
            "FILTER\n  SLICE"
        )
    )
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
        inner.explain(streaming=False, engine="cpu"),
        "JOIN inner on k\n  SCAN frame\n  FILTER\n    SCAN frame\n",
    )
    var left = (
        df.lazy()
        .join(other().lazy(), "k", "left")
        .filter(col("label").eq(lit(String("A"))))
    )
    assert_true(
        left.explain(streaming=False, engine="cpu").startswith(
            "FILTER\n  JOIN left"
        )
    )
    var sorted = df.lazy().sort(["w"]).filter(col("v") > lit(Int64(2)))
    assert_true(
        sorted.explain(streaming=False, engine="cpu").startswith(
            "SORT w\n  FILTER"
        )
    )
    assert_true(
        sorted.collect().equals(df.sort(["w"]).filter(col("v") > lit(Int64(2))))
    )
    # Below a group_by when it reads only the keys: every row of a group
    # shares them, so the same groups survive with the same values and
    # first-occurrence order. A filter on an aggregate stays above, as
    # does a key filter that is not row-local.
    var keyed = (
        df.lazy()
        .group_by(["k"], maintain_order=True)
        .agg([col("v").sum().alias("s"), col("w").mean().alias("m")])
        .filter(
            (col("s") > lit(Int64(0)))
            & (col("m") > lit(0.0))
            & col("k").ne(lit(String("b")))
        )
    )
    var keyed_plan = keyed.explain(streaming=False, engine="cpu")
    assert_true(keyed_plan.startswith("FILTER\n  GROUP_BY"), keyed_plan)
    assert_true("GROUP_BY k AGG s, m\n    FILTER\n" in keyed_plan, keyed_plan)
    same(
        keyed,
        df.group_by(["k"], maintain_order=True)
        .agg([col("v").sum().alias("s"), col("w").mean().alias("m")])
        .filter(
            (col("s") > lit(Int64(0)))
            & (col("m") > lit(0.0))
            & col("k").ne(lit(String("b")))
        ),
    )
    var nulls_first = (
        df.lazy()
        .group_by(["v"], maintain_order=True)
        .agg([col("w").sum().alias("s")])
        .filter(col("v").is_null() | (col("v") > lit(Int64(3))))
    )
    assert_true(
        nulls_first.explain(streaming=False, engine="cpu").startswith(
            "GROUP_BY"
        ),
    )
    same(
        nulls_first,
        df.group_by(["v"], maintain_order=True)
        .agg([col("w").sum().alias("s")])
        .filter(col("v").is_null() | (col("v") > lit(Int64(3)))),
    )
    var not_row_local = (
        df.lazy()
        .group_by(["v"], maintain_order=True)
        .agg([col("w").sum().alias("s")])
        .filter(col("v") >= col("v").max())
    )
    assert_true(
        not_row_local.explain(streaming=False, engine="cpu").startswith(
            "FILTER\n  GROUP_BY"
        )
    )
    same(
        not_row_local,
        df.group_by(["v"], maintain_order=True)
        .agg([col("w").sum().alias("s")])
        .filter(col("v") >= col("v").max()),
    )
    var whole_column = df.lazy().sort(["w"]).filter(col("w") > col("w").mean())
    assert_true(
        whole_column.explain(streaming=False, engine="cpu").startswith(
            "FILTER\n  SORT"
        )
    )
    assert_true(
        whole_column.collect().equals(
            df.sort(["w"]).filter(col("w") > col("w").mean())
        )
    )
    var projected = (
        df.lazy()
        .sort(["w"], descending=True)
        .select_exprs([col("note").alias("renamed")])
    )
    assert_true(
        projected.collect().equals(
            df.sort(["w"], descending=True).select_exprs(
                [col("note").alias("renamed")]
            )
        )
    )


def test_projection_and_slice_pushdown_into_csv() raises:
    write_csv(frame(), PATH)
    var q = scan_csv(PATH).select(["k", "v"]).head(2)
    assert_equal(
        q.explain(streaming=False, engine="cpu"),
        "SELECT k, v\n  SLICE 0 2\n    SCAN CSV "
        + PATH
        + " [project k, v] [n_rows 2]\n",
    )
    assert_true(q.collect().equals(frame().select(["k", "v"]).head(2)))
    # A slice is not pushed below an aggregate projection.
    var agg = scan_csv(PATH).select(col("v").sum()).head(1)
    assert_true(agg.explain(streaming=False, engine="cpu").startswith("SLICE"))
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
    assert_true(
        typed.explain(streaming=False, engine="cpu").endswith(
            "[project w, note]\n"
        )
    )
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


def test_filter_copies_only_columns_read_after_it() raises:
    # The stream's filter copies only what the next step reads: a selection
    # (here one that renames), a grouped reduction with its keys, or a join
    # input. Columns only the predicate reads must still filter correctly.
    var n = 20_000
    var d = List[Int32](capacity=n)
    var q = List[Float64](capacity=n)
    var p = List[Float64](capacity=n)
    var g = List[String](capacity=n)
    var id = List[Int64](capacity=n)
    for i in range(n):
        d.append(Int32(i % 365))
        q.append(Float64(i % 50))
        p.append(Float64(i % 1000) / 10)
        g.append("g" + String(i % 7))
        id.append(Int64(i % 900))
    var df = DataFrame(
        [
            Series("d", Column[Int32](d^)),
            Series("q", Column[Float64](q^)),
            Series("p", Column[Float64](p^)),
            Series("g", Column[String](g^)),
            Series("id", Column[Int64](id^)),
        ]
    )
    var predicate = (
        (col("d") >= lit(Int32(30)))
        & (col("d") < lit(Int32(200)))
        & (col("q") < lit(Float64(24)))
    )
    same(
        df.lazy()
        .filter(predicate)
        .select((col("p") * col("q")).sum().alias("r")),
        df.filter(predicate).select((col("p") * col("q")).sum().alias("r")),
    )
    same(
        df.lazy()
        .filter(predicate)
        .group_by("g", maintain_order=True)
        .agg([col("p").sum().alias("s"), col("id").n_unique().alias("u")]),
        df.filter(predicate)
        .group_by("g", maintain_order=True)
        .agg([col("p").sum().alias("s"), col("id").n_unique().alias("u")]),
    )
    same(
        df.lazy().filter(predicate).select_exprs([col("id").alias("key")]),
        df.filter(predicate).select_exprs([col("id").alias("key")]),
    )
    var names = List[String]()
    var ids = List[Int64]()
    for i in range(0, 900, 3):
        names.append("n" + String(i))
        ids.append(Int64(i))
    var dim = DataFrame(
        [
            Series("id", Column[Int64](ids^)),
            Series("name", Column[String](names^)),
        ]
    )
    same(
        df.lazy()
        .filter(predicate)
        .join(dim.lazy(), "id")
        .group_by("name", maintain_order=True)
        .agg([col("p").sum().alias("s")]),
        df.filter(predicate)
        .join(dim, "id")
        .group_by("name", maintain_order=True)
        .agg([col("p").sum().alias("s")]),
    )


def test_shared_subplans_run_once_and_match() raises:
    # A grouped subplan used twice (a self-join of an aggregate, as TPC-DS
    # q65 and q1 do) is executed once and both uses read the result; the
    # same plan with the copies differing is not shared. Both match the
    # eager result and the unoptimized plan.
    var n = 30_000
    var k = List[Int64](capacity=n)
    var g = List[String](capacity=n)
    var v = List[Float64](capacity=n)
    for i in range(n):
        k.append(Int64(i % 700))
        g.append("g" + String(i % 9))
        v.append(Float64(i % 101))
    var df = DataFrame(
        [
            Series("k", Column[Int64](k^)),
            Series("g", Column[String](g^)),
            Series("v", Column[Float64](v^)),
        ]
    )
    var totals = (
        df.lazy()
        .filter(col("v") > lit(Float64(3)))
        .group_by("k", maintain_order=True)
        .agg([col("v").sum().alias("s"), col("v").len().alias("c")])
    )
    var eager_totals = (
        df.filter(col("v") > lit(Float64(3)))
        .group_by("k", maintain_order=True)
        .agg([col("v").sum().alias("s"), col("v").len().alias("c")])
    )
    var averages = totals.group_by("c", maintain_order=True).agg(
        [col("s").mean().alias("avg_s")]
    )
    var eager_averages = eager_totals.group_by("c", maintain_order=True).agg(
        [col("s").mean().alias("avg_s")]
    )
    same(
        totals.join(averages, "c").filter(col("s") > col("avg_s")),
        eager_totals.join(eager_averages, "c").filter(col("s") > col("avg_s")),
    )
    # The copies differ (one more filter): nothing shared, same answer.
    var other = (
        df.lazy()
        .filter(col("v") > lit(Float64(3)))
        .filter(col("g").ne(lit(String("g4"))))
        .group_by("k", maintain_order=True)
        .agg([col("v").sum().alias("s2")])
    )
    var eager_other = (
        df.filter(col("v") > lit(Float64(3)))
        .filter(col("g").ne(lit(String("g4"))))
        .group_by("k", maintain_order=True)
        .agg([col("v").sum().alias("s2")])
    )
    same(totals.join(other, "k"), eager_totals.join(eager_other, "k"))
    # Two scans of the same frame are not a shared subplan: a join of a
    # frame with itself still answers.
    same(
        df.lazy().join(df.lazy().select(["k", "g"]), ["k", "g"]),
        df.join(df.select(["k", "g"]), ["k", "g"]),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
