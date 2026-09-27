"""Cross-batch equivalence and bounded-executor lifecycle regression tests."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
)
from dataframe import (
    Expr,
    Column,
    DataFrame,
    DataType,
    Series,
    CsvSchema,
    CsvField,
    col,
    lit,
    scan_csv,
    scan_parquet,
    read_parquet,
    write_csv,
)

comptime PATH = "/tmp/dataframe_lazy_streaming_test.csv"


def sample() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "k",
                Column[String](
                    ["b", "a", "b", "c", "a", "d", "b"],
                    [True, True, True, False, True, True, True],
                ),
            ),
            Series(
                "v",
                Column[Int64](
                    [1, 2, 3, 4, 5, 6, 7],
                    [True, False, True, True, True, True, True],
                ),
            ),
            Series("f", Column[Float64]([1, 2, 3, 4, 5, 6, 7])),
        ]
    )


def test_ordered_transforms_head_and_breakers() raises:
    var source = sample()
    for size in [1, 2, 3, 8]:
        var plan = (
            source.lazy()
            .filter(col("v") > 1)
            .with_columns((col("v") * 2).alias("twice"))
            .select(["k", "twice"])
        )
        assert_true(
            plan.collect(batch_size=size).equals(plan.collect(streaming=False))
        )
        assert_true(
            plan.slice(1, 3)
            .collect(batch_size=size)
            .equals(plan.slice(1, 3).collect(streaming=False))
        )
        assert_true(
            plan.sort("twice", descending=True)
            .collect(batch_size=size)
            .equals(
                plan.sort("twice", descending=True).collect(streaming=False)
            )
        )
        assert_true(
            source.lazy()
            .select(lit(Int64(42)).alias("constant"))
            .collect(batch_size=size)
            .equals(source.select(lit(Int64(42)).alias("constant")))
        )
    assert_true("[stream]" in source.lazy().filter(col("v") > 1).explain())
    assert_true("[materialize]" in source.lazy().sort("v").explain())
    with assert_raises(contains="positive"):
        _ = source.lazy().collect(batch_size=0)


def test_state_merges_nulls_counts_moments_and_order() raises:
    var source = sample()
    var expressions: List[Expr] = [
        col("v").sum().alias("sum"),
        col("v").count().alias("count"),
        col("v").mean().alias("mean"),
        col("v").first().alias("first"),
        col("v").last().alias("last"),
        col("v").min().alias("min"),
        col("v").max().alias("max"),
        col("v").n_unique().alias("distinct"),
        col("v").len().alias("rows"),
        (col("v").sum() + col("v").count()).alias("combined"),
    ]
    var plan = source.lazy().group_by("k", maintain_order=True).agg(expressions)
    for size in [1, 2, 3, 8]:
        assert_true(
            plan.collect(batch_size=size).equals(plan.collect(streaming=False))
        )
    var global_plan = source.lazy().select_exprs(
        [col("v").sum(), col("v").count().alias("count")]
    )
    assert_true(
        global_plan.collect(batch_size=2).equals(
            global_plan.collect(streaming=False)
        )
    )
    var empty = source.clear().lazy().group_by("k").agg(expressions)
    assert_true(
        empty.collect(batch_size=2).equals(empty.collect(streaming=False))
    )


def test_partial_integer_overflow_is_not_finalized() raises:
    var source = DataFrame(
        [Series("v", Column[Int64]([9223372036854775807, 1, -1]))]
    )
    assert_equal(
        source.lazy()
        .select(col("v").sum())
        .collect(batch_size=2)
        .item()
        .int64(),
        9223372036854775807,
    )
    var bad = DataFrame([Series("v", Column[Int64]([9223372036854775807, 1]))])
    with assert_raises(contains="overflow"):
        _ = bad.lazy().select(col("v").sum()).collect(batch_size=1)


def test_csv_quotes_eof_and_late_error() raises:
    with open(PATH, "w") as file:
        file.write('k,v\n"a\nb",1\na,2\n"a\nb",3\nc,4')
    var plan = (
        scan_csv(PATH)
        .filter(col("v") > 1)
        .group_by("k", maintain_order=True)
        .agg(col("v").sum())
    )
    for size in [1, 2, 3]:
        assert_true(
            plan.collect(batch_size=size).equals(plan.collect(streaming=False))
        )
    with open(PATH, "w") as file:
        file.write("v\n1\n2\n3\ninvalid\n")
    var schema = CsvSchema([CsvField.int64("v")])
    with assert_raises():
        _ = scan_csv(PATH, schema).select(col("v").sum()).collect(batch_size=1)
    with open(PATH, "w") as file:
        file.write("v\n")
    assert_equal(
        scan_csv(PATH, schema)
        .select(col("v").sum())
        .collect(batch_size=1)
        .item()
        .int64(),
        0,
    )


def test_randomized_state_and_join_differential() raises:
    for seed in range(6):
        var state = UInt64(seed + 1)
        var keys = List[Int64]()
        var values = List[Int64]()
        var floats = List[Float64]()
        var flags = List[Bool]()
        var valid = List[Bool]()
        for i in range(137 + seed):
            state = state * 6364136223846793005 + 1
            var value = Int64((state >> 32) % 101) - 50
            keys.append(Int64(i % (seed + 3)))
            values.append(value)
            floats.append(Float64(value) / 4)
            flags.append(value > 0)
            valid.append(i % 7 != 0)
        var source = DataFrame(
            [
                Series("k", Column[Int64](keys^, valid)),
                Series("v", Column[Int64](values^, valid)),
                Series("f", Column[Float64](floats^, valid)),
                Series("b", Column[Bool](flags^, valid)),
            ]
        )
        var query = (
            source.lazy()
            .group_by("k", maintain_order=True)
            .agg(
                [
                    col("v").sum(min_count=3).alias("sum"),
                    col("f").var().alias("var"),
                    col("f").std().alias("std"),
                    col("f").mean().alias("mean"),
                    col("v").null_count().alias("nulls"),
                    col("b").any(ignore_nulls=False).alias("any"),
                    col("b").all(ignore_nulls=False).alias("all"),
                    (col("v").sum() + col("v").count()).alias("combined"),
                ]
            )
        )
        var expected = query.collect(streaming=False)
        for size in [1, 17, 64]:
            var actual = query.collect(batch_size=size)
            assert_equal(actual.columns(), expected.columns())
            assert_equal(actual.height(), expected.height())
            for c in range(actual.width()):
                if actual._columns[c].dtype() == DataType.FLOAT64:
                    for r in range(actual.height()):
                        var left = actual._columns[c].get(r)
                        var right = expected._columns[c].get(r)
                        assert_equal(left.is_null(), right.is_null())
                        if not left.is_null():
                            assert_almost_equal(
                                left.float64(),
                                right.float64(),
                                rtol=1e-10,
                                atol=1e-10,
                            )
                else:
                    assert_true(actual._columns[c].equals(expected._columns[c]))
        var lookup = DataFrame(
            [
                Series("k", Column[Int64]([0, 1, 1, 9])),
                Series("label", Column[String](["a", "b", "c", "d"])),
            ]
        )
        for how in ["inner", "left", "semi", "anti", "right", "full"]:
            var joined = source.lazy().join(lookup.lazy(), "k", how=how)
            assert_true(
                joined.collect(batch_size=17).equals(
                    joined.collect(streaming=False)
                )
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
