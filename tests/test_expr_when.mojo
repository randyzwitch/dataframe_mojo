"""when/then/otherwise: branch selection, nulls, and masked evaluation."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import Column, DataFrame, Expr, Series, col, lit, null, when

comptime MIN = Int64(-9223372036854775807) - 1
comptime MAX = Int64(9223372036854775807)


def frame() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [0, 5, -3, 7, 12, -8],
                    [True, True, True, False, True, True],
                ),
            ),
            Series("k", Column[String](["a", "b", "a", "b", "a", "c"])),
        ]
    )


def ints(series: Series) raises -> List[String]:
    var out = List[String]()
    for i in range(len(series)):
        out.append(String(series.get(i)))
    return out^


def test_branch_selection_and_null_fallthrough() raises:
    var df = frame()
    var sign = df.select(
        when(col("x") > lit(Int64(0)))
        .then(lit(String("pos")))
        .when(col("x") < lit(Int64(0)))
        .then(lit(String("neg")))
        .otherwise(lit(String("zero")))
        .alias("sign")
    )
    assert_equal(
        ints(sign.column("sign")),
        [String("zero"), "pos", "neg", "zero", "pos", "neg"],
    )
    # No otherwise: unmatched rows are null. The chain converts implicitly.
    var big = df.select(when(col("x") > lit(Int64(4))).then(col("x")))
    assert_equal(
        ints(big.column("x")),
        [String("null"), "5", "null", "null", "12", "null"],
    )
    var buckets = df.select(
        when(col("x") < lit(Int64(-5)))
        .then(lit(Int64(0)))
        .when(col("x") < lit(Int64(1)))
        .then(lit(Int64(1)))
        .when(col("x") < lit(Int64(6)))
        .then(lit(Int64(2)))
        .otherwise(lit(Int64(3)))
        .alias("bucket")
    )
    assert_equal(
        ints(buckets.column("bucket")),
        [String("1"), "2", "1", "3", "3", "0"],
    )
    var null_branch = df.select(
        when(col("x") > lit(Int64(0)))
        .then(null("int64"))
        .otherwise(col("x"))
        .alias("v")
    )
    assert_equal(null_branch.column("v").null_count(), 3)


def test_masked_branches_do_not_raise() raises:
    var df = frame()
    # MAX + x overflows only for x > 0, and those rows take the other branch.
    var safe = df.select(
        when(col("x") <= lit(Int64(0)))
        .then(lit(MAX) + col("x"))
        .otherwise(col("x"))
        .alias("v")
    )
    assert_equal(safe.column("v").int64().value(0), MAX)
    assert_equal(safe.column("v").int64().value(1), Int64(5))
    assert_equal(safe.column("v").int64().value(5), MAX - 8)
    # The otherwise branch is masked too.
    var other = df.select(
        when(col("x") > lit(Int64(0)))
        .then(col("x"))
        .otherwise(lit(MIN) + col("x") * lit(Int64(-1)))
        .alias("v")
    )
    assert_equal(other.column("v").int64().value(0), MIN)
    # A selected overflow still raises.
    with assert_raises(contains="overflow"):
        _ = df.select(
            when(col("x") > lit(Int64(0)))
            .then(lit(MAX) + col("x"))
            .otherwise(col("x"))
        )
    # Nested conditionals narrow the mask further.
    var nested = df.select(
        when(col("x") > lit(Int64(0)))
        .then(
            when(col("x") > lit(Int64(10)))
            .then(col("x"))
            .otherwise(lit(MAX) - lit(Int64(10)) + col("x"))
        )
        .otherwise(-col("x"))
        .alias("v")
    )
    assert_equal(nested.column("v").int64().value(1), MAX - 5)
    assert_equal(nested.column("v").int64().value(4), Int64(12))
    assert_equal(nested.column("v").int64().value(2), Int64(3))
    # Unary and floor-division overflow are masked the same way.
    var edge = DataFrame([Series("x", Column[Int64]([MIN, 4, -1]))])
    var negated = edge.select(
        when(col("x") > lit(MIN))
        .then(-col("x"))
        .otherwise(lit(Int64(0)))
        .alias("v")
    )
    assert_equal(negated.column("v").int64().value(0), Int64(0))
    assert_equal(negated.column("v").int64().value(1), Int64(-4))
    # Only x=4 is selected; MIN // -1 in the unselected row must not raise.
    var divided = edge.select(
        when(col("x") > lit(Int64(0))).then(lit(MIN) // col("x")).alias("v")
    )
    assert_true(divided.column("v").int64().is_null(0))
    assert_equal(divided.column("v").int64().value(1), MIN // 4)
    assert_true(divided.column("v").int64().is_null(2))
    with assert_raises(contains="floor division overflow"):
        _ = edge.select(lit(MIN) // col("x"))
    # A scalar branch that no row selects is not evaluated for its error.
    var scalar = df.select(
        when(col("x") > lit(Int64(100)))
        .then(lit(MAX) + lit(Int64(1)))
        .otherwise(col("x"))
        .alias("v")
    )
    assert_equal(scalar.column("v").int64().value(1), Int64(5))


def test_scalar_aggregate_and_grouped_conditions() raises:
    var df = frame()
    var scalar = DataFrame([], height=3).select(
        when(lit(True)).then(lit(Int64(1))).otherwise(lit(Int64(2))).alias("v")
    )
    assert_equal(scalar.height(), 1)
    assert_equal(scalar.column("v").int64().value(0), Int64(1))
    var sums = df.with_columns(
        when(col("x").sum() > lit(Int64(0)))
        .then(col("x").max())
        .otherwise(col("x").min())
        .alias("pick")
    )
    assert_equal(sums.column("pick").int64().value(0), Int64(12))
    var grouped = df.group_by("k", maintain_order=True).agg(
        when(col("x").sum() > lit(Int64(0)))
        .then(col("x").max())
        .otherwise(col("x").min())
        .alias("pick")
    )
    assert_equal(ints(grouped.column("pick")), [String("12"), "5", "-8"])


def test_bind_errors() raises:
    var df = frame()
    with assert_raises(contains="when requires a bool predicate, found int64"):
        _ = df.select(when(col("x")).then(col("x")))
    with assert_raises(contains="branches require matching dtypes"):
        _ = df.select(
            when(col("x") > lit(Int64(0)))
            .then(col("x"))
            .otherwise(lit(Float64(0)))
        )
    with assert_raises(contains="branches require matching dtypes"):
        _ = df.clear().select(
            when(col("x") > lit(Int64(0)))
            .then(col("x"))
            .when(col("x") < lit(Int64(0)))
            .then(col("k"))
        )


def test_naming_filtering_and_batch_independence() raises:
    var df = frame()
    var named = df.select(
        when(col("x") > lit(Int64(0))).then(col("x")).otherwise(lit(Int64(0)))
    )
    assert_equal(named.columns()[0], "x")
    assert_equal(
        df.select(
            when(col("x") > lit(Int64(0))).then(col("x")).alias("y")
        ).columns()[0],
        "y",
    )
    var kept = df.filter(
        when(col("k").eq(lit(String("a"))))
        .then(col("x") > lit(Int64(0)))
        .otherwise(lit(True))
    )
    assert_equal(ints(kept.column("x")), [String("5"), "null", "12", "-8"])
    var e: Expr = (
        when(col("x") > lit(Int64(3)))
        .then(col("x") * lit(Int64(2)))
        .when(col("x").is_null())
        .then(lit(Int64(-1)))
        .otherwise(lit(MAX) + col("x") - lit(MAX))
    )
    var reference = df.select(e.alias("v"), batch_size=64)
    for size in range(1, 8):
        assert_true(df.select(e.alias("v"), batch_size=size).equals(reference))
    assert_equal(df.clear().select(e.alias("v")).height(), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
