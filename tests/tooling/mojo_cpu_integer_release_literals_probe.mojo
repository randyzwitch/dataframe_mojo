"""Independent CPU literals for two newly observed checked producer routes."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Expr, Series, col, lit


def main() raises:
    var wide = DataFrame(
        [
            Series("g", Column[Int32]([1, 1, 2, 2])),
            Series("x", Column[Int64]([1, 2, 3, 4])),
        ]
    )
    var expressions: List[Expr] = [
        col("x").n_unique().alias("distinct"),
        col("x").sum().alias("total"),
    ]
    var query = wide.lazy().group_by("g").agg(expressions).head(1)
    for optimize in [False, True]:
        for streaming in [False, True]:
            var result = query.collect(
                engine="cpu", optimize=optimize, streaming=streaming
            )
            assert_equal(result.height(), 1)
            var key = result.column("g").int32()._get(0)
            assert_true(key == 1 or key == 2)
            assert_equal(result.item(0, "distinct").int64(), 2)
            assert_equal(
                result.item(0, "total").int64(), Int64(3 if key == 1 else 7)
            )
    print(
        "PASS CPU grouped distinct + Int64 SUM before head1 literals",
        flush=True,
    )
    var bad = DataFrame(
        [
            Series("g", Column[Int32]([1, 1])),
            Series("x", Column[Int16]([32767, 1])),
        ]
    )
    var right = DataFrame([Series("g", Column[Int32]([1]))])
    var ordinary = (
        bad.lazy()
        .select_exprs([col("g"), (col("x") + lit(Int16(2))).alias("x")])
        .join(right.lazy(), "g")
    )
    with assert_raises(contains="overflow"):
        _ = ordinary.collect(engine="cpu", streaming=False, optimize=False)
    print("PASS CPU checked Int16 child overflow before JOIN", flush=True)
