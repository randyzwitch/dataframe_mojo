"""Independent integer/temporal producer order and recovery witnesses."""
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from dataframe.metal import MetalRuntime


def main() raises:
    var runtime = MetalRuntime()
    var source = DataFrame(
        [
            Series("t", Column[Int64]([0, 1710037800000000])).with_dtype(
                DataType.datetime("us")
            ),
            Series("v", Column[Int64]([Int64.MAX, 0])),
        ]
    )
    # The unused checked integer producer precedes localization. A later
    # ordering/head cannot hide it or replace its diagnostic with the gap.
    var integer_first = (
        source.lazy()
        .with_columns((col("v") + lit(Int64(1))).alias("unused"))
        .with_columns(
            col("t").dt().replace_time_zone("America/New_York").alias("z")
        )
        .sort("z")
        .head(0)
    )
    with assert_raises(contains="overflow"):
        _ = integer_first.collect(engine="cpu", streaming=False, optimize=False)
    with assert_raises(contains="overflow"):
        _ = integer_first.collect(accelerator=runtime, optimize=False)
    # In the reverse phase order, localization must finish before the later
    # integer producer even though that producer faults at an earlier row.
    var temporal_first = (
        source.lazy()
        .with_columns(
            col("t").dt().replace_time_zone("America/New_York").alias("z")
        )
        .with_columns((col("v") + lit(Int64(1))).alias("unused"))
        .sort("z")
        .head(0)
    )
    with assert_raises(contains="non-existent"):
        _ = temporal_first.collect(
            engine="cpu", streaming=False, optimize=False
        )
    with assert_raises(contains="non-existent"):
        _ = temporal_first.collect(accelerator=runtime, optimize=False)
    # Within one certified producer the original projection rank is observed
    # before row rank, in both sibling orders.
    for gap_first in [False, True]:
        var gap = col("t").dt().replace_time_zone("America/New_York").alias("z")
        var overflow = (col("v") + lit(Int64(1))).alias("unused")
        var expressions: List[Expr] = [
            gap.copy(),
            overflow.copy(),
        ] if gap_first else [overflow.copy(), gap.copy()]
        var siblings = source.lazy().with_columns(expressions).sort("t").head(0)
        with assert_raises(
            contains="non-existent" if gap_first else "overflow"
        ):
            _ = siblings.collect(engine="cpu", streaming=False, optimize=False)
        with assert_raises(
            contains="non-existent" if gap_first else "overflow"
        ):
            _ = siblings.collect(accelerator=runtime, optimize=False)
    var safe = (
        source.head(1)
        .lazy()
        .with_columns(
            col("t").dt().replace_time_zone("America/New_York").alias("z")
        )
        .sort("z")
    )
    var recovered = safe.collect(accelerator=runtime, optimize=False)
    assert_equal(recovered.item(0, "z").int64(), Int64(18000000000))
    assert_equal(
        recovered["z"].dtype(), DataType.datetime("us", "America/New_York")
    )
    print(
        "PASS temporal/integer prefix and sibling fault order, head0 and recovery"
    )
