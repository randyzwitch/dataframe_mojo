"""List and Struct columns that a plan only moves travel by row id."""
# contract-group: execution.apple_parity
from std.testing import assert_equal, assert_false, assert_true
from dataframe import (
    Column,
    DataFrame,
    ListColumn,
    Series,
    StructColumn,
    col,
    lit,
)
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from dataframe.accelerator import nested_transport
from tests.support.metal_outcomes import same_outcome


def _frames() raises -> Tuple[DataFrame, DataFrame]:
    var offsets: List[Int64] = [0, 2, 2, 5, 6, 6, 8]
    var items = Series("item", Column[Int64]([1, 2, 3, 4, 5, 6, 7, 8]))
    var texts = Series(
        "item", Column[String](["a", "b", "c", "d", "e", "f", "g", "h"])
    )
    var bits: List[UInt8] = [0b111011]  # row 2 is a null list
    var lists = ListColumn(offsets.copy(), items^, bits^)
    var words = ListColumn(offsets^, texts^)
    var record = StructColumn(
        [
            Series("a", Column[Int64]([10, 20, 30, 40, 50, 60])),
            Series("b", Column[String](["u", "v", "w", "x", "y", "z"])),
        ]
    )
    var left = DataFrame(
        [
            Series("k", Column[Int32]([2, 0, 1, 2, 0, 1])),
            Series("x", Column[Int64]([5, 3, 9, 1, 7, 2])),
            Series("l", lists^),
            Series("w", words^),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int32]([0, 1, 2, 3, 4, 5])),
            Series("s", record^),
        ]
    )
    return (left^, right^)


def test_nested_transport_admission_is_pass_through_only() raises:
    var frames = _frames()
    var left = frames[0].copy()
    var right = frames[1].copy()
    for moved in [
        left.lazy().filter(col("x") > lit(Int64(2))),
        left.lazy().sort("x").head(3),
        left.lazy().join(right.lazy(), ["k"]),
        left.lazy().group_by("k").agg(col("x").sum()),
    ]:
        assert_true(nested_transport.applies(moved))
    # Computing on a nested value is not a pass-through.
    for computed in [
        left.lazy().select(col("l").list().len()),
        right.lazy().select(col("s").field("a")),
        left.lazy().sort("l"),
        left.lazy().unique(),
        left.lazy().explode("l"),
        left.lazy().group_by("k").agg(col("l").first()),
    ]:
        assert_false(nested_transport.applies(computed))
    # Plans without nested columns keep their ordinary route.
    assert_false(
        nested_transport.applies(left.select(["k", "x"]).lazy().select(["k"]))
    )
    var described = MetalRuntime(device_id=999999).describe(
        left.lazy().filter(col("x") > lit(Int64(2)))
    )
    assert_true("row ids" in described)


# requires: metal
def test_mojo_gpu_nested_transport_matches_cpu() raises:
    var frames = _frames()
    var left = frames[0].copy()
    var right = frames[1].copy()
    same_outcome(left.lazy().select(["k", "l", "w"]))
    same_outcome(left.lazy().filter(col("x") > lit(Int64(2))))
    same_outcome(left.lazy().sort("x"))
    same_outcome(left.lazy().sort("x", descending=True).head(3))
    same_outcome(left.lazy().slice(1, 3))
    same_outcome(
        left.lazy().select_exprs(
            [col("l").alias("lists"), (col("x") * lit(Int64(2))).alias("x2")]
        )
    )
    same_outcome(left.lazy().join(right.lazy(), ["k"]).sort(["x"]))
    same_outcome(
        right.lazy().join(left.lazy(), ["k"], how="left").sort(["k", "x"])
    )
    same_outcome(left.lazy().concat(left.lazy().head(2)))
    same_outcome(
        left.lazy()
        .sort("x")
        .unique(["k"], keep="first", maintain_order=True)
        .sort("k")
    )
    same_outcome(left.lazy().group_by("k").agg(col("x").sum()).sort("k"))
    # Every row null and an empty selection keep the nested dtype.
    same_outcome(left.lazy().filter(col("x") < lit(Int64(0))))
    var result = (
        left.lazy()
        .filter(col("k") == lit(Int32(1)))
        .collect(accelerator=MetalRuntime())
    )
    assert_equal(result.height(), 2)
    assert_true(result.column("l").dtype().is_list())
    # Rows k == 1 are a null list and [7, 8].
    assert_equal(result.column("l").null_count(), 1)


# requires: metal
def test_mojo_gpu_nested_functions_in_general_plans_match_cpu() raises:
    var frames = _frames()
    var left = frames[0].copy()
    var right = frames[1].copy()
    # String struct children were refused by the direct struct route.
    same_outcome(right.lazy().select(col("s").field("b")))
    same_outcome(
        right.lazy()
        .filter(col("k") > lit(Int32(1)))
        .select_exprs([col("k"), col("s").field("a").alias("a")])
    )
    same_outcome(
        left.lazy()
        .filter(col("x") > lit(Int64(2)))
        .select_exprs([col("k"), col("l").list().len().alias("n")])
    )
    same_outcome(
        left.lazy()
        .join(right.lazy(), ["k"])
        .select_exprs(
            [
                col("k"),
                col("l").list().sum().alias("total"),
                col("s").field("b").alias("b"),
                col("l"),
            ]
        )
        .sort(["k", "total"])
    )
    same_outcome(left.lazy().select(col("w").list().join("-")))
    same_outcome(
        left.lazy().filter(col("l").list().contains(Int64(4))).select(["k"])
    )
    same_outcome(
        left.lazy()
        .group_by("k")
        .agg(col("l").list().max().max().alias("biggest"))
        .sort("k")
    )
    # Struct fields as keys and predicates.
    same_outcome(
        right.lazy()
        .with_columns(col("s").field("a").alias("a"))
        .filter(col("a") >= lit(Int64(30)))
        .sort("a", descending=True)
    )
