"""GPU slices use current resident row positions and retain stable order."""
from std.testing import TestSuite, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, StringColumn, col, lit
from dataframe.metal import MetalRuntime, metal_installed


def fixture() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "x",
                Column[Int16](
                    [5, 1, 4, 2, 3, 1, 0],
                    [True, True, False, True, True, True, True],
                ),
            ),
            Series("s", StringColumn(["é", "b", "c", "d", "e", "b", "g"])),
            Series(
                "b", Column[Bool]([True, False, True, False, True, False, True])
            ),
        ]
    )


def test_slice_offsets_lengths_nulls_and_empty() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = fixture()
    for offset in List[Int](
        [Int.MIN, -20, -7, -3, -1, 0, 1, 5, 7, 20, Int.MAX]
    ):
        for length in List[Int]([-1, 0, 1, 4, Int.MAX]):
            var query = frame.lazy().slice(offset, length)
            assert_true(
                query.collect(accelerator=runtime).equals(
                    query.collect(engine="cpu")
                )
            )
            var empty = frame.slice(0, 0).lazy().slice(offset, length)
            assert_true(
                empty.collect(accelerator=runtime).equals(
                    empty.collect(engine="cpu")
                )
            )


def test_slice_current_rows_after_filters_sort_groups_and_joins() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = fixture()
    var query = (
        frame.lazy()
        .filter(col("x") > lit(Int16(0)))
        .slice(-3, 2)
        .sort("x")
        .slice(1)
        .select(col("s").str().to_uppercase())
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    var grouped = (
        frame.lazy()
        .group_by("s", maintain_order=True)
        .agg([col("x").count().alias("n")])
        .slice(-3, 2)
    )
    assert_true(
        grouped.collect(accelerator=runtime).equals(
            grouped.collect(engine="cpu")
        )
    )
    var unique = (
        frame.lazy()
        .unique(["s"], keep="last", maintain_order=True)
        .slice(-3, 2)
    )
    assert_true(
        unique.collect(accelerator=runtime).equals(unique.collect(engine="cpu"))
    )
    var joined = (
        frame.lazy()
        .slice(1, 4)
        .join(frame.lazy().slice(-4), on="s", how="full")
        .slice(-3, 2)
    )
    assert_true(
        joined.collect(accelerator=runtime).equals(joined.collect(engine="cpu"))
    )
    var chained = frame.lazy().slice(1, 5).filter(col("b")).slice(1, 1).head(1)
    assert_true(
        chained.collect(accelerator=runtime).equals(
            chained.collect(engine="cpu")
        )
    )


def test_slice_before_reductions_and_invalid_length_recovery() raises:
    if not metal_installed():
        return
    var frame = fixture()
    var runtime = MetalRuntime()
    var query = (
        frame.lazy()
        .filter(col("x") > lit(Int16(0)))
        .slice(-3, 2)
        .select_exprs([col("x").sum(), col("x").count().alias("n")])
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    assert_true("without row compaction" not in runtime.describe(query))
    with assert_raises(contains="Slice length must be nonnegative"):
        _ = frame.lazy().slice(2, -2).collect(accelerator=runtime)
    var valid = frame.lazy().slice(-2)
    assert_true(
        valid.collect(accelerator=runtime).equals(valid.collect(engine="cpu"))
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
