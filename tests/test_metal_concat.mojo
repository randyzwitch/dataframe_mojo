"""UNION ALL preserves source row order and cells on the resident GPU DAG."""
from std.testing import TestSuite, assert_true, assert_raises
from std.memory import bitcast
from dataframe import DataFrame, Series, Column, StringColumn, col
from dataframe.metal import MetalRuntime, metal_installed
from dataframe.dtype import DataType


def fixture() raises -> DataFrame:
    var nan = bitcast[DType.float32](UInt32(0x7FC00011))
    return DataFrame(
        [
            Series("x", Column[Int16]([3, 1, 3, 2], [False, True, True, True])),
            Series("text", StringColumn(["é", "b", "é", "c"])),
            Series("f", Column[Float32]([-0.0, nan, 0.0, 1.0])),
            Series(
                "__dataframe_concat_side", Column[UInt64]([UInt64.MAX, 0, 1, 2])
            ),
            Series(
                "x__dataframe_concat_right",
                Column[Bool]([True, False, True, False]),
            ),
        ]
    )


def test_concat_order_columns_nulls_extremes_and_empty() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = fixture()
    for left_rows in List[Int]([0, 1, 4]):
        for right_rows in List[Int]([0, 2, 4]):
            var left = frame.slice(0, left_rows).lazy()
            var right = (
                frame.slice(0, right_rows)
                .lazy()
                .select_exprs(
                    [
                        col("text"),
                        col("x__dataframe_concat_right"),
                        col("f"),
                        col("__dataframe_concat_side"),
                        col("x"),
                    ]
                )
            )
            var query = left.concat(right)
            assert_true(
                query.collect(accelerator=runtime).equals(
                    query.collect(engine="cpu")
                )
            )
            var chained = (
                query.concat(left).slice(-5, 3).filter(col("x").is_not_null())
            )
            assert_true(
                chained.collect(accelerator=runtime).equals(
                    chained.collect(engine="cpu")
                )
            )


def test_concat_unique_group_aggregate_join_and_profile() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = fixture()
    var combined = frame.lazy().concat(frame.slice(1, 2).lazy())
    var distinct = combined.unique(["text"], keep="last", maintain_order=True)
    assert_true(
        distinct.collect(accelerator=runtime).equals(
            distinct.collect(engine="cpu")
        )
    )
    var grouped = combined.group_by("text", maintain_order=True).agg(
        [col("x").sum().alias("s")]
    )
    assert_true(
        grouped.collect(accelerator=runtime).equals(
            grouped.collect(engine="cpu")
        )
    )
    var aggregate = combined.select(col("x").sum())
    assert_true(
        aggregate.collect(accelerator=runtime).equals(
            aggregate.collect(engine="cpu")
        )
    )
    var joined = combined.join(frame.lazy(), on="text", how="full").head(7)
    assert_true(
        joined.collect(accelerator=runtime).equals(joined.collect(engine="cpu"))
    )
    var measured = combined.profile(accelerator=runtime)
    assert_true(measured[0].equals(combined.collect(engine="cpu")))
    assert_true(measured[1].item(0, "query_wall_ms").float64() > 0)
    with assert_raises(contains="budget"):
        _ = combined.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
    assert_true(
        combined.collect(accelerator=runtime).equals(
            combined.collect(engine="cpu")
        )
    )


def test_concat_fixed_logical_metadata() raises:
    if not metal_installed():
        return
    var frame = DataFrame(
        [
            Series(
                "date", Column[Int64]([0, -1, 20000], [True, False, True])
            ).with_dtype(DataType.DATE),
            Series("decimal", Column[Int32]([100, -200, 300])).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
        ]
    )
    var query = frame.lazy().concat(frame.slice(1).lazy())
    assert_true(
        query.collect(accelerator=MetalRuntime()).equals(
            query.collect(engine="cpu")
        )
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
