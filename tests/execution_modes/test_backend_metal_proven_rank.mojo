"""Static no-error proofs admit rank inputs without hiding checked children."""
from std.testing import assert_true, assert_raises
from tests.execution_modes.test_backend_metal import require_metal
from dataframe import DataFrame, Series, Column, col, lit, Expr
from dataframe.metal import MetalRuntime, _capabilities
from dataframe.accelerator.rows import lower_rows
from tests.execution_modes.test_backend_metal_proven_arithmetic import (
    frame,
    check,
)


def test_proven_rank_preflight_preserves_opt_in_and_unsafe_children() raises:
    var source = frame()
    var query = source.lazy().select((col("x") + lit(Int64(0))).rank())
    var plan = lower_rows(query, _capabilities())
    assert_true(len(plan.steps) == 2)
    var disabled = _capabilities()
    disabled.proven_integer_operations = False
    with assert_raises(contains="checked rank inputs"):
        _ = lower_rows(query, disabled)
    for unsafe in List[Expr](
        [
            col("x") + lit(Int64(1)),
            (col("x") + lit(Int64(1))) * lit(Int64(0)),
            col("x").cast("int32"),
            col("u").cast("int64"),
            col("x") // lit(Int64(-1)),
        ]
    ):
        with assert_raises(contains="checked rank inputs"):
            _ = lower_rows(source.lazy().select(unsafe.rank()), _capabilities())


def test_native_metal_proven_rank_gpu_methods_limits_partitions_and_slices() raises:
    _ = require_metal()
    var source = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [Int64.MIN, -7, 0, 1, Int64.MAX, 2, 3, 4],
                    [True, True, False, True, True, True, True, True],
                ),
            ),
            Series(
                "u", Column[UInt64]([UInt64.MAX, 0, 1, 2, 3, 4, 0, UInt64.MAX])
            ),
            Series("small", Column[Int8]([-128, -7, 0, 1, 127, 2, 3, 4])),
            Series(
                "k",
                Column[Int32](
                    [0, 0, 1, 1, 0, 0, 1, 1],
                    [True, True, False, False, True, True, True, True],
                ),
            ),
        ]
    )
    var bounded = col("small").clip(lit(Int8(-4)), lit(Int8(4)))
    for expression in List[Expr](
        [
            col("x") + lit(Int64(0)),
            col("x") // lit(Int64(2)),
            col("u") * lit(UInt64(1)),
            col("small").cast("int64"),
            -bounded,
            bounded.abs(),
            (col("x") > 0).cast("int64"),
        ]
    ):
        for method in List[String](
            ["average", "min", "max", "dense", "ordinal"]
        ):
            var ranked = expression.rank(method, descending=True)
            check(source.lazy().select(ranked))
            check(source.lazy().select(ranked.over("k")))
    check(source.slice(1, 5).lazy().select((col("x") // lit(Int64(2))).rank()))
    check(source.slice(0, 0).lazy().select((col("x") + lit(Int64(0))).rank()))


def test_native_metal_proven_rank_gpu_resident_consumers_head_and_rejections() raises:
    _ = require_metal()
    var source = frame()
    var query = source.lazy().with_columns(
        (col("x") // lit(Int64(2))).rank("dense").alias("rank")
    )
    check(query.head(0))
    check(query.filter(col("rank") > 0).sort("rank").head(3))
    check(query.select(col("rank").count()))
    check(
        query.join(
            source.lazy().select(col("u").alias("right")), how="cross"
        ).head(11)
    )
    with assert_raises(contains="checked rank inputs"):
        _ = (
            source.lazy()
            .select((col("x") + lit(Int64(1))).rank())
            .collect(accelerator=MetalRuntime())
        )
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
