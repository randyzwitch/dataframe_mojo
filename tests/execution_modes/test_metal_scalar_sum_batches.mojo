"""Wide scalar SUMs retain exact integer state across bounded GPU tiles."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from dataframe.accelerator.scalar_sum_batches import (
    prepare,
    partial_query,
    TILE_ROWS,
)
from tests.support.coverage_compare import assert_frame
from tests.execution_modes.test_backend_metal import require_metal


def sums(frame: DataFrame, expression: Expr, min_count: Int = 0) -> LazyFrame:
    var outputs = List[Expr]()
    for i in range(16):
        outputs.append(
            expression.sum(min_count=min_count).alias("s" + String(i))
        )
    return frame.lazy().select_exprs(outputs)


def cancellation_source() raises -> DataFrame:
    var values = List[Int64](length=2 * TILE_ROWS + 1, fill=Int64.MAX)
    values[0] = 42
    for i in range(TILE_ROWS + 1, 2 * TILE_ROWS + 1):
        values[i] = -Int64.MAX
    return DataFrame(
        [
            Series("value", Column[Int64](values^)),
            Series.full_null("unused", DataType.STRING, 2 * TILE_ROWS + 1),
        ]
    ).slice(1, 2 * TILE_ROWS)


def test_metal_scalar_sum_batches_metadata_and_admission() raises:
    var frame = cancellation_source()
    var prepared = prepare(sums(frame, col("value")))
    assert_true(Bool(prepared))
    var plan = prepared.take()
    assert_equal(plan.batches, 2)
    assert_equal(plan.source.width(), 1)
    assert_equal(partial_query(plan, 0)._frames[0].height(), TILE_ROWS)
    assert_true(plan.reservation > 0)
    assert_true(not prepare(sums(frame, col("value"), 1)))
    assert_true(not prepare(sums(frame, col("value").cum_sum())))
    assert_true(not prepare(frame.lazy().select(col("value").sum())))
    assert_true(not prepare(sums(frame.slice(0, 10), col("value"))))


# requires: metal
def test_metal_scalar_sum_batches_cancellation_and_unsigned_nulls() raises:
    var frame = cancellation_source()
    var query = sums(frame, col("value"))
    var measured = query.profile(accelerator=require_metal())
    assert_frame(measured[0], query.collect(engine="cpu"))
    assert_true("integer_sum_batches" in measured[1].columns())
    assert_equal(measured[1].item(0, "integer_sum_batches").int64(), Int64(2))
    assert_equal(measured[0].item(0, "s0").int64(), Int64(0))
    var values = List[UInt64](length=TILE_ROWS + 1, fill=0)
    values[0] = UInt64.MAX
    var valid = List[Bool](length=TILE_ROWS + 1, fill=False)
    valid[0] = True
    var unsigned = DataFrame([Series("value", Column[UInt64](values^, valid))])
    var unsigned_query = sums(unsigned, col("value"))
    assert_frame(
        unsigned_query.collect(accelerator=require_metal()),
        unsigned_query.collect(engine="cpu"),
    )
    var absent = DataFrame(
        [Series.full_null("value", DataType.INT64, TILE_ROWS + 1)]
    )
    var empty_sums = sums(absent, col("value"))
    assert_frame(
        empty_sums.collect(accelerator=require_metal()),
        empty_sums.collect(engine="cpu"),
    )


# requires: metal
def test_metal_scalar_sum_batches_faults_and_budget() raises:
    var frame = cancellation_source()
    with assert_raises():
        _ = sums(frame, col("value")).collect(
            accelerator=MetalRuntime(memory_limit_bytes=0)
        )
    var values = List[Int64](length=TILE_ROWS + 1, fill=0)
    values[0] = Int64.MAX
    values[1] = Int64.MAX
    var overflow = DataFrame([Series("value", Column[Int64](values^))])
    var query = sums(overflow, col("value"))
    with assert_raises():
        _ = query.collect(engine="cpu")
    with assert_raises():
        _ = query.collect(accelerator=require_metal())
    var late_values = List[Int64](length=TILE_ROWS + 1, fill=0)
    late_values[TILE_ROWS] = Int64.MAX
    var late = DataFrame([Series("value", Column[Int64](late_values^))])
    var checked = sums(late, col("value") + 1)
    with assert_raises():
        _ = checked.collect(engine="cpu")
    with assert_raises():
        _ = checked.collect(accelerator=require_metal())
