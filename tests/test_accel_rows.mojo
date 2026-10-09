"""Resident-expression lowering must remain usable without MAX or a device."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit, scan_csv
from dataframe.accel_row_memory import (
    row_memory,
    checked_add,
    checked_mul,
    bitmap_bytes,
)
from dataframe.dtype import DataType
from dataframe.accel_rows import lower_rows, ROW_MAX_NODES


def test_resident_steps_bind_shared_names_and_dtypes() raises:
    var frame = DataFrame([Series("x", Column[Float32]([1, 2, 3]))])
    var query = (
        frame.lazy()
        .with_columns(((col("x") + 1) * 2).alias("y"))
        .filter((col("y") > 2) & col("x").is_not_null())
        .select_exprs([col("y"), (col("x") != 2).alias("b")])
        .head(2)
    )
    var plan = lower_rows(query)
    assert_equal(len(plan.steps), 4)
    assert_true(plan.steps[1].filter)
    assert_equal(plan.steps[1].gather_count, 2)
    assert_equal(plan.outputs[0].name, "y")
    assert_equal(plan.outputs[1].dtype, DataType.BOOL)
    assert_equal(plan.limit, 2)
    assert_true(not plan.reductions)
    var reduction = lower_rows(
        frame.lazy()
        .with_columns((col("x") * 2).alias("y"))
        .select_exprs([col("y").sum().alias("s"), col("x").count().alias("n")])
    )
    assert_true(reduction.reductions)
    assert_equal(reduction.outputs[1].dtype, DataType.INT64)


def test_rejections_happen_without_reading_files() raises:
    with assert_raises(contains="in-memory scan"):
        _ = lower_rows(scan_csv("/nonexistent/accel-rows.csv").select(col("x")))
    var frame = DataFrame([Series("x", Column[Float32]([1, 2]))])
    with assert_raises(contains="Unknown expression column"):
        _ = lower_rows(frame.lazy().select(col("missing")))
    with assert_raises(contains="scalar-only"):
        _ = lower_rows(frame.lazy().select(lit(Float32(1))))
    with assert_raises(contains="unsupported reduction"):
        _ = lower_rows(frame.lazy().select(col("x").mean()))
    var expression = col("x")
    for _ in range(ROW_MAX_NODES):
        expression = expression + 1
    with assert_raises(contains="64 row-local nodes"):
        _ = lower_rows(frame.lazy().select(expression))


def test_memory_preflight_counts_all_requested_buffers() raises:
    var frame = DataFrame([Series("x", Column[Float32]([1, 2, 3]))])
    var plan = lower_rows(
        frame.lazy()
        .filter(col("x") > 1)
        .select_exprs(
            [(col("x") * 2 + 1).alias("y"), (col("x") == 2).alias("b")]
        )
    )
    var memory = row_memory(plan)
    assert_equal(memory.summary.peak_bytes, 644)
    assert_equal(memory.upload_bytes, 476)
    assert_equal(memory.summary.launches, 9)
    assert_equal(memory.matrix, 12)
    assert_equal(memory.alternate, 12)
    memory.summary.require_budget(644)
    with assert_raises(contains="memory preflight"):
        memory.summary.require_budget(643)
    assert_equal(bitmap_bytes(0, 7), 0)
    assert_equal(bitmap_bytes(9, 7), 2)
    assert_equal(checked_mul(Int.MAX, 0), 0)
    with assert_raises(contains="overflows"):
        _ = checked_mul(Int.MAX, 2)
    with assert_raises(contains="overflows"):
        _ = checked_add(Int.MAX, 1)
    with assert_raises(contains="Invalid"):
        _ = bitmap_bytes(1, 8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
