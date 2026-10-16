"""CPU group/OVER feed and occurrence order, independent of GPU lowering."""
from std.testing import assert_raises, assert_true, assert_equal
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from dataframe.decimal import pow10
from dataframe.hashing import _encode_dense
from tests.support.lifecycle_jobs import WorkerSetting
from tests.public_contracts.test_checked_decimal_feed_order import (
    feed_expression,
)


def _frame(rows: Int, bad: Int, second: Int = 1) raises -> DataFrame:
    var left = List[Int128](length=rows, fill=0)
    var right = List[Int128](length=rows, fill=1)
    left[0] = 6 * pow10(37)
    left[second] = 6 * pow10(37)
    if bad >= 0:
        left[bad] = pow10(38) - 1
        right[bad] = 2
    return DataFrame(
        [
            Series("g", Column[Int32](List[Int32](length=rows, fill=0))),
            Series("a", Column[Int128](left^)).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("b", Column[Int128](right^)).with_dtype(
                DataType.decimal(38, 0)
            ),
        ]
    )


def test_checked_decimal_group_and_over_distinct_operand_batches() raises:
    var worker = WorkerSetting(1)
    var frame = _frame(1025, 1024)
    with assert_raises(contains="decimal precision 38"):
        _ = frame.group_by("g").agg(feed_expression(), batch_size=1024)
    with assert_raises(contains="decimal multiplication overflow"):
        _ = frame.group_by("g").agg(feed_expression(), batch_size=1025)
    var over = feed_expression().over("g")
    with assert_raises(contains="decimal precision 38"):
        _ = frame.select(over, batch_size=1024)
    with assert_raises(contains="decimal multiplication overflow"):
        _ = frame.select(over, batch_size=1025)
    # Public lazy defaults differ: eager GROUP uses 1024, SELECT/OVER 8192.
    with assert_raises(contains="decimal precision 38"):
        _ = (
            frame.lazy()
            .group_by("g")
            .agg(feed_expression())
            .head(0)
            .collect(
                engine="cpu",
                optimize=False,
                streaming=False,
            )
        )
    with assert_raises(contains="decimal multiplication overflow"):
        _ = (
            frame.lazy()
            .select(over)
            .head(0)
            .collect(
                engine="cpu",
                optimize=False,
                streaming=False,
            )
        )
    _ = worker^


def test_checked_decimal_group_range_workers_before_partial_merges() raises:
    var frame = _frame(32768, 1024)
    var serial = WorkerSetting(1)
    with assert_raises(contains="decimal precision 38"):
        _ = frame.group_by("g").agg(feed_expression(), batch_size=1024)
    _ = serial^
    var parallel = WorkerSetting(2)
    # Low-cardinality range jobs use fixed 4096-row grouped feeds. The input
    # fault is inside the first feed even when agg's requested batch is 1024.
    for batch in [1024, 8192]:
        with assert_raises(contains="decimal multiplication overflow"):
            _ = frame.group_by("g").agg(feed_expression(), batch_size=batch)
    # Range workers must succeed before group partials are merged. With no
    # input fault each worker holds 6e37; only their merge exceeds precision.
    with assert_raises(contains="decimal precision 38"):
        _ = _frame(131072, -1, 65536).group_by("g").agg(feed_expression())
    with assert_raises(contains="decimal multiplication overflow"):
        _ = _frame(131072, 131071, 65536).group_by("g").agg(feed_expression())
    # A worker-zero prefix fault precedes a worker-one operand fault.
    with assert_raises(contains="decimal precision 38"):
        _ = _frame(131072, 131071).group_by("g").agg(feed_expression())
    _ = parallel^


def test_checked_decimal_over_partition_job_order_precedes_source_fault_order() raises:
    var rows = 8192
    var groups = List[Int32](length=rows, fill=0)
    for row in range(rows):
        groups[row] = Int32(1 - row % 2)
    var left = List[Int128](length=rows, fill=0)
    var right = List[Int128](length=rows, fill=1)
    # Dense ids are assigned in first-seen order (key 1 precedes key 0).
    # Partition 0's prefix is therefore rows 0,2; row 1 belongs to job 1.
    left[1] = pow10(38) - 1
    right[1] = 2
    left[0] = 6 * pow10(37)
    left[2] = 6 * pow10(37)
    var frame = DataFrame(
        [
            Series("g", Column[Int32](groups^)),
            Series("a", Column[Int128](left^)).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("b", Column[Int128](right^)).with_dtype(
                DataType.decimal(38, 0)
            ),
        ]
    )
    var partitions = _encode_dense(frame.column("g"), True)
    assert_true(Bool(partitions))
    assert_equal(partitions.value().ids[0], 0)
    assert_equal(partitions.value().ids[1], 1)
    assert_equal(partitions.value().ids[2], 0)
    var serial = WorkerSetting(1)
    print("CPU OVER serial: source batch row 1 multiplication fault")
    with assert_raises(contains="decimal multiplication overflow"):
        _ = frame.select(feed_expression().over("g"))
    _ = serial^
    var parallel = WorkerSetting(2)
    # The first-seen key 1 is the first whole-partition job; its rows 0,2
    # overflow a prefix before job 1 observes source row 1's product fault.
    print("CPU OVER two workers: first-seen partition rows 0,2 prefix fault")
    with assert_raises(contains="decimal precision 38"):
        _ = frame.select(feed_expression().over("g"))
    _ = parallel^


def test_checked_decimal_multiple_occurrences_follow_original_ast_order() raises:
    var worker = WorkerSetting(1)
    var maximum = pow10(38) - 1
    var frame = DataFrame(
        [
            Series("g", Column[Int32]([0, 1])),
            Series("price", Column[Int128]([maximum, 1])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("one", Column[Int128]([2, 1])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("discount", Column[Int128]([0, -maximum])).with_dtype(
                DataType.decimal(38, 0)
            ),
        ]
    )
    var discount_first = (col("price") * (col("one") - col("discount"))).sum()
    var product_first = ((col("price") * col("one")) - col("discount")).sum()
    var expressions: List[Expr] = [discount_first.copy(), product_first.copy()]
    for index in range(len(expressions)):
        var expression = expressions[index].copy()
        var message = (
            "decimal precision 38" if index
            == 0 else "decimal multiplication overflow"
        )
        for route in range(3):
            if route == 0:
                with assert_raises(contains=message):
                    _ = frame.select(expression)
            elif route == 1:
                with assert_raises(contains=message):
                    _ = frame.group_by("g").agg(expression)
            else:
                with assert_raises(contains=message):
                    _ = frame.select(expression.over("g"))
    _ = worker^
