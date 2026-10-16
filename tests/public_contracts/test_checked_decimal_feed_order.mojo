"""Independent CPU event-order oracle for checked Decimal SUM operands."""
from std.testing import assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from dataframe.decimal import pow10
from tests.support.lifecycle_jobs import WorkerSetting


def feed_fault_fixture() raises -> DataFrame:
    # Each early product fits precision 38, but their sum does not. A later
    # product exceeds Int128: which error wins depends on the operand batch.
    var left = List[Int128](length=8193, fill=0)
    var right = List[Int128](length=8193, fill=1)
    left[0] = 6 * pow10(37)
    left[1] = 6 * pow10(37)
    left[8192] = pow10(38) - 1
    right[8192] = 2
    return DataFrame(
        [
            Series("a", Column[Int128](left^)).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("b", Column[Int128](right^)).with_dtype(
                DataType.decimal(38, 0)
            ),
        ]
    )


def feed_expression() -> Expr:
    return (col("a") * col("b")).sum().alias("total")


def test_checked_decimal_sum_operand_batches_precede_prefix_observation() raises:
    var workers = WorkerSetting(1)
    var frame = feed_fault_fixture()
    with assert_raises(contains="value exceeds decimal precision 38"):
        _ = frame.select(feed_expression(), batch_size=8192)
    with assert_raises(contains="decimal multiplication overflow"):
        _ = frame.select(feed_expression(), batch_size=8193)
    # Lazy eager SELECT uses the same 8192-row batch. A later empty observer
    # cannot erase an already observed original expression fault.
    var query = frame.lazy().select(feed_expression())
    with assert_raises(contains="value exceeds decimal precision 38"):
        _ = query.collect(engine="cpu", optimize=False, streaming=False)
    with assert_raises(contains="value exceeds decimal precision 38"):
        _ = query.head(0).collect(engine="cpu", optimize=False, streaming=False)
    _ = workers^
