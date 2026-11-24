"""Independent CPU event-order oracle for checked Decimal SUM operands."""
from std.testing import assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from dataframe.decimal import pow10
from tests.support.lifecycle_jobs import WorkerSetting


def feed_fault_fixture(rows: Int = 8193, bad: Int = 8192) raises -> DataFrame:
    # Each early product fits precision 38, but their sum does not. A later
    # product exceeds Int128: which error wins depends on the operand batch.
    var left = List[Int128](length=rows, fill=0)
    var right = List[Int128](length=rows, fill=1)
    left[0] = 6 * pow10(37)
    left[1] = 6 * pow10(37)
    left[bad] = pow10(38) - 1
    right[bad] = 2
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
    # The eager select runs as a plan, whose reduction inputs are computed
    # one data chunk (STANDARD_VECTOR_SIZE rows) at a time, before that
    # chunk updates the state, at any batch size: an operand fault inside
    # the first chunk (row 1024) comes before the precision fault its rows
    # 0 and 1 make, and one in a later chunk (row 4096) after it.
    var first_chunk = feed_fault_fixture(1025, 1024)
    with assert_raises(contains="decimal multiplication overflow"):
        _ = first_chunk.select(feed_expression(), batch_size=1024)
    with assert_raises(contains="decimal multiplication overflow"):
        _ = first_chunk.select(feed_expression(), batch_size=1025)
    var later_chunk = feed_fault_fixture(4097, 4096)
    with assert_raises(contains="value exceeds decimal precision 38"):
        _ = later_chunk.select(feed_expression(), batch_size=8192)
    var frame = feed_fault_fixture()
    with assert_raises(contains="value exceeds decimal precision 38"):
        _ = frame.select(feed_expression(), batch_size=8193)
    # The lazy plan's executor evaluates the whole input in 8192-row
    # batches. A later empty observer cannot erase an already observed
    # original expression fault.
    var query = frame.lazy().select(feed_expression())
    with assert_raises(contains="value exceeds decimal precision 38"):
        _ = query.collect(engine="cpu", optimize=False, streaming=False)
    with assert_raises(contains="value exceeds decimal precision 38"):
        _ = query.head(0).collect(engine="cpu", optimize=False, streaming=False)
    _ = workers^
