"""Bounded wide outputs retain untouched source payload until final assembly."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series, StringColumn, col
from dataframe.accelerator.late_payload import columns, prepare, restore
from tests.support.coverage_compare import assert_frame
from tests.execution_modes.test_backend_metal import require_metal


def source() raises -> DataFrame:
    var frame = DataFrame(
        [
            Series("order", Column[Int64]([9, 2, 7, 1, 5, 0])),
            Series(
                "text",
                StringColumn(["keep", "drop", "keep", "keep", "drop", "keep"]),
            ),
            Series(
                "__dataframe_source_row",
                Column[Int64]([11, 12, 13, 14, 15, 16]),
            ),
            Series(
                "nullable",
                Column[Int64](
                    [1, 2, 3, 4, 5, 6], [True, False, True, False, True, True]
                ),
            ),
        ]
    )
    for i in range(32):
        frame = frame.with_column(
            Series(
                "payload" + String(i),
                Column[Int64](
                    [
                        Int64(i),
                        Int64(i + 1),
                        Int64(i + 2),
                        Int64(i + 3),
                        Int64(i + 4),
                        Int64(i + 5),
                    ]
                ),
            )
        )
    return frame^


def test_metal_late_payload_preserves_source_rows_and_schema() raises:
    var frame = source()
    var query = (
        frame.lazy()
        .filter(col("text").str().contains("keep"))
        .sort("order")
        .slice(1, 2)
    )
    var reads = columns(query)
    assert_true(Bool(reads))
    var plan = prepare(query, reads.take(), -1)
    assert_equal(plan.query._frames[0].width(), 3)
    assert_equal(plan.identity, "__dataframe_source_row_")
    var selected = plan.query.collect(engine="cpu")
    assert_frame(restore(selected, plan), query.collect(engine="cpu"))
    # The original immutable frame and every untouched payload field survive.
    assert_equal(frame.width(), 36)
    var empty = frame.lazy().sort("order").slice(0, 0)
    var empty_reads = columns(empty)
    var empty_plan = prepare(empty, empty_reads.take(), -1)
    assert_frame(
        restore(empty_plan.query.collect(engine="cpu"), empty_plan),
        empty.collect(engine="cpu"),
    )


def test_metal_late_payload_admission_and_budget_guards() raises:
    var frame = source()
    assert_true(not columns(frame.lazy().sort("order")))
    assert_true(
        not columns(
            frame.lazy()
            .with_columns([(col("order") + 1).alias("changed")])
            .head(2)
        )
    )
    assert_true(
        not columns(
            frame.lazy().group_by("text").agg([col("order").sum()]).head(2)
        )
    )
    assert_true(not columns(frame.lazy().slice(0, 4097)))
    var query = frame.lazy().sort("order").head(2)
    var raised = False
    try:
        var reads = columns(query)
        _ = prepare(query, reads.take(), 0)
    except:
        raised = True
    assert_true(raised)
    var reads = columns(query)
    var plan = prepare(query, reads.take(), -1)
    raised = False
    try:
        _ = restore(
            DataFrame([Series(plan.identity, Column[Int64]([-1]))]), plan
        )
    except:
        raised = True
    assert_true(raised)


# requires: metal
def test_metal_late_payload_public_gpu_results() raises:
    var frame = source()
    var query = (
        frame.lazy()
        .filter(col("text").str().contains("keep"))
        .sort("order")
        .slice(1, 2)
    )
    var expected = query.collect(engine="cpu")
    assert_frame(query.collect(accelerator=require_metal()), expected)
    assert_frame(
        query.collect(accelerator=require_metal(), optimize=False), expected
    )
