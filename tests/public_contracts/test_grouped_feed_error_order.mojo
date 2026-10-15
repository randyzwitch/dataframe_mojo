"""Literal anchors for CPU input-batch, state-update and finish error phases."""
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, when
from tests.support.lifecycle_jobs import WorkerSetting
from dataframe.binding import bind
from dataframe.execution import evaluate


def decimal_feed_frame(fault_row: Int, rows: Int = 1025) raises -> DataFrame:
    var values = List[Int128](length=rows, fill=0)
    values[0] = 999
    values[1] = 1
    var integers = List[Int64](length=rows, fill=0)
    var selected = List[Bool](length=rows, fill=False)
    if fault_row >= 0:
        integers[fault_row] = 128
        selected[fault_row] = True
    return DataFrame(
        [
            Series("d", Column[Int128](values^)).with_dtype(
                DataType.decimal(3, 0, 128)
            ),
            Series("x", Column[Int64](integers^)),
            Series("keep", Column[Bool](selected^)),
            Series("k", Column[Int32](List[Int32](length=rows, fill=0))),
        ]
    )


def decimal_feed_operand() raises -> Expr:
    return (
        when(col("keep"))
        .then(col("x").cast(DataType.INT8).cast(DataType.decimal(3, 0, 128)))
        .otherwise(col("d"))
    )


def test_grouped_feed_input_batch_precedes_decimal_update_same_batch_only() raises:
    # At row1, Decimal128 precision3 state would become1000. An input error
    # anywhere in that complete1024-row batch happens before state.update.
    for workers in [1, 4]:
        var setting = WorkerSetting(workers)
        for fault_row in [2, 1023, 1024]:
            var frame = decimal_feed_frame(fault_row)
            with assert_raises(
                contains="cast from int64 to int8 failed" if fault_row
                < 1024 else "decimal precision"
            ):
                _ = frame.group_by("k").agg(
                    decimal_feed_operand().sum(), batch_size=1024
                )
            with assert_raises(
                contains="cast from int64 to int8 failed" if fault_row
                < 1024 else "decimal precision"
            ):
                _ = frame.group_by("k").agg(
                    decimal_feed_operand().mean(), batch_size=1024
                )
            # Changing the actual eager GroupBy batch boundary changes phase
            # precedence. This is a contract, not a global fixed8192 assumption.
            with assert_raises(contains="decimal precision"):
                _ = frame.group_by("k").agg(
                    decimal_feed_operand().sum(), batch_size=1
                )
        for fault_row in [1023, 1024]:
            var decimals = List[Int128](length=1025, fill=0)
            decimals[0] = 999
            decimals[1] = 999
            decimals[2] = 1
            decimals[3] = 1
            var keys = List[Int32](length=1025, fill=0)
            for row in range(1025):
                keys[row] = Int32(row % 2)
            var frame = (
                decimal_feed_frame(fault_row)
                .with_column(
                    Series("d", Column[Int128](decimals^)).with_dtype(
                        DataType.decimal(3, 0, 128)
                    )
                )
                .with_column(Series("k", Column[Int32](keys^)))
            )
            with assert_raises(
                contains="cast from int64 to int8 failed" if fault_row
                < 1024 else "decimal precision"
            ):
                _ = frame.group_by("k").agg(decimal_feed_operand().sum())
        # Raising min_count suppresses final validity, not an update exception.
        with assert_raises(contains="decimal precision"):
            _ = (
                decimal_feed_frame(-1)
                .group_by("k")
                .agg(decimal_feed_operand().sum(min_count=Int.MAX))
            )
        _ = setting^


def integer_feed_frame(fault_row: Int) raises -> DataFrame:
    var rows = 1025
    var values = List[Int64](length=rows, fill=0)
    values[0] = Int64.MAX
    values[1] = 1
    var integers = List[Int64](length=rows, fill=0)
    if fault_row >= 0:
        integers[fault_row] = 128
    var keep = List[Bool](length=rows, fill=False)
    if fault_row >= 0:
        keep[fault_row] = True
    return DataFrame(
        [
            Series("v", Column[Int64](values^)),
            Series("x", Column[Int64](integers^)),
            Series("keep", Column[Bool](keep^)),
            Series("k", Column[Int32](List[Int32](length=rows, fill=0))),
            Series("base", Column[Int64](List[Int64](length=rows, fill=2))),
            Series("exp", Column[Int64](List[Int64](length=rows, fill=-1))),
        ]
    )


def test_grouped_feed_checked_ast_and_integer_sum_finish_order() raises:
    var frame = integer_feed_frame(1024)
    var strict = col("x").cast(DataType.INT8).cast(DataType.INT64)
    var input = when(col("keep")).then(strict.copy()).otherwise(col("v"))
    # Integer SUM accumulates wide and only checks its result at finish, so a
    # later input batch still wins over MAX+1 from the first batch.
    with assert_raises(contains="cast from int64 to int8 failed"):
        _ = frame.group_by("k").agg(input.sum())
    with assert_raises(contains="overflow"):
        _ = frame.group_by("k").agg(col("v").sum())
    for strict_first in [False, True]:
        var power = col("base").pow(col("exp"))
        var checked = strict + power if strict_first else power + strict
        # The selected final row has both errors. Earlier rows mask both.
        var expression = when(col("keep")).then(checked).otherwise(col("v"))
        for reducer in range(4):
            var reduced = (
                expression.sum() if reducer
                == 0 else expression.count() if reducer
                == 1 else expression.min() if reducer
                == 2 else expression.arg_max()
            )
            with assert_raises(
                contains="cast from int64 to int8 failed" if strict_first else "nonnegative exponent"
            ):
                _ = frame.group_by("k").agg(reduced)
    var empty = frame.clear().group_by("k").agg(input.sum())
    assert_equal(empty.height(), 0)


def test_grouped_feed_worker_errors_precede_worker_ordered_merges() raises:
    # Use the actual typed evaluate/_ReduceJob entry, so a GroupBy streaming
    # alternative cannot silently change the batch/job contract under test.

    var setting = WorkerSetting(2)
    var rows = 131072
    var groups = List[Int](length=rows, fill=0)
    for scenario in range(4):
        var frame = decimal_feed_frame(-1, rows)
        var values = List[Int128](length=rows, fill=0)
        var integers = List[Int64](length=rows, fill=0)
        var selected = List[Bool](length=rows, fill=False)
        if scenario == 0:
            # Job0 update fault wins over job1's input fault, regardless of
            # completion timing. The source-level run_jobs scan is job-ordered.
            values[0] = 999
            values[1] = 1
            integers[65538] = 128
            selected[65538] = True
        elif scenario == 1:
            values[65536] = 999
            values[65537] = 1
            integers[2] = 128
            selected[2] = True
        else:
            # Independently safe partials999 and1 overflow only when merged.
            values[0] = 999
            values[65536] = 1
            if scenario == 3:
                integers[66560] = 128
                selected[66560] = True
        frame = (
            frame.with_column(
                Series("d", Column[Int128](values^)).with_dtype(
                    DataType.decimal(3, 0, 128)
                )
            )
            .with_column(Series("x", Column[Int64](integers^)))
            .with_column(Series("keep", Column[Bool](selected^)))
        )
        var bound = bind(decimal_feed_operand().sum(), frame._columns)
        with assert_raises(
            contains="decimal precision" if scenario == 0
            or scenario == 2 else "cast from int64 to int8 failed"
        ):
            _ = evaluate(
                bound,
                frame._columns,
                rows,
                batch_size=1024,
                grouped=True,
                groups=groups,
                group_count=1,
            )
    _ = setting^
