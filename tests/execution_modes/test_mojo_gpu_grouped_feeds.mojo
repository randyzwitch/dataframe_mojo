"""Public grouped feeds preserve operand batches, update and finish phases."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, when
from dataframe.metal import MetalRuntime
from tests.support.lifecycle_jobs import WorkerSetting
from tests.public_contracts.test_grouped_feed_error_order import (
    decimal_feed_frame,
    decimal_feed_operand,
    integer_feed_frame,
)


def _with_list(frame: DataFrame, reduction: Expr) raises:
    # IMPLODE makes this request use eager _agg_whole even when collect's
    # streaming option is enabled. It is an actual CPU route certificate.
    var query = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                reduction.copy().alias("answer"),
                col("x").implode().alias("items"),
            ]
        )
    )
    var expected = query.collect(engine="cpu", streaming=False)
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    assert_true(actual.equals(expected))


# requires: metal
def test_mojo_gpu_grouped_feed_batch_input_update_and_integer_finish_order() raises:
    for workers in [1, 4]:
        var setting = WorkerSetting(workers)
        for fault in [2, 1023, 1024]:
            var frame = decimal_feed_frame(fault)
            for mean in [False, True]:
                var operand = decimal_feed_operand()
                var expression = operand.mean() if mean else operand.sum()
                var query = (
                    frame.lazy()
                    .group_by("k")
                    .agg(
                        [
                            expression.alias("answer"),
                            col("x").implode().alias("items"),
                        ]
                    )
                )
                with assert_raises(
                    contains="cast from int64 to int8 failed" if fault
                    < 1024 else "decimal precision"
                ):
                    _ = query.collect(engine="cpu", streaming=False)
                with assert_raises(
                    contains="strict cast failed" if fault
                    < 1024 else "decimal precision"
                ):
                    _ = query.collect(
                        accelerator=MetalRuntime(), engine="accel"
                    )
                # A trailing limit cannot hide preparation/feed failures.
                with assert_raises(
                    contains="strict cast failed" if fault
                    < 1024 else "decimal precision"
                ):
                    _ = query.head(0).collect(
                        accelerator=MetalRuntime(), engine="accel"
                    )
        var frame = integer_feed_frame(1024)
        var operand = (
            when(col("keep"))
            .then(col("x").cast(DataType.INT8).cast(DataType.INT64))
            .otherwise(col("v"))
        )
        var query = (
            frame.lazy()
            .group_by("k")
            .agg(
                [
                    operand.sum().alias("answer"),
                    col("x").implode().alias("items"),
                ]
            )
        )
        with assert_raises(contains="cast from int64 to int8 failed"):
            _ = query.collect(engine="cpu", streaming=False)
        with assert_raises(contains="strict cast failed"):
            _ = query.collect(accelerator=MetalRuntime(), engine="accel")
        _ = setting^


# requires: metal
def test_mojo_gpu_grouped_feed_checked_ast_original_projection_order_and_masks() raises:
    var frame = integer_feed_frame(1024)
    var strict = col("x").cast(DataType.INT8).cast(DataType.INT64)
    var power = col("base").pow(col("exp"))
    for first_cast in [False, True]:
        var checked = strict + power if first_cast else power + strict
        var selected = when(col("keep")).then(checked).otherwise(col("v"))
        for operation in range(4):
            var reduction = (
                selected.sum() if operation
                == 0 else selected.count() if operation
                == 1 else selected.min() if operation
                == 2 else selected.arg_max()
            )
            var query = (
                frame.lazy()
                .group_by("k")
                .agg(
                    [
                        reduction.alias("answer"),
                        col("x").implode().alias("items"),
                    ]
                )
            )
            with assert_raises(
                contains="cast from int64 to int8 failed" if first_cast else "nonnegative exponent"
            ):
                _ = query.collect(engine="cpu", streaming=False)
            with assert_raises(
                contains="strict cast failed" if first_cast else "nonnegative exponent"
            ):
                _ = query.collect(accelerator=MetalRuntime(), engine="accel")
    var decimals = decimal_feed_frame(-1).with_column(
        Series("bad", Column[Int64](List[Int64](length=1025, fill=128)))
    )
    var bad_list = col("bad").cast(DataType.INT8).implode().alias("items")
    var bad_sum = decimal_feed_operand().sum().alias("answer")
    for list_first in [False, True]:
        var expressions: List[Expr] = [
            bad_list.copy(),
            bad_sum.copy(),
        ] if list_first else [bad_sum.copy(), bad_list.copy()]
        var query = decimals.lazy().group_by("k").agg(expressions)
        with assert_raises(
            contains="cast from int64 to int8 failed" if list_first else "decimal precision"
        ):
            _ = query.collect(engine="cpu", streaming=False)
        with assert_raises(
            contains="strict cast" if list_first else "decimal precision"
        ):
            _ = query.collect(accelerator=MetalRuntime(), engine="accel")
    # Inactive row branches suppress both casts and negative powers. COUNT
    # avoids the independent MAX+1 integer SUM finish fault in this fixture.
    var safe = (
        when(col("keep"))
        .then(col("v"))
        .otherwise(when(col("keep")).then(strict + power).otherwise(col("v")))
    )
    _with_list(frame, safe.count())


# requires: metal
def test_mojo_gpu_grouped_feed_success_high_words_empty_profile_budget_and_provenance() raises:
    var values: List[Int128] = [Int128(1) << 80, -(Int128(1) << 80), 7, 11]
    var source = DataFrame(
        [
            Series("k", Column[Int32]([9, 2, 9, 2])),
            Series("d", Column[Int128](values^)).with_dtype(
                DataType.decimal(38, 0, 128)
            ),
            Series("x", Column[Int64]([0, 0, 0, 0])),
        ]
    )
    # Same-logical decimal arithmetic is conservatively checked and exercises
    # the private paired input/result planes with two nonadjacent groups.
    var zero: List[Int128] = [0, 0, 0, 0]
    source = source.with_column(
        Series("z", Column[Int128](zero^)).with_dtype(
            DataType.decimal(38, 0, 128)
        )
    )
    var query = (
        source.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                (col("d") + col("z")).sum().alias("answer"),
                col("x").implode().alias("items"),
            ]
        )
    )
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    assert_equal(measured[0].height(), 2)
    assert_equal(measured[0].item(0, "k").numeric[DType.int32](), Int32(9))
    assert_equal(
        measured[0].column("answer")._decimal_raw(0), (Int128(1) << 80) + 7
    )
    assert_equal(
        measured[0].column("answer")._decimal_raw(1), -(Int128(1) << 80) + 11
    )
    assert_true(
        measured[0].equals(query.collect(engine="cpu", streaming=False))
    )
    assert_equal(
        measured[1].item(0, "grouped_checked_feed_producers").int64(), Int64(1)
    )
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    assert_true(peak > 0)
    with assert_raises(contains="memory budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    assert_true(
        measured[0].equals(
            query.collect(
                accelerator=MetalRuntime(memory_limit_bytes=peak),
                engine="accel",
            )
        )
    )
    var empty = (
        source.clear()
        .lazy()
        .group_by("k")
        .agg(
            [
                (col("d") + col("z")).sum().alias("answer"),
                col("x").implode().alias("items"),
            ]
        )
    )
    var empty_result = empty.collect(
        accelerator=MetalRuntime(999999, memory_limit_bytes=0), engine="accel"
    )
    assert_equal(empty_result.height(), 0)
    assert_equal(
        empty_result.column("answer").dtype(), DataType.decimal(38, 0, 128)
    )
    # State-only SUM has no checked input AST but still needs the exact
    # source-row update schedule. Plain COL must create a private identity
    # program instead of silently bypassing the feed as a zero-step alias.
    var direct = (
        source.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [col("d").sum().alias("answer"), col("x").implode().alias("items")]
        )
    )
    var plain = direct.profile(accelerator=MetalRuntime(), engine="accel")
    assert_equal(
        plain[1].item(0, "grouped_checked_feed_producers").int64(), Int64(1)
    )
    assert_equal(
        plain[0].column("answer")._decimal_raw(0), (Int128(1) << 80) + 7
    )
    assert_equal(
        plain[0].column("answer")._decimal_raw(1), -(Int128(1) << 80) + 11
    )
    var state_values: List[Int128] = [999, 999, 1, 1]
    var state_source = DataFrame(
        [
            Series("k", Column[Int32]([0, 1, 1, 0])),
            Series("d", Column[Int128](state_values^)).with_dtype(
                DataType.decimal(3, 0, 128)
            ),
            Series("x", Column[Int64]([0, 0, 0, 0])),
        ]
    )
    for mean in [False, True]:
        var operation = col("d").mean() if mean else col("d").sum()
        var state_query = (
            state_source.lazy()
            .group_by("k")
            .agg([operation.alias("answer"), col("x").implode().alias("items")])
        )
        with assert_raises(contains="decimal precision"):
            _ = state_query.collect(engine="cpu", streaming=False)
        with assert_raises(contains="decimal precision"):
            _ = state_query.collect(accelerator=MetalRuntime(), engine="accel")
    with assert_raises(contains="source provenance"):
        _ = MetalRuntime(999999)._run(query, False, original_entry=False)
