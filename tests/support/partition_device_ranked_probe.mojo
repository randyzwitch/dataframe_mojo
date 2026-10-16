"""Private operand tokens are observed only by an explicit caller phase."""
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.accelerator.gpu.runtime import _preflight, execute_plan
from dataframe.accelerator.gpu.residency import Scope
from dataframe.accelerator.gpu.ranked_rows import evaluate, estimate
from dataframe.accelerator.gpu import ordered_faults
from tests.public_contracts.test_ranked_rows_resources import (
    ranked_operand,
    test_private_ranked_rows_resources_zero_rows_descriptors_and_wide_cells,
)


def run_probe() raises:
    test_private_ranked_rows_resources_zero_rows_descriptors_and_wide_cells()
    print("PASS private ranked rows metadata/empty/wide resource geometry")
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([128, 0, 0])),
            Series("base", Column[Int64]([2, 2, 2])),
            Series("exp", Column[Int64]([0, -1, 0])),
        ]
    )
    for cast_first in [False, True]:
        var scope = Scope()
        var staged = execute_plan(
            _preflight(frame.lazy().select(["x", "base", "exp"])),
            scope,
            export_values=False,
        )
        var source = staged[0].copy()
        var strict = col("x").cast(DataType.INT8).cast(DataType.INT64)
        var power = col("base").pow(col("exp"))
        var plan = ranked_operand(
            strict + power if cast_first else power + strict, frame
        )
        var before = scope.transfers()
        var baseline = scope.info()[0]
        var facts = estimate(plan, frame.height())
        # Both input faults exist. Evaluation itself succeeds and deliberately
        # does not observe either ranked output error.
        var value = evaluate(plan, source, scope)
        var after = scope.transfers()
        assert_equal(after[0] - before[0], facts.upload_bytes)
        assert_equal(after[1] - before[1], facts.download_bytes)
        assert_equal(after[2] - before[2], facts.kernel_launches)
        assert_equal(after[3] - before[3], facts.synchronizations)
        assert_equal(scope.info()[0], baseline + facts.matrix_bytes)
        assert_true(scope.info()[1] >= baseline + facts.peak_bytes)
        var ctx = scope.context()
        var control = ctx.enqueue_create_buffer[DType.uint32](2)
        var host = List[UInt32](length=2, fill=0)
        try:
            ctx.enqueue_function[ordered_faults.observe](
                value.tokens.unsafe_ptr(),
                control.unsafe_ptr(),
                UInt32(value.target),
                UInt32(value.rows),
                UInt32(value.rows),
                UInt32(65536),
                grid_dim=1,
                block_dim=1,
            )
            scope.record_launches(1)
            scope.read_control_into(control, host)
            scope.record_download(8)
            assert_equal(
                ordered_faults.category(host[0]),
                UInt32(1 if cast_first else 32),
            )
            assert_equal(host[1], UInt32(0 if cast_first else 1))
        except error:
            scope.synchronize()
            _ = control^
            _ = host^
            _ = value^
            raise error^
        _ = control^
        _ = host^
        _ = value^
        assert_equal(scope.info()[0], baseline)
        # Keep the baseline source owners live through the release check.
        _ = source^
        _ = staged^
        print(
            "PASS private ranked rows defer fault to caller phase", cast_first
        )
    # Ordinary failing source flags must never be reinterpreted as ranks.
    var scope = Scope()
    var bad = execute_plan(
        _preflight(
            frame.lazy().select(col("x").cast(DataType.INT8).alias("bad"))
        ),
        scope,
        export_values=False,
    )
    var plan = ranked_operand(col("bad").cast(DataType.INT64), bad[0].schema)
    with assert_raises(contains="already observed successful inputs"):
        _ = evaluate(plan, bad[0], scope)
    print("PASS private ranked rows reject foreign deferred source faults")
