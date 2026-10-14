"""Independent legacy anchors for scalar and partition worker association."""

from std.memory import bitcast
from std.testing import assert_true
from dataframe import DataFrame, Series, Column, col, Expr
from dataframe.metal import MetalRuntime
from tests.support.lifecycle_jobs import WorkerSetting
from tests.execution_modes.test_backend_metal_moments import (
    test_native_metal_moments_large_worker_merges_and_conditioned_accuracy,
)
from tests.execution_modes.test_backend_metal_higher_moments import (
    test_native_metal_higher_moments_workers_compositions_cost_budget_and_runtime_reuse,
)
from tests.execution_modes.test_backend_metal_pair_moments import (
    test_native_metal_pair_moments_large_workers_centered_accuracy_and_spearman_ties,
)
from tests.execution_modes.test_backend_metal_partition_aggregates import (
    test_native_metal_nested_partition_floating_worker_order,
    test_native_metal_nested_partition_checked_errors_are_fully_observed,
    test_native_metal_partition_checked_sum_observation,
)

from tests.execution_modes.test_mojo_gpu_decimals import (
    _check_decimal_over_worker_prefix,
)
from tests.execution_modes.test_mojo_gpu_partition_workers import (
    test_mojo_gpu_partition_workers_dense_and_hash_order_proofs,
)


def _compare(
    actual: DataFrame,
    expected: DataFrame,
    stage: String,
    workers: Int,
    rows: Int,
    tolerance: Float64,
) raises:
    for name in ["v", "s"]:
        var a = actual.column(name).float64()
        var b = expected.column(name).float64()
        var worst = 0
        var difference = Float64(0)
        for i in range(actual.height()):
            var gap = abs(a._get(i) - b._get(i))
            if gap > difference:
                difference = gap
                worst = i
        print(
            "STAGE",
            stage,
            "workers",
            workers,
            "rows",
            rows,
            "column",
            name,
            "worstrow",
            worst,
            "actual",
            a._get(worst),
            "CPU",
            b._get(worst),
            "difference",
            difference,
        )
        if difference > tolerance:
            raise Error(
                "statistical worker diagnostic exceeded existing tolerance"
            )


def _diagnose() raises:
    var runtime = MetalRuntime()
    for workers in [1, 2, 8]:
        var setting = WorkerSetting(workers)
        for rows in [65537, 131073, 262147]:
            var values = List[Float64]()
            var keys = List[Int32]()
            for i in range(rows):
                values.append(Float64(1048576) + Float64(i % 17) / 8)
                keys.append(Int32(i % 3))
            var frame = DataFrame(
                [
                    Series("x", Column[Float64](values^)),
                    Series("k", Column[Int32](keys^)),
                ]
            )
            var expressions: List[Expr] = [
                col("x").var().alias("v"),
                col("x").std().alias("s"),
            ]
            _compare(
                frame.lazy()
                .select_exprs(expressions)
                .collect(accelerator=runtime),
                frame.select_exprs(expressions),
                "scalar",
                workers,
                rows,
                Float64(1e-12),
            )
            var windows: List[Expr] = [
                col("x").var().over("k").alias("v"),
                col("x").std().over("k").alias("s"),
            ]
            _compare(
                frame.lazy().select_exprs(windows).collect(accelerator=runtime),
                frame.select_exprs(windows),
                "window",
                workers,
                rows,
                Float64(1e-12),
            )
            var grouped = (
                frame.lazy().group_by("k", maintain_order=True).agg(expressions)
            )
            _compare(
                grouped.collect(accelerator=runtime),
                grouped.collect(engine="cpu"),
                "grouped",
                workers,
                rows,
                Float64(1e-9),
            )
        _ = setting^


def main() raises:
    var setting = WorkerSetting(1)
    var values = List[Float64]()
    for i in range(65537):
        values.append(Float64(1048576) + Float64(i % 17) / 8)
    var frame = DataFrame([Series("x", Column[Float64](values^))])
    var query = frame.lazy().select(col("x").var().alias("v"))
    var actual = query.collect(accelerator=MetalRuntime())
    var expected = frame.select(col("x").var().alias("v"))
    var a = actual.column("v").float64()._get(0)
    var b = expected.column("v").float64()._get(0)
    print("65537 rows, 1 worker: actual", a, "CPU", b, "difference", a - b)
    print("variance words", bitcast[DType.uint64](a), bitcast[DType.uint64](b))
    assert_true(abs(a - b) <= Float64(1e-12))
    _ = setting^
    _diagnose()
    print("RUN moments worker merges, scalar/grouped/window")
    test_native_metal_moments_large_worker_merges_and_conditioned_accuracy()
    print("PASS moments worker merges, scalar/grouped/window")
    print("RUN higher moments workers and consumers")
    test_native_metal_higher_moments_workers_compositions_cost_budget_and_runtime_reuse()
    print("PASS higher moments workers and consumers")
    print("RUN paired moments workers, centered anchors and Spearman")
    test_native_metal_pair_moments_large_workers_centered_accuracy_and_spearman_ties()
    print("PASS paired moments workers, centered anchors and Spearman")
    print("RUN proved dense/hash partition association")
    test_mojo_gpu_partition_workers_dense_and_hash_order_proofs()
    print("PASS proved dense/hash partition association")
    print("RUN safe Decimal128 contiguous partition prefixes")
    _check_decimal_over_worker_prefix()
    print("PASS safe Decimal128 contiguous partition prefixes")
    print("RUN floating partition aggregate worker order")
    var nested_setting = WorkerSetting(3)
    test_native_metal_nested_partition_floating_worker_order()
    _ = nested_setting^
    print("PASS floating partition aggregate worker order")

    print("RUN partition checked-error materialization boundaries")
    test_native_metal_nested_partition_checked_errors_are_fully_observed()
    test_native_metal_partition_checked_sum_observation()
    print("PASS partition checked-error materialization boundaries")
