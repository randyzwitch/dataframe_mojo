"""CPU-equivalent proved contiguous partition association on the GPU."""

from dataframe import Column, DataFrame, Expr, Series, col
from dataframe_metal.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_moments import near
from tests.support.lifecycle_jobs import WorkerSetting


# requires: mojo_gpu
def test_mojo_gpu_partition_workers_dense_and_hash_order_proofs() raises:
    var setting = WorkerSetting(8)
    var runtime = MetalRuntime()
    for groups in [3, 257]:
        var xs = List[Float64]()
        var ys = List[Float64]()
        var keys = List[UInt64]()
        var valid = List[Bool]()
        var pattern: List[Float64] = [1e16, 1, -1e16, 1]
        for i in range(131073):
            xs.append(Float64(1048576) + Float64(i % 17) / 8)
            ys.append(pattern[i % 4])
            if groups == 3:
                # Dense CPU ids encounter3,2,1; GPU lexical ids encounter
                # 1,2,3. Each whole group must be its own contiguous run.
                keys.append(UInt64(3 - i % 3))
                valid.append(True)
            else:
                # A full-width span disables dense encoding. CPU hash order
                # differs from GPU lexical order; every possible whole-group
                # run is nevertheless below the64k worker threshold.
                keys.append(UInt64(i % 257 + 1) * 0x100000001)
                valid.append(i % 11 != 0)
        var frame = DataFrame(
            [
                Series("x", Column[Float64](xs^)),
                Series("y", Column[Float64](ys^)),
                Series("k", Column[UInt64](keys^, valid^)),
            ]
        )
        var expressions: List[Expr] = [
            col("x").var().over("k").alias("variance"),
            col("x").std().over("k").alias("standard"),
            col("y").sum().over("k").alias("sum"),
            col("y").mean().over("k").alias("mean"),
        ]
        near(
            frame.lazy().select_exprs(expressions).collect(accelerator=runtime),
            frame.select_exprs(expressions),
        )
    _ = setting^
