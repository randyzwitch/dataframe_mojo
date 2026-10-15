"""Grouped List siblings use the CPU fused local moment update."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.metal import MetalRuntime
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_mojo_gpu_grouped_implode_var_std_siblings_local_fma_literal() raises:
    var values: List[Float64] = [-1e16, 3, 2, 2, 1e16, -1e16]
    var source = DataFrame(
        [
            Series("k", Column[Int64]([0, 0, 0, 0, 0, 0])),
            Series("x", Column[Float64](values.copy())),
        ]
    )
    for frame in [source.copy(), offset_layout(source), chunk_layout(source)]:
        var query = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(
                [
                    col("x").implode().alias("items"),
                    col("x").var(0).alias("variance"),
                    col("x").std(0).alias("standard"),
                ]
            )
        )
        for gpu in [False, True]:
            var actual = query.collect(
                accelerator=MetalRuntime(), streaming=False, optimize=False
            ) if gpu else query.collect(
                engine="cpu", streaming=False, optimize=False
            )
            assert_equal(actual.height(), 1)
            assert_equal(actual.item(0, "k").int64(), 0)
            assert_equal(actual.column("variance").dtype(), DataType.FLOAT64)
            assert_equal(actual.column("standard").dtype(), DataType.FLOAT64)
            assert_true(not actual.item(0, "variance").is_null())
            assert_true(not actual.item(0, "standard").is_null())
            # PR731 actual-M1 CPU anchors. Separate m2 multiply/add returns
            # variance ...92a4 and standard ...d850 for these same values.
            assert_equal(
                bitcast[DType.uint64](actual.item(0, "variance").float64()),
                UInt64(0x4682A039D5A192A3),
            )
            assert_equal(
                bitcast[DType.uint64](actual.item(0, "standard").float64()),
                UInt64(0x433869E78FA4D84F),
            )
            var items = actual.column("items").list_column()
            assert_equal(items._offsets[], [Int64(0), Int64(6)])
            assert_true(
                items.child().equals(
                    Series("items", Column[Float64](values.copy()))
                )
            )
