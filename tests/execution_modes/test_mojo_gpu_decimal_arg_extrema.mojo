"""Decimal arg extrema use exact words and UInt32 logical row ordinals."""
from dataframe import col
from dataframe.metal import MetalRuntime
from tests.public_contracts.test_decimal_arg_extrema import (
    fixture,
    expressions,
    assert_indices,
    parallel_fixture,
)
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.lifecycle_jobs import WorkerSetting


def test_mojo_gpu_decimal_arg_extrema_exact_widths_scale_groups_and_layouts() raises:
    var runtime = MetalRuntime()
    for width in [32, 64, 128]:
        for scale in [0, 2, 6]:
            var raw = fixture(width, scale)
            for frame in [raw.copy(), offset_layout(raw), chunk_layout(raw)]:
                assert_indices(
                    frame.lazy()
                    .select_exprs(expressions())
                    .collect(engine="accel", accelerator=runtime),
                    [2],
                    [3],
                )
                assert_indices(
                    frame.lazy()
                    .group_by("g")
                    .agg(expressions())
                    .sort("g")
                    .collect(engine="accel", accelerator=runtime),
                    [0, 1, 2, -1],
                    [1, 1, 0, -1],
                )
                assert_indices(
                    frame.head(0)
                    .lazy()
                    .select_exprs(expressions())
                    .collect(engine="accel", accelerator=runtime),
                    [-1],
                    [-1],
                )
                assert_indices(
                    frame.head(0)
                    .lazy()
                    .group_by("g")
                    .agg(expressions())
                    .collect(engine="accel", accelerator=runtime),
                    [],
                    [],
                )
                assert_indices(
                    frame.tail(1)
                    .lazy()
                    .select_exprs(expressions())
                    .collect(engine="accel", accelerator=runtime),
                    [-1],
                    [-1],
                )
                # Group ordinals include null rows and spread back to every row.
                assert_indices(
                    frame.lazy()
                    .select_exprs(
                        [
                            col("x").arg_min().over("g").alias("lo"),
                            col("x").arg_max().over("g").alias("hi"),
                        ]
                    )
                    .collect(engine="accel", accelerator=runtime),
                    [1, 0, 1, 0, 1, 2, 2, 2, -1],
                    [1, 1, 1, 1, 1, 0, 0, 0, -1],
                )


def test_mojo_gpu_decimal_arg_extrema_preserve_high_words_and_first_ties() raises:
    var setting = WorkerSetting(2)
    var runtime = MetalRuntime()
    var raw = parallel_fixture()
    for frame in [raw.copy(), offset_layout(raw), chunk_layout(raw)]:
        assert_indices(
            frame.lazy()
            .select_exprs(expressions())
            .collect(engine="accel", accelerator=runtime),
            [1],
            [73728],
        )
    _ = setting^
