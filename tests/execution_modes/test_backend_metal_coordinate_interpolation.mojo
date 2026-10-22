"""Coordinate interpolation retains independent value and coordinate validity."""
# contract-group: execution.backends
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.accelerator.gpu.runtime import (
    capabilities as _capabilities,
)
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, LazyFrame, col, lit, when
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.capabilities import RowCapabilities
from dataframe_metal.accelerator.rows import lower_rows, ROW_CUMULATIVE_KEY
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_moments import near
from tests.execution_modes.test_backend_metal_interpolation import fixture
from tests.execution_modes.test_backend_metal_decimal_casts import (
    fixture as decimals,
)


def source(n: Int = 33) raises -> DataFrame:
    var frame = fixture(n)
    var positions = List[Float64]()
    var valid = List[Bool]()
    for i in range(n):
        positions.append(Float64((i * i) % 101))
        valid.append(i % 11 != 4)
    return frame.hstack(
        DataFrame([Series("axis", Column[Float64](positions^, valid))])
    )


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    near(
        query.collect(accelerator=runtime),
        query.collect(engine="cpu", streaming=False),
    )


def test_coordinate_interpolation_capability_private_axis_metadata_and_checked_inputs() raises:
    assert_true(not RowCapabilities().coordinate_interpolation)
    var frame = source()
    var query = frame.lazy().select(
        col("x").interpolate_by(col("axis") + 1).over("k")
    )
    var caps = _capabilities()
    caps.coordinate_interpolation = False
    with assert_raises(
        contains="coordinate interpolation requires backend support"
    ):
        _ = lower_rows(query, caps)
    var plan = lower_rows(query, _capabilities())
    var found = False
    for step in plan.steps:
        if plan.code[step.start * 4] != ROW_CUMULATIVE_KEY:
            continue
        found = True
        assert_equal(plan.code[step.start * 4 + 3], Int64(15))
        var axis = Int(plan.code[step.start * 4 + 2])
        assert_true(axis >= len(frame.columns()) and axis < step.slot)
        assert_true(plan.slot_dtypes[axis] == DataType.FLOAT64)
        assert_true(plan.slot_dtypes[step.slot] == DataType.FLOAT64)
        assert_equal(step.nodes, 2)
    assert_true(found)
    _ = Program(plan)
    assert_true(
        "unsupported" not in MetalRuntime(device_id=999999).describe(query)
    )
    with assert_raises(contains="row-valued"):
        _ = lower_rows(
            frame.lazy().select(col("x").interpolate_by(lit(1))),
            _capabilities(),
        )
    # Metal decides faults on the CPU, so it needs no fault-order boundary;
    # a backend without that still refuses.
    var checked_axis = frame.lazy().select(
        col("x").interpolate_by(col("x") + lit(Int64.MAX))
    )
    var strict = _capabilities()
    strict.adjudicated_faults = False
    with assert_raises(contains="checked interpolation axes"):
        _ = lower_rows(checked_axis, strict)
    _ = lower_rows(checked_axis, _capabilities())


# requires: metal
def test_native_metal_coordinate_interpolation_numeric_decimals_partitions_and_chunked_sources() raises:
    var runtime = require_metal()
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        var frame = (
            source(259)
            .slice(1, 257)
            .with_columns(
                [
                    col("x").cast(DataType.of(D)),
                    col("axis").cast(DataType.of(D)),
                ]
            )
        )
        for partitioned in [False, True]:
            var expression = col("x").interpolate_by(col("axis"))
            if partitioned:
                expression = expression.over(["k", "j"])
            check(frame.lazy().select(expression), runtime)
            check(frame.head(0).lazy().select(expression), runtime)
        var x = Series._from_chunks(
            [frame.column("x").slice(0, 31), frame.column("x").slice(31, 226)]
        )
        var axis = Series._from_chunks(
            [
                frame.column("axis").slice(0, 100),
                frame.column("axis").slice(100, 0),
                frame.column("axis").slice(100, 157),
            ]
        )
        check(
            DataFrame([x^, axis^])
            .lazy()
            .select(col("x").interpolate_by(col("axis"))),
            runtime,
        )
    for scale in [0, 3, 18, 38]:
        var frame = decimals(33, scale)
        for name in ["small", "large", "wide"]:
            check(
                frame.lazy().select(
                    col(name).interpolate_by(col("wide")).over("k")
                ),
                runtime,
            )


# requires: metal
def test_native_metal_coordinate_interpolation_independent_distances_duplicate_axes_and_consumers() raises:
    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [99, 2, 99, 99, 8, 99, 14, 99],
                    [False, True, False, False, True, False, True, False],
                ),
            ),
            Series("axis", Column[Float64]([0, 1, 2, 4, 7, 8, 10, 12])),
        ]
    )
    var actual = (
        frame.lazy()
        .select(col("x").interpolate_by(col("axis")))
        .collect(accelerator=runtime)
    )
    assert_true(
        actual.column("x").equals(
            Series(
                "x",
                Column[Float64](
                    [0, 2, 3, 5, 8, 10, 14, 0],
                    [False, True, True, True, True, True, True, False],
                ),
            )
        )
    )
    var variations: List[List[Float64]] = [
        [0, 1, 2, 4, 1, 8, 10, 12],
        [0, Float64("nan"), 2, 4, 7, 8, 10, 12],
        [0, Float64("inf"), 2, 4, Float64("inf"), 8, 10, 12],
        [0, 1, Float64("-inf"), 4, 7, 8, 10, 12],
    ]
    for axes in variations:
        var varied = frame.with_column(
            Series(
                "axis",
                Column[Float64](
                    axes.copy(),
                    [True, True, False, True, True, True, False, True],
                ),
            )
        )
        check(
            varied.lazy().select(col("x").interpolate_by(col("axis"))), runtime
        )
    var wide = DataFrame(
        [
            Series(
                "x",
                Column[UInt64](
                    [UInt64.MAX, 0, UInt64(1) << 63], [True, False, True]
                ),
            ),
            Series(
                "axis",
                Column[UInt64]([UInt64.MAX - 2, UInt64.MAX - 1, UInt64.MAX]),
            ),
        ]
    )
    check(wide.lazy().select(col("x").interpolate_by(col("axis"))), runtime)
    var original = source()
    check(
        original.lazy()
        .filter(col("k") > 0)
        .with_columns(
            col("x").interpolate_by(col("axis") + 1).over("k").alias("filled")
        )
        .filter(col("filled") > 3)
        .sort("filled"),
        runtime,
    )
    check(
        original.lazy()
        .with_columns(col("x").interpolate_by(col("axis")).alias("filled"))
        .group_by("k")
        .agg(col("filled").mean())
        .sort("k"),
        runtime,
    )
    check(
        original.lazy().select(
            col("x").interpolate_by(col("axis")).over("k").shift(-1)
        ),
        runtime,
    )
    check(
        original.lazy().select(
            when(col("x") > 4)
            .then(col("x").interpolate_by(col("axis")))
            .otherwise(lit(Float64(0)))
        ),
        runtime,
    )


# requires: metal
def test_native_metal_coordinate_interpolation_profile_exact_budget_cost_and_runtime_reuse() raises:
    var runtime = require_metal()
    var query = (
        source(513)
        .lazy()
        .filter(col("k") > 0)
        .select(col("x").interpolate_by(col("axis") + 1).over("k"))
    )
    var estimate = runtime.estimate_auto(query)
    assert_true(";coordinate-interpolation-v1" in estimate.family)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    var exact = MetalRuntime(memory_limit_bytes=estimate.peak_bytes)
    var measured = query.profile(accelerator=exact)
    assert_equal(
        measured[1].item(0, "peak_requested_bytes").int64(),
        Int64(estimate.peak_bytes),
    )
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=estimate.peak_bytes - 1)
        )
    check(query, exact)
