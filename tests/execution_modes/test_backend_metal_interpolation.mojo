"""Linear and nearest interpolation follow original partition row order."""
# contract-group: execution.backends
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.accelerator.gpu.runtime import (
    capabilities as _capabilities,
)
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, Expr, LazyFrame, col, lit, when
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.capabilities import RowCapabilities
from dataframe_metal.accelerator.rows import lower_rows, ROW_CUMULATIVE_KEY
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_moments import near
from tests.execution_modes.test_backend_metal_shift_fill import exact
from tests.execution_modes.test_backend_metal_decimal_casts import (
    fixture as decimals,
)


def fixture(n: Int = 33) raises -> DataFrame:
    var values = List[Int64]()
    var valid = List[Bool]()
    var keys = List[Int32]()
    var key_valid = List[Bool]()
    var second = List[Int32]()
    for i in range(n):
        values.append(Int64(i % 97))
        valid.append(i % 11 != 0 and i % 7 != 1 and i % 7 != 2)
        keys.append(Int32(i % 3))
        key_valid.append(i % 5 != 0)
        second.append(Int32(i % 2))
    return DataFrame(
        [
            Series("x", Column[Int64](values^, valid)),
            Series("k", Column[Int32](keys^, key_valid)),
            Series("j", Column[Int32](second^)),
        ]
    )


def check(
    query: LazyFrame, runtime: MetalRuntime, nearest: Bool = False
) raises:
    var actual = query.collect(accelerator=runtime)
    var expected = query.collect(engine="cpu", streaming=False)
    if nearest:
        exact(actual, expected)
    else:
        near(actual, expected)


def test_interpolation_capability_private_types_partition_metadata_and_errors() raises:
    assert_true(not RowCapabilities().interpolation_windows)
    var frame = fixture()
    for method in ["linear", "nearest"]:
        var query = frame.lazy().select(col("x").interpolate(method).over("k"))
        var caps = _capabilities()
        caps.interpolation_windows = False
        with assert_raises(
            contains="interpolation windows require backend support"
        ):
            _ = lower_rows(query, caps)
        var plan = lower_rows(query, _capabilities())
        var found = False
        for step in plan.steps:
            if plan.code[step.start * 4] != ROW_CUMULATIVE_KEY:
                continue
            found = True
            assert_equal(
                plan.code[step.start * 4 + 3],
                Int64(13 if method == "linear" else 14),
            )
            assert_equal(plan.code[step.start * 4 + 2], Int64(0))
            assert_equal(step.nodes, 2)
            assert_true(
                plan.slot_dtypes[step.slot]
                == (DataType.FLOAT64 if method == "linear" else DataType.INT64)
            )
        assert_true(found)
        _ = Program(plan)
        assert_true(
            "unsupported" not in MetalRuntime(device_id=999999).describe(query)
        )
    with assert_raises(contains="linear or nearest"):
        _ = lower_rows(
            frame.lazy().select(col("x").interpolate("spline")), _capabilities()
        )
    with assert_raises(contains="row-valued"):
        _ = lower_rows(
            frame.lazy().select(lit(1).interpolate()), _capabilities()
        )


# requires: metal
def test_native_metal_interpolation_numeric_widths_decimals_null_partitions_slices_and_chunks() raises:
    var runtime = require_metal()
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        var source = fixture(259).slice(1, 257)
        source = source.with_columns(col("x").cast(DataType.of(D)))
        for method in ["linear", "nearest"]:
            check(
                source.lazy().select(col("x").interpolate(method)),
                runtime,
                method == "nearest",
            )
            check(
                source.lazy().select(col("x").interpolate(method).over("k")),
                runtime,
                method == "nearest",
            )
            check(
                source.lazy().select(
                    col("x").interpolate(method).over(["k", "j"])
                ),
                runtime,
                method == "nearest",
            )
            check(
                source.head(0).lazy().select(col("x").interpolate(method)),
                runtime,
                method == "nearest",
            )
        var chunked = Series._from_chunks(
            [
                source.column("x").slice(0, 31),
                source.column("x").slice(31, 0),
                source.column("x").slice(31, 226),
            ]
        )
        check(
            DataFrame([chunked^]).lazy().select(col("x").interpolate()), runtime
        )
    for scale in [0, 3, 18, 38]:
        var source = decimals(33, scale)
        for name in ["small", "large", "wide"]:
            for method in ["linear", "nearest"]:
                check(
                    source.lazy().select(
                        col(name).interpolate(method).over("k")
                    ),
                    runtime,
                    method == "nearest",
                )
    var wide = DataFrame(
        [
            Series(
                "x",
                Column[UInt64](
                    [0, UInt64.MAX, 0, UInt64(1) << 63],
                    [False, True, False, True],
                ),
            )
        ]
    )
    check(wide.lazy().select(col("x").interpolate()), runtime)
    check(wide.lazy().select(col("x").interpolate("nearest")), runtime, True)


# requires: metal
def test_native_metal_interpolation_independent_gaps_payloads_and_resident_consumers() raises:
    var runtime = require_metal()
    var source = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [99, 2, 99, 99, 8, 99, 14, 99],
                    [False, True, False, False, True, False, True, False],
                ),
            )
        ]
    )
    var actual = (
        source.lazy()
        .select_exprs(
            [
                col("x").interpolate().alias("linear"),
                col("x").interpolate("nearest").alias("nearest"),
            ]
        )
        .collect(accelerator=runtime)
    )
    var valid: List[Bool] = [False, True, True, True, True, True, True, False]
    assert_true(
        actual.column("linear").equals(
            Series("linear", Column[Float64]([0, 2, 4, 6, 8, 11, 14, 0], valid))
        )
    )
    assert_true(
        actual.column("nearest").equals(
            Series("nearest", Column[Int64]([0, 2, 2, 8, 8, 14, 14, 0], valid))
        )
    )
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        0,
        0x7FF8000000000042,
        0,
        0xFFF0000000000042,
        0,
        0x7FF0000000000000,
        0,
        0xFFF0000000000000,
        0,
    ]
    var numbers = List[Float64]()
    var nv = List[Bool]()
    for i in range(len(words)):
        numbers.append(bitcast[DType.float64](words[i]))
        nv.append(i % 2 == 1)
    var exceptional = DataFrame([Series("x", Column[Float64](numbers^, nv^))])
    check(
        exceptional.lazy().select(col("x").interpolate("nearest")),
        runtime,
        True,
    )
    check(exceptional.lazy().select(col("x").interpolate()), runtime)
    var frame = fixture()
    check(
        frame.lazy()
        .with_columns(col("x").interpolate().over("k").alias("filled"))
        .filter(col("filled") > 4)
        .sort("filled"),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(col("x").interpolate().alias("filled"))
        .group_by("k")
        .agg(col("filled").mean())
        .sort("k"),
        runtime,
    )
    check(
        frame.lazy().select(
            col("x").interpolate("nearest").over("k").shift(-1).forward_fill(1)
        ),
        runtime,
        True,
    )
    check(
        frame.lazy().select(
            when(col("x") > 4)
            .then(col("x").interpolate())
            .otherwise(lit(Float64(0)))
        ),
        runtime,
    )
    var empty = DataFrame(
        [Series("x", Column[Float64]([1, 2, 3], [False, False, False]))]
    )
    check(empty.lazy().select(col("x").interpolate()), runtime)
    check(empty.lazy().select(col("x").interpolate("nearest")), runtime, True)


# requires: metal
def test_native_metal_interpolation_profile_budget_family_and_runtime_reuse() raises:
    var runtime = require_metal()
    var query = fixture(513).lazy().select(col("x").interpolate().over("k"))
    var estimate = runtime.estimate_auto(query)
    assert_true(";interpolation-v1" in estimate.family)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    assert_true(";rolling-v1" not in estimate.family)
    assert_true(";rolling-moments-v2" not in estimate.family)
    var exact_runtime = MetalRuntime(memory_limit_bytes=estimate.peak_bytes)
    var profile = query.profile(accelerator=exact_runtime)
    assert_equal(
        profile[1].item(0, "peak_requested_bytes").int64(),
        Int64(estimate.peak_bytes),
    )
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=estimate.peak_bytes - 1)
        )
    check(query, exact_runtime)
