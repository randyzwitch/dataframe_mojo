"""Duration windows preserve sorted time bounds, masks and checked observations."""
# contract-group: execution.backends
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.memory import bitcast
from std.math import abs
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    Series,
    Column,
    DataType,
    Expr,
    LazyFrame,
    col,
    lit,
    when,
)
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.metal import MetalRuntime
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import (
    lower_rows,
    ROW_DURATION_KEY,
    ROW_DURATION_BOUNDS,
)
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_moments import near
from tests.execution_modes.test_backend_metal_decimal_casts import (
    fixture as decimals,
)


def period(dtype: DataType) -> String:
    if dtype.is_date():
        return "3d"
    return (
        "3ms" if dtype.per_second()
        == 1000 else "3us" if dtype.per_second()
        == 1000000 else "3ns"
    )


def fixture(dtype: DataType, n: Int = 35) raises -> DataFrame:
    var x = List[Int64]()
    var valid = List[Bool]()
    var times = List[Int64]()
    var keys = List[Int32]()
    var key_valid = List[Bool]()
    var second = List[Int32]()
    for i in range(n):
        x.append(Int64(i % 17))
        valid.append(i % 5 != 1 and i % 7 != 0)
        times.append(Int64(i // 2 - 10))
        keys.append(Int32(i % 3))
        key_valid.append(i % 7 != 0)
        second.append(Int32(i % 2))
    return DataFrame(
        [
            Series("x", Column[Int64](x^, valid)),
            Series("t", Column[Int64](times^)).with_dtype(dtype),
            Series("k", Column[Int32](keys^, key_valid)),
            Series("j", Column[Int32](second^)),
        ]
    )


def expressions(
    every: String, closed: String, minimum: Int = 0, ddof: Int = 1
) -> List[Expr]:
    return [
        col("x").rolling_sum_by("t", every, closed, minimum).alias("s"),
        col("x").rolling_mean_by("t", every, closed, minimum).alias("m"),
        col("x").rolling_min_by("t", every, closed, minimum).alias("mn"),
        col("x").rolling_max_by("t", every, closed, minimum).alias("mx"),
        col("x").rolling_var_by("t", every, closed, minimum, ddof).alias("v"),
        col("x").rolling_std_by("t", every, closed, minimum, ddof).alias("sd"),
    ]


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    near(
        query.collect(accelerator=runtime),
        query.collect(engine="cpu", streaming=False),
    )


def fails(query: LazyFrame, runtime: MetalRuntime, message: String) raises:
    with assert_raises(contains=message):
        _ = query.collect(engine="cpu", streaming=False)
    with assert_raises(contains=message):
        _ = query.collect(accelerator=runtime)


def test_duration_rolling_capability_intervals_private_indices_types_and_scopes() raises:
    assert_true(not RowCapabilities().duration_rolling)
    var source = fixture(DataType.DATE)
    var query = source.lazy().select_exprs(expressions("3d", "both", 2, 0))
    var caps = _capabilities()
    caps.duration_rolling = False
    with assert_raises(
        contains="duration rolling windows require backend support"
    ):
        _ = lower_rows(query, caps)
    var plan = lower_rows(query, _capabilities())
    var count = 0
    for step in plan.steps:
        if plan.code[step.start * 4] != ROW_DURATION_KEY:
            continue
        count += 1
        var axis = Int(plan.code[step.start * 4 + 2])
        assert_true(plan.slot_dtypes[axis].is_date())
        assert_equal(bitcast[DType.int64](plan.literals[step.start]), Int64(3))
        assert_equal(
            plan.code[(step.start + 1) * 4], Int64(ROW_DURATION_BOUNDS)
        )
        assert_equal(plan.code[(step.start + 1) * 4 + 1], Int64(2))
        assert_equal(plan.code[(step.start + 1) * 4 + 2], Int64(3))
        assert_equal(
            plan.code[(step.start + 1) * 4 + 3],
            Int64(0 if plan.code[step.start * 4 + 3] >= 4 else 1),
        )
    assert_equal(count, 6)
    _ = Program(plan)
    assert_true(
        "unsupported" not in MetalRuntime(device_id=999999).describe(query)
    )
    for every in ["1ns"]:
        with assert_raises(contains="fixed positive interval aligned"):
            _ = lower_rows(
                source.lazy().select(col("x").rolling_sum_by("t", every)),
                _capabilities(),
            )
    with assert_raises(contains="positive"):
        _ = lower_rows(
            source.lazy().select(col("x").rolling_sum_by("t", "0d")),
            _capabilities(),
        )
    with assert_raises(contains="closed must"):
        _ = lower_rows(
            source.lazy().select(col("x").rolling_sum_by("t", "3d", "neither")),
            _capabilities(),
        )
    var aware = fixture(DataType.datetime("us", "America/New_York"))
    with assert_raises(contains="timezone-free"):
        _ = lower_rows(
            aware.lazy().select(col("x").rolling_mean_by("t", "3us")),
            _capabilities(),
        )


# requires: metal
def test_native_metal_duration_rolling_numeric_temporal_units_closed_partitions_slices_and_chunks() raises:
    var runtime = require_metal()
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        for dtype in List[DataType](
            [
                DataType.DATE,
                DataType.datetime("ms"),
                DataType.datetime("us"),
                DataType.datetime("ns"),
            ]
        ):
            var frame = (
                fixture(dtype)
                .with_columns(col("x").cast(DataType.of(D)))
                .slice(1, 33)
            )
            for closed in ["left", "right", "both", "none"]:
                var exprs = expressions(period(dtype), closed)
                check(frame.lazy().select_exprs(exprs), runtime)
                for i in range(len(exprs)):
                    exprs[i] = exprs[i].over(["k", "j"])
                check(frame.lazy().select_exprs(exprs), runtime)
            check(
                frame.head(0)
                .lazy()
                .select_exprs(
                    expressions(period(dtype), "right", 0, Int(Int64.MAX))
                ),
                runtime,
            )
        var frame = fixture(DataType.DATE).with_columns(
            col("x").cast(DataType.of(D))
        )
        var x = Series._from_chunks(
            [frame.column("x").slice(0, 9), frame.column("x").slice(9, 26)]
        )
        var t = Series._from_chunks(
            [
                frame.column("t").slice(0, 17),
                frame.column("t").slice(17, 0),
                frame.column("t").slice(17, 18),
            ]
        )
        check(
            DataFrame([x^, t^])
            .lazy()
            .select_exprs(expressions("3d", "both", 2, 0)),
            runtime,
        )
    var decimal = decimals(35, 3)
    var axis = fixture(DataType.DATE).column("t")
    decimal = decimal.with_column(axis^)
    for name in ["small", "large", "wide"]:
        check(
            decimal.lazy().select_exprs(
                [
                    col(name).rolling_min_by("t", "3d"),
                    col(name).rolling_max_by("t", "3d").alias("max"),
                ]
            ),
            runtime,
        )
        check(
            decimal.lazy().select(
                col(name).cast(DataType.FLOAT64).rolling_mean_by("t", "3d")
            ),
            runtime,
        )


# requires: metal
def test_native_metal_duration_rolling_checked_indices_unsigned_sums_and_finite_outlier_recovery() raises:
    var runtime = require_metal()
    var source = fixture(DataType.DATE, 3)
    var unsorted = source.with_column(
        Series("t", Column[Int64]([0, 2, 1])).with_dtype(DataType.DATE)
    )
    fails(
        unsorted.lazy().select(col("x").rolling_sum_by("t", "1d")),
        runtime,
        "not sorted",
    )
    var nulls = source.with_column(
        Series("t", Column[Int64]([0, 1, 2], [True, False, True])).with_dtype(
            DataType.DATE
        )
    )
    fails(
        nulls.lazy().select(col("x").rolling_mean_by("t", "1d")),
        runtime,
        "contains nulls",
    )
    var underflow = DataFrame(
        [
            Series("x", Column[Int64]([1], [False])),
            Series("t", Column[Int64]([Int64.MIN])).with_dtype(DataType.DATE),
        ]
    )
    fails(
        underflow.lazy().select(col("x").rolling_sum_by("t", "1d", "right", 0)),
        runtime,
        "overflow",
    )
    var wide = DataFrame(
        [
            Series("x", Column[UInt64]([UInt64.MAX, 1, UInt64.MAX])),
            Series("t", Column[Int64]([0, 3, 6])).with_dtype(DataType.DATE),
        ]
    )
    check(wide.lazy().select_exprs(expressions("1d", "right", 0, 0)), runtime)
    var overflow = DataFrame(
        [
            Series("x", Column[UInt64]([UInt64.MAX, 1])),
            Series("t", Column[Int64]([0, 0])).with_dtype(DataType.DATE),
        ]
    )
    fails(
        overflow.lazy().select(col("x").rolling_sum_by("t", "1d")),
        runtime,
        "overflow",
    )
    check(
        overflow.lazy().select(col("x").rolling_sum_by("t", "1d", "right", 3)),
        runtime,
    )
    var outlier = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [0, 1e12, 1e12 + 0.75, 0], [True, True, True, False]
                ),
            ),
            Series("t", Column[Int64]([0, 1, 2, 3])).with_dtype(DataType.DATE),
        ]
    )
    var actual = (
        outlier.lazy()
        .select_exprs(
            [
                col("x").rolling_var(3, 1, 1).alias("rv"),
                col("x").rolling_std(3, 1, 1).alias("rs"),
                col("x").rolling_var_by("t", "3d").alias("tv"),
                col("x").rolling_std_by("t", "3d").alias("ts"),
            ]
        )
        .collect(accelerator=runtime)
    )
    for name in ["rv", "tv"]:
        assert_equal(actual.column(name).float64()._get(3), Float64(0.28125))
    for name in ["rs", "ts"]:
        assert_true(
            abs(
                actual.column(name).float64()._get(3)
                - Float64(0.5303300858899106)
            )
            <= Float64(1e-12)
        )


# requires: metal
def test_native_metal_duration_rolling_consumers_profile_family_budget_and_runtime_reuse() raises:
    var runtime = require_metal()
    var frame = fixture(DataType.DATE)
    check(
        frame.lazy()
        .with_columns(col("x").rolling_sum_by("t", "3d").over("k").alias("s"))
        .filter(col("s") > 3)
        .sort("s"),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(col("x").rolling_mean_by("t", "3d").alias("m"))
        .group_by("k")
        .agg(col("m").mean())
        .sort("k"),
        runtime,
    )
    check(
        frame.lazy().select(
            col("x").rolling_var_by("t", "3d").over("k").shift(-1)
        ),
        runtime,
    )
    check(
        frame.lazy().select(
            when(col("k") > 0)
            .then(col("x").rolling_mean_by("t", "3d"))
            .otherwise(lit(Float64(0)))
        ),
        runtime,
    )
    var query = (
        frame.lazy()
        .filter(col("k") > 0)
        .select(col("x").rolling_mean_by("t", "3d").over("k"))
    )
    var estimate = runtime.estimate_auto(query)
    assert_true(";duration-rolling-v1" in estimate.family)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
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
    var narrow = (
        frame.with_columns(col("x").cast(DataType.FLOAT32))
        .lazy()
        .select(col("x").rolling_sum_by("t", "3d"))
    )
    assert_true(";mojo-metal-v1;ops=" in runtime.estimate_auto(narrow).family)
