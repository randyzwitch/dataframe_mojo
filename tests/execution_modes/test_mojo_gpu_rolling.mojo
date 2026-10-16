"""Direct Mojo rolling and interpolation queries, including checked time bounds."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    Expr,
    LazyFrame,
    col,
    lit,
    when,
)
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_moments import near
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.execution_modes.test_mojo_gpu_ordering import _layout
from tests.execution_modes.test_backend_metal_coordinate_interpolation import (
    source as coordinate_source,
)
from tests.execution_modes.test_backend_metal_duration_rolling import (
    fixture as duration_fixture,
)
from tests.execution_modes.test_backend_metal_calendar_rolling import (
    fixture as calendar_fixture,
)
from tests.execution_modes.test_backend_metal_fixed_zone_rolling import (
    local_fixture,
)


def _rolling_expressions() -> List[Expr]:
    return [
        col("x").rolling_sum(7, 2).over("k").alias("sum"),
        col("x").rolling_mean(3, 1).over(["k", "j"]).alias("mean"),
        col("x").rolling_min(17, 0).over("k").alias("min"),
        col("x").rolling_max(1, 0).alias("max"),
        col("x").rolling_var(7, 1, 0).over("k").alias("var"),
        col("x").rolling_std(3, 1, 1).over("k").alias("std"),
        col("x").interpolate().over("k").alias("linear"),
        col("x").interpolate("nearest").over(["k", "j"]).alias("nearest"),
        col("x").interpolate_by(col("axis")).over("k").alias("coordinate"),
    ]


def _rolling_cases() raises -> List[LazyFrame]:
    var queries = List[LazyFrame]()
    comptime for D in NUMERIC_DTYPES:
        var source = coordinate_source(263).slice(3, 257)
        source = source.with_columns(col("x").cast(DataType.of(D)))
        queries.append(source.lazy().select_exprs(_rolling_expressions()))
        var columns = List[Series]()
        for name in source.columns():
            columns.append(_layout(source.column(name), True))
        queries.append(
            DataFrame(columns^).lazy().select_exprs(_rolling_expressions())
        )
    var source = coordinate_source(17)
    queries.append(source.head(0).lazy().select_exprs(_rolling_expressions()))
    queries.append(
        source.lazy().select_exprs(
            [
                col("x").rolling_sum(Int(Int64.MAX), 0).alias("sum"),
                col("x").rolling_mean(3, 300).alias("suppressed_mean"),
                col("x").rolling_var(Int(Int64.MAX), 0, 0).alias("var"),
                col("x")
                .rolling_std(3, 0, Int(Int64.MAX))
                .alias("suppressed_std"),
            ]
        )
    )
    var nulls = DataFrame(
        [
            Series.full_null("x", DataType.FLOAT64, 17),
            Series("k", Column[Int32](List[Int32](length=17, fill=0))),
            Series("j", Column[Int32](List[Int32](length=17, fill=0))),
            Series("axis", Column[Float64](List[Float64](length=17, fill=1))),
        ]
    )
    queries.append(nulls.lazy().select_exprs(_rolling_expressions()))
    return queries^


def _duration_expressions(period: String, closed: String) -> List[Expr]:
    return [
        col("x")
        .rolling_sum_by("t", period, closed, 0)
        .over(["k", "j"])
        .alias("sum"),
        col("x")
        .rolling_mean_by("t", period, closed, 0)
        .over(["k", "j"])
        .alias("mean"),
        col("x")
        .rolling_min_by("t", period, closed, 0)
        .over(["k", "j"])
        .alias("min"),
        col("x")
        .rolling_max_by("t", period, closed, 0)
        .over(["k", "j"])
        .alias("max"),
        col("x")
        .rolling_var_by("t", period, closed, 0, 0)
        .over(["k", "j"])
        .alias("var"),
        col("x")
        .rolling_std_by("t", period, closed, 0, 1)
        .over(["k", "j"])
        .alias("std"),
    ]


def _duration_cases() raises -> List[LazyFrame]:
    var queries = List[LazyFrame]()
    for dtype in [
        DataType.DATE,
        DataType.datetime("ms"),
        DataType.datetime("us"),
        DataType.datetime("ns"),
        DataType.datetime("us", "UTC"),
        DataType.datetime("us", "+05:45"),
        DataType.datetime("ns", "-03:30"),
    ]:
        var source = duration_fixture(dtype, 35)
        var period = (
            "3d" if dtype.is_date() else "3ms" if dtype.per_second()
            == 1000 else "3us" if dtype.per_second()
            == 1000000 else "3ns"
        )
        var calendar = calendar_fixture(dtype)
        if dtype.is_datetime() and dtype.time_zone() not in ["", "UTC"]:
            calendar = local_fixture(dtype)
        for closed in ["left", "right", "both", "none"]:
            queries.append(
                source.lazy().select_exprs(
                    _duration_expressions(period, closed)
                )
            )
            queries.append(
                calendar.lazy().select_exprs(
                    _duration_expressions("1mo2d", closed)
                )
            )
        queries.append(
            source.head(0)
            .lazy()
            .select_exprs(_duration_expressions(period, "none"))
        )
    var unsigned = DataFrame(
        [
            Series("x", Column[UInt64]([UInt64.MAX, 1, 2])),
            Series("t", Column[Int64]([0, 1, 2])).with_dtype(DataType.DATE),
        ]
    )
    queries.append(
        unsigned.lazy().select(col("x").rolling_sum_by("t", "1d", "right", 0))
    )
    var nan = Float64("nan")
    var floats = duration_fixture(DataType.DATE, 8).with_column(
        Series("x", Column[Float64]([nan, 2, -0.0, 0, nan, 3, 4, 5]))
    )
    queries.append(
        floats.lazy().select_exprs(_duration_expressions("3d", "both"))
    )
    return queries^


def _floating_cases() raises -> List[LazyFrame]:
    var queries = List[LazyFrame]()
    for dtype in [DataType.FLOAT32, DataType.FLOAT64]:
        var tiny = bitcast[DType.float32](UInt32(1))
        var source = DataFrame(
            [
                Series(
                    "x",
                    Column[Float64](
                        [
                            0,
                            -0.0,
                            Float64(tiny),
                            -Float64(tiny),
                            Float64("nan"),
                            Float64("inf"),
                            -Float64("inf"),
                            3,
                            5,
                            7,
                            9,
                            11,
                        ],
                        [
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                            True,
                            True,
                            True,
                            True,
                        ],
                    ),
                ).cast(dtype),
            ]
        )
        queries.append(
            source.lazy().select_exprs(
                [
                    col("x").rolling_sum(3, 0).alias("sum"),
                    col("x").rolling_mean(3, 1).alias("mean"),
                    col("x").rolling_min(3, 0).alias("min"),
                    col("x").rolling_max(3, 0).alias("max"),
                    col("x").rolling_var(3, 1, 0).alias("var"),
                    col("x").rolling_std(3, 1, 1).alias("std"),
                    col("x").interpolate().alias("linear"),
                    col("x").interpolate("nearest").alias("nearest"),
                ]
            )
        )
    var values = List[Float64]()
    var valid = List[Bool]()
    for row in range(257):
        values.append(1e15 + Float64(row % 17))
        valid.append(row % 7 != 3)
    queries.append(
        DataFrame([Series("x", Column[Float64](values^, valid^))])
        .lazy()
        .select_exprs(
            [
                col("x").rolling_var(17, 2, 0).alias("var"),
                col("x").rolling_std(17, 2, 1).alias("std"),
            ]
        )
    )
    return queries^


def _failure_cases() raises -> List[Tuple[LazyFrame, String]]:
    var cases = List[Tuple[LazyFrame, String]]()
    var sums = DataFrame([Series("x", Column[Int64]([Int64.MAX, 1, -1]))])
    cases.append(
        (sums.lazy().select(col("x").rolling_sum(2, 2)).head(0), "overflow")
    )
    var unsigned = DataFrame([Series("x", Column[UInt64]([UInt64.MAX]))])
    cases.append(
        (unsigned.lazy().select(col("x").rolling_mean(1, 100)), "overflow")
    )
    for kind in range(3):
        var time: List[Int64] = [0, 2, 1]
        var valid: List[Bool] = [True, True, True]
        var message = "not sorted"
        if kind == 1:
            time = [0, 1, 2]
            valid[1] = False
            message = "null"
        elif kind == 2:
            time = [Int64.MIN, Int64.MIN + 1, Int64.MIN + 2]
            message = "bounds overflow"
        var source = DataFrame(
            [
                Series("x", Column[Int64]([1, 2, 3])),
                Series("t", Column[Int64](time^, valid^)).with_dtype(
                    DataType.DATE
                ),
            ]
        )
        cases.append(
            (
                source.lazy()
                .select(col("x").rolling_sum_by("t", "1d"))
                .head(0),
                message,
            )
        )
    return cases^


# requires: mojo_gpu
def test_mojo_gpu_rolling_numeric_layouts_thresholds_and_interpolation() raises:
    for query in _rolling_cases():
        near(
            query.collect(accelerator=MetalRuntime(), engine="accel"),
            query.collect(engine="cpu", streaming=False),
        )


# requires: mojo_gpu
def test_mojo_gpu_rolling_duration_bounds_calendar_zones_and_unsigned_storage() raises:
    for query in _duration_cases():
        near(
            query.collect(accelerator=MetalRuntime(), engine="accel"),
            query.collect(engine="cpu", streaming=False),
        )
    var source = calendar_fixture(DataType.DATE)
    var result = (
        source.lazy()
        .select(col("x").rolling_sum_by("t", "1mo", "both", 0))
        .collect(accelerator=MetalRuntime(), engine="accel")
    )
    assert_equal(result.column("x").int64()._get(3), Int64(14))


# requires: mojo_gpu
def test_mojo_gpu_rolling_floating_recovery_zero_nan_and_variance_association() raises:
    for query in _floating_cases():
        near(
            query.collect(accelerator=MetalRuntime(), engine="accel"),
            query.collect(engine="cpu", streaming=False),
        )
    var source = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [0, 0, 9, 0, 0], [False, True, True, True, False]
                ),
            )
        ]
    )
    var query = source.lazy().select_exprs(
        [
            col("x").interpolate().alias("linear"),
            col("x").interpolate("nearest").alias("nearest"),
        ]
    )
    same(
        query.collect(accelerator=MetalRuntime(), engine="accel"),
        query.collect(engine="cpu", streaming=False),
    )


# requires: mojo_gpu
def test_mojo_gpu_rolling_errors_compositions_exact_budget_and_recovery() raises:
    var runtime = MetalRuntime()
    for pair in _failure_cases():
        with assert_raises(contains=pair[1]):
            _ = pair[0].collect(accelerator=runtime, engine="accel")
    var source = coordinate_source(35)
    var query = (
        source.lazy()
        .filter(col("k").is_not_null())
        .with_columns(
            col("x").interpolate_by(col("axis")).over("k").alias("filled")
        )
        .with_columns(
            col("filled")
            .rolling_mean(3, 1)
            .over("k")
            .rolling_var(5, 1, 0)
            .alias("v")
        )
        .sort("v")
        .slice(1, 7)
        .select(["filled", "v"])
    )
    var profile = query.profile(accelerator=runtime, engine="accel")
    near(profile[0], query.collect(engine="cpu", streaming=False))
    var peak = Int(profile[1].item(0, "peak_requested_bytes").int64())
    assert_true(peak > 0)
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    near(
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ),
        profile[0],
    )
