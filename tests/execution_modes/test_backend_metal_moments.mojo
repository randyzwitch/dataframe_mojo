"""Stable binary64 moments through scalar, grouped and partitioned Metal."""
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.math import sqrt, isnan, isfinite
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, Expr, LazyFrame, col, lit, when
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.metal import MetalRuntime
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from dataframe.reductions import VarState
from tests.support.lifecycle_jobs import WorkerSetting
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.execution_modes.test_backend_metal_decimal_casts import (
    fixture as decimal_fixture,
)


def near(actual: DataFrame, expected: DataFrame) raises:
    assert_equal(actual.columns(), expected.columns())
    assert_equal(actual.height(), expected.height())
    for name in actual.columns():
        assert_equal(actual.column(name).dtype(), expected.column(name).dtype())
        if actual.column(name).dtype() != DataType.FLOAT64:
            assert_true(actual.column(name).equals(expected.column(name)))
            continue
        var a = actual.column(name).float64()
        var b = expected.column(name).float64()
        for i in range(actual.height()):
            assert_equal(a.is_valid(i), b.is_valid(i))
            if not a.is_valid(i):
                continue
            var x = a._get(i)
            var y = b._get(i)
            if x == y or (isnan(x) and isnan(y)):
                continue
            assert_true(isfinite(x) and isfinite(y))
            assert_true(abs(x - y) <= Float64(1e-12) * max(Float64(1), abs(y)))


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    # CPU grouped materialization can change its operation association, even
    # with one worker. Independent native tests check every state word; API
    # comparisons also require the same schema, ordering and validity.
    near(
        query.collect(accelerator=runtime),
        query.collect(engine="cpu", streaming=False),
    )


def expressions(name: String, ddof: Int) -> List[Expr]:
    return [col(name).var(ddof).alias("v"), col(name).std(ddof).alias("s")]


def test_moments_capability_precision_metadata_and_private_casts() raises:
    assert_true(not RowCapabilities().moments)
    var setting = WorkerSetting(8)
    var frame = DataFrame([Series("x", Column[UInt64]([0, UInt64.MAX]))])
    for ddof in List[Int]([0, 1, 2, Int(Int64.MAX)]):
        var query = frame.lazy().select_exprs(expressions("x", ddof))
        var caps = _capabilities()
        caps.moments = False
        with assert_raises(
            contains="variance and standard deviation require backend support"
        ):
            _ = lower_rows(query, caps)
        caps = _capabilities()
        caps.float64 = False
        caps.float64_emulated = False
        with assert_raises(contains="Float64 arithmetic"):
            _ = lower_rows(query, caps)
        var plan = lower_rows(query, _capabilities())
        for output in plan.outputs:
            assert_equal(plan.slot_dtypes[output.slot], DataType.FLOAT64)
            assert_equal(
                output.min_count & 0x7FFFFFFF, min(ddof, Int(Int32.MAX))
            )
            assert_equal(output.min_count >> 47, 8)
            assert_equal((output.min_count >> 31) & 65535, 0)
        _ = Program(plan)
        assert_true(
            "unsupported" not in MetalRuntime(device_id=999999).describe(query)
        )
    with assert_raises(contains="ddof"):
        _ = lower_rows(frame.lazy().select(col("x").var(-1)), _capabilities())
    with assert_raises(contains="explicit Float64 cast"):
        _ = lower_rows(
            decimal_fixture(3, 2).lazy().select(col("wide").var()),
            _capabilities(),
        )
    _ = setting^


def test_native_metal_moments_numeric_widths_ddof_empty_null_and_literal_results() raises:
    _ = require_metal()
    var setting = WorkerSetting(1)
    var runtime = MetalRuntime()
    comptime for D in NUMERIC_DTYPES:
        var values = List[Scalar[D]]()
        var valid = List[Bool]()
        var keys = List[Int32]()
        var key_valid = List[Bool]()
        for i in range(263):
            values.append(Scalar[D](i % 11))
            valid.append(i % 5 != 0)
            keys.append(Int32(i % 3))
            key_valid.append(i % 7 != 0)
        var frame = DataFrame(
            [
                Series("x", Column[Scalar[D]](values^, valid^)),
                Series("k", Column[Int32](keys^, key_valid^)),
            ]
        ).slice(3, 257)
        for ddof in List[Int]([0, 1, 2, 100, Int(Int64.MAX)]):
            check(frame.lazy().select_exprs(expressions("x", ddof)), runtime)
            check(
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg(expressions("x", ddof)),
                runtime,
            )
            check(
                frame.lazy().select_exprs(
                    [
                        col("x").var(ddof).over("k").alias("v"),
                        col("x").std(ddof).over("k").alias("s"),
                    ]
                ),
                runtime,
            )
        for rows in List[Int]([0, 1]):
            check(
                frame.head(rows).lazy().select_exprs(expressions("x", 1)),
                runtime,
            )
            check(
                frame.head(rows)
                .lazy()
                .group_by("k", maintain_order=True)
                .agg(expressions("x", 0)),
                runtime,
            )
            check(
                frame.head(rows).lazy().select(col("x").std().over("k")),
                runtime,
            )
    var frame = DataFrame([Series("x", Column[Int64]([1, 2, 3, 4]))])
    same(
        frame.lazy()
        .select_exprs(expressions("x", 0))
        .collect(accelerator=runtime),
        DataFrame(
            [
                Series("v", Column[Float64]([1.25])),
                Series("s", Column[Float64]([sqrt(Float64(1.25))])),
            ]
        ),
    )
    same(
        frame.lazy()
        .select_exprs(expressions("x", 1))
        .collect(accelerator=runtime),
        DataFrame(
            [
                Series("v", Column[Float64]([Float64(5) / 3])),
                Series("s", Column[Float64]([sqrt(Float64(5) / 3)])),
            ]
        ),
    )
    var nulls = DataFrame(
        [
            Series("x", Column[Float64]([1, 2, 3], [False, False, False])),
            Series("k", Column[Int32]([0, 0, 1])),
        ]
    )
    for ddof in List[Int]([0, 1]):
        check(nulls.lazy().select_exprs(expressions("x", ddof)), runtime)
        check(
            nulls.lazy()
            .group_by("k", maintain_order=True)
            .agg(expressions("x", ddof)),
            runtime,
        )
        check(nulls.lazy().select(col("x").var(ddof).over("k")), runtime)
    _ = setting^


def test_native_metal_moments_ieee_words_full_unsigned_and_decimal_casts() raises:
    _ = require_metal()
    var setting = WorkerSetting(1)
    var runtime = MetalRuntime()
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        1,
        0x8000000000000001,
        0x000FFFFFFFFFFFFF,
        0x0010000000000000,
        0x3FF0000000000001,
        0xBFF0000000000000,
        0x7FEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000042,
        0x7FF0000000000042,
    ]
    for word in words:
        var frame = DataFrame(
            [
                Series(
                    "x",
                    Column[Float64](
                        [bitcast[DType.float64](word), 1, -1, 2, 4],
                        [True, True, False, True, True],
                    ),
                ),
                Series("k", Column[Int32]([0, 1, 0, 1, 0])),
            ]
        )
        for ddof in List[Int]([0, 1, 4]):
            check(frame.lazy().select_exprs(expressions("x", ddof)), runtime)
            check(
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg(expressions("x", ddof)),
                runtime,
            )
            check(frame.lazy().select(col("x").std(ddof).over("k")), runtime)
    var unsigned = DataFrame(
        [
            Series(
                "x",
                Column[UInt64](
                    [UInt64.MAX, UInt64.MAX - 1024, UInt64.MAX - 4096, 0]
                ),
            )
        ]
    )
    check(unsigned.lazy().select_exprs(expressions("x", 1)), runtime)
    for scale in List[Int]([0, 2, 9]):
        var frame = decimal_fixture(13, scale)
        check(
            frame.lazy().select_exprs(
                [
                    col("wide").cast(DataType.FLOAT64).var().alias("v"),
                    col("large").cast(DataType.FLOAT64).std().alias("s"),
                    col("small").cast(DataType.FLOAT64).var(0).alias("small"),
                ]
            ),
            runtime,
        )
    _ = setting^


def test_native_metal_moments_large_worker_merges_and_conditioned_accuracy() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    for workers in List[Int]([1, 2, 8]):
        var setting = WorkerSetting(workers)
        for rows in List[Int]([65537, 131073, 262147]):
            var values = List[Float64]()
            var keys = List[Int32]()
            var serial = VarState()
            var residual_sum = Float64(0)
            var residual_squares = Float64(0)
            for i in range(rows):
                var value = Float64(1048576) + Float64(i % 17) / 8
                values.append(value)
                keys.append(Int32(i % 3))
                serial.add(value)
                var residual = Float64(i % 17) / 8
                residual_sum += residual
                residual_squares += residual * residual
            var frame = DataFrame(
                [
                    Series("x", Column[Float64](values^)),
                    Series("k", Column[Int32](keys^)),
                ]
            )
            var query = frame.lazy().select_exprs(expressions("x", 1))
            var plan = lower_rows(query, _capabilities())
            for output in plan.outputs:
                assert_equal(output.min_count >> 47, workers)
            var actual = query.collect(accelerator=runtime)
            # Check both materializing and streaming CPU routes.
            near(actual, frame.select_exprs(expressions("x", 1)))
            var reference = serial.variance(1).value()
            var variance = actual.column("v").float64()._get(0)
            assert_true(abs(variance - reference) < Float64(1e-9))
            var centered = (
                residual_squares - residual_sum * residual_sum / Float64(rows)
            ) / Float64(rows - 1)
            assert_true(abs(variance - centered) < Float64(1e-9))
            var streamed = query.collect(engine="cpu")
            assert_true(
                abs(variance - streamed.column("v").float64()._get(0))
                < Float64(1e-9)
            )
            var window = frame.lazy().select_exprs(
                [
                    col("x").var().over("k").alias("v"),
                    col("x").std().over("k").alias("s"),
                ]
            )
            near(
                window.collect(accelerator=runtime),
                frame.select_exprs(
                    [
                        col("x").var().over("k").alias("v"),
                        col("x").std().over("k").alias("s"),
                    ]
                ),
            )
            var grouped = (
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg(expressions("x", 1))
            )
            var a = grouped.collect(accelerator=runtime)
            var b = grouped.collect(engine="cpu")
            assert_true(a.column("k").equals(b.column("k")))
            for i in range(a.height()):
                assert_true(
                    abs(
                        a.column("v").float64()._get(i)
                        - b.column("v").float64()._get(i)
                    )
                    < Float64(1e-9)
                )
                assert_true(
                    abs(
                        a.column("s").float64()._get(i)
                        - b.column("s").float64()._get(i)
                    )
                    < Float64(1e-9)
                )
        _ = setting^


def test_native_metal_moments_nested_consumers_preflight_cost_budget_and_reuse() raises:
    _ = require_metal()
    var setting = WorkerSetting(1)
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Int32](
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [True, False, True, True, True, True, False, True],
                ),
            ),
            Series("k", Column[Int32]([0, 1, 0, 1, 0, 1, 0, 1])),
            Series("j", Column[Int32]([1, 1, 0, 0, 1, 1, 0, 0])),
        ]
    )
    check(
        frame.lazy().select_exprs(
            [
                col("x").var().over("k").std(0).over("j").alias("nested"),
                col("x").std(0).over("j").shift(1).over("k").alias("shift"),
                when(col("k") == 0)
                .then(col("x").var().over("k"))
                .otherwise(col("x").std(0).over("j"))
                .alias("conditional"),
            ]
        ),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(col("x").var().over("k").alias("v"))
        .filter(col("v") > 0)
        .sort("j")
        .head(5)
        .select_exprs([col("v").std().alias("s"), col("x").sum().alias("sum")]),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(col("x").std(0).over("k").alias("s"))
        .group_by("j", maintain_order=True)
        .agg(col("s").var()),
        runtime,
    )
    var right = DataFrame(
        [
            Series("k", Column[Int32]([1, 0])),
            Series("r", Column[Float64]([2, 3])),
        ]
    )
    check(
        frame.lazy()
        .with_columns(col("x").var().over("k").alias("v"))
        .join(right.lazy(), "k")
        .select((col("v") + col("r")).std()),
        runtime,
    )
    check(
        frame.lazy().select(
            (col("x").cast(DataType.INT64) + lit(Int64(1))).var()
        ),
        runtime,
    )
    check(
        frame.lazy().select(
            (col("x").var() + col("x").std(0)).alias("combined")
        ),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(col("x").std().alias("scalar"))
        .filter(col("scalar") > 0)
        .head(3),
        runtime,
    )
    check(
        frame.lazy().select(
            when(col("x").var() > 0)
            .then(col("x").std())
            .otherwise(lit(Float64(0)))
        ),
        runtime,
    )
    var unproven = frame.lazy().select((col("x") + lit(Int32(1))).var()).head(1)
    assert_true(
        "unsupported" in MetalRuntime(device_id=999999).describe(unproven)
    )
    var query = frame.lazy().select(col("x").var().over("k"))
    var estimate = runtime.estimate_auto(frame.lazy().select(col("x").var()))
    assert_true(";moments-v1" in estimate.family)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    same(
        query.collect(engine="auto", accelerator=runtime),
        query.collect(engine="cpu"),
    )
    var profiled = query.profile(accelerator=runtime)
    same(profiled[0], query.collect(engine="cpu"))
    for i in range(profiled[1].height() - 1):
        assert_equal(
            profiled[1].column("host_result_bytes").int64()._get(i), Int64(0)
        )
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    for ddof in List[Int]([0, 1, 7, Int(Int64.MAX), 2, 1]):
        check(frame.lazy().select_exprs(expressions("x", ddof)), runtime)
    _ = setting^
