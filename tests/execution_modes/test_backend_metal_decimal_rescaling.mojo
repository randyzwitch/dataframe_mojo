"""Exact decimal scale changes, integer conversions, and observed cast errors."""
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, Expr, DataType, col, lit, when
from dataframe.decimal import pow10
from dataframe.metal import MetalRuntime
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_decimal_casts import fixture
from tests.execution_modes.test_backend_metal_float64_emulation import same


def test_decimal_rescaling_capability_and_strict_metadata() raises:
    assert_true(not RowCapabilities().decimal_casts)
    var frame = fixture(3, 2)
    for strict in List[Bool]([True, False]):
        var query = frame.lazy().select(
            col("wide").cast(DataType.decimal(9, 1, 32), strict=strict)
        )
        var plan = lower_rows(query, _capabilities())
        var descriptors = Program(plan)
        assert_equal(descriptors.code[4], Int64(217))
        assert_equal(descriptors.code[6], Int64(-1))
        assert_equal(
            descriptors.code[7],
            Int64(2 | (1 << 6) | (9 << 12) | (plan.code[7] << 18)),
        )
        var caps = _capabilities()
        caps.decimal_casts = False
        with assert_raises(contains="Decimal128 row arithmetic"):
            _ = lower_rows(query, caps)
    var runtime = MetalRuntime(device_id=999999)
    assert_true(
        "unsupported"
        not in runtime.describe(
            frame.lazy().select(col("wide").cast(DataType.INT64, strict=False))
        )
    )


def test_native_metal_decimal_rescaling_all_scales_widths_and_numeric_targets() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    for scale in range(39):
        var frame = fixture(31, scale)
        var expressions = List[Expr]()
        for name in List[String](["wide", "large", "small"]):
            var targets: List[DataType] = [
                DataType.INT8,
                DataType.UINT8,
                DataType.INT16,
                DataType.UINT16,
                DataType.INT32,
                DataType.UINT32,
                DataType.INT64,
                DataType.UINT64,
                DataType.BOOL,
                DataType.FLOAT32,
                DataType.decimal(9, min(scale // 2, 9), 32),
                DataType.decimal(18, min(scale + 1, 18), 64),
                DataType.decimal(38, min(scale + 1, 38)),
            ]
            for target in targets:
                expressions.append(
                    col(name)
                    .cast(target, strict=False)
                    .alias("c" + String(len(expressions)))
                )
        var query = frame.lazy().select_exprs(expressions)
        same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    # Literal answers establish tie rounding independently of CPU comparison.
    var frame = DataFrame(
        [
            Series(
                "v", Column[Int128]([5, 15, 25, 35, -5, -15, -25, -35, 4, 6])
            ).with_dtype(DataType.decimal(38, 1)),
        ]
    )
    var query = frame.lazy().select(col("v").cast(DataType.INT64))
    var result = query.collect(accelerator=runtime)
    var expected: List[Int64] = [0, 2, 2, 4, 0, -2, -2, -4, 0, 1]
    for i in range(len(expected)):
        assert_equal(result.column("v").int64()._get(i), expected[i])
    same(result, query.collect(engine="cpu"))
    for length in List[Int]([0, 1, 7]):
        var sliced = (
            frame.slice(2, length)
            .lazy()
            .select(col("v").cast(DataType.decimal(3, 0, 32)))
        )
        same(sliced.collect(accelerator=runtime), sliced.collect(engine="cpu"))


def test_native_metal_integer_decimal_casts_exact_words_nulls_and_consumers() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "signed",
                Column[Int64](
                    [Int64.MIN, -1, 0, 1, Int64.MAX],
                    [True, True, False, True, True],
                ),
            ),
            Series(
                "unsigned",
                Column[UInt64]([UInt64.MAX, 0, 1, 1 << 53 | 1, 1 << 63]),
            ),
            Series("small", Column[Int8]([Int8.MIN, -1, 0, 1, Int8.MAX])),
            Series("flag", Column[Bool]([True, False, True, False, True])),
            Series("k", Column[Int32]([0, 1, 0, 1, 0])),
        ]
    )
    for scale in range(39):
        var expressions = List[Expr]()
        for name in List[String](["signed", "unsigned", "small", "flag"]):
            for target in List[DataType](
                [
                    DataType.decimal(9, min(scale, 9), 32),
                    DataType.decimal(18, min(scale, 18), 64),
                    DataType.decimal(38, scale),
                ]
            ):
                expressions.append(
                    col(name)
                    .cast(target, strict=False)
                    .alias("c" + String(len(expressions)))
                )
        var query = frame.lazy().select_exprs(expressions)
        same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    var query = (
        frame.lazy()
        .with_columns(col("signed").cast(DataType.decimal(38, 2)).alias("v"))
        .filter(col("v").is_not_null())
        .sort("v")
        .head(3)
    )
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    var joined = (
        frame.lazy()
        .join(
            frame.lazy().select_exprs(
                [
                    col("k"),
                    col("unsigned").cast(DataType.decimal(38, 0)).alias("v"),
                ]
            ),
            "k",
        )
        .head(9)
    )
    same(joined.collect(accelerator=runtime), joined.collect(engine="cpu"))
    var aggregate = frame.lazy().select(
        col("small").cast(DataType.decimal(38, 2)).sum().cast(DataType.INT64)
    )
    same(
        aggregate.collect(accelerator=runtime), aggregate.collect(engine="cpu")
    )
    var window = frame.lazy().select(
        col("small")
        .cast(DataType.decimal(38, 2))
        .max()
        .over("k")
        .cast(DataType.INT64)
    )
    same(window.collect(accelerator=runtime), window.collect(engine="cpu"))
    assert_true(
        ";mojo-metal-v1;ops="
        in runtime.estimate_auto(
            frame.lazy().select(col("small").cast(DataType.decimal(38, 2)))
        ).family
    )


def test_native_metal_decimal_cast_errors_observation_lax_budget_and_reuse() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "v",
                Column[Int128](
                    [999, 1000, -1000, 999], [True, True, True, False]
                ),
            ).with_dtype(DataType.decimal(38, 0)),
            Series("p", Column[Bool]([True, False, False, False])),
        ]
    )
    var target = DataType.decimal(3, 0, 32)
    var bad = frame.lazy().select(col("v").cast(target)).head(1)
    with assert_raises(contains="cast"):
        _ = bad.collect(engine="cpu", optimize=False)
    with assert_raises(contains="strict casts require"):
        _ = bad.collect(accelerator=runtime)
    with assert_raises(contains="cast"):
        _ = (
            frame.lazy()
            .select(col("v").cast(target))
            .collect(accelerator=runtime)
        )
    var lax = frame.lazy().select(col("v").cast(target, strict=False))
    same(lax.collect(accelerator=runtime), lax.collect(engine="cpu"))
    for predicate in List[Expr]([lit(False), col("p")]):
        var query = frame.lazy().select(
            when(predicate)
            .then(col("v").cast(target))
            .otherwise(lit("0").cast(target))
        )
        same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    var filtered = frame.lazy().filter(col("p")).select(col("v").cast(target))
    same(filtered.collect(accelerator=runtime), filtered.collect(engine="cpu"))
    var valid = frame.slice(0, 1).lazy().select(col("v").cast(target))
    with assert_raises(contains="budget"):
        _ = valid.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
    same(valid.collect(accelerator=runtime), valid.collect(engine="cpu"))
