"""Decimal-to-Float64 casts round the integer first, then divide by scale."""
# contract-group: execution.backends
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.memory import bitcast
from tests.execution_modes.test_backend_metal import require_metal
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, Expr, col, lit, when
from dataframe.decimal import pow10
from dataframe.dtype import DataType
from dataframe.metal import (
    MetalRuntime,
)
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from tests.execution_modes.test_backend_metal_scalar_broadcasts import (
    same,
    check,
)


def fixture(n: Int, scale: Int) raises -> DataFrame:
    var wide = List[Int128]()
    var large = List[Int64]()
    var small = List[Int32]()
    var keys = List[Int64]()
    var valid = List[Bool]()
    var maximum = pow10(38) - 1
    var half = (Int128(1) << 100) + (Int128(1) << 47)
    var boundaries = List[Int128](
        [
            Int128(0),
            Int128(1),
            Int128(-1),
            maximum,
            -maximum,
            (Int128(1) << 53) - 1,
            (Int128(1) << 53) + 1,
            half - 1,
            half,
            half + 1,
            -half - 1,
            -half,
            -half + 1,
        ]
    )
    for i in range(n):
        wide.append(boundaries[i % len(boundaries)])
        var direction = Int64(-1) if i % 2 else Int64(1)
        large.append(direction * (Int64(999999999999999999) - Int64(i % 31)))
        small.append(Int32(direction) * (Int32(999999999) - Int32(i % 31)))
        keys.append(Int64(i % 7))
        valid.append(i % 17 != 15)
    return DataFrame(
        [
            Series("wide", Column[Int128](wide^, valid)).with_dtype(
                DataType.decimal(38, scale)
            ),
            Series("large", Column[Int64](large^, valid)).with_dtype(
                DataType.decimal(18, min(scale, 18), 64)
            ),
            Series("small", Column[Int32](small^, valid)).with_dtype(
                DataType.decimal(9, min(scale, 9), 32)
            ),
            Series("k", Column[Int64](keys^)),
        ]
    )


def cast_columns() -> List[Expr]:
    return [
        col("wide").cast(DataType.FLOAT64),
        col("large").cast(DataType.FLOAT64),
        col("small").cast(DataType.FLOAT64),
    ]


def test_decimal_cast_capability_scale_descriptors_and_rejections() raises:
    assert_true(not RowCapabilities().decimal_float_cast)
    for scale in range(39):
        var query = fixture(3, scale).lazy().select_exprs(cast_columns())
        var plan = lower_rows(query, _capabilities())
        var descriptors = Program(plan)
        var casts = 0
        for i in range(0, len(descriptors.code), 4):
            if descriptors.code[i] == 213:
                var actual_scale = scale if casts == 0 else min(
                    scale, 18 if casts == 1 else 9
                )
                assert_equal(descriptors.code[i + 2], Int64(-1))
                assert_equal(
                    descriptors.code[i + 3],
                    Int64(actual_scale) | (plan.code[i + 3] << 18),
                )
                assert_equal(
                    descriptors.literals[i // 4],
                    bitcast[DType.uint64](Float64(pow10(actual_scale))),
                )
                assert_equal(descriptors.node_types[i // 4], Int64(11))
                casts += 1
        assert_equal(casts, 3)
    var caps = _capabilities()
    caps.decimal_float_cast = False
    caps.decimal_casts = False
    with assert_raises(contains="logical"):
        _ = lower_rows(
            fixture(3, 2).lazy().select(col("large").cast(DataType.FLOAT64)),
            caps,
        )
    with assert_raises(contains="Decimal128 row arithmetic"):
        _ = lower_rows(
            fixture(3, 2).lazy().select(col("wide").cast(DataType.FLOAT64)),
            caps,
        )
    with assert_raises(contains="logical"):
        _ = lower_rows(
            fixture(3, 2).lazy().select(col("large").cast(DataType.FLOAT32)),
            caps,
        )
    var runtime = MetalRuntime(device_id=999999)
    var query = fixture(3, 38).lazy().select_exprs(cast_columns())
    assert_true("unsupported" not in runtime.describe(query))


# requires: metal
def test_native_metal_decimal_cast_all_scales_widths_and_integer_rounding() raises:
    _ = require_metal()
    for scale in range(39):
        check(fixture(257, scale).lazy().select_exprs(cast_columns()))
        check(
            fixture(257, scale)
            .lazy()
            .select(col("wide").cast(DataType.FLOAT64, strict=False))
        )
    # Dividing the exact decimal first and rounding once has different results
    # from the CPU's separately rounded integer conversion and division.
    var frame = DataFrame(
        [
            Series(
                "a",
                Column[Int64](
                    [999999999999999999, -999999999999999999, 9007199254740993]
                ),
            ).with_dtype(DataType.decimal(18, 18, 64))
        ]
    )
    check(frame.lazy().select(col("a").cast(DataType.FLOAT64)))


# requires: metal
def test_native_metal_decimal_cast_empty_nulls_slices_sort_and_resident_filters() raises:
    _ = require_metal()
    for n in List[Int]([0, 1, 4097, 65537]):
        var frame = fixture(n, 38)
        check(frame.lazy().select_exprs(cast_columns()))
        check(frame.lazy().slice(-300, 257).select_exprs(cast_columns()))
        check(frame.lazy().sort("wide").head(31).select_exprs(cast_columns()))
        check(
            frame.lazy()
            .filter(col("wide").cast(DataType.FLOAT64) < lit(Float64(0)))
            .select_exprs(cast_columns())
        )
    var nulls = DataFrame(
        [
            Series(
                "a",
                Column[Int128](
                    [pow10(38) - 1, -pow10(38) + 1], List[Bool]([False, False])
                ),
            ).with_dtype(DataType.decimal(38, 38))
        ]
    )
    check(nulls.lazy().select(col("a").cast(DataType.FLOAT64)))
    check(
        nulls.lazy().select(
            when(col("a").is_null())
            .then(lit(Float64(7)))
            .otherwise(col("a").cast(DataType.FLOAT64))
            .alias("a")
        )
    )


# requires: metal
def test_native_metal_decimal_cast_aggregates_windows_broadcasts_and_joins() raises:
    _ = require_metal()
    var frame = fixture(257, 18)
    check(
        frame.lazy().select(
            col("large").sum().cast(DataType.FLOAT64).alias("sum")
        )
    )
    check(
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(col("large").sum().alias("sum"))
        .select(col("sum").cast(DataType.FLOAT64))
    )
    check(
        frame.lazy().select_exprs(
            [
                col("k"),
                col("large").sum().cast(DataType.FLOAT64).alias("sum"),
                col("wide").cast(DataType.FLOAT64),
            ]
        )
    )
    check(
        frame.lazy().select(
            col("large").cast(DataType.FLOAT64).mean().over("k").alias("mean")
        )
    )
    # Floating SUM permits reassociation in both backends. Use exactly
    # representable sums to check cast/group composition; adversarial cast
    # rounding is checked bit for bit before aggregation in the earlier tests.
    var exact = DataFrame(
        [
            Series(
                "x",
                Column[Int128](
                    [1, 2, -3, 4, -5, 6, 0, 7],
                    List[Bool](
                        [True, True, True, True, True, True, True, False]
                    ),
                ),
            ).with_dtype(DataType.decimal(38, 0)),
            Series("k", Column[Int64]([0, 1, 0, 1, 0, 1, 0, 1])),
        ]
    )
    check(
        exact.lazy()
        .group_by("k", maintain_order=True)
        .agg(col("x").cast(DataType.FLOAT64).sum().alias("sum"))
    )
    var right = DataFrame([Series("k", Column[Int64]([1, 2, 2, 4]))])
    check(frame.lazy().join(right.lazy(), "k").select_exprs(cast_columns()))
    check(
        frame.lazy()
        .join(right.lazy(), "k")
        .select(col("large").sum().cast(DataType.FLOAT64))
    )
    check(
        frame.lazy().select(
            when(col("k") == lit(Int64(2)))
            .then(col("wide").cast(DataType.FLOAT64))
            .otherwise(col("large").sum().cast(DataType.FLOAT64))
            .alias("value")
        )
    )


# requires: metal
def test_native_metal_decimal_cast_profile_budget_and_recovery() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var query = fixture(4097, 38).lazy().select_exprs(cast_columns())
    var estimate = runtime.estimate_auto(query)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    var measured = query.profile(accelerator=runtime)
    same(measured[0], query.collect(engine="cpu"))
    var tiny = MetalRuntime(memory_limit_bytes=1)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=tiny)
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
