"""Scaled decimal means preserve the CPU quotient/remainder rounding."""
from std.memory import bitcast
from tests.execution_modes.test_backend_metal import require_metal
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit, when
from dataframe.dtype import DataType
from dataframe.metal import (
    MetalRuntime,
    _capabilities,
    _Descriptors,
)
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.capabilities import RowCapabilities
from tests.execution_modes.test_backend_metal_scalar_broadcasts import (
    same,
    check,
)


def fixture(n: Int, scale: Int, sign: Int = 0) raises -> DataFrame:
    var a = List[Int64]()
    var b = List[Int32]()
    var integers = List[Int64]()
    var keys = List[String]()
    var valid = List[Bool]()
    var key_valid = List[Bool]()
    for i in range(n):
        var direction = sign if sign else (1 if i % 3 else -1)
        a.append(Int64(direction) * (Int64(999999999999999999) - Int64(i % 31)))
        b.append(Int32(direction) * (Int32(999999999) - Int32(i % 31)))
        integers.append(1)
        keys.append(String(i % 7))
        valid.append(i % 13 != 2)
        key_valid.append(i % 11 != 3)
    return DataFrame(
        [
            Series("a", Column[Int64](a^, valid)).with_dtype(
                DataType.decimal(18, scale, 64)
            ),
            Series("b", Column[Int32](b^, valid^)).with_dtype(
                DataType.decimal(9, min(scale, 9), 32)
            ),
            Series("k", Column[String](keys^, key_valid^)),
            Series("i", Column[Int64](integers^)),
        ]
    )


def test_decimal_mean_capabilities_and_scale_descriptors() raises:
    assert_true(not RowCapabilities().decimal_mean)
    assert_true(not RowCapabilities().decimal128_reductions)
    var frame = fixture(3, 7)
    var query = frame.lazy().select(col("a").mean())
    var caps = _capabilities()
    var plan = lower_rows(query, caps)
    assert_equal(plan.outputs[0].dtype, DataType.FLOAT64)
    var descriptors = _Descriptors(plan)
    assert_equal(descriptors.outputs[4], Int64(212))
    assert_equal(descriptors.outputs[5], Int64(7))
    caps.decimal_mean = False
    with assert_raises(contains="decimal mean requires"):
        _ = lower_rows(query, caps)
    var wide = DataFrame(
        [Series("a", Column[Int128]([1])).with_dtype(DataType.decimal(38, 2))]
    )
    var wide_query = wide.lazy().select(col("a").mean())
    var wide_descriptors = _Descriptors(lower_rows(wide_query, _capabilities()))
    assert_equal(wide_descriptors.outputs[4], Int64(216))
    var metadata = wide_descriptors.outputs[5]
    assert_equal(metadata & 0xFFFFFFFF, Int64(0))
    assert_equal((metadata >> 32) & 63, Int64(38))
    assert_equal((metadata >> 38) & 63, Int64(2))
    assert_true(((metadata >> 44) & 0xFFFF) > 0)
    var grouped_query = (
        wide.lazy().group_by("a").agg(col("a").mean().alias("mean"))
    )
    var grouped_wide = _Descriptors(lower_rows(grouped_query, _capabilities()))
    var saw_wide_mean = False
    for i in range(0, len(grouped_wide.code), 4):
        if grouped_wide.code[i] == 203 and grouped_wide.code[i + 3] == 216:
            metadata = bitcast[DType.int64](grouped_wide.literals[i // 4])
            assert_equal(metadata & 0xFFFFFFFF, Int64(0))
            assert_equal((metadata >> 32) & 63, Int64(38))
            assert_equal((metadata >> 38) & 63, Int64(2))
            assert_true(((metadata >> 44) & 0xFFFF) > 0)
            saw_wide_mean = True
    assert_true(saw_wide_mean)
    var no_wide_reductions = _capabilities()
    no_wide_reductions.decimal128_reductions = False
    with assert_raises(contains="Decimal128 input reductions"):
        _ = lower_rows(wide_query, no_wide_reductions)
    # Scale is a raw integer payload: a numerical Float64 conversion would
    # silently encode the wrong divisor in the native grouped descriptor.
    for scale in range(19):
        var scaled = (
            fixture(3, scale)
            .lazy()
            .group_by("i")
            .agg([col("a").mean().alias("a"), col("b").mean().alias("b")])
        )
        var grouped = _Descriptors(lower_rows(scaled, _capabilities()))
        var means = 0
        for i in range(0, len(grouped.code), 4):
            if grouped.code[i] == 203 and grouped.code[i + 3] == 212:
                assert_equal(
                    bitcast[DType.int64](grouped.literals[i // 4]),
                    Int64(scale if means == 0 else min(scale, 9)),
                )
                means += 1
        assert_equal(means, 2)
    var runtime = MetalRuntime(device_id=999999)
    assert_true("unsupported" not in runtime.describe(query))


def test_native_metal_decimal_mean_all_scales_signs_and_grouped_rounding() raises:
    _ = require_metal()
    for scale in range(19):
        for sign in List[Int]([-1, 0, 1]):
            var frame = fixture(257, scale, sign)
            check(
                frame.lazy().select_exprs(
                    [
                        col("a").mean().alias("a"),
                        col("b").mean().alias("b"),
                        col("a").sum().alias("sum"),
                        col("a").count().alias("count"),
                        col("i").sum().alias("integer_sum"),
                    ]
                )
            )
            check(
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg(
                    [
                        col("a").mean().alias("a"),
                        col("b").mean().alias("b"),
                        col("a").sum().alias("sum"),
                        col("a").count().alias("count"),
                        col("i").sum().alias("integer_sum"),
                    ]
                )
            )


def test_native_metal_decimal_mean_empty_nulls_partial_boundaries_and_resident_rows() raises:
    _ = require_metal()
    for n in List[Int]([0, 1, 4097, 65537]):
        var frame = fixture(n, 18)
        check(frame.lazy().select_exprs([col("a").mean(), col("b").mean()]))
        check(
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg([col("a").mean(), col("b").mean()])
        )
        check(frame.lazy().sort("k").slice(-300, 257).select(col("a").mean()))
        check(frame.lazy().filter(col("k") == lit("2")).select(col("a").mean()))
    var nulls = DataFrame(
        [
            Series(
                "a", Column[Int64]([1, -1], List[Bool]([False, False]))
            ).with_dtype(DataType.decimal(18, 18, 64))
        ]
    )
    check(nulls.lazy().select(col("a").mean()))
    check(nulls.lazy().select_exprs([col("a"), col("a").mean().alias("mean")]))
    # Total-first Float64 conversion differs by two ULPs for these values.
    var rounding = DataFrame(
        [
            Series(
                "a",
                Column[Int64](
                    [753613813067637343, 542142670295939064, -91891836467610593]
                ),
            ).with_dtype(DataType.decimal(18, 18, 64))
        ]
    )
    check(rounding.lazy().select(col("a").mean()))
    var cancel = DataFrame(
        [
            Series(
                "a",
                Column[Int64](
                    [
                        999999999999999999,
                        999999999999999999,
                        -999999999999999999,
                        -999999999999999999,
                    ]
                ),
            ).with_dtype(DataType.decimal(18, 9, 64))
        ]
    )
    check(cancel.lazy().select(col("a").mean()))


def test_native_metal_decimal_mean_broadcast_conditionals_join_and_group_consumers() raises:
    _ = require_metal()
    var frame = fixture(8193, 18)
    check(
        frame.lazy().select_exprs(
            [
                col("k"),
                col("a").mean().alias("mean"),
                col("b").mean().alias("small"),
            ]
        )
    )
    check(
        frame.lazy()
        .filter(col("a").is_not_null())
        .select(
            when(col("k").is_null())
            .then(col("a").mean())
            .otherwise(col("b").mean())
            .alias("mean")
        )
    )
    check(
        frame.lazy().select(
            when(lit(False))
            .then(col("a").mean())
            .otherwise(lit(Float64(7)))
            .alias("mean")
        )
    )
    var right = DataFrame([Series("k", Column[String](["1", "2", "2"]))])
    check(
        frame.lazy()
        .join(right.lazy(), "k")
        .select_exprs([col("a").mean(), col("b").mean()])
    )
    check(
        frame.lazy()
        .join(right.lazy(), "k")
        .select_exprs([col("k"), col("a").mean().alias("mean")])
    )
    check(
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(col("a").mean())
        .sort("a")
        .head(3)
    )


def test_native_metal_decimal_mean_profile_cost_budget_and_recovery() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var query = (
        fixture(4097, 18)
        .lazy()
        .select_exprs([col("a").mean(), col("b").mean()])
    )
    var estimate = runtime.estimate_auto(query)
    assert_true("metal-decimal-means-v1" in estimate.family)
    assert_true(estimate.peak_bytes > 0)
    var measured = query.profile(accelerator=runtime)
    same(measured[0], query.collect(engine="cpu"))
    var tiny = MetalRuntime(memory_limit_bytes=1)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=tiny)
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
