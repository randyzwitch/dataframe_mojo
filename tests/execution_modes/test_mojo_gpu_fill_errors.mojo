"""Fill operations eagerly observe both inputs, under outer WHEN masks."""

from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.metal import MetalRuntime
from dataframe.accelerator.gpu.runtime import _preflight
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source() raises -> DataFrame:
    return DataFrame(
        [
            Series("i", Column[Int64]([7, 8])),
            Series("f", Column[Float64]([7.0, 8.0])),
            Series("bad", Column[Int64]([128, 0])),
            Series("keep", Column[Bool]([False, False])),
        ]
    )


def _fill(nan: Bool, lax: Bool = False) -> Expr:
    var right = col("bad").cast(DataType.INT8, strict=not lax)
    if nan:
        return col("f").fill_nan(right.cast(DataType.FLOAT64))
    return col("i").fill_null(right.cast(DataType.INT64))


def test_mojo_gpu_fill_observes_unselected_row_rhs_strict_cast() raises:
    var original = _source()
    for variant in range(3):
        var frame = original.copy() if variant == 0 else (
            offset_layout(original) if variant == 1 else chunk_layout(original)
        )
        for nan in [False, True]:
            var query = frame.lazy().select(_fill(nan).alias("v"))
            # Prove terminal ROWS casts reach device execution rather than a
            # scalar normalizer or an unsupported-composition rejection.
            _ = _preflight(query)
            with assert_raises(contains="failed at row 0"):
                _ = query.collect(engine="cpu", streaming=False)
            with assert_raises(contains="strict cast failed"):
                _ = query.collect(accelerator=MetalRuntime(), engine="accel")


def test_mojo_gpu_fill_outer_when_masks_rhs_fault_and_lax_cast_is_safe() raises:
    var frame = _source()
    for nan in [False, True]:
        var fallback = lit(Float64(0)) if nan else lit(Int64(0))
        var masked = frame.lazy().select(
            when(col("keep")).then(_fill(nan)).otherwise(fallback).alias("v")
        )
        _ = _preflight(masked)
        var actual = masked.collect(accelerator=MetalRuntime(), engine="accel")
        var expected = masked.collect(engine="cpu", streaming=False)
        assert_equal(actual.equals(expected), True)
        assert_equal(actual.height(), 2)
        for row in range(2):
            if nan:
                assert_equal(actual.item(row, "v").float64(), Float64(0))
            else:
                assert_equal(actual.item(row, "v").int64(), Int64(0))
        var relaxed = frame.lazy().select(_fill(nan, lax=True).alias("v"))
        _ = _preflight(relaxed)
        var kept = relaxed.collect(accelerator=MetalRuntime(), engine="accel")
        assert_equal(
            kept.equals(relaxed.collect(engine="cpu", streaming=False)), True
        )
        for row in range(2):
            if nan:
                assert_equal(kept.item(row, "v").float64(), Float64(7 + row))
            else:
                assert_equal(kept.item(row, "v").int64(), Int64(7 + row))
