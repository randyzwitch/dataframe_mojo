"""Scalar GPU mode has independently anchored typed, ordered winners."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit
from dataframe.metal import MetalRuntime
from dataframe.accelerator.distributions import lower_scalar_modes
from tests.public_contracts.test_scalar_mode import (
    mode_sources,
    mode_expected,
    float_mode_source,
    assert_float_mode_bits,
)
from tests.support.coverage_compare import assert_frame, assert_value
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_mojo_gpu_scalar_mode_typed_winners_nulls_empty_and_scalar_siblings() raises:
    for source in mode_sources():
        var raw = DataFrame([source.copy()])
        var expected = mode_expected(source)
        for layout in range(3):
            var frame = (
                raw.copy() if layout
                == 0 else offset_layout(raw) if layout
                == 1 else chunk_layout(raw)
            )
            var query = frame.lazy().select(col("x").mode())
            var actual = query.collect(
                accelerator=MetalRuntime(), engine="accel"
            )
            assert_frame(actual, expected)
            var empty = (
                frame.clear()
                .lazy()
                .select(col("x").mode())
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
            assert_equal(empty.height(), 0)
            assert_equal(empty.dtypes(), expected.dtypes())
            var nulls = (
                frame.tail(2)
                .lazy()
                .select(col("x").mode())
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
            assert_equal(nulls.height(), 1)
            assert_true(nulls.item().is_null())
            var siblings = (
                frame.lazy()
                .select_exprs(
                    [
                        col("x").mode(),
                        lit(Int64(7)).alias("scalar"),
                        col("x").len().alias("rows"),
                    ]
                )
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
            assert_equal(siblings.height(), 3)
            for row in range(3):
                assert_value(siblings.item(row, "x"), expected.item(row, "x"))
                assert_equal(siblings.item(row, "scalar").int64(), Int64(7))
                assert_equal(siblings.item(row, "rows").int64(), Int64(6))
            var empty_siblings = (
                frame.clear()
                .lazy()
                .select_exprs([col("x").mode(), lit(Int64(7)).alias("scalar")])
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
            assert_equal(empty_siblings.height(), 0)


def test_mojo_gpu_scalar_mode_float_equality_first_bits_and_budget() raises:
    for single in [False, True]:
        var source = float_mode_source(single)
        var expected = DataFrame([source.take([4, 0, 6, 2, 8])])
        for layout in range(3):
            var raw = DataFrame([source.copy()])
            var frame = (
                raw.copy() if layout
                == 0 else offset_layout(raw) if layout
                == 1 else chunk_layout(raw)
            )
            var query = frame.lazy().select(col("x").mode())
            var actual = query.collect(
                accelerator=MetalRuntime(), engine="accel"
            )
            assert_frame(actual, expected)
            assert_float_mode_bits(actual, single)
            if layout == 0:
                var measured = query.profile(accelerator=MetalRuntime())
                var peak = Int(
                    measured[1].item(0, "query_peak_requested_bytes").int64()
                )
                assert_true(peak > 0)
                assert_frame(
                    query.collect(
                        accelerator=MetalRuntime(memory_limit_bytes=peak),
                        engine="accel",
                    ),
                    expected,
                )
                with assert_raises(contains="budget"):
                    _ = query.collect(
                        accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
                        engine="accel",
                    )

    # A valid first signaling NaN is quieted by the CPU Float32 histogram's
    # Float64 state conversion. Hidden invalid payloads cannot witness this.
    for first in [UInt32(0x7F800042), UInt32(0xFF800073)]:
        var words: List[UInt32] = [
            first,
            UInt32(0x7FC00099),
            UInt32(0x80000000),
            UInt32(0),
        ]
        var values = List[Float32]()
        for word in words:
            values.append(bitcast[DType.float32](word))
        var raw = DataFrame([Series("x", Column[Float32](values^))])
        for layout in range(3):
            var frame = (
                raw.copy() if layout
                == 0 else offset_layout(raw) if layout
                == 1 else chunk_layout(raw)
            )
            var query = frame.lazy().select(col("x").mode())
            var cpu = query.collect(engine="cpu", streaming=False)
            assert_equal(cpu.height(), 2)
            var cpu_column = cpu.column("x")
            ref cpu_values = cpu_column._data[Column[Float32]]
            assert_equal(
                bitcast[DType.uint32](cpu_values._get(0)), UInt32(0x80000000)
            )
            assert_equal(
                bitcast[DType.uint32](cpu_values._get(1)),
                first | UInt32(0x00400000),
            )
            var actual = query.collect(
                accelerator=MetalRuntime(), engine="accel"
            )
            assert_equal(actual.height(), 2)
            var gpu_column = actual.column("x")
            ref gpu_values = gpu_column._data[Column[Float32]]
            assert_equal(
                bitcast[DType.uint32](gpu_values._get(0)), UInt32(0x80000000)
            )
            assert_equal(
                bitcast[DType.uint32](gpu_values._get(1)),
                first | UInt32(0x00400000),
            )
            assert_frame(actual, cpu)


def test_mojo_gpu_scalar_mode_reports_remaining_alignment_and_literal_shape() raises:
    var frame = DataFrame([Series("x", Column[Int64]([3, 1, 3, 1]))])
    with assert_raises(contains="ordinal alignment"):
        _ = lower_scalar_modes(
            frame.lazy().select_exprs(
                [col("x").mode().alias("a"), col("x").mode().alias("b")]
            )
        )
    for empty in [False, True]:
        var input = frame.clear() if empty else frame.copy()
        var query = input.lazy().select(lit(Int64(7)).mode())
        with assert_raises(
            contains="Aggregate input must be row-valued without nested aggregates"
        ):
            _ = query.collect(engine="cpu", streaming=False)
        with assert_raises(
            contains="Aggregate input must be row-valued without nested aggregates"
        ):
            _ = query.collect(accelerator=MetalRuntime(), engine="accel")
