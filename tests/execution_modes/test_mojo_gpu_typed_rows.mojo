"""Native Float32 row programs retain CPU values, nulls, and source metadata."""
# contract-group: execution.backends
from std.memory import ArcPointer, bitcast
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from tests.support.coverage_generators import offset_layout, chunk_layout


def _assert_typed_rows_profile(
    query: LazyFrame,
) raises -> Tuple[DataFrame, DataFrame]:
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    var expected = query.collect(engine="cpu", streaming=False, optimize=False)
    var collected = query.collect(accelerator=MetalRuntime(), engine="accel")
    assert_true(measured[0].equals(expected))
    assert_true(collected.equals(expected))
    assert_equal(measured[1].item(0, "runtime_implementation").string(), "mojo")
    return measured^


# requires: mojo_gpu
def test_mojo_gpu_typed_rows_float32_projection_and_arithmetic15() raises:
    var source = DataFrame(
        [Series("x", Column[Float32]([-9.5, -1, 0, 0.25, 3, 100.5]))]
    )
    var arithmetic: Expr = col("x")
    for step in range(15):
        if step % 3 == 0:
            arithmetic = arithmetic + lit(Float32(1.25))
        elif step % 3 == 1:
            arithmetic = arithmetic * lit(Float32(0.5))
        else:
            arithmetic = arithmetic - lit(Float32(0.125))
    var query = source.lazy().select(arithmetic.alias("value"))
    var measured = _assert_typed_rows_profile(query)
    assert_equal(
        measured[1].item(0, "algorithm").string(),
        "mojo_gpu_native_float32_rows",
    )
    assert_equal(measured[0].column("value").dtype(), DataType.FLOAT32)


# requires: mojo_gpu
def test_mojo_gpu_typed_rows_nullable_offset_slices_chunks_and_field_metadata() raises:
    var values = List[Float32]()
    var valid = List[Bool]()
    for row in range(19):
        values.append(Float32(row - 8) * Float32(0.5))
        valid.append(row % 5 != 2)
    var source_column = Series("x", Column[Float32](values^, valid^))
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    source_column._field_metadata = ArcPointer(metadata^)
    var original = DataFrame([source_column.copy()])
    var sliced = original.slice(3, 13)
    for frame in [offset_layout(sliced), chunk_layout(sliced)]:
        var filtered = frame.lazy().filter(col("x") >= lit(Float32(-1)))
        var query = filtered.select(col("x").alias("alias"))
        var measured = _assert_typed_rows_profile(query)
        assert_equal(
            measured[1].item(0, "algorithm").string(),
            "mojo_gpu_native_float32_filter",
        )
        assert_equal(measured[0].column("alias").dtype(), DataType.FLOAT32)
        assert_true(Bool(measured[0].column("alias")._field_metadata))
        assert_equal(
            measured[0].column("alias")._field_metadata.value()[],
            original.column("x")._field_metadata.value()[],
        )
        var computed = filtered.select(
            (col("x") + lit(Float32(2))).alias("shifted")
        )
        var computed_result = _assert_typed_rows_profile(computed)
        assert_false(Bool(computed_result[0].column("shifted")._field_metadata))
        # The current native matcher handles one output; a multi-output query
        # still returns the exact CPU result through the general row path.
        var multiple = filtered.select_exprs(
            [
                col("x").alias("alias"),
                (col("x") + lit(Float32(2))).alias("shifted"),
            ]
        )
        var fallback = _assert_typed_rows_profile(multiple)
        assert_true(
            fallback[1].item(0, "algorithm").string()
            != "mojo_gpu_native_float32_filter"
        )


# requires: mojo_gpu
def test_mojo_gpu_typed_rows_filter_head_after_filter_and_zero_rows() raises:
    var source = DataFrame(
        [Series("x", Column[Float32]([-3, -2, -1, 0, 1, 2, 3]))]
    )
    var query = (
        source.lazy()
        .filter(col("x") > lit(Float32(-1)))
        .select((col("x") * lit(Float32(3))).alias("v"))
        .head(2)
    )
    var measured = _assert_typed_rows_profile(query)
    assert_equal(
        measured[1].item(0, "algorithm").string(),
        "mojo_gpu_native_float32_filter",
    )
    assert_equal(measured[0].height(), 2)
    assert_equal(measured[0].item(0, "v").float32(), Float32(0))
    assert_equal(measured[0].item(1, "v").float32(), Float32(3))

    var none = (
        source.lazy()
        .filter(col("x") > lit(Float32(10)))
        .select((col("x") + lit(Float32(1))).alias("v"))
    )
    var empty = _assert_typed_rows_profile(none)
    assert_equal(empty[0].height(), 0)
    assert_equal(
        empty[1].item(0, "algorithm").string(),
        "mojo_gpu_native_float32_filter",
    )
    var zero_input = (
        source.clear().lazy().select((col("x") + lit(Float32(1))).alias("v"))
    )
    var zero = _assert_typed_rows_profile(zero_input)
    assert_equal(zero[0].height(), 0)
    assert_equal(zero[1].item(0, "initialization_ms").float64(), Float64(0))
    assert_equal(zero[1].item(0, "synchronizations").int64(), Int64(0))


# requires: mojo_gpu
def test_mojo_gpu_typed_rows_float32_special_values_and_sum_fallback() raises:
    var patterns: List[UInt32] = [
        0x00000000,
        0x80000000,
        0x00000001,
        0x80000001,
        0x7F800000,
        0xFF800000,
        0x7FC00042,
    ]
    var values = List[Float32]()
    for pattern in patterns:
        values.append(bitcast[DType.float32](pattern))
    var source = DataFrame([Series("x", Column[Float32](values^))])
    var projection = source.lazy().select(col("x").alias("copy"))
    var identity = projection.profile(
        accelerator=MetalRuntime(), engine="accel"
    )
    var identity_collect = projection.collect(
        accelerator=MetalRuntime(), engine="accel"
    )
    for row in range(len(patterns)):
        assert_equal(
            bitcast[DType.uint32](
                identity[0].column("copy").float32()._get(row)
            ),
            patterns[row],
        )
        assert_equal(
            bitcast[DType.uint32](
                identity_collect.column("copy").float32()._get(row)
            ),
            patterns[row],
        )

    var arithmetic = source.lazy().select(
        (col("x") + lit(Float32(0))).alias("sum_zero")
    )
    var arithmetic_profile = arithmetic.profile(
        accelerator=MetalRuntime(), engine="accel"
    )
    assert_equal(
        arithmetic_profile[1].item(0, "algorithm").string(),
        "mojo_gpu_native_float32_rows",
    )
    var cpu_arithmetic = arithmetic.collect(
        engine="cpu", streaming=False, optimize=False
    )
    var gpu_arithmetic = arithmetic.collect(
        accelerator=MetalRuntime(), engine="accel"
    )
    for row in range(len(patterns)):
        var cpu_value = cpu_arithmetic.column("sum_zero").float32()._get(row)
        var gpu_value = gpu_arithmetic.column("sum_zero").float32()._get(row)
        if isnan(cpu_value):
            assert_true(isnan(gpu_value))
            assert_true(
                isnan(
                    arithmetic_profile[0].column("sum_zero").float32()._get(row)
                )
            )
        else:
            var cpu_bits = bitcast[DType.uint32](cpu_value)
            assert_equal(bitcast[DType.uint32](gpu_value), cpu_bits)
            assert_equal(
                bitcast[DType.uint32](
                    arithmetic_profile[0].column("sum_zero").float32()._get(row)
                ),
                cpu_bits,
            )

    # Reductions are deliberately outside native Float32 row-program support.
    var finite_source = DataFrame(
        [Series("x", Column[Float32]([Float32(1.5), Float32(2.25)]))]
    )
    var reduced = finite_source.lazy().select(col("x").sum().alias("total"))
    var measured = reduced.profile(accelerator=MetalRuntime(), engine="accel")
    var expected = reduced.collect(
        engine="cpu", streaming=False, optimize=False
    )
    assert_true(measured[0].equals(expected))
    assert_equal(measured[0].item(0, "total").float32(), Float32(3.75))
    assert_true(
        measured[1].item(0, "algorithm").string()
        != "mojo_gpu_native_float32_rows"
    )


# requires: mojo_gpu
def test_mojo_gpu_typed_filter_multiple_blocks_and_query_budget() raises:
    var values = List[Float32]()
    var valid = List[Bool]()
    for row in range(4097):
        values.append(Float32(row % 97) - Float32(48))
        valid.append(row % 13 != 0)
    var frame = DataFrame([Series("x", Column[Float32](values^, valid^))])
    var query = (
        frame.lazy()
        .filter(col("x") > lit(Float32(0)))
        .select(
            (col("x") * lit(Float32(0.25)) + lit(Float32(1))).alias("value")
        )
    )
    var measured = _assert_typed_rows_profile(query)
    assert_equal(
        measured[1].item(0, "algorithm").string(),
        "mojo_gpu_native_float32_filter",
    )
    var peak = Int(measured[1].item(0, "peak_requested_bytes").int64())
    var exact = query.collect(
        engine="accel", accelerator=MetalRuntime(memory_limit_bytes=peak)
    )
    assert_true(exact.equals(measured[0]))
    with assert_raises(contains="memory budget"):
        _ = query.collect(
            engine="accel",
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
        )
    assert_true(
        query.collect(engine="accel", accelerator=MetalRuntime()).equals(
            measured[0]
        )
    )
