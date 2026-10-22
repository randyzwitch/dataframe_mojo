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
def test_mojo_gpu_typed_rows_computed_head_preserves_limits() raises:
    var original = DataFrame(
        [
            Series(
                "x",
                Column[Float32](
                    [-3, -2, -1, 0, 1, 2, 3],
                    [True, False, True, True, False, True, True],
                ),
            )
        ]
    )
    for source in [
        original.copy(),
        offset_layout(original),
        chunk_layout(original),
    ]:
        var runtime = MetalRuntime()
        for limit in [0, 1, 4, 30]:
            var query = (
                source.lazy()
                .select((col("x") * lit(Float32(3))).alias("v"))
                .head(limit)
            )
            for optimize in [False, True]:
                var expected = query.collect(
                    engine="cpu", optimize=optimize, streaming=False
                )
                var measured = query.profile(
                    engine="accel",
                    accelerator=runtime,
                    optimize=optimize,
                    streaming=False,
                )
                var collected = query.collect(
                    engine="accel",
                    accelerator=runtime,
                    optimize=optimize,
                    streaming=False,
                )
                assert_equal(collected.height(), min(limit, source.height()))
                assert_true(collected.equals(expected))
                assert_true(measured[0].equals(expected))
                if not optimize and limit > 0:
                    assert_equal(
                        measured[1].item(0, "algorithm").string(),
                        "mojo_gpu_native_float32_rows",
                    )


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


# requires: mojo_gpu
def test_mojo_gpu_typed_float32_join_uses_narrow_native_input_bridge() raises:
    var values: List[Float32] = [
        Float32(0),
        Float32(1),
        Float32(2),
        Float32(3),
        Float32(4),
        Float32(5),
        Float32(6),
    ]
    var valid: List[Bool] = [True, False, True, True, True, False, True]
    var original = DataFrame([Series("k", Column[Float32](values^, valid^))])
    var right = DataFrame(
        [
            Series(
                "k",
                Column[Float32](
                    [Float32(3), Float32(4), Float32(5), Float32(7)]
                ),
            ),
            Series("payload", Column[Int64]([30, 40, 50, 70])),
        ]
    )
    for frame in [
        offset_layout(original.slice(1, 5)),
        chunk_layout(original.slice(1, 5)),
    ]:
        var query = (
            frame.lazy()
            .select((col("k") + lit(Float32(1))).alias("k"))
            .join(right.lazy(), on="k", how="left")
        )
        var runtime = MetalRuntime()
        assert_true(
            "certified native Float32 JOIN input" in runtime.describe(query)
        )
        var measured = _assert_typed_rows_profile(query)
        var native_rows = -1
        var native_bytes = -1
        var bridge_bytes = -1
        var join_seen = False
        for row in range(measured[1].height()):
            var algorithm = measured[1].item(row, "algorithm").string()
            if algorithm == "mojo_gpu_native_float32_rows":
                native_rows = Int(measured[1].item(row, "output_rows").int64())
                native_bytes = Int(
                    measured[1].item(row, "resident_output_bytes").int64()
                )
                assert_equal(
                    measured[1].item(row, "materialization_boundary").string(),
                    "Mojo GPU resident table",
                )
                assert_equal(
                    measured[1].item(row, "download_bytes").int64(),
                    Int64(0),
                )
            if algorithm == "mojo_gpu_native_hash_join":
                bridge_bytes = Int(
                    measured[1]
                    .item(row, "native_input_conversion_bytes")
                    .int64()
                )
                join_seen = True
        assert_true(native_rows >= 0)
        assert_equal(native_bytes, native_rows * 4 + (native_rows + 7) // 8)
        assert_true(join_seen)
        assert_equal(bridge_bytes, 0)
        assert_true(measured[0].equals(query.collect(engine="cpu")))

        var peak = Int(
            measured[1].item(0, "query_peak_requested_bytes").int64()
        )
        var exact = query.collect(
            engine="accel",
            accelerator=MetalRuntime(memory_limit_bytes=peak),
        )
        assert_true(exact.equals(measured[0]))
        with assert_raises(contains="memory budget"):
            _ = query.collect(
                engine="accel",
                accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            )

    # The native hash join declines Bool payloads, so this join converts the
    # native producer at the generic matrix's right-hand target offset.
    var raw_left = DataFrame(
        [
            Series("k", Column[Float32]([Float32(4), Float32(5)])),
            Series("flag", Column[Bool]([True, False])),
        ]
    )
    var raw_right = DataFrame(
        [Series("source", Column[Float32]([Float32(3), Float32(4)]))]
    )
    var right_query = raw_left.lazy().join(
        raw_right.lazy().select((col("source") + lit(Float32(1))).alias("k")),
        on="k",
    )
    var right_measured = _assert_typed_rows_profile(right_query)
    var right_bridge_bytes = -1
    for row in range(right_measured[1].height()):
        if (
            right_measured[1].item(row, "algorithm").string()
            == "mojo_gpu_sorted_join"
        ):
            right_bridge_bytes = Int(
                right_measured[1]
                .item(row, "native_input_conversion_bytes")
                .int64()
            )
    assert_equal(right_bridge_bytes, 2 * 21)


# requires: mojo_gpu
def test_mojo_gpu_typed_float32_join_bridge_float_keys_empty_and_compacted_filter() raises:
    # JOIN key semantics equate every NaN payload and both signed zeros, while
    # null keys remain unmatched. The native hash join reads the retained
    # Float32 encodings and validity directly, with the same semantics.
    var nan_left = bitcast[DType.float32](UInt32(0x7FC00035))
    var nan_right = bitcast[DType.float32](UInt32(0xFFC00002))
    var special_left = DataFrame(
        [
            Series(
                "k",
                Column[Float32](
                    [
                        nan_left,
                        Float32(-0.0),
                        Float32(0.0),
                        Float32(0.0),
                        Float32(8),
                    ],
                    [True, True, True, False, True],
                ),
            )
        ]
    )
    var special_right = DataFrame(
        [
            Series(
                "k",
                Column[Float32](
                    [
                        nan_right,
                        Float32(0.0),
                        Float32(-0.0),
                        Float32(0.0),
                        Float32(9),
                    ],
                    [True, True, True, False, True],
                ),
            )
        ]
    )
    var special_query = (
        special_left.lazy()
        .select((col("k") * lit(Float32(1))).alias("k"))
        .join(special_right.lazy(), on="k", how="full")
    )
    var special = _assert_typed_rows_profile(special_query)
    assert_equal(special[0].height(), 9)
    var special_rows = -1
    var special_bridge_bytes = -1
    for row in range(special[1].height()):
        var algorithm = special[1].item(row, "algorithm").string()
        if algorithm == "mojo_gpu_native_float32_rows":
            special_rows = Int(special[1].item(row, "output_rows").int64())
        elif algorithm == "mojo_gpu_native_hash_join":
            special_bridge_bytes = Int(
                special[1].item(row, "native_input_conversion_bytes").int64()
            )
    assert_equal(special_rows, 5)
    assert_equal(special_bridge_bytes, 0)

    # Empty computed input still gets a certified native table, and the JOIN
    # reads it with no conversion while the right side remains observable.
    var empty = DataFrame([Series("k", Column[Float32]([]))])
    var empty_right = DataFrame(
        [Series("k", Column[Float32]([Float32(1), Float32(2)]))]
    )
    var empty_query = (
        empty.lazy()
        .select((col("k") + lit(Float32(1))).alias("k"))
        .join(empty_right.lazy(), on="k", how="right")
    )
    var empty_measured = _assert_typed_rows_profile(empty_query)
    assert_equal(empty_measured[0].height(), 2)
    var empty_native_rows = -1
    var empty_bridge_bytes = -1
    for row in range(empty_measured[1].height()):
        var algorithm = empty_measured[1].item(row, "algorithm").string()
        if algorithm == "mojo_gpu_native_float32_rows":
            empty_native_rows = Int(
                empty_measured[1].item(row, "output_rows").int64()
            )
        elif algorithm == "mojo_gpu_native_hash_join":
            empty_bridge_bytes = Int(
                empty_measured[1]
                .item(row, "native_input_conversion_bytes")
                .int64()
            )
    assert_equal(empty_native_rows, 0)
    assert_equal(empty_bridge_bytes, 0)

    # The filter compacts more than one validity byte before the JOIN. Null
    # source rows are excluded by the filter, and the JOIN reads the compacted
    # native table directly.
    var many_values = List[Float32]()
    var many_valid = List[Bool]()
    for row in range(48):
        many_values.append(Float32(row))
        many_valid.append(row % 7 != 0)
    var many = DataFrame(
        [Series("k", Column[Float32](many_values^, many_valid^))]
    )
    var many_right = DataFrame(
        [Series("k", Column[Float32]([Float32(7), Float32(20), Float32(40)]))]
    )
    var compacted_query = (
        many.lazy()
        .filter(col("k") > lit(Float32(5)))
        .select((col("k") + lit(Float32(1))).alias("k"))
        .join(many_right.lazy(), on="k", how="left")
    )
    var compacted = _assert_typed_rows_profile(compacted_query)
    var compacted_native_rows = -1
    var compacted_native_bytes = -1
    var compacted_bridge_bytes = -1
    for row in range(compacted[1].height()):
        var algorithm = compacted[1].item(row, "algorithm").string()
        if algorithm == "mojo_gpu_native_float32_filter":
            compacted_native_rows = Int(
                compacted[1].item(row, "output_rows").int64()
            )
            compacted_native_bytes = Int(
                compacted[1].item(row, "resident_output_bytes").int64()
            )
        elif algorithm == "mojo_gpu_native_hash_join":
            compacted_bridge_bytes = Int(
                compacted[1].item(row, "native_input_conversion_bytes").int64()
            )
    assert_true(compacted_native_rows > 8)
    assert_equal(
        compacted_native_bytes,
        compacted_native_rows * 4 + (compacted_native_rows + 7) // 8,
    )
    assert_equal(compacted_bridge_bytes, 0)


# requires: mojo_gpu
def test_mojo_gpu_typed_float32_join_keeps_multicolumn_producer_generic() raises:
    var left = DataFrame(
        [
            Series("k", Column[Float32]([Float32(1), Float32(2)])),
            Series("v", Column[Float32]([Float32(10), Float32(20)])),
        ]
    )
    var right = DataFrame(
        [Series("k", Column[Float32]([Float32(1), Float32(2)]))]
    )
    var query = (
        left.lazy()
        .with_columns((col("v") + lit(Float32(1))).alias("v"))
        .join(right.lazy(), on="k")
    )
    var runtime = MetalRuntime()
    assert_true(
        "0 certified native Float32 JOIN input" in runtime.describe(query)
    )
    var measured = _assert_typed_rows_profile(query)
    var generic_producer = False
    for row in range(measured[1].height()):
        if measured[1].item(row, "operator").string() == "ROW EXPRESSIONS":
            generic_producer = (
                measured[1].item(row, "algorithm").string()
                == "mojo_gpu_row_program"
            )
            assert_equal(
                measured[1].item(row, "native_input_conversion_bytes").int64(),
                Int64(0),
            )
    assert_true(generic_producer)

    # The internal native JOIN producer certificate admits only no-limit
    # plans, so a limited producer stays on the generic matrix path.
    var limited_source = DataFrame(
        [Series("k", Column[Float32]([Float32(1), Float32(2)]))]
    )
    var limited_query = (
        limited_source.lazy()
        .select((col("k") + lit(Float32(1))).alias("k"))
        .head(1)
        .join(right.lazy(), on="k")
    )
    assert_true(
        "0 certified native Float32 JOIN input"
        in MetalRuntime().describe(limited_query)
    )
    var limited_measured = _assert_typed_rows_profile(limited_query)
    assert_true(limited_measured[0].equals(limited_query.collect(engine="cpu")))


# requires: mojo_gpu
def test_mojo_gpu_typed_float32_join_keeps_branched_producers_scoped() raises:
    var source = DataFrame(
        [
            Series(
                "k",
                Column[Float32]([Float32(1), Float32(2), Float32(3)]),
            )
        ]
    )
    var branch = source.lazy().select((col("k") + lit(Float32(1))).alias("k"))
    var query = branch.join(branch, on="k")
    var runtime = MetalRuntime()
    assert_true(
        "2 certified native Float32 JOIN input" in runtime.describe(query)
    )
    var measured = _assert_typed_rows_profile(query)
    var typed_regions = 0
    var conversion_bytes = -1
    for row in range(measured[1].height()):
        var algorithm = measured[1].item(row, "algorithm").string()
        if algorithm == "mojo_gpu_native_float32_rows":
            typed_regions += 1
        elif algorithm == "mojo_gpu_native_hash_join":
            conversion_bytes = Int(
                measured[1].item(row, "native_input_conversion_bytes").int64()
            )
    assert_equal(typed_regions, 2)
    assert_equal(conversion_bytes, 0)
