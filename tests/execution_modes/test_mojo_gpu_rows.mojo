"""Direct Mojo GPU contracts; no native dataframe runtime is loaded."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col, lit, when
from dataframe.metal import MetalRuntime


# requires: mojo_gpu
def test_mojo_gpu_integer_widths_nullable_slices_chunks_and_chained_programs() raises:
    comptime types = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    )
    comptime for k in range(len(types)):
        comptime D = types[k]
        var values = List[Scalar[D]]()
        var valid = List[Bool]()
        for row in range(1_007):
            values.append(Scalar[D](row % 17 + 1))
            valid.append(row % 11 != 3)
        var source = Series(
            "x", Column[Scalar[D]](values.copy(), valid.copy())
        ).slice(3, 1_001)
        var chunked = Series._from_chunks(
            [source.slice(0, 257), source.slice(257, 0), source.slice(257, 744)]
        )
        for input in List[Series]([source.copy(), chunked.copy()]):
            var frame = DataFrame([input.copy()])
            var actual = (
                frame.lazy()
                .select_exprs(
                    [
                        ((col("x") + 2) * 3 - 1).alias("v"),
                        when(col("x") == 3)
                        .then(col("x") * 4)
                        .otherwise(col("x") - 1)
                        .alias("chosen"),
                        col("x").is_null().alias("missing"),
                    ]
                )
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
            assert_equal(actual.height(), 1_001)
            assert_equal(actual.column("v").dtype(), DataType.of(D))
            for row in range(1_001):
                var present = valid[row + 3]
                assert_equal(actual.item(row, "v").is_null(), not present)
                assert_equal(actual.item(row, "chosen").is_null(), not present)
                assert_equal(actual.item(row, "missing").bool(), not present)
                if present:
                    var value = Int64(values[row + 3])
                    assert_equal(
                        Int64(actual.column("v").numeric[D]()._get(row)),
                        (value + 2) * 3 - 1,
                    )
                    assert_equal(
                        Int64(actual.column("chosen").numeric[D]()._get(row)),
                        value * 4 if value == 3 else value - 1,
                    )


# requires: mojo_gpu
def test_mojo_gpu_conditional_errors_are_observed_only_in_selected_values() raises:
    var source = DataFrame(
        [
            Series("x", Column[Int64]([Int64.MAX, 3])),
            Series("use", Column[Bool]([False, True])),
        ]
    )
    var query = source.lazy().select(
        when(col("use")).then(col("x") + 1).otherwise(lit(Int64(0))).alias("v")
    )
    var result = query.collect(accelerator=MetalRuntime(), engine="accel")
    assert_equal(result.item(0, "v").int64(), Int64(0))
    assert_equal(result.item(1, "v").int64(), Int64(4))
    with assert_raises():
        _ = (
            source.lazy()
            .select(col("x") + 1)
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
    # A failed collection leaves no queued work behind and the next succeeds.
    _ = query.collect(accelerator=MetalRuntime(), engine="accel")


# requires: mojo_gpu
def test_mojo_gpu_integer_endpoints_and_width_overflow_match_exact_anchors() raises:
    comptime types = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    )
    comptime for k in range(len(types)):
        comptime D = types[k]
        var original: List[Scalar[D]] = [Scalar[D].MAX, Scalar[D].MIN, 0]
        var frame = DataFrame([Series("x", Column[Scalar[D]](original.copy()))])
        var result = (
            frame.lazy()
            .select_exprs(
                [
                    (col("x") + 0).alias("plus"),
                    (col("x") - 0).alias("minus"),
                    (col("x") * 1).alias("times"),
                ]
            )
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
        for name in ["plus", "minus", "times"]:
            for row in range(3):
                assert_equal(
                    result.column(name).numeric[D]()._get(row), original[row]
                )
        with assert_raises():
            _ = (
                frame.lazy()
                .select(col("x") + 1)
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
        with assert_raises():
            _ = (
                frame.lazy()
                .select(col("x") - 1)
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
        with assert_raises():
            _ = (
                frame.lazy()
                .select(col("x") * 2)
                .collect(accelerator=MetalRuntime(), engine="accel")
            )


# requires: mojo_gpu
def test_mojo_gpu_preserves_float64_storage_and_reports_direct_execution() raises:
    var patterns: List[UInt64] = [
        0,
        0x8000000000000000,
        1,
        0x7FF0000000000000,
        0x7FF8000000000042,
        0xFFF0000000000001,
    ]
    var values = List[Float64]()
    for pattern in patterns:
        values.append(bitcast[DType.float64](pattern))
    var source = DataFrame([Series("x", Column[Float64](values^))])
    var result = (
        source.lazy()
        .select(col("x").alias("copy"))
        .profile(accelerator=MetalRuntime(), engine="accel")
    )
    for row in range(len(patterns)):
        assert_equal(
            bitcast[DType.uint64](result[0].column("copy").float64()._get(row)),
            patterns[row],
        )
    assert_equal(result[1].item(0, "runtime_implementation").string(), "mojo")
    assert_equal(
        result[1].item(0, "algorithm").string(), "mojo_gpu_row_program"
    )
    var nullable = DataFrame(
        [Series("x", Column[Float64]([0, 0], [False, True]))]
    )
    var filled = (
        nullable.lazy()
        .select(col("x").fill_null(1))
        .collect(accelerator=MetalRuntime(), engine="accel")
    )
    assert_equal(filled.item(0, "x").float64(), Float64(1))
    assert_equal(filled.item(1, "x").float64(), Float64(0))
    with assert_raises():
        _ = (
            source.lazy()
            .select(col("x"))
            .collect(
                accelerator=MetalRuntime(memory_limit_bytes=0), engine="accel"
            )
        )


# requires: mojo_gpu
def test_mojo_gpu_boolean_kleene_nulls_empty_rows_and_zero_columns() raises:
    var source = DataFrame(
        [
            Series(
                "a",
                Column[Bool](
                    [True, False, True, False], [True, True, False, False]
                ),
            ),
            Series(
                "b",
                Column[Bool](
                    [False, True, False, True], [False, False, True, True]
                ),
            ),
        ]
    )
    var query = source.lazy().select_exprs(
        [
            (col("a") & col("b")).alias("and"),
            (col("a") | col("b")).alias("or"),
            (~col("a")).alias("not"),
        ]
    )
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    var expected = query.collect(engine="cpu")
    for name in ["and", "or", "not"]:
        for row in range(4):
            assert_equal(
                actual.item(row, name).is_null(),
                expected.item(row, name).is_null(),
            )
            if not expected.item(row, name).is_null():
                assert_equal(
                    actual.item(row, name).bool(),
                    expected.item(row, name).bool(),
                )
    var empty = (
        source.slice(0, 0)
        .lazy()
        .select(col("a"))
        .collect(accelerator=MetalRuntime(), engine="accel")
    )
    assert_equal(empty.height(), 0)
    var no_columns = (
        source.lazy()
        .drop(["a", "b"])
        .collect(accelerator=MetalRuntime(), engine="accel")
    )
    assert_equal(no_columns.height(), 4)
    assert_equal(no_columns.width(), 0)


# requires: mojo_gpu
def test_mojo_gpu_floating_arithmetic_predicates_rounding_and_subnormals() raises:
    from tests.execution_modes.test_backend_metal_float64_emulation import (
        fixture,
        same,
    )

    var frames = List[DataFrame]()
    frames.append(fixture())
    var boundaries: List[UInt32] = [
        0,
        1,
        2,
        0x007FFFFF,
        0x00800000,
        0x00800001,
        0x17800000,
        0x3DCCCCCD,
        0x3F000000,
        0x3F7FFFFF,
        0x3F800000,
        0x3F800001,
        0x3F800800,
        0x40000000,
        0x4B000000,
        0x7F7FFFFF,
        0x7F800000,
        0x7F800042,
        0x7FC00042,
    ]
    var positive = len(boundaries)
    for i in range(positive):
        boundaries.append(boundaries[i] | 0x80000000)
    var left = List[Float32]()
    var right = List[Float32]()
    var valid = List[Bool]()
    for a in boundaries:
        for b in boundaries:
            left.append(bitcast[DType.float32](a))
            right.append(bitcast[DType.float32](b))
            valid.append(len(left) % 13 != 0)
    frames.append(
        DataFrame(
            [
                Series("a", Column[Float32](left^, valid^)),
                Series("b", Column[Float32](right^)),
            ]
        )
    )
    for frame in frames:
        var query = frame.lazy().select_exprs(
            [
                (col("a") + col("b")).alias("add"),
                (col("a") - col("b")).alias("subtract"),
                (col("a") * col("b")).alias("multiply"),
                (col("a") / col("b")).alias("divide"),
                (col("a") % col("b")).alias("modulo"),
                (col("a") // col("b")).alias("floor_divide"),
                (-col("a")).alias("negative"),
                col("a").abs().alias("absolute"),
                col("a").sqrt().alias("root"),
                col("a").floor().alias("floor"),
                col("a").ceil().alias("ceil"),
                col("a").round(0).alias("round"),
                col("a").round(2).alias("round_hundredths"),
                col("a").round(-2).alias("round_hundreds"),
                (col("a") == col("b")).alias("eq"),
                (col("a") != col("b")).alias("ne"),
                (col("a") < col("b")).alias("lt"),
                (col("a") <= col("b")).alias("le"),
                (col("a") > col("b")).alias("gt"),
                (col("a") >= col("b")).alias("ge"),
                col("a").is_nan().alias("nan"),
                col("a").is_not_nan().alias("not_nan"),
                col("a").is_finite().alias("finite"),
                col("a").is_infinite().alias("infinite"),
                col("a").fill_nan(col("b")).alias("filled"),
                col("a").clip(-1, 1).alias("clipped"),
                ((col("a") + col("b")) * 0.5).alias("composed"),
            ]
        )
        same(
            query.collect(accelerator=MetalRuntime(), engine="accel"),
            query.collect(engine="cpu"),
        )
    # Check the specified NaN payload behavior independently of host NaN rules.
    var nan = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [bitcast[DType.float64](UInt64(0xFFF0000000000042))]
                ),
            )
        ]
    )
    var quieted = (
        nan.lazy()
        .select(col("x") + 0)
        .collect(accelerator=MetalRuntime(), engine="accel")
    )
    assert_equal(
        bitcast[DType.uint64](quieted.column("x").float64()._get(0)),
        UInt64(0xFFF8000000000042),
    )


# requires: mojo_gpu
def test_mojo_gpu_transcendentals_power_and_numeric_casts() raises:
    from tests.execution_modes.test_backend_metal_transcendentals import (
        fixture,
        check,
    )
    from tests.execution_modes.test_backend_metal_floating_power import (
        anchor,
        check as check_power,
    )
    from tests.execution_modes.test_backend_metal_float64_emulation import same

    var runtime = MetalRuntime()
    for dtype in [DataType.FLOAT32, DataType.FLOAT64]:
        var frame = DataFrame([fixture().column("x").cast(dtype)])
        check(
            frame.lazy().select_exprs(
                [col("x").exp().alias("e"), col("x").log().alias("l")]
            ),
            runtime,
        )
    anchor[DType.float32](runtime)
    anchor[DType.float64](runtime)
    var near_one = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [
                        1.0000000000000002,
                        0.9999999999999999,
                        1.01,
                        1.1,
                        1e-300,
                        1e300,
                    ]
                ),
            )
        ]
    )
    for exponent in [-3e17, 3e17, -64.5, -0.5, 0.5, 64.5]:
        check_power(
            near_one.lazy().select(col("x").pow(lit(exponent))), runtime
        )
    var integers = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [Int64.MIN, -129, -128, -1, 0, 1, 127, 128, Int64.MAX]
                ),
            )
        ]
    )
    var converted = integers.lazy().select_exprs(
        [
            col("x").cast(DataType.FLOAT64).alias("wide"),
            col("x").cast(DataType.FLOAT32).alias("narrow"),
            col("x").cast(DataType.INT8, strict=False).alias("tiny"),
            col("x").cast(DataType.UINT64, strict=False).alias("unsigned"),
            col("x").cast(DataType.BOOL, strict=False).alias("truth"),
        ]
    )
    same(
        converted.collect(accelerator=runtime, engine="accel"),
        converted.collect(engine="cpu"),
    )
    with assert_raises():
        _ = (
            integers.lazy()
            .select(col("x").cast(DataType.INT8))
            .collect(accelerator=runtime, engine="accel")
        )
    var floating = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [
                        -129.5,
                        -128.9,
                        -0.5,
                        -0.0,
                        0.0,
                        127.9,
                        128,
                        Float64("nan"),
                        Float64("inf"),
                        Float64("-inf"),
                        9223372036854775808.0,
                        18446744073709551616.0,
                    ]
                ),
            )
        ]
    )
    var query = floating.lazy().select_exprs(
        [
            col("x").cast(DataType.INT8, strict=False).alias("i8"),
            col("x").cast(DataType.INT64, strict=False).alias("i64"),
            col("x").cast(DataType.UINT64, strict=False).alias("u64"),
            col("x").cast(DataType.FLOAT32).alias("f32"),
            col("x").cast(DataType.BOOL, strict=False).alias("truth"),
        ]
    )
    same(
        query.collect(accelerator=runtime, engine="accel"),
        query.collect(engine="cpu"),
    )
    with assert_raises():
        _ = (
            floating.lazy()
            .select(col("x").cast(DataType.BOOL))
            .collect(accelerator=runtime, engine="accel")
        )
    # A strict cast in an unselected branch must not fail the result.
    var safe = integers.lazy().select(
        when(col("x") == 1)
        .then(col("x").cast(DataType.INT8))
        .otherwise(lit(Int8(0)))
        .alias("v")
    )
    same(
        safe.collect(accelerator=runtime, engine="accel"),
        safe.collect(engine="cpu"),
    )


# requires: mojo_gpu
def test_mojo_gpu_filters_slices_preserve_order_nulls_chunks_and_bits() raises:
    from tests.execution_modes.test_backend_metal_float64_emulation import same

    var ids = List[Int64]()
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        1,
        0x7FF0000000000000,
        0x7FF8000000000042,
        0xFFF0000000000001,
    ]
    var floats = List[Float64]()
    var valid = List[Bool]()
    var keep = List[Bool]()
    var keep_valid = List[Bool]()
    for row in range(1_033):
        ids.append(Int64(row))
        floats.append(bitcast[DType.float64](words[row % len(words)]))
        valid.append(row % 7 != 2)
        keep.append(row % 3 != 1)
        keep_valid.append(row % 11 != 4)
    var input = DataFrame(
        [
            Series("id", Column[Int64](ids^)),
            Series("f", Column[Float64](floats^, valid^)),
            Series("keep", Column[Bool](keep^, keep_valid^)),
        ]
    ).slice(3, 1_027)
    var columns = List[Series]()
    for column in input._columns:
        columns.append(
            Series._from_chunks(
                [
                    column.slice(0, 257),
                    column.slice(257, 0),
                    column.slice(257, 770),
                ]
            )
        )
    var chunked = DataFrame(columns^)
    for frame in [input.copy(), chunked.copy()]:
        var query = (
            frame.lazy()
            .filter(col("keep"))
            .filter(col("id") >= 128)
            .select_exprs(
                [col("id").alias("left"), col("id").alias("right"), col("f")]
            )
        )
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        same(actual, query.collect(engine="cpu"))
        var expected_row = 128
        for row in range(actual.height()):
            while expected_row % 3 == 1 or expected_row % 11 == 4:
                expected_row += 1
            assert_equal(actual.item(row, "left").int64(), Int64(expected_row))
            assert_equal(actual.item(row, "right").int64(), Int64(expected_row))
            assert_equal(actual.item(row, "f").is_null(), expected_row % 7 == 2)
            if expected_row % 7 != 2:
                assert_equal(
                    bitcast[DType.uint64](
                        actual.column("f").float64()._get(row)
                    ),
                    words[expected_row % len(words)],
                )
            expected_row += 1
        for offset in [-2_000, -257, -1, 0, 1, 255, 2_000]:
            for length in [0, 1, 193, Int.MAX]:
                var sliced = (
                    frame.lazy()
                    .filter(col("keep"))
                    .slice(offset, length)
                    .with_columns((col("f") + 0.5).alias("value"))
                )
                same(
                    sliced.collect(accelerator=MetalRuntime(), engine="accel"),
                    sliced.collect(engine="cpu"),
                )
        var head = query.head(17)
        same(
            head.collect(accelerator=MetalRuntime(), engine="accel"),
            head.collect(engine="cpu"),
        )
        for selected in [False, True]:
            var constant = frame.lazy().filter(lit(selected))
            same(
                constant.collect(accelerator=MetalRuntime(), engine="accel"),
                constant.collect(engine="cpu"),
            )
        var empty = frame.head(0).lazy().filter(col("keep"))
        same(
            empty.collect(accelerator=MetalRuntime(), engine="accel"),
            empty.collect(engine="cpu"),
        )
        var no_columns = (
            frame.lazy().filter(col("keep")).drop(["id", "f", "keep"])
        )
        var no_values = no_columns.collect(
            accelerator=MetalRuntime(), engine="accel"
        )
        assert_equal(no_values.width(), 0)
        assert_equal(
            no_values.height(),
            frame.lazy().filter(col("keep")).collect(engine="cpu").height(),
        )


# requires: mojo_gpu
def test_mojo_gpu_compaction_observes_checked_work_on_surviving_rows() raises:
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([Int64.MAX, 3])),
            Series("keep", Column[Bool]([False, True])),
        ]
    )
    var safe = (
        frame.lazy().filter(col("keep")).select((col("x") + 1).alias("v"))
    )
    var result = safe.collect(accelerator=MetalRuntime(), engine="accel")
    assert_equal(result.height(), 1)
    assert_equal(result.item(0, "v").int64(), Int64(4))
    with assert_raises():
        _ = (
            frame.lazy()
            .filter(~col("keep"))
            .select(col("x") + 1)
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
    _ = safe.collect(accelerator=MetalRuntime(), engine="accel")
    var profile = safe.profile(accelerator=MetalRuntime(), engine="accel")
    assert_true(profile[1].item(0, "kernel_launches").int64() >= 5)
    var peak = Int(profile[1].item(0, "peak_requested_bytes").int64())
    with assert_raises(contains="budget"):
        _ = safe.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    _ = safe.collect(
        accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
    )
