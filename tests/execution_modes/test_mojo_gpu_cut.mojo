"""Explicit CUT executes resident checked operands before label consumers."""
# contract-group: expr.cut
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.math import isnan
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe_metal.metal import MetalRuntime
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source() raises -> DataFrame:
    var inf = Float64(1) / Float64(0)
    var nan = Float64(0) / Float64(0)
    var column = Series(
        "x",
        Column[Float64](
            [-inf, -1, 0, 1, 2, inf, nan, 99],
            [True, True, True, True, True, True, True, False],
        ),
    )
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    column._field_metadata = ArcPointer(metadata^)
    return DataFrame([column^])


def _check_ordered_scalar_faults(
    runtime: MetalRuntime, frame: DataFrame
) raises:
    var bad_cast = lit(Int64(128)).cast(DataType.INT8).cast(DataType.INT64)
    var bad_power = lit(Int64(2)).pow(lit(Int64(-1)))
    for empty in [False, True]:
        var source = frame.clear() if empty else frame.copy()
        for cast_first in [False, True]:
            var scalar = (bad_cast + bad_power) if cast_first else (
                bad_power + bad_cast
            )
            var query = source.lazy().select_exprs(
                [
                    scalar.alias("checked"),
                    col("x").cut([0], ["lower", "upper"]).alias("bin"),
                ]
            )
            with assert_raises(
                contains="cast from int64 to int8 failed" if cast_first else "nonnegative exponent"
            ):
                _ = query.collect(engine="cpu", optimize=False, streaming=False)
            with assert_raises(
                contains="strict cast failed" if cast_first else "nonnegative exponent"
            ):
                _ = query.collect(accelerator=runtime)
        for selected in [False, True]:
            var scalar = when(lit(selected)).then(bad_cast).otherwise(bad_power)
            var query = source.lazy().select_exprs(
                [
                    scalar.alias("checked"),
                    col("x").cut([0], ["lower", "upper"]).alias("bin"),
                ]
            )
            with assert_raises(
                contains="cast from int64 to int8 failed" if selected else "nonnegative exponent"
            ):
                _ = query.collect(engine="cpu", optimize=False, streaming=False)
            with assert_raises(
                contains="strict cast failed" if selected else "nonnegative exponent"
            ):
                _ = query.collect(accelerator=runtime)
        var masked = (
            when(lit(False)).then(bad_cast + bad_power).otherwise(lit(Int64(7)))
        )
        var query = source.lazy().select_exprs(
            [
                masked.alias("checked"),
                col("x").cut([0], ["lower", "upper"]).alias("bin"),
            ]
        )
        var actual = query.collect(accelerator=runtime)
        assert_equal(actual.height(), 0 if empty else frame.height())
        for row in range(actual.height()):
            assert_equal(actual.item(row, "checked").int64(), Int64(7))
        assert_frame(
            actual, query.collect(engine="cpu", optimize=False, streaming=False)
        )
    # Plain literal String/conditional siblings only recode nodes in place;
    # they keep the ranked AST shape while restoring ordinary String values.
    var labels = when(lit(False)).then(lit("discard")).otherwise(lit("é🙂"))
    var string_query = frame.lazy().select_exprs(
        [
            labels.alias("label"),
            col("x").cut([0], ["lower", "upper"]).alias("bin"),
        ]
    )
    var string_result = string_query.collect(accelerator=runtime)
    for row in range(frame.height()):
        assert_equal(string_result.item(row, "label").string(), "é🙂")
    assert_frame(
        string_result,
        string_query.collect(engine="cpu", optimize=False, streaming=False),
    )
    # Eager reduction preparation precedes even an earlier checked scalar
    # operand, including when the entire value branch will stay unselected.
    var overflow = DataFrame([Series("x", Column[Int64]([Int64.MAX, 1]))])
    for hidden in [False, True]:
        var scalar = bad_cast + col("x").sum() + bad_power
        if hidden:
            scalar = when(lit(False)).then(scalar).otherwise(lit(Int64(0)))
        var query = overflow.lazy().select_exprs(
            [
                scalar.alias("checked"),
                col("x").cut([0], ["lower", "upper"]).alias("bin"),
            ]
        )
        with assert_raises(contains="overflow"):
            _ = query.collect(engine="cpu", optimize=False, streaming=False)
        with assert_raises(contains="overflow"):
            _ = query.collect(accelerator=runtime)


# requires: metal
def test_mojo_gpu_cut_literals_default_labels_nullable_layouts_and_struct_breakpoints() raises:
    var raw = _source()
    var inf = Float64(1) / Float64(0)
    var right: List[String] = ["low", "low", "low", "mid", "high", "high"]
    var left: List[String] = ["low", "low", "mid", "high", "high", "high"]
    var default_right: List[String] = [
        "(-inf, 0]",
        "(-inf, 0]",
        "(-inf, 0]",
        "(0, 1]",
        "(1, inf]",
        "(1, inf]",
    ]
    var default_left: List[String] = [
        "[-inf, 0)",
        "[-inf, 0)",
        "[0, 1)",
        "[1, inf)",
        "[1, inf)",
        "[1, inf)",
    ]
    var runtime = MetalRuntime()
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        for closed in [False, True]:
            var query = frame.lazy().select_exprs(
                [
                    col("x")
                    .cut([0, 1], ["low", "mid", "high"], closed)
                    .alias("named"),
                    col("x").cut([0, 1], left_closed=closed).alias("default"),
                ]
            )
            var actual = query.collect(accelerator=runtime)
            assert_equal(actual.height(), 8)
            assert_equal(actual.column("named").dtype(), DataType.STRING)
            assert_false(Bool(actual.column("named")._field_metadata))
            for row in range(8):
                assert_equal(actual.item(row, "named").is_null(), row >= 6)
                assert_equal(actual.item(row, "default").is_null(), row >= 6)
                if row < 6:
                    assert_equal(
                        actual.item(row, "named").string(),
                        left[row] if closed else right[row],
                    )
                    assert_equal(
                        actual.item(row, "default").string(),
                        default_left[row] if closed else default_right[row],
                    )
            assert_frame(
                actual,
                query.collect(engine="cpu", optimize=False, streaming=False),
            )
            var packed_query = frame.lazy().select(
                col("x")
                .cut([0, 1], ["low", "mid", "high"], closed, True)
                .alias("bin")
            )
            actual = packed_query.collect(accelerator=runtime)
            var structure = actual.column("bin").struct_column()
            assert_false(Bool(actual.column("bin")._field_metadata))
            assert_false(Bool(structure.field("category")._field_metadata))
            assert_false(Bool(structure.field("breakpoint")._field_metadata))
            for row in range(8):
                assert_true(structure._valid(row))
                assert_equal(
                    actual.item(row, "bin").struct_field("category").is_null(),
                    row >= 6,
                )
                assert_equal(
                    actual.item(row, "bin")
                    .struct_field("breakpoint")
                    .is_null(),
                    row >= 6,
                )
                if row < 6:
                    assert_equal(
                        actual.item(row, "bin")
                        .struct_field("category")
                        .string(),
                        left[row] if closed else right[row],
                    )
                    var expected = inf
                    if not closed and row <= 2 or closed and row < 2:
                        expected = 0
                    elif not closed and row == 3 or closed and row == 2:
                        expected = 1
                    assert_equal(
                        actual.item(row, "bin")
                        .struct_field("breakpoint")
                        .float64(),
                        expected,
                    )
            assert_frame(
                actual,
                packed_query.collect(
                    engine="cpu", optimize=False, streaming=False
                ),
            )
            var empty = (
                frame.clear()
                .lazy()
                .select(
                    col("x").cut(
                        [0, 1], left_closed=closed, include_breaks=True
                    )
                )
                .collect(accelerator=runtime)
            )
            assert_equal(empty.height(), 0)
            assert_equal(
                empty.column("x").dtype(),
                DataType.struct(
                    ["breakpoint", "category"],
                    [DataType.FLOAT64, DataType.STRING],
                ),
            )


# requires: metal
def test_mojo_gpu_cut_single_nan_edge_empty_edges_filter_and_computed_operands() raises:
    var nan = Float64(0) / Float64(0)
    var source = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [-1, 0, 1, 2, nan, 99],
                    [True, True, True, True, True, False],
                ),
            ),
            Series("id", Column[Int64]([0, 1, 2, 3, 4, 5])),
        ]
    )
    var runtime = MetalRuntime()
    var result = (
        source.lazy()
        .select(col("x").cut([nan], ["é🙂", "last"], include_breaks=True))
        .collect(accelerator=runtime)
    )
    for row in range(4):
        assert_equal(
            result.item(row, "x").struct_field("category").string(), "é🙂"
        )
        assert_true(
            isnan(result.item(row, "x").struct_field("breakpoint").float64())
        )
    for row in range(4, 6):
        assert_true(result.item(row, "x").struct_field("category").is_null())
    result = (
        source.lazy()
        .select(col("x").cut(List[Float64]()))
        .collect(accelerator=runtime)
    )
    for row in range(4):
        assert_equal(result.item(row, "x").string(), "(-inf, inf]")
    var derived = (
        source.lazy()
        .with_columns(col("x").cut([0, 1], ["low", "mid", "high"]).alias("bin"))
        .filter(col("bin") == "mid")
        .select(["id", "bin"])
    )
    result = derived.collect(accelerator=runtime)
    assert_equal(result.height(), 1)
    assert_equal(result.item(0, "id").int64(), Int64(2))
    assert_equal(result.item(0, "bin").string(), "mid")
    assert_frame(
        result, derived.collect(engine="cpu", optimize=False, streaming=False)
    )
    var direct = (
        source.lazy()
        .filter(col("x").cut([0, 1], ["low", "mid", "high"]) == "mid")
        .select(["id"])
    )
    assert_equal(direct.collect(accelerator=runtime).item().int64(), Int64(2))
    var computed = (
        DataFrame([Series("x", Column[Int64]([-2, -1, 0, 1]))])
        .lazy()
        .select((col("x") + 1).cut([0, 1], ["a", "b", "c"]))
    )
    result = computed.collect(accelerator=runtime)
    var expected: List[String] = ["a", "a", "b", "c"]
    for row in range(4):
        assert_equal(result.item(row, "x").string(), expected[row])

    comptime kinds = (DType.int32, DType.int64, DType.int128)
    comptime widths = (32, 64, 128)
    comptime for index in range(3):
        comptime D = kinds[index]
        var values: List[Scalar[D]] = [-20, -10, 0, 10, 20]
        var decimal = Series("x", Column[Scalar[D]](values^)).with_dtype(
            DataType.decimal(8, 1, widths[index])
        )
        var result = (
            DataFrame([decimal^])
            .lazy()
            .select(col("x").cut([0, 1], ["a", "b", "c"]))
            .collect(accelerator=runtime)
        )
        var labels: List[String] = ["a", "a", "a", "b", "c"]
        for row in range(5):
            assert_equal(result.item(row, "x").string(), labels[row])


# requires: metal
def test_mojo_gpu_cut_eager_operand_and_deferred_control_error_order_recovery() raises:
    var source = DataFrame([Series("x", Column[Int64]([Int64.MAX]))])
    var safe = DataFrame([Series("x", Column[Int64]([0]))])
    var runtime = MetalRuntime()
    var hidden = (
        when(lit(False)).then((col("x") + 1).cut([1, 1])).otherwise(lit("safe"))
    )
    for head in [False, True]:
        var query = source.lazy().select(hidden)
        if head:
            query = query.head(0)
        with assert_raises(contains="overflow"):
            _ = query.collect(engine="cpu", optimize=False, streaming=False)
        with assert_raises(contains="overflow"):
            _ = query.collect(accelerator=runtime)
    with assert_raises(contains="breaks must be unique"):
        _ = safe.lazy().select(hidden).collect(accelerator=runtime)
    with assert_raises(contains="labels must have one more"):
        _ = (
            source.lazy()
            .select((col("x") + 1).cut([1, 1], ["wrong"]))
            .collect(accelerator=runtime)
        )
    for empty in [False, True]:
        var frame = safe.clear() if empty else safe.copy()
        with assert_raises(contains="breaks must be unique"):
            _ = (
                frame.lazy()
                .select(col("x").cut([1, 1]))
                .collect(accelerator=runtime)
            )
    # CPU evaluates earlier sibling expressions before a later CUT state.
    var siblings = source.lazy().select_exprs(
        [(col("x") + 1).alias("earlier"), col("x").cut([1, 1]).alias("later")]
    )
    with assert_raises(contains="overflow"):
        _ = siblings.collect(engine="cpu", optimize=False, streaming=False)
    with assert_raises(contains="overflow"):
        _ = siblings.collect(accelerator=runtime)
    assert_equal(
        safe.lazy()
        .select(col("x").cut([1], ["ok", "other"]))
        .collect(accelerator=runtime)
        .item()
        .string(),
        "ok",
    )


# requires: metal
def test_mojo_gpu_cut_budget_profiles_and_pending_families_are_explicit() raises:
    var source = _source()
    var query = source.lazy().select(
        col("x").cut([0, 1], ["low", "mid", "high"])
    )
    var measured = query.profile(accelerator=MetalRuntime())
    var report = measured[1].copy()
    assert_true(report.height() >= 3)
    for row in range(report.height() - 1):
        assert_equal(report.item(row, "host_result_bytes").int64(), Int64(0))
    assert_true(report.item(0, "query_kernel_launches").int64() > 0)
    assert_true(report.item(0, "query_upload_bytes").int64() > 0)
    var peak = Int(report.item(0, "query_peak_requested_bytes").int64())
    assert_frame(
        query.collect(accelerator=MetalRuntime(memory_limit_bytes=peak)),
        measured[0],
    )
    var limited = MetalRuntime(memory_limit_bytes=peak - 1)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=limited)
    assert_equal(
        DataFrame([Series("n", Column[Int64]([7]))])
        .lazy()
        .select(["n"])
        .collect(accelerator=limited)
        .item()
        .int64(),
        Int64(7),
    )
    var quantiles = source.lazy().select(col("x").qcut([0.5]))
    assert_frame(
        quantiles.collect(accelerator=MetalRuntime()),
        quantiles.collect(engine="cpu", optimize=False, streaming=False),
    )
    with assert_raises(contains="intermediate Struct"):
        _ = (
            source.lazy()
            .select(col("x").cut([0], include_breaks=True).field("category"))
            .collect(accelerator=MetalRuntime())
        )


# requires: metal
def test_mojo_gpu_cut_scalar_reduction_siblings_observe_expression_order_and_empty_errors() raises:
    var source = DataFrame([Series("x", Column[Int64]([Int64.MAX, 1]))])
    var runtime = MetalRuntime()
    for sum_first in [False, True]:
        for hidden in [False, True]:
            var sum = col("x").sum()
            if hidden:
                sum = when(lit(False)).then(sum).otherwise(lit(Int64(0)))
            var cuts = col("x").cut([1, 1]).alias("bin")
            var fields: List[Expr] = [
                sum.alias("total"),
                cuts.copy(),
            ] if sum_first else [cuts.copy(), sum.alias("total")]
            var query = source.lazy().select_exprs(fields)
            var message = "overflow" if sum_first else "breaks must be unique"
            with assert_raises(contains=message):
                _ = query.collect(engine="cpu", optimize=False, streaming=False)
            with assert_raises(contains=message):
                _ = query.collect(accelerator=runtime)
    var safe = DataFrame([Series("x", Column[Int64]([0, 1]))])
    _check_ordered_scalar_faults(runtime, safe)
    for empty in [False, True]:
        var frame = safe.clear() if empty else safe.copy()
        for cast_first in [False, True]:
            var cast = lit(Int64(128)).cast(DataType.INT8).alias("checked")
            var cut = col("x").cut([1, 1]).alias("bin")
            var fields: List[Expr] = [
                cast.copy(),
                cut.copy(),
            ] if cast_first else [cut.copy(), cast.copy()]
            var query = frame.lazy().select_exprs(fields)
            with assert_raises(
                contains="cast from int64 to int8 failed" if cast_first else "breaks must be unique"
            ):
                _ = query.collect(engine="cpu", optimize=False, streaming=False)
            with assert_raises(
                contains="strict cast failed" if cast_first else "breaks must be unique"
            ):
                _ = query.collect(accelerator=runtime)
    # All siblings bind before the first value expression: a later bad label
    # count must precede an earlier unsafe scalar cast.
    var bind_first = safe.lazy().select_exprs(
        [
            lit(Int64(128)).cast(DataType.INT8).alias("checked"),
            col("x").cut([0, 1], ["wrong"]).alias("bin"),
        ]
    )
    with assert_raises(contains="labels must have one more"):
        _ = bind_first.collect(accelerator=runtime)
    var literals = (
        when(lit(False))
        .then(lit(Int64(128)).cast(DataType.INT8))
        .otherwise(lit(Int8(7)))
        .alias("checked")
    )
    var query = safe.lazy().select_exprs(
        [
            col("x").sum().alias("total"),
            literals.copy(),
            col("x").cut([0], ["low", "high"]).alias("bin"),
        ]
    )
    var measured = query.profile(accelerator=runtime)
    var actual = measured[0].copy()
    assert_equal(actual.height(), 2)
    for row in range(2):
        assert_equal(actual.item(row, "total").int64(), Int64(1))
        assert_equal(actual.item(row, "checked").int8(), Int8(7))
        assert_equal(
            actual.item(row, "bin").string(), "low" if row == 0 else "high"
        )
    assert_frame(
        actual, query.collect(engine="cpu", optimize=False, streaming=False)
    )
    assert_equal(measured[1].item(0, "cut_staging_bytes").int64(), Int64(2))
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    assert_frame(
        query.collect(accelerator=MetalRuntime(memory_limit_bytes=peak)), actual
    )
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=peak - 1))
    var empty = (
        safe.clear()
        .lazy()
        .select_exprs(
            [
                lit(Int64(7)).alias("scalar"),
                col("x").sum().alias("total"),
                col("x").cut([0], ["low", "high"]).alias("bin"),
            ]
        )
        .collect(accelerator=runtime)
    )
    assert_equal(empty.height(), 0)
    assert_equal(empty.columns(), ["scalar", "total", "bin"])


# requires: metal
def test_mojo_gpu_cut_literal_only_window_rejections_and_literal_sibling_broadcast() raises:
    var runtime = MetalRuntime(device_id=999999)
    for rows in [0, 2]:
        var ordinary = DataFrame(
            [Series("x", Column[Int64](List[Int64](length=rows, fill=0)))]
        )
        var no_columns = DataFrame(List[Series](), height=rows)
        for source in [ordinary.copy(), no_columns.copy()]:
            for sibling in [False, True]:
                var expressions: List[Expr] = [
                    lit(Float64(1)).cut([0]).alias("bin")
                ]
                if sibling:
                    expressions.insert(0, lit(Int64(7)).alias("scalar"))
                var query = source.lazy().select_exprs(expressions)
                with assert_raises(
                    contains="Window operations require a row-valued input"
                ):
                    _ = query.collect(
                        engine="cpu", optimize=False, streaming=False
                    )
                with assert_raises(
                    contains="Window operations require a row-valued input"
                ):
                    _ = query.collect(accelerator=runtime)
    var frame = DataFrame([Series("x", Column[Int64]([0, 1]))])
    var query = frame.lazy().select_exprs(
        [
            lit(Int64(7)).alias("scalar"),
            col("x").cut([0], ["low", "high"]).alias("bin"),
        ]
    )
    var actual = query.collect(accelerator=MetalRuntime())
    assert_equal(actual.height(), 2)
    for row in range(2):
        assert_equal(actual.item(row, "scalar").int64(), Int64(7))
        assert_equal(
            actual.item(row, "bin").string(), "low" if row == 0 else "high"
        )
