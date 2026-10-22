"""Resident custom-label QCUT controls, exact edges and ordered failures."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe_metal.metal import MetalRuntime
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


# requires: metal
def test_mojo_gpu_qcut_custom_labels_layouts_closure_and_consumers() raises:
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000042))
    var raw = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [0, 1, 2, 3, 4, nan, 99],
                    [True, True, True, True, True, True, False],
                ),
            ),
            Series("id", Column[Int64]([0, 1, 2, 3, 4, 5, 6])),
        ]
    )
    var runtime = MetalRuntime()
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        for closed in [False, True]:
            var query = frame.lazy().select(
                col("x")
                .qcut([0.5], ["é🙂", "upper"], left_closed=closed)
                .alias("bin")
            )
            var actual = query.collect(accelerator=runtime)
            assert_frame(
                actual,
                query.collect(engine="cpu", optimize=False, streaming=False),
            )
            for row in range(7):
                if row >= 5:
                    assert_true(actual.item(row, "bin").is_null())
                else:
                    assert_equal(
                        actual.item(row, "bin").string(),
                        "é🙂" if row < (2 if closed else 3) else "upper",
                    )
            query = (
                frame.lazy()
                .with_columns(
                    col("x")
                    .qcut([0.5], ["é🙂", "upper"], left_closed=closed)
                    .alias("bin")
                )
                .filter(col("bin") == "upper")
                .select(["id", "bin"])
            )
            actual = query.collect(accelerator=runtime)
            assert_frame(
                actual,
                query.collect(engine="cpu", optimize=False, streaming=False),
            )
            assert_equal(actual.height(), 3 if closed else 2)
            assert_equal(
                actual.item(0, "id").int64(), Int64(2 if closed else 3)
            )
    for values in [raw.clear(), raw.head(1)]:
        var query = values.lazy().select(
            col("x").qcut(List[Float64](), ["only"])
        )
        var actual = query.collect(accelerator=runtime)
        assert_frame(
            actual, query.collect(engine="cpu", optimize=False, streaming=False)
        )
        if values.height():
            assert_equal(actual.item().string(), "only")


# requires: metal
def test_mojo_gpu_qcut_duplicate_labels_null_nan_and_control_order() raises:
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000042))
    var source = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [0, 0, 0, 1, nan, 99], [True, True, True, True, True, False]
                ),
            )
        ]
    )
    var query = source.lazy().select(
        col("x")
        .qcut(
            [0, 0.25, 0.5],
            ["low", "skip1", "skip2", "high"],
            allow_duplicates=True,
        )
        .alias("bin")
    )
    var runtime = MetalRuntime()
    var actual = query.collect(accelerator=runtime)
    assert_frame(
        actual, query.collect(engine="cpu", optimize=False, streaming=False)
    )
    for row in range(3):
        assert_equal(actual.item(row, "bin").string(), "low")
    assert_equal(actual.item(3, "bin").string(), "high")
    assert_true(actual.item(4, "bin").is_null())
    assert_true(actual.item(5, "bin").is_null())
    for duplicates in [False, True]:
        query = source.lazy().select(
            col("x").qcut(
                [0.5, 0.5, 2], ["a", "b", "c", "d"], allow_duplicates=duplicates
            )
        )
        var message = (
            "between 0 and 1" if duplicates else "breaks must be unique"
        )
        with assert_raises(contains=message):
            _ = query.collect(engine="cpu", optimize=False, streaming=False)
        with assert_raises(contains=message):
            _ = query.collect(accelerator=runtime)
    for empty in [False, True]:
        var frame = source.clear() if empty else source.copy()
        query = frame.lazy().select(col("x").qcut([-1], ["a", "b"]))
        with assert_raises(contains="between 0 and 1"):
            _ = query.collect(accelerator=runtime)
    var missing = DataFrame(
        [Series("x", Column[Float64]([nan, 99], [True, False]))]
    )
    query = missing.lazy().select(
        col("x").qcut([0.25, 0.75], ["a", "b", "c"], include_breaks=True)
    )
    actual = query.collect(accelerator=runtime)
    assert_frame(
        actual, query.collect(engine="cpu", optimize=False, streaming=False)
    )
    for row in range(2):
        assert_true(actual.item(row, "x").struct_field("category").is_null())
        assert_true(actual.item(row, "x").struct_field("breakpoint").is_null())


# requires: metal
def test_mojo_gpu_qcut_exact_breakpoints_association_and_zero_permutation() raises:
    # These are independent rational IEEE anchors, not tolerance comparisons.
    var cases: List[UInt64] = [
        0xC024000000000000,
        0,
        0x3FE6666666666666,
        0xC008000000000000,
        0xC024000000000000,
        0x4008000000000000,
        0x3FD3333333333333,
        0xC018666666666666,
        0xC341C37937E08000,
        0x3FF0000000000000,
        0x3FECCCCCCCCCCCCD,
        0xC30C6BF526340000,
        0xC04FCC0D54832E81,
        0xC03BE2DBD1AF6896,
        0x3FE6666666666666,
        0xC0434C9DBC57F28E,
    ]
    var runtime = MetalRuntime()
    for at in range(4):
        var values: List[Float64] = [
            bitcast[DType.float64](cases[4 * at]),
            bitcast[DType.float64](cases[4 * at + 1]),
        ]
        var q: List[Float64] = [bitcast[DType.float64](cases[4 * at + 2])]
        var source = DataFrame([Series("x", Column[Float64](values^))])
        var query = source.lazy().select(
            col("x")
            .qcut(q, ["lower", "upper"], include_breaks=True)
            .alias("bin")
        )
        var actual = query.collect(accelerator=runtime)
        assert_frame(
            actual, query.collect(engine="cpu", optimize=False, streaming=False)
        )
        assert_equal(
            bitcast[DType.uint64](
                actual.item(0, "bin").struct_field("breakpoint").float64()
            ),
            cases[4 * at + 3],
        )
    var negative = bitcast[DType.float64](UInt64(0x8000000000000000))
    var source = DataFrame(
        [Series("x", Column[Float64]([negative, negative, negative]))]
    )
    var query = source.lazy().select(
        col("x").qcut([0.5], ["low", "high"], include_breaks=True).alias("bin")
    )
    var actual = query.collect(accelerator=runtime)
    assert_frame(
        actual, query.collect(engine="cpu", optimize=False, streaming=False)
    )
    for row in range(3):
        assert_equal(
            bitcast[DType.uint64](
                actual.item(row, "bin").struct_field("breakpoint").float64()
            ),
            UInt64(0x8000000000000000),
        )
    var values = List[Float64](length=27, fill=0)
    values[0] = 1
    values[1] = -1
    values[26] = negative
    source = DataFrame([Series("x", Column[Float64](values^))])
    var q: List[Float64] = [bitcast[DType.float64](UInt64(0x3FA3B13B13B13B14))]
    query = source.lazy().select(
        col("x").qcut(q, ["low", "high"], include_breaks=True).alias("bin")
    )
    actual = query.collect(accelerator=runtime)
    assert_equal(
        bitcast[DType.uint64](
            actual.item(1, "bin").struct_field("breakpoint").float64()
        ),
        UInt64(0x8000000000000000),
    )


# requires: metal
def test_mojo_gpu_qcut_eager_producers_scalar_siblings_budget_and_recovery() raises:
    var bad = DataFrame([Series("x", Column[Int64]([Int64.MAX, 1]))])
    var safe = DataFrame([Series("x", Column[Int64]([0, 1]))])
    var runtime = MetalRuntime()
    var hidden = (
        when(lit(False))
        .then((col("x") + 1).qcut([-1], ["a", "b"]))
        .otherwise(lit("safe"))
    )
    for head in [False, True]:
        var query = bad.lazy().select(hidden)
        if head:
            query = query.head(0)
        with assert_raises(contains="overflow"):
            _ = query.collect(engine="cpu", optimize=False, streaming=False)
        with assert_raises(contains="overflow"):
            _ = query.collect(accelerator=runtime)
    with assert_raises(contains="between 0 and 1"):
        _ = safe.lazy().select(hidden).collect(accelerator=runtime)
    var scalar = lit(Int64(128)).cast(DataType.INT8)
    var query = safe.lazy().select_exprs(
        [scalar.alias("s"), col("x").qcut([-1], ["a", "b"]).alias("bin")]
    )
    with assert_raises(contains="strict cast failed"):
        _ = query.collect(accelerator=runtime)
    query = bad.lazy().select_exprs(
        [
            col("x").qcut([-1], ["a", "b"]).alias("bin"),
            col("x").sum().alias("sum"),
        ]
    )
    with assert_raises(contains="between 0 and 1"):
        _ = query.collect(engine="cpu", optimize=False, streaming=False)
    with assert_raises(contains="between 0 and 1"):
        _ = query.collect(accelerator=runtime)
    query = safe.lazy().select(
        col("x").qcut([0.5], ["low", "high"]).alias("bin")
    )
    var measured = query.profile(accelerator=runtime)
    var report = measured[1].copy()
    var saw_controls = False
    for row in range(report.height()):
        if report.item(row, "algorithm").string() == "mojo_gpu_qcut_controls":
            saw_controls = True
            assert_equal(
                report.item(row, "host_result_bytes").int64(), Int64(0)
            )
            # 8-byte dynamic edge control plus three public resident error
            # columns (source x, observed operand, category), no value export.
            assert_equal(
                report.item(row, "download_bytes").int64(), Int64(8 + 4 * 2 * 3)
            )
    assert_true(saw_controls)
    var peak = Int(report.item(0, "query_peak_requested_bytes").int64())
    assert_frame(
        query.collect(accelerator=MetalRuntime(memory_limit_bytes=peak)),
        measured[0],
    )
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=peak - 1))
    assert_frame(query.collect(accelerator=runtime), measured[0])
    var default_query = safe.lazy().select(col("x").qcut([0.5]))
    assert_frame(
        default_query.collect(accelerator=runtime),
        default_query.collect(engine="cpu", optimize=False, streaming=False),
    )
    with assert_raises(contains="intermediate Struct"):
        _ = (
            safe.lazy()
            .select(
                col("x")
                .qcut([0.5], ["a", "b"], include_breaks=True)
                .field("category")
            )
            .collect(accelerator=runtime)
        )
