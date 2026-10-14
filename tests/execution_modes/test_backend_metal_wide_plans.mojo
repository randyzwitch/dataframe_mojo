"""Metal shape extensions retain strict preflight and exact row semantics."""
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from tests.execution_modes.test_backend_metal import require_metal
from std.testing import assert_true, assert_raises
from std.memory import bitcast
from dataframe.dtype import DataType
from dataframe import DataFrame, Series, Column, col, lit, Expr
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.joins import lower_join_dag
from tests.execution_modes.test_backend_metal_float64_emulation import same


def wide_frame(
    columns: Int, rows: Int = 19, prefix: String = "c"
) raises -> DataFrame:
    var fields = List[Series]()
    for c in range(columns):
        var values = List[Int64]()
        var valid = List[Bool]()
        for r in range(rows):
            values.append(Int64.MAX - Int64(c * 64 + r))
            valid.append((c + r) % 7 != 3)
        fields.append(
            Series(prefix + String(c), Column[Int64](values^, valid^))
        )
    return DataFrame(fields^)


def numeric_frame() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "g",
                Column[Int32](
                    [2, 1, 2, 1, 3, 3, 2],
                    [True, True, True, False, True, False, True],
                ),
            ),
            Series(
                "x",
                Column[Float64](
                    [-0.0, 1.5, -2.25, 4.0, 0.0, 8.5, -3.0],
                    [True, True, True, False, True, True, True],
                ),
            ),
        ]
    )


def check(query: LazyFrame) raises:
    same(query.collect(accelerator=MetalRuntime()), query.collect(engine="cpu"))


def negations(nodes: Int) -> Expr:
    var expression = col("x")
    for _ in range(nodes - 1):
        expression = -expression
    return expression.alias("long")


def test_wide_metal_metadata_and_default_backend_limits() raises:
    var defaults = RowCapabilities()
    var metal = _capabilities()
    assert_true(defaults.max_nodes == 64 and defaults.max_slots == 64)
    assert_true(defaults.max_steps == 64 and defaults.max_join_columns == 64)
    assert_true(metal.max_nodes == 128 and metal.max_slots == 256)
    assert_true(metal.max_steps == 128 and metal.max_join_columns == 256)
    var frame = wide_frame(256, 1)
    var plan = lower_rows(frame.lazy(), metal)
    assert_true(plan.source.width() == 256 and len(plan.outputs) == 256)
    assert_true(plan.slots == 256)
    with assert_raises(contains="64 source columns"):
        _ = lower_rows(frame.lazy(), defaults)
    with assert_raises(contains="256 source columns"):
        _ = lower_rows(wide_frame(257, 1).lazy(), metal)
    assert_true(
        lower_rows(
            wide_frame(257, 1).lazy().select(col("c256")), metal
        ).source.width()
        == 1
    )
    var numeric = numeric_frame()
    for size in List[Int]([64, 65, 127, 128]):
        assert_true(
            lower_rows(numeric.lazy().select(negations(size)), metal)
            .steps[0]
            .nodes
            == size
        )
    with assert_raises(contains="64 row-local nodes"):
        _ = lower_rows(
            numeric.select(["x"]).lazy().select(negations(65)), defaults
        )
    with assert_raises(contains="128 row-local nodes"):
        _ = lower_rows(numeric.lazy().select(negations(129)), metal)
    var sorted = numeric.select(["x"]).lazy()
    for _ in range(128):
        sorted = sorted.sort("x")
    assert_true(len(lower_rows(sorted, metal).steps) == 128)
    with assert_raises(contains="64 plan steps"):
        _ = lower_rows(sorted, RowCapabilities(order=True))
    with assert_raises(contains="128 plan steps"):
        _ = lower_rows(sorted.sort("x"), metal)
    var ordinary_join = (
        wide_frame(33, 1, "l")
        .lazy()
        .join(wide_frame(32, 1, "r").lazy(), how="cross")
    )
    with assert_raises(contains="64 combined input columns"):
        _ = lower_join_dag(ordinary_join, defaults)
    var wide_join = (
        wide_frame(129, 1, "l")
        .lazy()
        .join(wide_frame(127, 1, "r").lazy(), how="cross")
    )
    assert_true(lower_join_dag(wide_join, metal).joins == 1)
    var too_wide = (
        wide_frame(129, 1, "l")
        .lazy()
        .join(wide_frame(128, 1, "r").lazy(), how="cross")
    )
    with assert_raises(contains="256 combined input columns"):
        _ = lower_join_dag(too_wide, metal)
    # One computed value beyond a full source reaches the slot bound before
    # the independent expression/step limits; all source outputs remain live.
    with assert_raises(contains="256 value slots"):
        _ = lower_rows(
            wide_frame(256, 1)
            .lazy()
            .with_columns(col("c0").cast(DataType.FLOAT64).alias("computed")),
            metal,
        )


def test_native_metal_wide_metal_transport_bitmaps_slices_and_projection() raises:
    _ = require_metal()
    for width in List[Int]([64, 65, 127, 128, 129, 255, 256]):
        var frame = wide_frame(width)
        check(frame.lazy())
        check(frame.slice(3, 13).lazy())
        check(frame.slice(0, 0).lazy())
    check(wide_frame(257).lazy().select(col("c256")))
    with assert_raises(contains="budget"):
        _ = (
            wide_frame(256)
            .lazy()
            .collect(accelerator=MetalRuntime(memory_limit_bytes=0))
        )
    with assert_raises(contains="256 source columns"):
        _ = wide_frame(257).lazy().collect(accelerator=MetalRuntime())


def test_native_metal_wide_metal_programs_steps_and_resident_consumers() raises:
    _ = require_metal()
    var frame = numeric_frame()
    for size in List[Int]([65, 127, 128]):
        check(frame.lazy().select(negations(size)))
    var projections = List[Expr]()
    for i in range(70):
        projections.append(col("x").alias("v" + String(i)))
    check(frame.lazy().select_exprs(projections))
    check(
        frame.lazy()
        .select_exprs(projections)
        .filter(col("v69") < 2)
        .select(col("v0"))
    )
    var filtered = frame.lazy()
    for _ in range(66):
        filtered = filtered.filter(col("x") > lit(Float64(-100)))
    check(filtered.select(col("x")))
    var sorted = frame.lazy()
    for i in range(65):
        sorted = sorted.sort("x", descending=i % 2 == 1)
    check(sorted)
    check(sorted.head(3))


def test_native_metal_wide_metal_global_and_grouped_reductions() raises:
    _ = require_metal()
    var frame = numeric_frame()
    var globals = List[Expr]()
    for i in range(70):
        globals.append(
            (col("x").sum() if i % 2 == 0 else col("x").count()).alias(
                "a" + String(i)
            )
        )
    check(frame.lazy().select_exprs(globals))
    check(frame.slice(0, 0).lazy().select_exprs(globals))
    var aggregates = List[Expr]()
    for i in range(32):
        aggregates.append(
            (
                col("x").sum(min_count=1) if i % 2 == 0 else col("x").count()
            ).alias("a" + String(i))
        )
    var grouped = (
        frame.lazy().group_by("g", maintain_order=True).agg(aggregates)
    )
    assert_true(lower_rows(grouped, _capabilities()).slots > 64)
    check(grouped)
    check(grouped.sort("g").select_exprs([col("g"), col("a31"), col("a30")]))
    check(frame.slice(0, 0).lazy().group_by("g").agg(aggregates))


def test_native_metal_wide_metal_join_fields_keys_empty_and_budget() raises:
    _ = require_metal()
    var left = wide_frame(129, 6, "l").slice(1, 4)
    var right = wide_frame(127, 5, "r").slice(1, 3)
    check(left.lazy().join(right.lazy(), how="cross"))
    check(left.lazy().join(right.slice(0, 0).lazy(), how="cross"))
    for how in List[String](["inner", "left", "right", "full"]):
        check(
            left.lazy().join(
                right.lazy(), left_on=["l0"], right_on=["r0"], how=how
            )
        )
    with assert_raises(contains="budget"):
        _ = (
            left.lazy()
            .join(right.lazy(), how="cross")
            .collect(accelerator=MetalRuntime(memory_limit_bytes=0))
        )
    var measured = (
        left.lazy()
        .join(right.lazy(), how="cross")
        .profile(accelerator=MetalRuntime())
    )
    same(
        measured[0],
        left.lazy().join(right.lazy(), how="cross").collect(engine="cpu"),
    )
    assert_true(measured[0].width() == 256 and measured[1].height() > 0)
    var last = measured[1].height() - 1
    assert_true(measured[1].column("executor").string()._get(last) == "metal")
    assert_true(measured[1].column("download_bytes").int64()._get(last) > 0)


def mixed_wide_frame(columns: Int, prefix: String = "m") raises -> DataFrame:
    var fields = List[Series]()
    for c in range(columns):
        var valid = List[Bool]()
        for r in range(19):
            valid.append((c + r) % 7 != 3)
        var name = prefix + String(c)
        if c % 4 == 0:
            var values = List[Int128]()
            for r in range(19):
                var magnitude = (Int128(1) << 100) + Int128(c * 31 + r)
                values.append(-magnitude if r % 2 else magnitude)
            fields.append(
                Series(name, Column[Int128](values^, valid^)).with_dtype(
                    DataType.decimal(38, 9)
                )
            )
        elif c % 4 == 1:
            var values = List[Float64]()
            for r in range(19):
                var bits = UInt64(1) if r % 3 == 0 else (
                    UInt64(0x8000000000000000) if r % 3
                    == 1 else UInt64(0x3FF0000000000001)
                )
                values.append(bitcast[DType.float64](bits))
            fields.append(Series(name, Column[Float64](values^, valid^)))
        elif c % 4 == 2:
            var values = List[Bool]()
            for r in range(19):
                values.append((c + r) % 2 == 0)
            fields.append(Series(name, Column[Bool](values^, valid^)))
        else:
            var values = List[Int64]()
            for r in range(19):
                values.append(Int64.MAX - Int64(c * 31 + r))
            fields.append(Series(name, Column[Int64](values^, valid^)))
    return DataFrame(fields^)


def test_native_metal_256_column_mixed_payloads_physical_words_and_resident_boundaries() raises:
    _ = require_metal()
    var frame = mixed_wide_frame(256)
    check(frame.lazy())
    check(frame.slice(3, 13).lazy())
    check(frame.slice(0, 0).lazy())
    check(frame.lazy().sort("m0").head(7))
    # 255 live payloads plus the filter predicate exactly fill 256 slots.
    check(wide_frame(255).lazy().filter(col("c0") > lit(Int64.MAX - 8)))
    var left = mixed_wide_frame(129, "l").slice(1, 4)
    var right = mixed_wide_frame(127, "r").slice(2, 3)
    for how in List[String](["inner", "left", "right", "full"]):
        check(
            left.lazy().join(
                right.lazy(), left_on=["l0"], right_on=["r0"], how=how
            )
        )
    check(left.lazy().join(right.lazy(), how="cross").head(5))
