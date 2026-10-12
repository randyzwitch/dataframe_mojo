"""Metal shape extensions retain strict preflight and exact row semantics."""
from std.testing import TestSuite, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, col, lit, Expr
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime, metal_installed, _capabilities
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.joins import lower_join_dag
from test_metal_float64_emulation import same


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
    assert_true(metal.max_nodes == 128 and metal.max_slots == 128)
    assert_true(metal.max_steps == 128 and metal.max_join_columns == 128)
    var frame = wide_frame(128, 1)
    var plan = lower_rows(frame.lazy(), metal)
    assert_true(plan.source.width() == 128 and len(plan.outputs) == 128)
    assert_true(plan.slots == 128)
    with assert_raises(contains="64 source columns"):
        _ = lower_rows(frame.lazy(), defaults)
    with assert_raises(contains="128 source columns"):
        _ = lower_rows(wide_frame(129, 1).lazy(), metal)
    assert_true(
        lower_rows(
            wide_frame(129, 1).lazy().select(col("c128")), metal
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
        wide_frame(65, 1, "l")
        .lazy()
        .join(wide_frame(63, 1, "r").lazy(), how="cross")
    )
    assert_true(lower_join_dag(wide_join, metal).joins == 1)
    var too_wide = (
        wide_frame(65, 1, "l")
        .lazy()
        .join(wide_frame(64, 1, "r").lazy(), how="cross")
    )
    with assert_raises(contains="128 combined input columns"):
        _ = lower_join_dag(too_wide, metal)
    # Slots remain independently bounded even when each expression is small.
    var expressions = List[Expr]()
    for i in range(128):
        expressions.append((col("x") + lit(Float64(i))).alias("x" + String(i)))
    with assert_raises(contains="128 value slots"):
        _ = lower_rows(numeric.lazy().select_exprs(expressions), metal)


def test_wide_metal_transport_bitmaps_slices_and_projection() raises:
    if not metal_installed():
        return
    for width in List[Int]([64, 65, 127, 128]):
        var frame = wide_frame(width)
        check(frame.lazy())
        check(frame.slice(3, 13).lazy())
        check(frame.slice(0, 0).lazy())
    check(wide_frame(129).lazy().select(col("c128")))
    with assert_raises(contains="budget"):
        _ = (
            wide_frame(128)
            .lazy()
            .collect(accelerator=MetalRuntime(memory_limit_bytes=0))
        )
    with assert_raises(contains="128 source columns"):
        _ = wide_frame(129).lazy().collect(accelerator=MetalRuntime())


def test_wide_metal_programs_steps_and_resident_consumers() raises:
    if not metal_installed():
        return
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


def test_wide_metal_global_and_grouped_reductions() raises:
    if not metal_installed():
        return
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


def test_wide_metal_join_fields_keys_empty_and_budget() raises:
    if not metal_installed():
        return
    var left = wide_frame(65, 6, "l").slice(1, 4)
    var right = wide_frame(63, 5, "r").slice(1, 3)
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
    assert_true(measured[0].width() == 128 and measured[1].height() > 0)
    var last = measured[1].height() - 1
    assert_true(measured[1].column("executor").string()._get(last) == "metal")
    assert_true(measured[1].column("download_bytes").int64()._get(last) > 0)


def main() raises:
    var suite = TestSuite()
    suite.test[test_wide_metal_metadata_and_default_backend_limits]()
    suite.test[test_wide_metal_transport_bitmaps_slices_and_projection]()
    suite.test[test_wide_metal_programs_steps_and_resident_consumers]()
    suite.test[test_wide_metal_global_and_grouped_reductions]()
    suite.test[test_wide_metal_join_fields_keys_empty_and_budget]()
    suite^.run()
