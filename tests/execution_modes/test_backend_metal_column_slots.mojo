"""Column aliases share resident values without changing visible schemas."""
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from tests.execution_modes.test_backend_metal import require_metal
from std.testing import assert_true, assert_raises
from dataframe import DataFrame, Series, Column, col, lit, Expr
from dataframe.lazy import LazyFrame
from dataframe.string_column import StringColumn
from dataframe.metal import MetalRuntime
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.execution_modes.test_backend_metal_wide_plans import (
    wide_frame,
    numeric_frame,
)


def aliases(count: Int) -> List[Expr]:
    var result = List[Expr]()
    for i in range(count):
        result.append(col("x").alias("v" + String(i)))
    return result^


def check(query: LazyFrame) raises:
    same(query.collect(accelerator=MetalRuntime()), query.collect(engine="cpu"))


def test_column_slot_preflight_reuses_values_and_bounds_outputs() raises:
    var source = DataFrame([Series("x", Column[Int32]([3, 1, 2]))])
    var query = source.lazy().select_exprs(aliases(256))
    var plan = lower_rows(query, _capabilities())
    assert_true(plan.source.width() == 1 and plan.slots == 1)
    assert_true(len(plan.steps) == 0 and len(plan.outputs) == 256)
    assert_true(len(plan.code) == 0 and len(plan.literals) == 0)
    assert_true(plan.reused_columns)
    for output in plan.outputs:
        assert_true(output.slot == 0)
    with assert_raises(contains="256 output columns"):
        _ = lower_rows(
            source.lazy().select_exprs(aliases(257)), _capabilities()
        )
    var two = source.lazy().select_exprs(aliases(2))
    var default_plan = lower_rows(two, RowCapabilities())
    assert_true(default_plan.slots == 3 and len(default_plan.steps) == 2)
    assert_true(not default_plan.reused_columns)
    var reused = lower_rows(two, _capabilities())
    assert_true(reused.slots == 1 and len(reused.steps) == 0)
    var identity = List[Expr]()
    for name in wide_frame(256, 1).columns():
        identity.append(col(name))
    var wide = lower_rows(
        wide_frame(256, 1).lazy().select_exprs(identity), _capabilities()
    )
    assert_true(wide.slots == 256 and len(wide.steps) == 0)
    var boundary = (
        source.lazy()
        .select_exprs(aliases(256))
        .filter(col("v0") > 0)
        .sort("v1")
    )
    var gathered = lower_rows(boundary, _capabilities())
    assert_true(len(gathered.steps) == 2)
    assert_true(gathered.steps[0].gather_count == 1)
    assert_true(gathered.steps[1].gather_count == 1)
    assert_true(len(gathered.gathers) == 2)
    assert_true(gathered.gathers[0] == 0 and gathered.gathers[1] == 0)
    # Computations still allocate values and enforce the independent slot cap.
    with assert_raises(contains="256 value slots"):
        _ = lower_rows(
            wide_frame(256, 1)
            .lazy()
            .with_columns((col("c0") + lit(Int64(0))).alias("computed")),
            _capabilities(),
        )


def test_native_metal_column_slot_gpu_aliases_filters_order_slices_and_replacement() raises:
    _ = require_metal()
    var raw = wide_frame(256)
    var identity = List[Expr]()
    for name in raw.columns():
        identity.append(col(name))
    check(raw.lazy().select_exprs(identity))
    check(raw.slice(3, 13).lazy().select_exprs(identity))
    var raw_aliases = List[Expr]()
    for i in range(256):
        raw_aliases.append(col("c0").alias("v" + String(i)))
    check(raw.lazy().select_exprs(raw_aliases))
    var source = numeric_frame().select(["x"])
    var query = source.lazy().select_exprs(aliases(256))
    check(query)
    check(query.head(0))
    check(query.filter(col("v127") < 3))
    check(query.sort("v126", descending=True))
    check(query.slice(1, 4))
    check(query.filter(col("v127") < 3).sort("v1").slice(1, 2))
    check(source.slice(0, 0).lazy().select_exprs(aliases(256)))
    check(source.slice(1, 5).lazy().select_exprs(aliases(256)))
    var changed = (
        source.lazy()
        .with_columns(col("x").alias("copy"))
        .with_columns((col("x") * lit(Float64(2))).alias("x"))
    )
    check(changed.select_exprs([col("copy"), col("x")]))
    check(changed.filter(col("copy") != col("x")).sort("x").select(col("copy")))
    # Simultaneous replacement must use the original input for every alias.
    var pair = DataFrame(
        [
            Series("a", Column[Int64]([Int64.MAX, -3, 7])),
            Series("b", Column[Int64]([1, 2, Int64.MIN])),
        ]
    )
    check(pair.lazy().with_columns([col("b").alias("a"), col("a").alias("b")]))
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=0))


def test_native_metal_column_slot_gpu_resident_joins_reductions_and_dictionaries() raises:
    _ = require_metal()
    var source = numeric_frame().select(["x"])
    var query = source.lazy().select_exprs(aliases(64))
    var right = source.lazy().select_exprs([col("x").alias("right")])
    check(query.join(right, how="cross").head(13))
    check(
        query.select_exprs(
            [col("v0").sum().alias("sum"), col("v63").count().alias("count")]
        )
    )
    check(
        query.group_by("v0", maintain_order=True).agg(
            [col("v63").count().alias("count")]
        )
    )
    var text = DataFrame(
        [
            Series(
                "text",
                StringColumn(
                    ["google", "plain", "google", "", "other"],
                    [True, True, False, True, True],
                ),
            ),
            Series("n", Column[Int32]([1, 2, 3, 4, 5])),
        ]
    )
    var named = text.lazy().with_columns(col("text").alias("copy"))
    check(named.filter(col("copy").str().contains("google")).sort("n"))
    check(named.select_exprs([col("copy"), col("text")]).sort("copy"))
    var facts = MetalRuntime().estimate_auto(
        source.lazy().select_exprs(aliases(2))
    )
    assert_true(";mojo-metal-v1;ops=" in facts.family)
    with assert_raises(contains="unsupported"):
        _ = MetalRuntime().estimate_auto(query)
    var ordinary = MetalRuntime().estimate_auto(
        source.lazy().select((col("x") * lit(Float64(2))).alias("twice"))
    )
    assert_true(facts.family != ordinary.family)
    var measured = query.profile(accelerator=MetalRuntime())
    same(measured[0], query.collect(engine="cpu"))
    assert_true(measured[1].height() > 0)
    var last = measured[1].height() - 1
    assert_true(measured[1].column("executor").string()._get(last) == "metal")
