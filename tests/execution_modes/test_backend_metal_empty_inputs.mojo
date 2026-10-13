"""Zero-column sources and resident consumers retain independent row metadata."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    Series,
    Column,
    Expr,
    LazyFrame,
    col,
    lit,
    all,
    by_dtype,
)
from dataframe.metal import MetalRuntime, _capabilities, _Descriptors
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.selectors import lower_selectors
from tests.execution_modes.test_backend_metal import require_metal


def empty(rows: Int) raises -> DataFrame:
    return DataFrame(List[Series](), height=rows)


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    var actual = query.collect(accelerator=runtime)
    var expected = query.collect(engine="cpu", streaming=False)
    assert_equal(actual.columns(), expected.columns())
    if actual.height() != expected.height():
        print(query.explain(optimize=False, streaming=False))
    assert_equal(actual.height(), expected.height())
    assert_equal(actual.width(), expected.width())
    for name in actual.columns():
        assert_true(actual.column(name).equals(expected.column(name)))


def test_empty_inputs_capability_schema_descriptors_and_empty_selectors() raises:
    assert_true(not RowCapabilities().empty_inputs)
    var query = empty(7).lazy()
    var caps = _capabilities()
    caps.empty_inputs = False
    with assert_raises(contains="source columns"):
        _ = lower_rows(query, caps)
    var plan = lower_rows(query, _capabilities())
    assert_equal(plan.source.height(), 7)
    assert_equal(plan.source.width(), 0)
    assert_equal(plan.slots, 0)
    assert_equal(len(plan.outputs), 0)
    var descriptors = _Descriptors(plan)
    var request = descriptors.request(plan, -1, False)
    assert_equal(request[1], 7)
    assert_equal(request[3], 0)
    assert_equal(request[4], 0)
    assert_true(
        "unsupported" not in MetalRuntime(device_id=999999).describe(query)
    )
    var selected = lower_selectors(query.select(all()))
    var result = selected.collect(engine="cpu", streaming=False)
    assert_equal(result.height(), 1)
    assert_equal(result.width(), 0)


def test_native_metal_empty_inputs_rows_slices_filters_and_literal_broadcasts() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    for rows in List[Int]([0, 1, 17, 257, 513]):
        var frame = empty(rows)
        check(frame.lazy(), runtime)
        check(frame.lazy().select_exprs(List[Expr]()), runtime)
        check(frame.lazy().drop(List[String]()), runtime)
        check(frame.lazy().filter(lit(True)), runtime)
        check(frame.lazy().filter(lit(False)), runtime)
        check(frame.lazy().head(0), runtime)
        check(frame.lazy().head(3), runtime)
        check(frame.lazy().slice(3, 9), runtime)
        check(frame.lazy().slice(-3, 2), runtime)
        check(frame.lazy().with_columns(lit(Int32(7)).alias("x")), runtime)
        check(frame.lazy().with_columns(lit(Float64(1.25)).alias("x")), runtime)
        check(frame.lazy().with_columns(lit(True).alias("x")), runtime)
        check(frame.lazy().with_columns(lit("β").alias("x")), runtime)
        check(
            frame.lazy()
            .with_columns(lit(Int32(7)).alias("x"))
            .group_by("x")
            .agg(col("x").count().alias("n")),
            runtime,
        )


def test_native_metal_empty_inputs_selectors_and_resident_scalar_consumers() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    for rows in List[Int]([0, 1, 17]):
        var frame = empty(rows)
        check(frame.lazy().select(all()), runtime)
        check(frame.lazy().select(by_dtype(["uint8"])), runtime)
        check(frame.lazy().select(all()).head(0), runtime)
        check(
            frame.lazy().select(all()).with_columns(lit(Float64(3)).alias("y")),
            runtime,
        )
        var source = DataFrame(
            [Series("x", Column[Int32](List[Int32](length=rows, fill=7)))]
        )
        check(source.lazy().drop(["x"]).select(all()), runtime)
        check(
            source.lazy()
            .select(col("x").count().alias("n"))
            .drop(["n"])
            .with_columns(lit(Float64(3)).alias("y")),
            runtime,
        )
        check(
            source.lazy()
            .select(col("x").count().alias("n"))
            .drop(["n"])
            .filter(lit(True)),
            runtime,
        )


def test_native_metal_empty_inputs_profiles_cost_budget_and_runtime_reuse() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var query = empty(513).lazy()
    var profiled = query.profile(accelerator=runtime)
    assert_equal(profiled[0].width(), 0)
    assert_equal(profiled[0].height(), 513)
    assert_true(profiled[1].height() > 0)
    var estimate = runtime.estimate_auto(query)
    assert_true(estimate.peak_bytes > 0)
    assert_true(";empty-inputs-v1" in estimate.family)
    assert_true(";metal-empty-inputs-v1" in estimate.family)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    check(query, runtime)
