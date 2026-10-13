"""Zero columns preserve row cardinality through native materialization."""
from std.testing import assert_true, assert_equal, assert_raises
from dataframe import (
    DataFrame,
    Series,
    Column,
    StringColumn,
    LazyFrame,
    Expr,
    col,
    lit,
    by_dtype,
    exclude,
)
from dataframe.metal import MetalRuntime, _capabilities, _Descriptors
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.selectors import lower_selectors
from tests.execution_modes.test_backend_metal import require_metal


def fixture(rows: Int) raises -> DataFrame:
    var values = List[Int32]()
    var keys = List[Int32]()
    for i in range(rows):
        values.append(Int32(i % 11))
        keys.append(Int32(i % 3))
    return DataFrame(
        [
            Series("x", Column[Int32](values^)),
            Series("k", Column[Int32](keys^)),
        ]
    )


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    var actual = query.collect(accelerator=runtime)
    var expected = query.collect(engine="cpu", streaming=False)
    assert_equal(actual.columns(), expected.columns())
    assert_equal(actual.width(), 0)
    assert_equal(actual.height(), expected.height())


def test_empty_outputs_capability_metadata_and_distinct_cpu_selection_shapes() raises:
    assert_true(not RowCapabilities().empty_outputs)
    var frame = fixture(5)
    var empty = frame.lazy().select_exprs(List[Expr]())
    var caps = _capabilities()
    caps.empty_outputs = False
    with assert_raises(contains="at least one output column"):
        _ = lower_rows(empty, caps)
    var plan = lower_rows(empty, _capabilities())
    assert_equal(len(plan.outputs), 0)
    _ = _Descriptors(plan)
    assert_equal(empty.collect(engine="cpu").height(), 5)
    var unmatched = frame.lazy().select(by_dtype(["uint8"]))
    assert_equal(unmatched.collect(engine="cpu").height(), 1)
    var lowered = lower_selectors(unmatched)
    assert_equal(lowered.collect(engine="cpu").height(), 1)
    assert_equal(lowered.collect(engine="cpu").width(), 0)
    var runtime = MetalRuntime(device_id=999999)
    assert_true("unsupported" not in runtime.describe(empty))
    assert_true("unsupported" not in runtime.describe(unmatched))


def test_native_metal_empty_results_rows_filters_slices_groups_and_joins() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    for rows in List[Int]([0, 1, 17, 257, 513]):
        var frame = fixture(rows)
        check(frame.lazy().select_exprs(List[Expr]()), runtime)
        check(frame.lazy().drop(["x", "k"]), runtime)
        check(
            frame.lazy()
            .filter(col("x") > lit(Int32(4)))
            .select_exprs(List[Expr]())
            .head(3),
            runtime,
        )
        check(frame.lazy().slice(3, 9).select_exprs(List[Expr]()), runtime)
        check(
            frame.lazy()
            .group_by("k")
            .agg(col("x").count().alias("n"))
            .drop(["k", "n"]),
            runtime,
        )
    var frame = fixture(13)
    check(
        frame.lazy().join(frame.lazy(), ["k"]).select_exprs(List[Expr]()),
        runtime,
    )
    var query = (
        frame.lazy().filter(col("x") > lit(Int32(4))).select_exprs(List[Expr]())
    )
    var profiled = query.profile(accelerator=runtime)
    assert_equal(profiled[0].width(), 0)
    assert_equal(profiled[0].height(), query.collect(engine="cpu").height())
    var estimate = runtime.estimate_auto(query)
    assert_true(estimate.peak_bytes > 0)
    assert_true(";metal-empty-outputs-v1" in estimate.family)
    assert_true(";empty-outputs-v1" in estimate.family)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    check(query, runtime)


def test_native_metal_empty_selectors_scalar_shape_strings_partitions_and_limits() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    for rows in List[Int]([0, 1, 17]):
        var frame = fixture(rows)
        check(frame.lazy().select(by_dtype(["uint8"])), runtime)
        check(frame.lazy().select(exclude(["x", "k"])), runtime)
        check(
            frame.lazy()
            .filter(col("x") > lit(Int32(4)))
            .select(by_dtype(["uint8"])),
            runtime,
        )
        check(frame.lazy().select(by_dtype(["uint8"])).head(0), runtime)
        check(frame.lazy().select(by_dtype(["uint8"])).head(7), runtime)
    var strings = DataFrame(
        [
            Series("s", StringColumn(["a", "β", "a"])),
            Series("x", Column[Float64]([1, 2, 3])),
        ]
    )
    check(strings.lazy().select_exprs(List[Expr]()), runtime)
    check(strings.lazy().select(by_dtype(["uint8"])), runtime)
    check(
        strings.lazy()
        .with_columns(col("x").rolling_var(2, 1, 0).alias("v"))
        .select(by_dtype(["uint8"])),
        runtime,
    )
