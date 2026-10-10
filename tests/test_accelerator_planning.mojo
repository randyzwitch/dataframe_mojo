"""Resident-expression lowering must remain usable without MAX or a device."""
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    Series,
    StringColumn,
    col,
    lit,
    scan_csv,
)
from dataframe.accelerator.memory import (
    row_memory,
    checked_add,
    checked_mul,
    bitmap_bytes,
)
from dataframe.dtype import DataType
from dataframe.expr import SUM
from dataframe.accelerator.rows import lower_rows, ROW_MAX_NODES
from dataframe.accelerator import RowCapabilities


def test_resident_steps_bind_shared_names_and_dtypes() raises:
    var frame = DataFrame([Series("x", Column[Float32]([1, 2, 3]))])
    var query = (
        frame.lazy()
        .with_columns(((col("x") + 1) * 2).alias("y"))
        .filter((col("y") > 2) & col("x").is_not_null())
        .select_exprs([col("y"), (col("x") != 2).alias("b")])
        .head(2)
    )
    var plan = lower_rows(query)
    assert_equal(len(plan.steps), 4)
    assert_true(plan.steps[1].filter)
    assert_equal(plan.steps[1].gather_count, 2)
    assert_equal(plan.outputs[0].name, "y")
    assert_equal(plan.outputs[1].dtype, DataType.BOOL)
    assert_equal(plan.limit, 2)
    assert_true(not plan.reductions)
    var reduction = lower_rows(
        frame.lazy()
        .with_columns((col("x") * 2).alias("y"))
        .select_exprs([col("y").sum().alias("s"), col("x").count().alias("n")])
    )
    assert_true(reduction.reductions)
    assert_equal(reduction.outputs[1].dtype, DataType.INT64)


def test_rejections_happen_without_reading_files() raises:
    with assert_raises(contains="in-memory scan"):
        _ = lower_rows(scan_csv("/nonexistent/accel-rows.csv").select(col("x")))
    var frame = DataFrame([Series("x", Column[Float32]([1, 2]))])
    with assert_raises(contains="Unknown expression column"):
        _ = lower_rows(frame.lazy().select(col("missing")))
    with assert_raises(contains="scalar-only"):
        _ = lower_rows(frame.lazy().select(lit(Float32(1))))
    with assert_raises(contains="unsupported reduction"):
        _ = lower_rows(frame.lazy().select(col("x").median()))
    var expression = col("x")
    for _ in range(ROW_MAX_NODES):
        expression = expression + 1
    with assert_raises(contains="64 row-local nodes"):
        _ = lower_rows(frame.lazy().select(expression))


def test_memory_preflight_counts_all_requested_buffers() raises:
    var frame = DataFrame([Series("x", Column[Float32]([1, 2, 3]))])
    var plan = lower_rows(
        frame.lazy()
        .filter(col("x") > 1)
        .select_exprs(
            [(col("x") * 2 + 1).alias("y"), (col("x") == 2).alias("b")]
        )
    )
    var memory = row_memory(plan)
    assert_equal(memory.summary.peak_bytes, 644)
    assert_equal(memory.upload_bytes, 476)
    assert_equal(memory.summary.launches, 9)
    assert_equal(memory.matrix, 12)
    assert_equal(memory.alternate, 12)
    memory.summary.require_budget(644)
    with assert_raises(contains="memory preflight"):
        memory.summary.require_budget(643)
    assert_equal(bitmap_bytes(0, 7), 0)
    assert_equal(bitmap_bytes(9, 7), 2)
    assert_equal(checked_mul(Int.MAX, 0), 0)
    with assert_raises(contains="overflows"):
        _ = checked_mul(Int.MAX, 2)
    with assert_raises(contains="overflows"):
        _ = checked_add(Int.MAX, 1)
    with assert_raises(contains="Invalid"):
        _ = bitmap_bytes(1, 8)


def test_integer_lowering_keeps_literal_words_and_checked_boundaries() raises:
    var exact = Int64(9007199254740993)
    var frame = DataFrame([Series("x", Column[Int64]([1, 2, 3]))])
    var plan = lower_rows(frame.lazy().select(col("x") + lit(exact)))
    assert_equal(plan.dtype, DataType.INT64)
    assert_equal(bitcast[DType.int64](plan.literals[1]), exact)
    with assert_raises(contains="terminal projections"):
        _ = lower_rows(
            frame.lazy()
            .with_columns((col("x") + 1).alias("y"))
            .filter(col("x") > 1)
        )
    with assert_raises(contains="terminal projections"):
        _ = lower_rows(frame.lazy().filter((col("x") + 1) > 1))
    with assert_raises(contains="terminal projections"):
        _ = lower_rows(frame.lazy().select(col("x") + 1).head(1))
    with assert_raises(contains="integer sum with final head"):
        _ = lower_rows(frame.lazy().select(col("x").sum()).head(0))
    with assert_raises(contains="scalar integer arithmetic"):
        _ = lower_rows(
            frame.lazy().select_exprs(
                [col("x"), (lit(Int64.MAX) + 1).alias("scalar")]
            )
        )
    var memory = row_memory(
        lower_rows(
            frame.lazy()
            .filter(col("x") > 1)
            .select_exprs(
                [(col("x") * 2 + 1).alias("y"), (col("x") == 2).alias("b")]
            )
        )
    )
    # Float64-equivalent buffers: 740 bytes. Integers add 24 bytes of error
    # positions, 8 bytes of wider partials, and 16 bytes of metadata/output.
    assert_equal(memory.summary.peak_bytes, 788)
    assert_equal(memory.upload_bytes, 496)
    assert_equal(memory.summary.launches, 10)
    memory.summary.require_budget(788)
    with assert_raises(contains="memory preflight"):
        memory.summary.require_budget(787)


def test_projection_prunes_wide_mixed_sources_and_preserves_inputs() raises:
    var columns = List[Series]([Series("x", Column[Float32]([1, 2, 3]))])
    for i in range(70):
        columns.append(Series("unused" + String(i), Column[Int64]([1, 2, 3])))
    columns.append(Series("text", StringColumn(["a", "b", "c"])))
    var frame = DataFrame(columns^)
    var query = (
        frame.lazy()
        .filter(col("x") > 1)
        .select_exprs(
            [col("x").mean().alias("mean"), col("x").len().alias("len")]
        )
    )
    var before = query.explain(engine="cpu")
    var plan = lower_rows(query)
    assert_equal(plan.source.width(), 1)
    assert_equal(plan.source.height(), 3)
    assert_equal(plan.outputs[0].dtype, DataType.FLOAT64)
    assert_equal(plan.outputs[1].dtype, DataType.INT64)
    assert_equal(query.explain(engine="cpu"), before)
    assert_equal(frame.width(), 72)
    with assert_raises(contains="source columns"):
        _ = lower_rows(frame.lazy().filter(col("x") > 1))


def test_backend_capabilities_preserve_precision_contracts() raises:
    var native = RowCapabilities("native", float64=False, wide_integer=False)
    var single = DataFrame([Series("x", Column[Float32]([1, 2, 3]))])
    var double = DataFrame([Series("x", Column[Float64]([1, 2, 3]))])
    var integers = DataFrame([Series("x", Column[Int32]([1, 2, 3]))])
    var wide = DataFrame([Series("x", Column[Int64]([1, 2, 3]))])
    _ = lower_rows(single.lazy().select(col("x") * 2), native)
    _ = lower_rows(integers.lazy().select(col("x").sum()), native)
    _ = lower_rows(wide.lazy().select(col("x").count()), native)
    with assert_raises(contains="native unsupported [dtype]: Float64"):
        _ = lower_rows(double.lazy().select(col("x")), native)
    with assert_raises(contains="Float64 accumulation"):
        _ = lower_rows(single.lazy().select(col("x").sum()), native)
    with assert_raises(contains="Float64 arithmetic"):
        _ = lower_rows(single.lazy().select(col("x").mean()), native)
    with assert_raises(contains="Float64 arithmetic"):
        _ = lower_rows(integers.lazy().select(col("x").mean()), native)
    with assert_raises(contains="exact wide accumulation"):
        _ = lower_rows(wide.lazy().select(col("x").sum()), native)
    with assert_raises(contains="exact Int64 row bound"):
        native.require_reduction(DataType.INT32, SUM, Int(Int32.MAX) + 1)
    # An independently constrained provider can reject arithmetic while
    # preserving exact Int64 predicates and literals above 2**53.
    var restricted = RowCapabilities("restricted", int64_arithmetic=False)
    _ = lower_rows(
        wide.lazy().select(col("x") > lit(Int64(9007199254740993))), restricted
    )
    with assert_raises(contains="checked Int64 arithmetic"):
        _ = lower_rows(wide.lazy().select(col("x") + 1), restricted)


def test_memory_launch_dimensions_are_backend_owned() raises:
    var frame = DataFrame(
        [Series("x", Column[Float32](List[Float32](length=1025, fill=1)))]
    )
    var plan = lower_rows(frame.lazy().select(col("x")))
    var first = row_memory(plan, threads=256, max_blocks=1024)
    var second = row_memory(plan, threads=128, max_blocks=2)
    assert_equal(first.blocks, 5)
    assert_equal(second.blocks, 2)
    assert_equal(second.chunk, 640)
    with assert_raises(contains="must be positive"):
        _ = row_memory(plan, threads=0)
    with assert_raises(contains="must be positive"):
        _ = row_memory(plan, max_blocks=0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
