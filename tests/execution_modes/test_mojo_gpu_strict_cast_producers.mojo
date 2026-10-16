"""Strict-cast producers observe their complete original phase before consumers."""
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.metal import MetalRuntime
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_mojo_gpu_strict_cast_composition_filter_group_join_and_layouts() raises:
    var raw = DataFrame(
        [
            Series("x", Column[Float64]([1.25, 2.75, 1.75])),
            Series("k", Column[Int32]([0, 1, 0])),
        ]
    )
    var expected = DataFrame(
        [
            Series("segment", Column[Int32]([1, 2])),
            Series("n", Column[Int64]([2, 1])),
        ]
    )
    for layout in range(3):
        var source = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        var produced = source.lazy().select_exprs(
            [col("x").cast(DataType.INT32).alias("segment"), col("k")]
        )
        var grouped = produced.group_by("segment", maintain_order=True).agg(
            col("k").len().alias("n")
        )
        assert_frame(
            grouped.collect(engine="cpu", optimize=False, streaming=False),
            expected,
        )
        assert_frame(
            grouped.collect(accelerator=MetalRuntime(), optimize=False),
            expected,
        )
        var right = DataFrame([Series("segment", Column[Int32]([1, 2]))])
        var joined = (
            produced.filter(col("segment") > 0)
            .join(right.lazy(), "segment")
            .head(2)
        )
        assert_frame(
            joined.collect(accelerator=MetalRuntime(), optimize=False),
            joined.collect(engine="cpu", optimize=False, streaming=False),
        )
    print("PASS strict cast composition filter/group/join offset/chunk")


def test_mojo_gpu_strict_cast_complete_phase_faults_and_conditionals() raises:
    var source = DataFrame([Series("x", Column[Int64]([1, 128]))])
    var produced = source.lazy().with_columns(
        col("x").cast(DataType.INT8).alias("y")
    )
    for count in List[Int]([0, 1]):
        var failing = produced.filter(col("x") < 2).head(count)
        with assert_raises(contains="cast from int64 to int8 failed"):
            _ = failing.collect(engine="cpu", optimize=False, streaming=False)
        with assert_raises(contains="strict cast failed"):
            _ = failing.collect(accelerator=MetalRuntime(), optimize=False)
    var empty_right = DataFrame([Series("y", Column[Int8]())])
    var joined = produced.join(empty_right.lazy(), "y").head(0)
    with assert_raises(contains="cast from int64 to int8 failed"):
        _ = joined.collect(engine="cpu", optimize=False, streaming=False)
    with assert_raises(contains="strict cast failed"):
        _ = joined.collect(accelerator=MetalRuntime(), optimize=False)
    var safe = (
        source.lazy()
        .select(
            when(col("x") < 2)
            .then(col("x").cast(DataType.INT8))
            .otherwise(lit(Int8(0)))
            .alias("y")
        )
        .filter(col("y") >= 0)
    )
    var expected = DataFrame([Series("y", Column[Int8]([1, 0]))])
    assert_frame(
        safe.collect(engine="cpu", optimize=False, streaming=False), expected
    )
    assert_frame(
        safe.collect(accelerator=MetalRuntime(), optimize=False), expected
    )
    with assert_raises(contains="budget"):
        _ = safe.collect(
            accelerator=MetalRuntime(memory_limit_bytes=0), optimize=False
        )
    assert_frame(
        safe.collect(accelerator=MetalRuntime(), optimize=False), expected
    )
    print(
        "PASS strict cast phase-before-filter/head and conditional masks/budget/recovery"
    )


def test_mojo_gpu_strict_cast_projection_fault_order_before_consumer() raises:
    var source = DataFrame(
        [
            Series("x", Column[Int64]([1, 128])),
            Series("v", Column[Int64]([Int64.MAX, 0])),
        ]
    )
    for cast_first in [True, False]:
        var expressions = List[Expr]()
        var cast = col("x").cast(DataType.INT8).alias("y")
        var arithmetic = (col("v") + lit(Int64(1))).alias("a")
        expressions.append(cast.copy() if cast_first else arithmetic.copy())
        expressions.append(arithmetic.copy() if cast_first else cast.copy())
        var query = source.lazy().select_exprs(expressions).sort("y").head(0)
        with assert_raises(
            contains="cast from int64 to int8 failed" if cast_first else "overflow"
        ):
            _ = query.collect(engine="cpu", optimize=False, streaming=False)
        with assert_raises(
            contains="strict cast failed" if cast_first else "overflow"
        ):
            _ = query.collect(accelerator=MetalRuntime(), optimize=False)
    print("PASS strict cast projection order before downstream empty head")


def test_mojo_gpu_strict_cast_nullable_original_values_and_dtype() raises:
    var source = DataFrame(
        [
            Series("x", Column[Int64]([7, 128, 9], [True, False, True])),
            Series("k", Column[Int32]([0, 1, 2])),
        ]
    )
    var expected = DataFrame(
        [
            Series("x", Column[Int64]([7, 128, 9], [True, False, True])),
            Series("k", Column[Int32]([0, 1, 2])),
            Series("y", Column[Int8]([7, 0, 9], [True, False, True])),
        ]
    )
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        var query = (
            frame.lazy()
            .with_columns(col("x").cast(DataType.INT8).alias("y"))
            .sort("k")
        )
        assert_frame(
            query.collect(engine="cpu", optimize=False, streaming=False),
            expected,
        )
        assert_frame(
            query.collect(accelerator=MetalRuntime(), optimize=False), expected
        )
    print(
        "PASS strict cast null mask, original fields/dtypes, offset/chunk owners"
    )
