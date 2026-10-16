"""Scalar-only selections retain one-row CPU shapes through resident consumers."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    DataType,
    Series,
    Column,
    Expr,
    LazyFrame,
    col,
    lit,
    null,
)
from dataframe.accelerator.scalars import (
    has_scalar_projections,
    lower_scalar_projections,
)
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_moments import near


def empty(rows: Int) raises -> DataFrame:
    return DataFrame(List[Series](), height=rows)


def expressions() -> List[Expr]:
    return [
        lit(Int32(7)).alias("i"),
        lit(UInt64.MAX).alias("u"),
        (lit(Float64(1.25)) + lit(Float64(2.5))).alias("f"),
        lit(Float32(0.25)).alias("f32"),
        lit(True).alias("b"),
        null(DataType.FLOAT64).alias("n"),
        lit(Int64(-1)).cast(DataType.DATE).alias("date"),
    ]


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    near(
        query.collect(accelerator=runtime),
        query.collect(engine="cpu", streaming=False),
    )


def test_scalar_projection_normalization_shape_aliases_and_metadata() raises:
    var source = DataFrame(
        [Series("__metal_scalar_count_0", Column[Int32]([1, 2, 3]))]
    )
    for rows in List[Int]([0, 1, 3]):
        for frame in List[DataFrame]([source.head(rows), empty(rows)]):
            var query = frame.lazy().select_exprs(expressions())
            var before = query.explain(optimize=False)
            assert_true(has_scalar_projections(query))
            var lowered = lower_scalar_projections(query)
            assert_true(not has_scalar_projections(lowered))
            near(
                lowered.collect(engine="cpu", streaming=False),
                query.collect(engine="cpu", streaming=False),
            )
            assert_equal(query.explain(optimize=False), before)
            assert_true(
                "unsupported"
                not in MetalRuntime(device_id=999999).describe(query)
            )
    assert_true(
        not has_scalar_projections(source.lazy().select_exprs(List[Expr]()))
    )
    assert_true(
        not has_scalar_projections(
            source.lazy().select(col("__metal_scalar_count_0"))
        )
    )
    assert_true(
        not has_scalar_projections(
            source.lazy().select(col("__metal_scalar_count_0").sum())
        )
    )


# requires: metal
def test_native_metal_scalar_projections_empty_sources_types_and_literal_text() raises:
    var runtime = require_metal()
    var source = DataFrame([Series("x", Column[Int32]([1, 2, 3, 4]))])
    for rows in List[Int]([0, 1, 4]):
        for frame in List[DataFrame]([source.head(rows), empty(rows)]):
            check(frame.lazy().select_exprs(expressions()), runtime)
            check(frame.lazy().select(lit("β").alias("text")), runtime)
            check(
                frame.lazy().select_exprs(
                    [
                        lit("2000-02-29T12:34:56.123456")
                        .str()
                        .strptime("datetime[us]")
                        .alias("dt"),
                        lit("12:34:56.123456789")
                        .str()
                        .strptime("time")
                        .alias("clock"),
                    ]
                ),
                runtime,
            )
    var literal = (
        source.head(0)
        .lazy()
        .select(lit(Int32(7)).alias("i"))
        .collect(accelerator=runtime)
    )
    assert_equal(literal.height(), 1)
    assert_equal(literal.column("i").int32()._get(0), Int32(7))


# requires: metal
def test_native_metal_scalar_projection_consumers_limits_joins_and_collisions() raises:
    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series("x", Column[Int32]([1, 2, 3])),
            Series("__metal_scalar_count_0", Column[Int32]([4, 5, 6])),
        ]
    )
    var query = frame.lazy().select(lit(Int32(7)).alias("x"))
    check(query.select_exprs(List[Expr]()), runtime)
    check(query.head(0), runtime)
    check(query.head(3), runtime)
    check(query.slice(-1, 1), runtime)
    check(query.filter(col("x") > lit(Int32(3))), runtime)
    check(query.filter(col("x") < lit(Int32(3))), runtime)
    check(query.select(col("x").sum()), runtime)
    check(query.select(col("x").median()), runtime)
    check(query.select(lit(Int32(8)).alias("v")), runtime)
    check(query.join(frame.lazy(), ["x"]), runtime)
    check(
        query.join(frame.lazy().with_columns(lit(Int32(7)).alias("x")), ["x"]),
        runtime,
    )
    check(
        frame.lazy().select(lit(Int32(9)).alias("__metal_scalar_count_0")),
        runtime,
    )
    check(
        empty(0).lazy().select(lit(Int32(9)).alias("__metal_scalar_input")),
        runtime,
    )


# requires: metal
def test_native_metal_scalar_projection_profiles_budgets_and_runtime_reuse() raises:
    var runtime = require_metal()
    var query = empty(17).lazy().select_exprs(expressions())
    var profiled = query.profile(accelerator=runtime)
    near(profiled[0], query.collect(engine="cpu", streaming=False))
    assert_true(profiled[1].height() > 0)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    check(query, runtime)
