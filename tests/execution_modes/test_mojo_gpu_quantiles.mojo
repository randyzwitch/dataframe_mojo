"""Mojo quantiles and medians through scalar and grouped execution."""

from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Expr, LazyFrame, Series, col, lit
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_moments import near


def _expressions(probability: Float64) -> List[Expr]:
    return [
        col("x").median().alias("median"),
        col("x").quantile(probability, "linear").alias("linear"),
        col("x").quantile(probability, "lower").alias("lower"),
        col("x").quantile(probability, "higher").alias("higher"),
        col("x").quantile(probability, "nearest").alias("nearest"),
        col("x").quantile(probability, "midpoint").alias("midpoint"),
        col("x").mean().alias("mean"),
        col("x").var(0).alias("variance"),
        col("x").count().alias("count"),
    ]


def _check(query: LazyFrame) raises -> DataFrame:
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    near(actual, query.collect(engine="cpu", streaming=False))
    return actual^


def _layout_case[D: DType](n: Int, chunked: Bool) raises:
    var values = List[Scalar[D]]()
    var valid = List[Bool]()
    for row in range(n + 6):
        values.append(Scalar[D]((row * 17) % 37))
        valid.append(row % 7 != 0)
    var source = Series("x", Column[Scalar[D]](values^, valid^)).slice(3, n)
    if chunked:
        source = Series._from_chunks(
            [
                source.slice(0, n // 2),
                source.slice(n // 2, 0),
                source.slice(n // 2, n - n // 2),
            ]
        )
    var frame = DataFrame([source^])
    var actual = _check(frame.lazy().select_exprs(_expressions(0.37)))
    assert_equal(actual.height(), 1)
    if n == 0:
        assert_true(actual.item(0, "median").is_null())
        assert_true(actual.item(0, "linear").is_null())


# requires: mojo_gpu
def test_mojo_gpu_quantiles_numeric_layouts_empty_masks_and_mixed_reductions() raises:
    comptime types = (DType.int64, DType.uint64, DType.float32, DType.float64)
    comptime for i in range(len(types)):
        comptime D = types[i]
        for n in List[Int]([0, 1, 257]):
            _layout_case[D](n, False)
            _layout_case[D](n, True)


# requires: mojo_gpu
def test_mojo_gpu_quantiles_nonfinite_zero_and_interpolation_anchors() raises:
    var examples: List[List[Float64]] = [
        [],
        [4],
        [0, 8],
        [-0.0, 0.0, -0.0, 0.0],
        [Float64("nan"), Float64("nan")],
        [1, Float64("nan"), 3, 4],
        [1, Float64("inf"), 3, 4],
        [1, Float64("-inf"), Float64("inf"), 4],
    ]
    for values in examples:
        var frame = DataFrame([Series("x", Column[Float64](values.copy()))])
        var probabilities: List[Float64] = [0, 0.25, 0.5, 1]
        for probability in probabilities:
            _ = _check(frame.lazy().select_exprs(_expressions(probability)))
    var frame = DataFrame([Series("x", Column[Float64]([0, 8]))])
    var result = _check(frame.lazy().select_exprs(_expressions(0.25)))
    assert_equal(result.item(0, "median").float64(), Float64(4))
    assert_equal(result.item(0, "linear").float64(), Float64(2))
    assert_equal(result.item(0, "lower").float64(), Float64(0))
    assert_equal(result.item(0, "higher").float64(), Float64(8))
    assert_equal(result.item(0, "nearest").float64(), Float64(0))
    assert_equal(result.item(0, "midpoint").float64(), Float64(4))
    result = _check(frame.lazy().select_exprs(_expressions(0.5)))
    assert_equal(result.item(0, "nearest").float64(), Float64(8))
    var zeros = DataFrame([Series("x", Column[Float64]([-0.0, 0.0]))])
    result = _check(zeros.lazy().select_exprs(_expressions(0.5)))
    assert_equal(
        bitcast[DType.uint64](result.item(0, "median").float64()), UInt64(0)
    )
    var nulls = DataFrame(
        [Series("x", Column[Float64]([1, 2], [False, False]))]
    )
    result = _check(nulls.lazy().select_exprs(_expressions(0.5)))
    assert_true(result.item(0, "median").is_null())
    var wide = DataFrame(
        [Series("x", Column[UInt64]([0, UInt64.MAX, UInt64.MAX - 1]))]
    )
    _ = _check(wide.lazy().select_exprs(_expressions(0.37)))


# requires: mojo_gpu
def test_mojo_gpu_quantiles_grouped_filtered_sliced_and_checked_errors() raises:
    var keys = List[Int32]()
    var values = List[Float64]()
    var valid = List[Bool]()
    for row in range(1027):
        keys.append(Int32(row % 5))
        values.append(Float64((row * 17) % 37 - 18) / 8)
        valid.append(row % 7 != 0)
    var frame = DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("x", Column[Float64](values^, valid^)),
        ]
    )
    _ = _check(
        frame.lazy().group_by("k", maintain_order=True).agg(_expressions(0.37))
    )
    _ = _check(
        frame.head(0)
        .lazy()
        .group_by("k", maintain_order=True)
        .agg(_expressions(0.5))
    )
    _ = _check(
        frame.lazy()
        .filter(col("x") > lit(Float64(0)))
        .slice(1, 257)
        .select_exprs(_expressions(0.37))
    )
    var empty = _check(
        frame.lazy()
        .filter(col("x") > lit(Float64(99)))
        .select_exprs(_expressions(0.5))
    )
    assert_equal(empty.height(), 1)
    assert_true(empty.item(0, "median").is_null())
    var bad = DataFrame([Series("x", Column[Int64]([Int64.MAX, 1]))])
    with assert_raises(contains="overflow"):
        _ = (
            bad.lazy()
            .select((col("x") + lit(Int64(1))).median())
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
