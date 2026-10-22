"""Direct Mojo GPU moments, paired reductions, masks and mixed outputs."""

from std.math import isnan
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    cov,
    corr,
    lit,
)
from dataframe_metal.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_moments import near


def _expressions() -> List[Expr]:
    return [
        col("x").var(0).alias("variance"),
        col("x").var(1).alias("sample_variance"),
        col("x").var(Int(Int64.MAX)).alias("insufficient"),
        col("x").std(0).alias("standard"),
        col("x").skew().alias("skew"),
        col("x").skew(False).alias("sample_skew"),
        col("x").kurtosis().alias("fisher"),
        col("x").kurtosis(False).alias("pearson"),
        col("x").kurtosis(True, False).alias("sample_fisher"),
        col("x").kurtosis(False, False).alias("sample_pearson"),
        cov(col("x"), col("y"), 0).alias("covariance"),
        cov(col("x"), col("y"), 1).alias("sample_covariance"),
        cov(col("x"), col("y"), Int(Int64.MAX)).alias("large_ddof_covariance"),
        corr(col("x"), col("y")).alias("correlation"),
        col("x").count().alias("count"),
        col("x").sum().alias("sum"),
    ]


def _check(query: LazyFrame) raises -> DataFrame:
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    near(actual, query.collect(engine="cpu", streaming=False))
    return actual^


def _layout_case[D: DType](n: Int, chunked: Bool) raises:
    var xs = List[Scalar[D]]()
    var ys = List[Scalar[D]]()
    var xv = List[Bool]()
    var yv = List[Bool]()
    for i in range(n + 6):
        xs.append(Scalar[D](i % 11 + 1))
        ys.append(Scalar[D]((i % 11 + 1) * 2 + i % 3))
        xv.append(i % 7 != 3)
        yv.append(i % 5 != 2)
    var x = Series("x", Column[Scalar[D]](xs^, xv^)).slice(3, n)
    var y = Series("y", Column[Scalar[D]](ys^, yv^)).slice(3, n)
    if chunked:
        x = Series._from_chunks(
            [
                x.slice(0, n // 2),
                x.slice(n // 2, 0),
                x.slice(n // 2, n - n // 2),
            ]
        )
        y = Series._from_chunks(
            [
                y.slice(0, n // 3),
                y.slice(n // 3, 0),
                y.slice(n // 3, n - n // 3),
            ]
        )
    var frame = DataFrame([x^, y^])
    var result = _check(frame.lazy().select_exprs(_expressions()))
    assert_equal(result.height(), 1)
    assert_true(result.item(0, "insufficient").is_null())
    assert_true(
        result.item(0, "large_ddof_covariance").is_null()
        or result.item(0, "large_ddof_covariance").float64() == 0
    )


# requires: mojo_gpu
def test_mojo_gpu_statistics_numeric_layouts_empty_masks_and_mixed_outputs() raises:
    comptime types = (DType.int64, DType.uint64, DType.float32, DType.float64)
    comptime for i in range(len(types)):
        for n in [0, 1, 2, 3, 4, 257, 1027]:
            _layout_case[types[i]](n, False)
            _layout_case[types[i]](n, True)


# requires: mojo_gpu
def test_mojo_gpu_statistics_nonfinite_constant_and_scalar_anchors() raises:
    var examples: List[List[Float64]] = [
        [],
        [4],
        [4, 4, 4, 4],
        [1, 2, 3, 4],
        [1, Float64("nan"), 3, 4],
        [1, Float64("inf"), 3, 4],
        [1, Float64("-inf"), Float64("inf"), 4],
    ]
    for values in examples:
        var frame = DataFrame(
            [
                Series("x", Column[Float64](values.copy())),
                Series("y", Column[Float64](values.copy())),
            ]
        )
        _ = _check(frame.lazy().select_exprs(_expressions()))
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([1, 2, 3, 4])),
            Series("y", Column[Int64]([2, 4, 6, 8])),
        ]
    )
    var result = _check(frame.lazy().select_exprs(_expressions()))
    assert_equal(result.item(0, "variance").float64(), Float64(1.25))
    assert_equal(result.item(0, "covariance").float64(), Float64(2.5))
    assert_equal(result.item(0, "correlation").float64(), Float64(1))
    assert_equal(result.item(0, "skew").float64(), Float64(0))
    assert_true(abs(result.item(0, "fisher").float64() + 1.36) < 1e-12)
    assert_true(abs(result.item(0, "sample_fisher").float64() + 1.2) < 1e-12)
    var nulls = DataFrame(
        [
            Series(
                "x", Column[Float64]([1, 2, 3, 4], [False, False, False, False])
            ),
            Series("y", Column[Float64]([2, 4, 6, 8])),
        ]
    )
    result = _check(nulls.lazy().select_exprs(_expressions()))
    assert_true(result.item(0, "variance").is_null())
    assert_true(result.item(0, "skew").is_null())
    assert_true(result.item(0, "covariance").is_null())
    assert_true(isnan(result.item(0, "correlation").float64()))


# requires: mojo_gpu
def test_mojo_gpu_statistics_filtered_sliced_and_pair_intersection() raises:
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([1, 2, 3, 4, 5, 6, 7, 8])),
            Series(
                "y",
                Column[Float32](
                    [2, 4, 6, 8, 10, 12, 14, 16],
                    [True, False, True, True, True, False, True, True],
                ),
            ),
        ]
    )
    _ = _check(
        frame.lazy()
        .filter(col("x") > lit(Int64(1)))
        .slice(1, 4)
        .select_exprs(_expressions())
    )
    var empty = _check(
        frame.lazy()
        .filter(col("x") > lit(Int64(99)))
        .select_exprs(_expressions())
    )
    assert_equal(empty.height(), 1)
    assert_true(empty.item(0, "variance").is_null())
    assert_true(empty.item(0, "covariance").is_null())
    assert_true(isnan(empty.item(0, "correlation").float64()))
    var separate = DataFrame(
        [
            Series(
                "x", Column[Int64]([1, 2, 3, 4], [True, False, True, False])
            ),
            Series(
                "y", Column[Int64]([2, 4, 6, 8], [False, True, False, True])
            ),
        ]
    )
    var result = _check(separate.lazy().select_exprs(_expressions()))
    assert_true(result.item(0, "covariance").is_null())
    assert_true(isnan(result.item(0, "correlation").float64()))
