"""Exact signed-zero order statistics, source-order ties and resident consumers."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Expr, LazyFrame, Series, col, lit
from dataframe.metal import MetalRuntime
from dataframe.accelerator.gpu.runtime import _preflight, estimate_resources
from dataframe.expr import SUM


def _expressions(q: Float64) -> List[Expr]:
    return [
        col("x").quantile(q, "linear").alias("linear"),
        col("x").quantile(q, "lower").alias("lower"),
        col("x").quantile(q, "higher").alias("higher"),
        col("x").quantile(q, "nearest").alias("nearest"),
        col("x").quantile(q, "midpoint").alias("midpoint"),
    ]


def _exact(query: LazyFrame) raises -> DataFrame:
    var wanted = query.collect(engine="cpu", optimize=False, streaming=False)
    var actual = query.collect(accelerator=MetalRuntime())
    assert_equal(actual.height(), wanted.height())
    assert_equal(actual.width(), wanted.width())
    for name in wanted.columns():
        assert_equal(actual.column(name).dtype(), wanted.column(name).dtype())
        if not wanted.column(name).dtype().is_float():
            assert_true(actual.column(name).equals(wanted.column(name)))
            continue
        for row in range(wanted.height()):
            var value = wanted.item(row, name)
            if value.is_null():
                assert_true(actual.item(row, name).is_null())
            elif wanted._columns[wanted._index(name)].dtype().is_float():
                var a = bitcast[DType.uint64](actual.item(row, name).float64())
                var b = bitcast[DType.uint64](value.float64())
                if value.float64() != value.float64():
                    assert_true(
                        actual.item(row, name).float64()
                        != actual.item(row, name).float64()
                    )
                else:
                    assert_equal(a, b)
    return actual^


# requires: metal
def test_mojo_gpu_quantile_zero_scalar_exact_ranks_and_reciprocal() raises:
    var negative = bitcast[DType.float64](UInt64(0x8000000000000000))
    var frame = DataFrame(
        [Series("x", Column[Float64]([negative, negative, negative]))]
    )
    var expressions = _expressions(0.5)
    expressions.append(col("x").median().alias("median"))
    var actual = _exact(frame.lazy().select_exprs(expressions))
    for name in actual.columns():
        assert_equal(
            bitcast[DType.uint64](actual.item(0, name).float64()),
            UInt64(0x8000000000000000),
        )
    actual = _exact(
        frame.lazy()
        .select(col("x").median().alias("mid"))
        .with_columns((lit(Float64(1)) / col("mid")).alias("inverse"))
    )
    assert_equal(
        bitcast[DType.uint64](actual.item(0, "inverse").float64()),
        UInt64(0xFFF0000000000000),
    )
    var pairs: List[List[Float64]] = [[negative, 0], [0, negative]]
    for values in pairs:
        frame = DataFrame([Series("x", Column[Float64](values.copy()))])
        for q in List[Float64]([0.0, 0.5, 1.0]):
            _ = _exact(frame.lazy().select_exprs(_expressions(q)))


# requires: metal
def test_mojo_gpu_quantile_zero_quickselect_fallback_and_nonfinite() raises:
    # Independent literal: three-way quickselect swaps trailing -0 into rank1.
    var values = List[Float64](length=27, fill=0)
    values[0] = 1
    values[1] = -1
    values[26] = bitcast[DType.float64](UInt64(0x8000000000000000))
    var frame = DataFrame([Series("x", Column[Float64](values^))])
    var q = bitcast[DType.float64](UInt64(0x3FA3B13B13B13B14))
    var actual = _exact(frame.lazy().select_exprs(_expressions(q)))
    assert_equal(
        bitcast[DType.uint64](actual.item(0, "lower").float64()),
        UInt64(0x8000000000000000),
    )
    # Natural round-limit fallback, then std.sort's unstable zero permutation.
    # The pinned CPU oracle supplies its exact representation, never tolerance.
    for zeros in [2, 32]:
        values = List[Float64](length=1024, fill=0)
        for row in range(1024):
            values[row] = Float64(row - 1024)
            if row >= 1024 - zeros:
                values[row] = bitcast[DType.float64](UInt64(row % 2) << 63)
        frame = DataFrame([Series("x", Column[Float64](values^))])
        _ = _exact(frame.lazy().select_exprs(_expressions(1)))
    var words: List[UInt64] = [
        0x7FF8000000000042,
        0x8000000000000000,
        0xDEADBEEF,
        0x8000000000000000,
        0x8000000000000000,
    ]
    values = List[Float64]()
    for word in words:
        values.append(bitcast[DType.float64](word))
    frame = DataFrame(
        [Series("x", Column[Float64](values^, [True, True, False, True, True]))]
    )
    for probability in List[Float64]([0.0, 0.25, 0.5, 0.75, 1.0]):
        _ = _exact(frame.lazy().select_exprs(_expressions(probability)))


# requires: metal
def test_mojo_gpu_quantile_zero_grouped_disjoint_scratch_and_budget() raises:
    var values = List[Float64]()
    var keys = List[Int32]()
    for row in range(108):
        var group = row % 4
        keys.append(Int32(group))
        var local = row // 4
        var word = UInt64(0x8000000000000000)
        if group == 1:
            word = UInt64(local % 2) << 63
        elif group == 2:
            word = (
                0x3FF0000000000000 if local
                == 0 else 0xBFF0000000000000 if local
                == 1 else 0x8000000000000000 if local
                == 26 else UInt64(0)
            )
        elif group == 3 and local % 5 == 0:
            word = 0x7FF8000000000042
        values.append(bitcast[DType.float64](word))
    var frame = DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("x", Column[Float64](values^)),
        ]
    )
    var expressions = _expressions(0.5)
    expressions.append(col("x").median().alias("median"))
    var result = _exact(
        frame.lazy().group_by("k", maintain_order=True).agg(expressions)
    )
    assert_equal(
        bitcast[DType.uint64](result.item(0, "median").float64()),
        UInt64(0x8000000000000000),
    )
    _ = _exact(
        frame.lazy()
        .slice(1, 103)
        .group_by("k", maintain_order=True)
        .agg(_expressions(0))
    )
    var scalar = frame.lazy().select(col("x").quantile(0.5, "lower"))
    var plan = _preflight(scalar)
    var facts = estimate_resources(plan)
    plan.outputs[0].reduction = SUM
    var sum_facts = estimate_resources(plan)
    # Same descriptors/matrix/partial/output shape; only the new index plane.
    assert_equal(facts.peak_bytes - sum_facts.peak_bytes, frame.height() * 4)
    _ = scalar.collect(
        accelerator=MetalRuntime(memory_limit_bytes=facts.peak_bytes)
    )
    with assert_raises(contains="budget"):
        _ = scalar.collect(
            accelerator=MetalRuntime(memory_limit_bytes=facts.peak_bytes - 1)
        )
    _ = _exact(scalar)


# requires: metal
def test_mojo_gpu_quantile_linear_separate_rounding_literal_anchors() raises:
    # Exact integer/rational IEEE round-to-nearest ties-even anchors. FMA
    # differs by one or two ULPs, so numerical tolerance would miss this.
    var cases: List[UInt64] = [
        0xC024000000000000,
        0,
        0x3FE6666666666666,
        0xC008000000000000,
        0xC024000000000000,
        0x4008000000000000,
        0x3FD3333333333333,
        0xC018666666666666,
        0xC341C37937E08000,
        0x3FF0000000000000,
        0x3FECCCCCCCCCCCCD,
        0xC30C6BF526340000,
        0xC04FCC0D54832E81,
        0xC03BE2DBD1AF6896,
        0x3FE6666666666666,
        0xC0434C9DBC57F28E,
    ]
    for at in range(4):
        var a = bitcast[DType.float64](cases[4 * at])
        var b = bitcast[DType.float64](cases[4 * at + 1])
        var q = bitcast[DType.float64](cases[4 * at + 2])
        var frame = DataFrame([Series("x", Column[Float64]([a, b]))])
        var result = _exact(
            frame.lazy().select(col("x").quantile(q, "linear").alias("v"))
        )
        assert_equal(
            bitcast[DType.uint64](result.item(0, "v").float64()),
            cases[4 * at + 3],
        )
        frame = DataFrame(
            [
                Series("k", Column[Int32]([0, 1, 0, 1])),
                Series("x", Column[Float64]([a, a, b, b])),
            ]
        )
        result = _exact(
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg([col("x").quantile(q, "linear").alias("v")])
        )
        for group in range(2):
            assert_equal(
                bitcast[DType.uint64](result.item(group, "v").float64()),
                cases[4 * at + 3],
            )
