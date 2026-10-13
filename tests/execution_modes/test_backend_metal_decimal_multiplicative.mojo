"""Decimal products and quotients preserve checked integer CPU semantics."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    Series,
    Column,
    Expr,
    DataType,
    LazyFrame,
    col,
    lit,
    when,
)
from dataframe.decimal import pow10
from dataframe.metal import MetalRuntime, _capabilities, _Descriptors
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.execution_modes.test_backend_metal_decimal_comparisons import (
    decimal_series,
    safe_fixture,
)


def parity(query: LazyFrame, runtime: MetalRuntime) raises:
    var expected = DataFrame(List[Series]())
    var failure = String()
    try:
        expected = query.collect(engine="cpu")
    except e:
        failure = String(e)
    if failure.byte_length():
        if "decimal multiplication overflow" in failure:
            with assert_raises(contains="decimal multiplication overflow"):
                _ = query.collect(accelerator=runtime)
        elif "decimal precision" in failure:
            with assert_raises(contains="decimal precision"):
                _ = query.collect(accelerator=runtime)
        else:
            raise Error(failure)
        return
    same(query.collect(accelerator=runtime), expected)


def test_decimal_multiplicative_capability_proofs_and_metadata() raises:
    assert_true(not RowCapabilities().decimal_multiplicative)
    for expression in List[Expr]([col("a") * col("b"), col("a") / col("b")]):
        var query = safe_fixture().lazy().select(expression)
        var descriptors = _Descriptors(lower_rows(query, _capabilities()))
        assert_equal(descriptors.code[8], Int64(219))
        assert_equal(
            descriptors.code[11],
            Int64(
                1
                | 2 << 6
                | Int(expression._nodes[len(expression._nodes) - 1].op) << 12
            ),
        )
        var caps = _capabilities()
        caps.decimal_multiplicative = False
        with assert_raises(contains="unsupported"):
            _ = lower_rows(query, caps)
        assert_true(
            "unsupported" not in MetalRuntime(device_id=999999).describe(query)
        )
    var mixed_wide = DataFrame(
        [
            decimal_series("a", 128, 0, 1),
            decimal_series("b", 128, 1, 1),
        ]
    )
    with assert_raises(contains="static rescaling proof"):
        _ = lower_rows(
            mixed_wide.lazy().select(col("a") * col("b")), _capabilities()
        )


def test_native_metal_decimal_multiplication_division_all_scales_widths_nulls_and_slices() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var widths: List[Int] = [32, 64, 128, 128]
    var others: List[Int] = [64, 128, 32, 128]
    for pair in range(len(widths)):
        var left_max = (
            9 if widths[pair] == 32 else 18 if widths[pair] == 64 else 38
        )
        var right_max = (
            9 if others[pair] == 32 else 18 if others[pair] == 64 else 38
        )
        var magnitude = pow10(min(18, min(left_max, right_max))) - 1
        for scale in range(39):
            var common = min(scale, min(left_max, right_max))
            var frame = DataFrame(
                [
                    decimal_series("a", widths[pair], common, magnitude),
                    decimal_series("b", others[pair], common, magnitude),
                ]
            )
            var product = frame.lazy().select(col("a") * col("b"))
            same(
                product.collect(accelerator=runtime),
                product.collect(engine="cpu"),
            )
            var divided = DataFrame(
                [
                    decimal_series(
                        "a", widths[pair], min(scale, left_max), magnitude
                    ),
                    decimal_series(
                        "b",
                        others[pair],
                        min((scale * 7) % 39, right_max),
                        magnitude,
                    ),
                ]
            )
            parity(divided.lazy().select(col("a") / col("b")), runtime)
    for scale in range(39):
        var frame = DataFrame(
            [
                Series(
                    "a", Column[Int128]([25, 75, 1, -25, -75, 0])
                ).with_dtype(DataType.decimal(38, scale)),
                Series(
                    "b", Column[Int128]([200, 200, 3, 200, 200, 0])
                ).with_dtype(DataType.decimal(38, scale)),
            ]
        )
        var quotient = frame.lazy().select(col("a") / col("b"))
        same(
            quotient.collect(accelerator=runtime),
            quotient.collect(engine="cpu"),
        )
    for scale in range(39):
        var minimum_product = DataFrame(
            [
                Series("a", Column[Int128]([2, -2])).with_dtype(
                    DataType.decimal(38, scale)
                ),
                Series(
                    "b",
                    Column[Int128]([-(Int128(1) << 126), -(Int128(1) << 126)]),
                ).with_dtype(DataType.decimal(38, scale)),
            ]
        )
        parity(minimum_product.lazy().select(col("a") * col("b")), runtime)
    var frame = DataFrame(
        [
            Series("a", Column[Int128]([25, 75, -25, -75])).with_dtype(
                DataType.decimal(38, 2)
            ),
            Series("b", Column[Int128]([50, 50, 50, 50])).with_dtype(
                DataType.decimal(38, 2)
            ),
        ]
    )
    var product = frame.lazy().select(col("a") * col("b"))
    var result = product.collect(accelerator=runtime)
    var expected: List[Int128] = [12, 38, -12, -38]
    for i in range(4):
        assert_equal(result.column("a")._decimal_raw(i), expected[i])
    same(result, product.collect(engine="cpu"))
    var divisor = DataFrame(
        [
            frame.column("a"),
            Series("b", Column[Int128]([200, 200, 200, 200])).with_dtype(
                DataType.decimal(38, 2)
            ),
        ]
    )
    var quotient = divisor.lazy().select(col("a") / col("b"))
    result = quotient.collect(accelerator=runtime)
    for i in range(4):
        assert_equal(result.column("a")._decimal_raw(i), expected[i])
    same(result, quotient.collect(engine="cpu"))
    for length in List[Int]([0, 1, 3]):
        parity(
            divisor.slice(1, length).lazy().select(col("a") / col("b")), runtime
        )
    # Exercise wrapped remainder digits with a full-width denominator.
    for scale in List[Int]([0, 1, 18, 19, 37, 38]):
        var wide = DataFrame(
            [
                Series(
                    "a", Column[Int128]([pow10(38) - 2, -(pow10(38) - 2), 1])
                ).with_dtype(DataType.decimal(38, scale)),
                Series(
                    "b",
                    Column[Int128](
                        [pow10(38) - 1, pow10(38) - 1, pow10(38) - 1]
                    ),
                ).with_dtype(DataType.decimal(38, scale)),
            ]
        )
        parity(wide.lazy().select(col("a") / col("b")), runtime)


def test_native_metal_decimal_multiplicative_proven_resident_consumers() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var frame = safe_fixture()
    var query = (
        frame.lazy()
        .with_columns((col("a") * col("b")).alias("product"))
        .sort("product")
        .head(4)
    )
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    var grouped = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg((col("a") / col("b")).min().alias("q"))
    )
    same(grouped.collect(accelerator=runtime), grouped.collect(engine="cpu"))
    var window = frame.lazy().select((col("a") * col("b")).max().over("k"))
    same(window.collect(accelerator=runtime), window.collect(engine="cpu"))
    var right = DataFrame([Series("k", Column[Int32]([1, 0, 1]))])
    var joined = (
        frame.lazy()
        .with_columns((col("a") / col("b")).alias("q"))
        .join(right.lazy(), "k")
        .head(7)
    )
    same(joined.collect(accelerator=runtime), joined.collect(engine="cpu"))
    assert_true(
        "metal-decimal-multiplicative-v1"
        in runtime.estimate_auto(
            frame.lazy().select(col("a") / col("b"))
        ).family
    )


def test_native_metal_decimal_multiplicative_errors_conditionals_zero_divisor_and_reuse() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series("a", Column[Int128]([0, pow10(38) - 1])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("b", Column[Int128]([0, 2])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("p", Column[Bool]([True, False])),
        ]
    )
    var product = frame.lazy().select(col("a") * col("b"))
    with assert_raises(contains="decimal multiplication overflow"):
        _ = product.collect(engine="cpu")
    with assert_raises(contains="decimal multiplication overflow"):
        _ = product.collect(accelerator=runtime)
    with assert_raises(contains="checked decimal arithmetic"):
        _ = product.head(1).collect(accelerator=runtime)
    var masked = frame.lazy().select(
        when(col("p"))
        .then(col("a") * col("b"))
        .otherwise(lit("0").cast(DataType.decimal(38, 0)))
    )
    same(masked.collect(accelerator=runtime), masked.collect(engine="cpu"))
    var scaled = DataFrame(
        [
            Series("a", Column[Int128]([1])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("b", Column[Int128]([1])).with_dtype(
                DataType.decimal(38, 38)
            ),
        ]
    )
    var overflow = scaled.lazy().select(col("a") / col("b"))
    with assert_raises(contains="decimal precision"):
        _ = overflow.collect(engine="cpu")
    with assert_raises(contains="decimal precision"):
        _ = overflow.collect(accelerator=runtime)
    var zero = frame.slice(0, 1).lazy().select(col("a") / col("b"))
    var result = zero.collect(accelerator=runtime)
    assert_equal(result.column("a").null_count(), 1)
    same(result, zero.collect(engine="cpu"))
    with assert_raises(contains="budget"):
        _ = zero.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
    same(zero.collect(accelerator=runtime), zero.collect(engine="cpu"))
