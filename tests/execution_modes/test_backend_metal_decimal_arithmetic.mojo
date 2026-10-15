"""Decimal row addition/subtraction and width-preserving unary operations."""
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, Expr, DataType, col, lit, when
from dataframe.lazy import LazyFrame
from dataframe.decimal import pow10
from dataframe.metal import MetalRuntime
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.execution_modes.test_backend_metal_decimal_comparisons import (
    decimal_series,
    safe_fixture,
)


def arithmetic() -> List[Expr]:
    return [
        (col("a") + col("b")).alias("add"),
        (col("a") - col("b")).alias("sub"),
    ]


def test_decimal_unary_cpu_storage_widths_and_arithmetic_descriptors() raises:
    assert_true(not RowCapabilities().decimal_arithmetic)
    for width in List[Int]([32, 64, 128]):
        var frame = DataFrame([decimal_series("a", width, 2, 999)])
        for expression in List[Expr]([-col("a"), col("a").abs()]):
            var result = frame.lazy().select(expression).collect(engine="cpu")
            assert_equal(result.column("a").dtype(), frame.column("a").dtype())
            assert_equal(
                result.column("a")._decimal_raw(1),
                Int128(-1) if expression._nodes[len(expression._nodes) - 1].op
                == 50 else Int128(1),
            )
    var minimum32 = DataFrame(
        [
            Series("a", Column[Int32]([Int32.MIN])).with_dtype(
                DataType.decimal(9, 0, 32)
            )
        ]
    )
    var minimum64 = DataFrame(
        [
            Series("a", Column[Int64]([Int64.MIN])).with_dtype(
                DataType.decimal(18, 0, 64)
            )
        ]
    )
    for expression in List[Expr]([-col("a"), col("a").abs()]):
        with assert_raises(contains="overflow"):
            _ = minimum32.lazy().select(expression).collect(engine="cpu")
        with assert_raises(contains="overflow"):
            _ = minimum64.lazy().select(expression).collect(engine="cpu")
    var narrow = Program(
        lower_rows(
            safe_fixture().lazy().select(col("a") + col("b")), _capabilities()
        )
    )
    assert_equal(narrow.code[8], Int64(219))
    assert_equal(narrow.literals[2], UInt64(18))
    var wide = DataFrame(
        [decimal_series("a", 128, 1, 99), decimal_series("b", 128, 2, 99)]
    )
    var query = wide.lazy().select(col("a") + col("b"))
    var descriptors = Program(lower_rows(query, _capabilities()))
    assert_equal(descriptors.code[8], Int64(219))
    assert_equal(descriptors.code[11], Int64(1 | 2 << 6 | 5 << 12))
    assert_equal(descriptors.literals[2], UInt64(38))
    var caps = _capabilities()
    caps.decimal_arithmetic = False
    with assert_raises(contains="logical"):
        _ = lower_rows(query, caps)


def test_native_metal_decimal_add_subtract_all_widths_scales_and_unary_metadata() raises:
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
        for scale in range(39):
            var left_scale = min(scale, left_max)
            var right_scale = min((scale * 7) % 39, right_max)
            var magnitude = (
                pow10(
                    min(
                        min(left_max, right_max),
                        max(37 - abs(left_scale - right_scale), 0),
                    )
                )
                - 1
            )
            var frame = DataFrame(
                [
                    decimal_series("a", widths[pair], left_scale, magnitude),
                    decimal_series("b", others[pair], right_scale, magnitude),
                ]
            )
            var query = frame.lazy().select_exprs(arithmetic())
            same(
                query.collect(accelerator=runtime), query.collect(engine="cpu")
            )
            var unary = frame.lazy().select_exprs(
                [(-col("a")).alias("neg"), col("a").abs().alias("abs")]
            )
            same(
                unary.collect(accelerator=runtime), unary.collect(engine="cpu")
            )
    var frame = safe_fixture()
    var query = frame.lazy().select_exprs(arithmetic())
    var result = query.collect(accelerator=runtime)
    assert_equal(result.column("add")._decimal_raw(0), Int128(200))
    assert_equal(result.column("sub")._decimal_raw(1), Int128(-1))
    for length in List[Int]([0, 1, 4]):
        var sliced = frame.slice(1, length).lazy().select_exprs(arithmetic())
        same(sliced.collect(accelerator=runtime), sliced.collect(engine="cpu"))


def test_native_metal_decimal_arithmetic_proven_consumers_aggregates_and_windows() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var frame = safe_fixture()
    var query = (
        frame.lazy()
        .with_columns((col("a") + col("b")).abs().alias("v"))
        .sort("v")
        .head(4)
    )
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    var grouped = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg((col("a") + col("b")).sum().alias("v"))
    )
    same(grouped.collect(accelerator=runtime), grouped.collect(engine="cpu"))
    var window = frame.lazy().select((col("a") - col("b")).max().over("k"))
    same(window.collect(accelerator=runtime), window.collect(engine="cpu"))
    var right = DataFrame([Series("k", Column[Int32]([1, 0, 1]))])
    var joined = (
        frame.lazy()
        .with_columns((col("a") + col("b")).alias("v"))
        .join(right.lazy(), "k")
        .head(7)
    )
    same(joined.collect(accelerator=runtime), joined.collect(engine="cpu"))
    assert_true(
        ";mojo-metal-v1;ops="
        in runtime.estimate_auto(frame.lazy().select_exprs(arithmetic())).family
    )


def test_native_metal_decimal_arithmetic_checked_errors_conditionals_budget_and_reuse() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var maximum = pow10(38) - 1
    var frame = DataFrame(
        [
            Series("a", Column[Int128]([0, maximum, -maximum])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("b", Column[Int128]([0, maximum, maximum])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("p", Column[Bool]([True, False, False])),
        ]
    )
    for expression in arithmetic():
        var query = frame.lazy().select(expression)
        with assert_raises(contains="decimal precision"):
            _ = query.collect(engine="cpu")
        with assert_raises(contains="decimal precision"):
            _ = query.collect(accelerator=runtime)
        same(
            query.head(1).collect(accelerator=runtime),
            query.head(1).collect(engine="cpu"),
        )
        var masked = frame.lazy().select(
            when(col("p"))
            .then(expression)
            .otherwise(lit("0").cast(DataType.decimal(38, 0)))
        )
        same(masked.collect(accelerator=runtime), masked.collect(engine="cpu"))
    var scaled = DataFrame(
        [
            frame.column("a"),
            frame.column("b").with_dtype(DataType.decimal(38, 1)),
        ]
    )
    var bad_scale = scaled.lazy().select(col("a") + col("b"))
    with assert_raises(contains="decimal multiplication overflow"):
        _ = bad_scale.collect(engine="cpu")
    with assert_raises(contains="decimal multiplication overflow"):
        _ = bad_scale.collect(accelerator=runtime)
    var minimum = DataFrame(
        [
            Series("a", Column[Int128]([Int128.MIN])).with_dtype(
                DataType.decimal(38, 0)
            )
        ]
    )
    for expression in List[Expr]([-col("a"), col("a").abs()]):
        with assert_raises(contains="overflow"):
            _ = minimum.lazy().select(expression).collect(engine="cpu")
        with assert_raises(contains="overflow"):
            _ = minimum.lazy().select(expression).collect(accelerator=runtime)
    var wrapped = DataFrame(
        [
            Series("a", Column[Int128]([Int128.MAX, -Int128.MAX])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("b", Column[Int128]([Int128.MAX, -Int128.MAX])).with_dtype(
                DataType.decimal(38, 0)
            ),
        ]
    )
    var wrapping = wrapped.lazy().select(col("a") + col("b"))
    var result = wrapping.collect(accelerator=runtime)
    assert_equal(result.column("a")._decimal_raw(0), Int128(-2))
    assert_equal(result.column("a")._decimal_raw(1), Int128(2))
    same(result, wrapping.collect(engine="cpu"))
    var valid = frame.slice(0, 1).lazy().select_exprs(arithmetic())
    with assert_raises(contains="budget"):
        _ = valid.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
    same(valid.collect(accelerator=runtime), valid.collect(engine="cpu"))


def test_native_metal_decimal_add_subtract_literal_result_types_and_boundaries() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series("a", Column[Int32]([125, -125, 0])).with_dtype(
                DataType.decimal(3, 2, 32)
            ),
            Series("b", Column[Int32]([75, 75, 0])).with_dtype(
                DataType.decimal(3, 1, 32)
            ),
            Series("k", Column[Int32]([0, 0, 1])),
        ]
    )
    var query = frame.lazy().select_exprs(arithmetic())
    var expected = DataFrame(
        [
            Series("add", Column[Int32]([875, 625, 0])).with_dtype(
                DataType.decimal(5, 2, 32)
            ),
            Series("sub", Column[Int32]([-625, -875, 0])).with_dtype(
                DataType.decimal(5, 2, 32)
            ),
        ]
    )
    same(query.collect(engine="cpu"), expected)
    same(query.collect(accelerator=runtime), expected)
    var grouped = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg((col("a") + col("b")).sum().alias("v"))
    )
    var grouped_expected = DataFrame(
        [
            Series("k", Column[Int32]([0, 1])),
            Series("v", Column[Int128]([1500, 0])).with_dtype(
                DataType.decimal(38, 2)
            ),
        ]
    )
    same(grouped.collect(engine="cpu"), grouped_expected)
    same(grouped.collect(accelerator=runtime), grouped_expected)
    # with_dtype admits physical values beyond the declared precision.
    # This checked producer must fail before a later ordering/head prunes it.
    var invalid = DataFrame(
        [
            Series("a", Column[Int32]([1, 10000])).with_dtype(
                DataType.decimal(3, 0, 32)
            ),
            Series("b", Column[Int32]([0, 1])).with_dtype(
                DataType.decimal(3, 0, 32)
            ),
        ]
    )
    var producer = invalid.lazy().with_columns((col("a") + col("b")).alias("v"))
    for failing in List[LazyFrame](
        [producer.sort("v").head(1), producer.sort("v").head(0)]
    ):
        with assert_raises(contains="decimal precision"):
            _ = failing.collect(engine="cpu")
        with assert_raises(contains="decimal precision"):
            _ = failing.collect(accelerator=runtime)
    # Optimized head precedes a row-local projection on both engines.
    var head = producer.head(1).select("v")
    var head_expected = DataFrame(
        [Series("v", Column[Int32]([1])).with_dtype(DataType.decimal(4, 0, 32))]
    )
    same(head.collect(engine="cpu"), head_expected)
    same(head.collect(accelerator=runtime), head_expected)
