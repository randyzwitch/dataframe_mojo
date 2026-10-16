"""Exact decimal comparisons across scales and physical storage widths."""
# contract-group: execution.backends
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, Expr, DataType, col, lit, when
from dataframe.decimal import pow10
from dataframe.metal import MetalRuntime
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_float64_emulation import same


def comparisons() -> List[Expr]:
    return [
        (col("a") > col("b")).alias("gt"),
        (col("a") == col("b")).alias("eq"),
        (col("a") < col("b")).alias("lt"),
        (col("a") >= col("b")).alias("ge"),
        (col("a") <= col("b")).alias("le"),
        (col("a") != col("b")).alias("ne"),
    ]


def decimal_series(
    name: String, width: Int, scale: Int, magnitude: Int128
) raises -> Series:
    var raw: List[Int128] = [0, 1, -1, magnitude, -magnitude, 0, 1, -1]
    var valid: List[Bool] = [True, True, True, True, True, False, True, False]
    if width == 32:
        var values = List[Int32]()
        for value in raw:
            values.append(Int32(value))
        return Series(name, Column[Int32](values^, valid^)).with_dtype(
            DataType.decimal(9, scale, 32)
        )
    if width == 64:
        var values = List[Int64]()
        for value in raw:
            values.append(Int64(value))
        return Series(name, Column[Int64](values^, valid^)).with_dtype(
            DataType.decimal(18, scale, 64)
        )
    return Series(name, Column[Int128](raw^, valid^)).with_dtype(
        DataType.decimal(38, scale)
    )


def safe_fixture() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "a",
                Column[Int32](
                    [10, -10, 0, 1, 100, -100],
                    [True, True, False, True, True, True],
                ),
            ).with_dtype(DataType.decimal(9, 1, 32)),
            Series(
                "b", Column[Int64]([100, -99, 0, 9, 1001, -1000])
            ).with_dtype(DataType.decimal(18, 2, 64)),
            Series("k", Column[Int32]([0, 1, 0, 1, 0, 1])),
        ]
    )


def test_decimal_comparison_capability_and_packed_metadata() raises:
    assert_true(not RowCapabilities().decimal_comparisons)
    var query = safe_fixture().lazy().select(col("a") < col("b"))
    var plan = lower_rows(query, _capabilities())
    var descriptors = Program(plan)
    assert_equal(descriptors.code[8], Int64(218))
    assert_equal(descriptors.code[11], Int64(1 | (2 << 6) | (20 << 12)))
    var caps = _capabilities()
    caps.decimal_comparisons = False
    with assert_raises(contains="matching scales"):
        _ = lower_rows(query, caps)
    assert_true(
        "unsupported" not in MetalRuntime(device_id=999999).describe(query)
    )


# requires: metal
def test_native_metal_decimal_comparisons_width_scale_matrix_and_exact_words() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var widths: List[Int] = [32, 64, 128, 128]
    var other_widths: List[Int] = [64, 128, 32, 128]
    for pair in range(len(widths)):
        var left_width = widths[pair]
        var right_width = other_widths[pair]
        for scale in range(39):
            var left_max = (
                9 if left_width == 32 else 18 if left_width == 64 else 38
            )
            var right_max = (
                9 if right_width == 32 else 18 if right_width == 64 else 38
            )
            var left_scale = min(scale, left_max)
            var right_scale = min((scale * 7) % 39, right_max)
            var delta = abs(left_scale - right_scale)
            var magnitude = pow10(min(min(left_max, right_max), 38 - delta)) - 1
            var frame = DataFrame(
                [
                    decimal_series("a", left_width, left_scale, magnitude),
                    decimal_series("b", right_width, right_scale, magnitude),
                ]
            )
            var query = frame.lazy().select_exprs(comparisons())
            if (
                left_width == 64
                and right_width == 128
                and right_scale < left_scale
            ):
                with assert_raises(contains="static proof"):
                    _ = lower_rows(query, _capabilities())
                continue
            same(
                query.collect(accelerator=runtime), query.collect(engine="cpu")
            )
    var wide = DataFrame(
        [
            Series(
                "a",
                Column[Int128](
                    [
                        pow10(38) - 1,
                        (Int128(1) << 100) + 1,
                        -(Int128(1) << 100),
                        0,
                    ]
                ),
            ).with_dtype(DataType.decimal(38, 17)),
            Series(
                "b",
                Column[Int128](
                    [
                        pow10(38) - 2,
                        Int128(1) << 100,
                        -(Int128(1) << 100) + 1,
                        0,
                    ]
                ),
            ).with_dtype(DataType.decimal(38, 17)),
        ]
    )
    var query = wide.lazy().select_exprs(comparisons())
    var result = query.collect(accelerator=runtime)
    same(result, query.collect(engine="cpu"))
    assert_true(result.column("gt").bool()._get(0))
    assert_true(result.column("gt").bool()._get(1))
    assert_true(result.column("lt").bool()._get(2))
    assert_true(result.column("eq").bool()._get(3))
    for length in List[Int]([0, 1, 3]):
        var sliced = wide.slice(1, length).lazy().select_exprs(comparisons())
        same(sliced.collect(accelerator=runtime), sliced.collect(engine="cpu"))


# requires: metal
def test_native_metal_decimal_comparisons_resident_consumers_and_conditionals() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var frame = safe_fixture()
    var query = (
        frame.lazy()
        .with_columns((col("a") < col("b")).alias("less"))
        .filter(col("less"))
        .sort("a")
        .head(2)
    )
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    var conditional = frame.lazy().select(
        when(col("k") == lit(Int32(0)))
        .then(col("a") == col("b"))
        .otherwise(col("a") > col("b"))
    )
    same(
        conditional.collect(accelerator=runtime),
        conditional.collect(engine="cpu"),
    )
    var grouped = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg((col("a") < col("b")).cast(DataType.INT64).sum().alias("n"))
    )
    same(grouped.collect(accelerator=runtime), grouped.collect(engine="cpu"))
    var window = frame.lazy().select(
        (col("a") < col("b")).cast(DataType.INT64).sum().over("k")
    )
    same(window.collect(accelerator=runtime), window.collect(engine="cpu"))
    var right = DataFrame([Series("k", Column[Int32]([1, 0, 1]))])
    var joined = (
        frame.lazy().join(right.lazy(), "k").select_exprs(comparisons())
    )
    same(joined.collect(accelerator=runtime), joined.collect(engine="cpu"))
    assert_true(
        ";mojo-metal-v1;ops="
        in runtime.estimate_auto(
            frame.lazy().select_exprs(comparisons())
        ).family
    )


# requires: metal
def test_native_metal_decimal_comparison_overflow_observation_budget_and_reuse() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series("a", Column[Int128]([0, pow10(38) - 1])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("b", Column[Int128]([0, 1])).with_dtype(
                DataType.decimal(38, 38)
            ),
        ]
    )
    var query = frame.lazy().select(col("a") > col("b"))
    with assert_raises(contains="decimal multiplication overflow"):
        _ = query.collect(engine="cpu")
    with assert_raises(contains="decimal multiplication overflow"):
        _ = query.collect(accelerator=runtime)
    with assert_raises(contains="checked decimal comparisons"):
        _ = query.head(1).collect(accelerator=runtime)
    for predicate in List[Expr]([lit(False), lit(True)]):
        var conditional = frame.lazy().select(
            when(predicate).then(col("a") > col("b")).otherwise(lit(False))
        )
        with assert_raises(contains="checked decimal comparisons"):
            _ = conditional.collect(accelerator=runtime, optimize=False)
    var filtered = (
        frame.lazy()
        .filter(col("a") == lit("0").cast(DataType.decimal(38, 0)))
        .select(col("a") > col("b"))
    )
    same(filtered.collect(accelerator=runtime), filtered.collect(engine="cpu"))
    var valid = frame.slice(0, 1).lazy().select(col("a") > col("b"))
    with assert_raises(contains="budget"):
        _ = valid.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
    same(valid.collect(accelerator=runtime), valid.collect(engine="cpu"))
