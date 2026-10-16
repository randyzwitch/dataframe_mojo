"""Paired decimal storage executes through the direct Mojo GPU query path."""
from std.testing import assert_true, assert_equal, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.lazy import LazyFrame
from dataframe.decimal import pow10
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_decimal128 import (
    wide_fixture,
    sums_fixture,
    _sparse_decimal,
)
from tests.execution_modes.test_backend_metal_decimal_comparisons import (
    decimal_series,
)
from tests.public_contracts.test_duration_decimal_windows import (
    decimal_frame,
    decimal_expressions,
)
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.lifecycle_jobs import WorkerSetting


def _check(query: LazyFrame) raises:
    var expected = query.collect(engine="cpu", streaming=False)
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    if not actual.equals(expected):
        print("Decimal actual", actual)
        print("Decimal expected", expected)
    assert_true(actual.equals(expected))
    for name in expected.columns():
        if expected.column(name).dtype().is_decimal():
            for row in range(expected.height()):
                assert_equal(
                    actual.column(name)._decimal_valid(row),
                    expected.column(name)._decimal_valid(row),
                )
                if expected.column(name)._decimal_valid(row):
                    assert_equal(
                        actual.column(name)._decimal_raw(row),
                        expected.column(name)._decimal_raw(row),
                    )


# requires: mojo_gpu
def test_mojo_gpu_decimal_transport_expressions_casts_and_arithmetic() raises:
    for layout in range(3):
        var source = wide_fixture(257)
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        _check(
            frame.lazy().select_exprs(
                [
                    col("v"),
                    (-col("v")).alias("neg"),
                    col("v").abs().alias("abs"),
                    (col("v") + col("v")).alias("add"),
                    (col("v") - col("v")).alias("sub"),
                    col("v").cast(DataType.FLOAT64).alias("float"),
                    col("v")
                    .cast(DataType.decimal(38, 1), strict=False)
                    .alias("rescaled"),
                    when(col("k") < lit(Int32(3)))
                    .then(col("v"))
                    .otherwise(lit("-0.01").cast(DataType.decimal(38, 2)))
                    .alias("chosen"),
                    (
                        col("v") < lit("0.00").cast(DataType.decimal(38, 2))
                    ).alias("less"),
                ]
            )
        )
    for width in [32, 64, 128]:
        var frame = DataFrame(
            [
                decimal_series("a", width, 1, 999),
                decimal_series("b", width, 2, 789),
            ]
        )
        _check(
            frame.lazy().select_exprs(
                [
                    (col("a") + col("b")).alias("add"),
                    (col("a") - col("b")).alias("sub"),
                    (col("a") * col("a")).alias("product"),
                    (col("a") / col("a")).alias("quotient"),
                    (col("a") == col("b")).alias("equal"),
                    (col("a") >= col("b")).alias("greater"),
                    col("a").cast(DataType.FLOAT32).alias("float32"),
                    col("a")
                    .cast(DataType.INT32, strict=False)
                    .alias("integer"),
                ]
            )
        )

    # Selecting decimals with different widths or precisions widens to
    # Decimal128. Narrow negative selections must retain their sign word.
    var mixed = DataFrame(
        [
            Series("p", Column[Bool]([True, True, False, False])),
            Series(
                "a", Column[Int32]([-123, 1, -2, 0], [True, False, True, False])
            ).with_dtype(DataType.decimal(9, 1, 32)),
            Series("b", Column[Int64]([-456, -7, -8, -9])).with_dtype(
                DataType.decimal(18, 1, 64)
            ),
            Series("d", Column[Int64]([-45, -6, -7, -8])).with_dtype(
                DataType.decimal(17, 1, 64)
            ),
            Series(
                "c",
                Column[Int128](
                    [-789, -10, -11, -12], [False, True, True, False]
                ),
            ).with_dtype(DataType.decimal(38, 1)),
        ]
    )
    for source in [mixed.copy(), offset_layout(mixed), chunk_layout(mixed)]:
        var expressions = List[Expr]()
        for names in [
            List[String](["a", "b"]),
            List[String](["b", "c"]),
            List[String](["c", "a"]),
            List[String](["b", "d"]),
        ]:
            expressions.append(
                when(col("p"))
                .then(col(names[0]))
                .otherwise(col(names[1]))
                .alias("choose_" + names[0] + names[1])
            )
            expressions.append(
                col(names[0])
                .fill_null(col(names[1]))
                .alias("fill_" + names[0] + names[1])
            )
        _check(source.lazy().select_exprs(expressions))


# requires: mojo_gpu
def test_mojo_gpu_decimal_reductions_grouping_order_rank_and_join() raises:
    for n in [0, 1, 257, 65537]:
        var frame = wide_fixture(n)
        _check(
            frame.lazy().select_exprs(
                [
                    col("v").sum().alias("sum"),
                    col("v").sum(min_count=n + 1).alias("missing"),
                    col("v").mean().alias("mean"),
                    col("v").min().alias("min"),
                    col("v").max().alias("max"),
                    col("v").first().alias("first"),
                    col("v").last().alias("last"),
                ]
            )
        )
        _check(
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(
                [
                    col("v").sum().alias("sum"),
                    col("v").mean().alias("mean"),
                    col("v").min().alias("min"),
                    col("v").last().alias("last"),
                ]
            )
            .sort("sum")
        )
    var frame = chunk_layout(wide_fixture(257))
    _check(
        frame.lazy()
        .filter(col("k") < lit(Int32(6)))
        .sort("v", descending=True)
        .slice(-30, 17)
    )
    _check(
        frame.lazy()
        .group_by("v", maintain_order=True)
        .agg(col("k").count())
        .sort("v")
    )
    for method in ["average", "min", "max", "dense", "ordinal"]:
        _check(
            frame.lazy().select_exprs(
                [
                    col("v"),
                    col("v").rank(method).alias("rank"),
                    col("v").rank(method, True).over("k").alias("partitioned"),
                ]
            )
        )
    var right = wide_fixture(23).rename({"k": "id"})
    for how in ["inner", "left", "right", "full", "semi", "anti"]:
        _check(frame.lazy().join(right.lazy(), "v", how=how).sort("v"))
    var narrow = sums_fixture(257)
    _check(
        narrow.lazy().select_exprs(
            [
                col("a").sum().alias("a"),
                col("b").sum().alias("b"),
                col("a").mean().alias("mean"),
            ]
        )
    )
    _check(
        narrow.lazy()
        .group_by("k", maintain_order=True)
        .agg([col("a").sum().alias("a"), col("b").mean().alias("mean")])
    )


# requires: mojo_gpu
def test_mojo_gpu_decimal_windows_keep_both_words_and_precision() raises:
    var frame = wide_fixture(257)
    _check(
        frame.lazy().select_exprs(
            [
                col("v"),
                col("v").cum_sum().alias("sum"),
                col("v").cum_min().alias("min"),
                col("v").cum_max().alias("max"),
                col("v").shift(2).alias("shift"),
                col("v").forward_fill().alias("forward"),
                col("v").backward_fill().alias("backward"),
                col("v").rolling_min(7, min_samples=1).alias("rolling_min"),
                col("v").rolling_max(7, min_samples=1).alias("rolling_max"),
            ]
        )
    )
    var narrow = sums_fixture(257)
    _check(
        narrow.lazy().select_exprs(
            [
                col("a").cum_sum().alias("a"),
                col("b").cum_sum().alias("b"),
                col("a").cum_sum(True).alias("reverse"),
                col("a").cum_sum().over("k").alias("partitioned"),
            ]
        )
    )
    for width in [32, 64, 128]:
        var dtype = DataType.decimal(6, 3, width)
        _check(decimal_frame(dtype).lazy().select_exprs(decimal_expressions()))


def _check_decimal_over_worker_prefix() raises:
    # CPU contiguous OVER jobs give each 43,691-row group one worker.
    # Source-row workers would split this safe cancellation prefix and
    # overflow a partial containing +A,+A despite its final valid zero.
    var setting = WorkerSetting(2)
    var rows = 131073
    var amount = Int128(6) * pow10(37)
    var values = List[Int128](length=rows, fill=0)
    values[73725] = -amount
    values[73728] = amount
    values[73731] = amount
    values[73734] = -amount
    var keys = List[Int32](capacity=rows)
    for row in range(rows):
        keys.append(Int32(row % 3))
    var frame = DataFrame(
        [
            Series("x", Column[Int128](values^)).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("k", Column[Int32](keys^)),
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            col("x").sum().over("k").alias("sum"),
            col("x").mean().over("k").alias("mean"),
        ]
    )
    var expected = query.collect(engine="cpu", streaming=False)
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    assert_true(actual.equals(expected))
    for row in [0, 73725, 73728, rows - 1]:
        assert_equal(actual.column("sum")._decimal_raw(row), Int128(0))
        assert_equal(actual.item(row, "mean").float64(), Float64(0))
    _ = setting^


# requires: mojo_gpu
def test_mojo_gpu_decimal_prefix_merge_errors_and_cancellation() raises:
    _check_decimal_over_worker_prefix()
    for query in List[LazyFrame](
        [
            _sparse_decimal(3, 999, 1, 1).lazy().select(col("v").sum()),
            _sparse_decimal(131072, 600, 65536, 500)
            .lazy()
            .select(col("v").mean()),
            _sparse_decimal(131072, 600, 65536, 500, True)
            .lazy()
            .group_by("k")
            .agg(col("v").sum()),
        ]
    ):
        with assert_raises():
            _ = query.collect(engine="cpu", streaming=False)
        with assert_raises(contains="decimal precision"):
            _ = query.collect(accelerator=MetalRuntime(), engine="accel")
    var valid = _sparse_decimal(131072, 600, 65536, -500)
    _check(
        valid.lazy().select_exprs(
            [col("v").sum().alias("sum"), col("v").mean().alias("mean")]
        )
    )

    for width in [32, 64, 128]:
        for count in [0, 1, 8]:
            var source = DataFrame([decimal_series("v", width, 0, 999)]).head(
                count
            )
            for minimum in [
                Int(UInt32.MAX),
                Int(1) << 32,
                Int(1) << 33,
                Int.MAX,
            ]:
                _check(source.lazy().select(col("v").sum(min_count=minimum)))
                _check(
                    source.lazy()
                    .group_by("v", maintain_order=True)
                    .agg(col("v").sum(min_count=minimum).alias("sum"))
                )
    var maximum = pow10(38) - 1
    var overflow = DataFrame(
        [
            Series("a", Column[Int128]([maximum])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("b", Column[Int128]([maximum])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("two", Column[Int128]([2])).with_dtype(
                DataType.decimal(38, 0)
            ),
        ]
    )
    with assert_raises(contains="decimal precision"):
        _ = (
            overflow.lazy()
            .select(col("a") + col("b"))
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
    with assert_raises(contains="decimal multiplication overflow"):
        _ = (
            overflow.lazy()
            .select(col("a") * col("two"))
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
    with assert_raises(contains="strict cast"):
        _ = (
            overflow.lazy()
            .select(col("a").cast(DataType.INT32))
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
    var unary = DataFrame(
        [
            Series("v", Column[Int128]([Int128(1) << 127])).with_dtype(
                DataType.decimal(38, 0)
            )
        ]
    )
    with assert_raises(contains="integer overflow"):
        _ = (
            unary.lazy()
            .select(-col("v"))
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
