"""Exact two-word decimal transport and Decimal32/64 sums on Metal."""
from std.testing import assert_true, assert_equal, assert_raises
from tests.execution_modes.test_backend_metal import require_metal
from dataframe import DataFrame, Series, Column, Expr, col, lit, when
from dataframe.dtype import DataType
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime, _capabilities
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.cpu_work import cpu_features
from dataframe.accelerator.constants import fold_literal_text_parsing


def check(query: LazyFrame) raises:
    var actual = query.collect(accelerator=MetalRuntime())
    var expected = query.collect(engine="cpu")
    if not actual.equals(expected):
        print("Decimal128 actual:", actual)
        print("Decimal128 expected:", expected)
    assert_true(actual.equals(expected))
    for name in expected.columns():
        var dtype = expected.column(name).dtype()
        if dtype.is_decimal() and dtype.decimal_width() == 128:
            var a = actual.column(name).numeric[DType.int128]()
            var b = expected.column(name).numeric[DType.int128]()
            for i in range(expected.height()):
                assert_equal(a.is_valid(i), b.is_valid(i))
                if b.is_valid(i):
                    assert_equal(a.value(i), b.value(i))


def sums_fixture(n: Int, sign: Int = 1) raises -> DataFrame:
    var large = List[Int64]()
    var small = List[Int32]()
    var keys = List[Int32]()
    var valid = List[Bool]()
    var key_valid = List[Bool]()
    for i in range(n):
        var direction = sign if sign else (1 if i % 3 else -1)
        large.append(
            Int64(direction) * (Int64(999999999999999999) - Int64(i % 17))
        )
        small.append(Int32(direction) * (Int32(999999999) - Int32(i % 17)))
        keys.append(Int32(i % 11))
        valid.append(i % 7 != 2)
        key_valid.append(i % 13 != 3)
    return DataFrame(
        [
            Series("k", Column[Int32](keys^, key_valid^)),
            Series("a", Column[Int64](large^, valid)).with_dtype(
                DataType.decimal(18, 2, 64)
            ),
            Series("b", Column[Int32](small^, valid^)).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
        ]
    )


def wide_fixture(n: Int = 513) raises -> DataFrame:
    var raw = List[Int128]()
    var keys = List[Int32]()
    var valid = List[Bool]()
    for i in range(n):
        var bucket = i % 19
        # Equal low words with different high words, plus nearby low words,
        # exercise both halves of signed ordering and equality.
        var value = (Int128(bucket % 5) << 80) | Int128(bucket % 3)
        if bucket % 2:
            value = -value
        raw.append(value)
        keys.append(Int32(i % 7))
        valid.append(i % 13 != 0)
    return DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("v", Column[Int128](raw^, valid^)).with_dtype(
                DataType.decimal(38, 2)
            ),
        ]
    )


def test_native_metal_decimal128_literal_preflight_and_remaining_boundaries() raises:
    _ = require_metal()
    var frame = wide_fixture(3)
    for value in List[String](
        [
            "0.00",
            "-0.01",
            "184467440737095516.16",
            "-368934881474191032.32",
            "999999999999999999999999999999999999.99",
            "-999999999999999999999999999999999999.99",
        ]
    ):
        var query = frame.lazy().select_exprs(
            [
                col("k"),
                lit(value).cast(DataType.decimal(38, 2)).alias("literal"),
            ]
        )
        var folded = fold_literal_text_parsing(query)
        assert_true(
            folded.collect(engine="cpu").equals(query.collect(engine="cpu"))
        )
        _ = lower_rows(folded, _capabilities())
        check(query)
    var query = sums_fixture(3).lazy().select(col("a").sum())
    var plan = lower_rows(query, _capabilities())
    assert_equal(plan.outputs[0].dtype, DataType.decimal(38, 2))
    with assert_raises(contains="Decimal128 input reductions"):
        _ = lower_rows(frame.lazy().select(col("v").sum()), _capabilities())
    with assert_raises(contains="Decimal128 row arithmetic"):
        _ = lower_rows(
            frame.lazy().select(col("v") + col("v")), _capabilities()
        )
    with assert_raises(contains="Decimal128 input reductions"):
        _ = lower_rows(frame.lazy().select(col("v").arg_min()), _capabilities())
    var bad = frame.lazy().select(lit("1000.00").cast(DataType.decimal(3, 2)))
    with assert_raises(contains="literal text parsing"):
        _ = fold_literal_text_parsing(bad)


def test_native_metal_decimal_wide_scalar_sums_exact_extremes_nulls_and_empty() raises:
    _ = require_metal()
    for n in List[Int]([0, 1, 257, 4097]):
        for sign in List[Int]([-1, 0, 1]):
            var frame = sums_fixture(n, sign)
            check(
                frame.lazy().select_exprs(
                    [
                        col("a").sum().alias("a"),
                        col("b").sum().alias("b"),
                        col("a").sum(min_count=1).alias("nonempty"),
                        col("b").sum(min_count=n + 1).alias("missing"),
                        col("k").count().alias("count"),
                        col("k").mean().alias("mean"),
                    ]
                )
            )
            check(
                frame.lazy()
                .filter(col("k") < lit(Int32(5)))
                .select(col("a").sum())
            )
            check(frame.lazy().sort("k").slice(-100, 17).select(col("a").sum()))
    var nulls = DataFrame(
        [
            Series(
                "a", Column[Int64]([1, 2, 3], [False, False, False])
            ).with_dtype(DataType.decimal(18, 18, 64))
        ]
    )
    check(
        nulls.lazy().select_exprs(
            [col("a").sum(), col("a").sum(min_count=1).alias("missing")]
        )
    )


def test_native_metal_decimal_wide_grouped_sums_sort_filter_and_resident_copy() raises:
    _ = require_metal()
    for n in List[Int]([0, 257, 4097]):
        var frame = sums_fixture(n)
        for maintain_order in List[Bool]([True, False]):
            var grouped = (
                frame.lazy()
                .group_by("k", maintain_order=maintain_order)
                .agg(
                    [
                        col("a").sum().alias("a"),
                        col("b").sum(min_count=1).alias("b"),
                        col("a").sum(min_count=n + 1).alias("missing"),
                        col("k").count().alias("count"),
                    ]
                )
            )
            if not maintain_order:
                grouped = grouped.sort("k")
            check(grouped)
            check(
                grouped.sort("a", descending=True)
                .slice(1, 7)
                .select_exprs(
                    [
                        col("k"),
                        col("a"),
                        when(col("k").is_null())
                        .then(col("a"))
                        .otherwise(col("b"))
                        .alias("chosen"),
                    ]
                )
            )
            check(
                grouped.filter(
                    col("a") > lit("100.00").cast(DataType.decimal(38, 2))
                ).head(3)
            )
            check(
                grouped.select_exprs(
                    [col("a").count(), col("b").null_count().alias("missing")]
                )
            )
    # Hash grouping and restoration to public group order keep both words.
    var many = sums_fixture(65537)
    check(
        many.lazy().group_by("k").agg(col("a").sum()).sort("a", descending=True)
    )


def test_native_metal_decimal128_transport_comparisons_order_rank_and_keys() raises:
    _ = require_metal()
    for n in List[Int]([0, 1, 513]):
        var frame = wide_fixture(n)
        check(frame.lazy().select_exprs([col("v"), col("k")]))
        for descending in List[Bool]([False, True]):
            check(
                frame.lazy()
                .filter(col("k") < lit(Int32(6)))
                .sort("v", descending=descending)
                .slice(-30, 17)
            )
            for method in List[String](
                ["average", "min", "max", "dense", "ordinal"]
            ):
                check(
                    frame.lazy().select_exprs(
                        [
                            col("v"),
                            col("v").rank(method, descending).alias("rank"),
                            col("v")
                            .rank(method, descending)
                            .over("k")
                            .alias("partitioned"),
                        ]
                    )
                )
        check(
            frame.lazy().select_exprs(
                [
                    col("v"),
                    when(col("k") < lit(Int32(3)))
                    .then(col("v"))
                    .otherwise(col("v"))
                    .alias("chosen"),
                    (
                        col("v") == lit("0.00").cast(DataType.decimal(38, 2))
                    ).alias("equal"),
                    (
                        col("v") != lit("0.00").cast(DataType.decimal(38, 2))
                    ).alias("different"),
                    (
                        col("v") < lit("-0.01").cast(DataType.decimal(38, 2))
                    ).alias("less"),
                    (
                        col("v") <= lit("0.00").cast(DataType.decimal(38, 2))
                    ).alias("less_equal"),
                    (
                        col("v") >= lit("0.00").cast(DataType.decimal(38, 2))
                    ).alias("greater_equal"),
                ]
            )
        )
        check(
            frame.lazy()
            .group_by("v", maintain_order=True)
            .agg(col("k").count())
        )
        check(frame.lazy().group_by("v").agg(col("k").count()).sort("v"))
    # Large keys use hash grouping; both words participate in hashing/equality.
    var hashed = wide_fixture(4097)
    check(hashed.lazy().group_by("v").agg(col("k").count()).sort("v"))
    check(
        hashed.lazy().select_exprs(
            [col("v"), col("k").rank("dense").over("v").alias("rank")]
        )
    )
    var narrow = sums_fixture(17, 0)
    check(
        narrow.lazy().select_exprs(
            [
                col("k"),
                (col("a") > lit("-0.01").cast(DataType.decimal(38, 2))).alias(
                    "mixed"
                ),
                (lit("0.00").cast(DataType.decimal(38, 2)) >= col("a")).alias(
                    "reverse"
                ),
            ]
        )
    )


def test_native_metal_decimal128_joins_payloads_keys_and_scalar_broadcast() raises:
    _ = require_metal()
    var frame = wide_fixture(37)
    var right = wide_fixture(23).rename({"v": "w"})
    for how in List[String](["inner", "left", "right", "full", "semi", "anti"]):
        check(frame.lazy().join(right.lazy(), "k", how=how).sort("k"))
    var same_keys = wide_fixture(23).rename({"k": "id"})
    check(frame.lazy().join(same_keys.lazy(), "v", how="full").sort("v"))
    # Large one-to-one keys exercise the hash-join path without fanout.
    var words = List[Int128]()
    var ids = List[Int32]()
    var valid = List[Bool]()
    for i in range(4097):
        var word = (Int128(i % 37) << 80) | Int128(i)
        words.append(-word if i % 2 else word)
        ids.append(Int32(i))
        valid.append(i % 31 != 0)
    var hashed = DataFrame(
        [
            Series("v", Column[Int128](words^, valid^)).with_dtype(
                DataType.decimal(38, 2)
            ),
            Series("id", Column[Int32](ids^)),
        ]
    )
    var other = hashed.rename({"id": "other"})
    for how in List[String](["inner", "full"]):
        check(
            hashed.lazy().join(other.lazy(), "v", how=how).sort(["id", "other"])
        )
    var sums = sums_fixture(257)
    check(sums.lazy().with_columns(col("a").sum().alias("total")))
    check(
        sums.lazy()
        .select_exprs([col("k"), col("a").sum().alias("total")])
        .head(7)
    )
    var grouped = sums.lazy().group_by("k").agg(col("a").sum().alias("total"))
    var keys = DataFrame([Series("k", Column[Int32]([0, 1, 2, 10, 15]))])
    check(grouped.join(keys.lazy(), "k", how="full").sort("k"))
    check(
        sums.slice(0, 0)
        .lazy()
        .with_columns(col("a").sum(min_count=1).alias("total"))
    )


def test_native_metal_decimal128_extrema_first_last_and_resident_reductions() raises:
    _ = require_metal()
    for n in List[Int]([0, 1, 513]):
        var frame = wide_fixture(n)
        var expressions = List[Expr](
            [
                col("v").min().alias("min"),
                col("v").max().alias("max"),
                col("v").first().alias("first"),
                col("v").last().alias("last"),
                col("v").count().alias("count"),
                col("v").null_count().alias("nulls"),
            ]
        )
        check(frame.lazy().select_exprs(expressions))
        check(frame.lazy().group_by("k", maintain_order=True).agg(expressions))
    var frame = sums_fixture(257)
    var grouped = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(col("a").sum().alias("total"))
    )
    check(
        grouped.select_exprs(
            [col("total").min(), col("total").max().alias("max")]
        )
    )
    check(
        frame.lazy()
        .with_columns(col("a").sum().alias("total"))
        .group_by("k", maintain_order=True)
        .agg(col("total").first())
    )


def test_native_metal_decimal128_cost_memory_budget_and_literal_pipeline_cache() raises:
    var frame = wide_fixture(17)
    var query = frame.lazy().select_exprs([col("k"), col("v")])
    var work = cpu_features(query)
    assert_equal(work[0][2], Float64(17 * 20))
    assert_true("decimal128-v1" in work[1])
    _ = require_metal()
    var runtime = MetalRuntime()
    var estimate = runtime.estimate_auto(query)
    assert_equal(estimate.cpu[2], Float64(17 * 20))
    assert_true("metal-decimal128-v1" in estimate.family)
    var measured = query.profile(accelerator=runtime)
    assert_true(measured[0].equals(query.collect(engine="cpu")))
    assert_equal(
        measured[1].column("peak_requested_bytes").int64()._get(0),
        Int64(estimate.peak_bytes),
    )
    with assert_raises(contains="memory budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
    check(query)
    var literals = DataFrame([Series("k", Column[Int32]([1, 2, 3]))])
    var first = literals.lazy().select_exprs(
        [
            col("k"),
            lit("184467440737095516.16")
            .cast(DataType.decimal(38, 2))
            .alias("literal"),
        ]
    )
    var second = literals.lazy().select_exprs(
        [
            col("k"),
            lit("368934881474191032.32")
            .cast(DataType.decimal(38, 2))
            .alias("literal"),
        ]
    )
    check(first)
    var folded = fold_literal_text_parsing(second)
    assert_equal(runtime.estimate_auto(folded).accelerator[13], Float64(0))
    check(second)
