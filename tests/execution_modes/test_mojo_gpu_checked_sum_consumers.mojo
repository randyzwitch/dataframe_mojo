"""Checked integer scalar totals finish before downstream consumers."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime, _capabilities
from dataframe.accelerator.joins import lower_join_dag, _checked_scalar_sum
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.aggregates import lower_aggregate_compositions
from tests.support.coverage_generators import offset_layout, chunk_layout


def _collect_sum(
    query: LazyFrame, gpu: Bool, optimize: Bool = False
) raises -> DataFrame:
    return query.collect(
        accelerator=MetalRuntime(),
        engine="accel",
        streaming=False,
        optimize=optimize,
    ) if gpu else query.collect(
        engine="cpu", streaming=False, optimize=optimize
    )


def _sum_source() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "v", Column[Int64]([2, 99, -1, 4], [True, False, True, True])
            ),
            Series("u", Column[UInt64]([UInt64.MAX, 1, 1, 0])),
        ]
    )


def test_mojo_gpu_checked_sum_consumers_literals_layouts_nulls_and_empty() raises:
    var source = _sum_source()
    for frame in [source.copy(), offset_layout(source), chunk_layout(source)]:
        var producer = frame.lazy().select_exprs(
            [
                col("v").sum().alias("total"),
                col("v").sum(4).alias("insufficient"),
                col("v").count().alias("count"),
            ]
        )
        var query = producer.with_columns(
            (col("total") + lit(Int64(2))).alias("next")
        )
        for gpu in [False, True]:
            var actual = _collect_sum(query, gpu)
            assert_equal(actual.height(), 1)
            assert_equal(actual.column("total").dtype(), DataType.INT64)
            assert_equal(actual.item(0, "total").int64(), Int64(5))
            assert_equal(actual.item(0, "next").int64(), Int64(7))
            assert_equal(actual.item(0, "count").int64(), Int64(3))
            assert_true(actual.item(0, "insufficient").is_null())
            assert_equal(
                _collect_sum(producer.filter(col("total") < 0), gpu).height(), 0
            )
            var joined = producer.join(producer, on="total", how="inner")
            assert_equal(
                _collect_sum(joined, gpu).item(0, "total").int64(), Int64(5)
            )
        var empty = (
            frame.head(0)
            .lazy()
            .select_exprs(
                [
                    col("v").sum().alias("total"),
                    col("v").sum(1).alias("missing"),
                ]
            )
            .with_columns((col("total") + lit(Int64(3))).alias("next"))
        )
        for gpu in [False, True]:
            var actual = _collect_sum(empty, gpu)
            assert_equal(actual.height(), 1)
            assert_equal(actual.item(0, "next").int64(), Int64(3))
            assert_true(actual.item(0, "missing").is_null())
    var wide = DataFrame(
        [
            Series("v", Column[Int64]([Int64.MAX, 1, -1])),
            Series("u", Column[UInt64]([UInt64.MAX, 1, 0])),
        ]
    )
    var signed = wide.lazy().select(col("v").sum().alias("total")).head(1)
    for gpu in [False, True]:
        assert_equal(
            _collect_sum(signed, gpu).item(0, "total").int64(), Int64.MAX
        )
    var unsigned = (
        DataFrame([Series("u", Column[UInt64]([UInt64.MAX, 0]))])
        .lazy()
        .select(col("u").sum())
        .head(1)
    )
    for gpu in [False, True]:
        var actual = _collect_sum(unsigned, gpu)
        assert_equal(actual.column("u").dtype(), DataType.UINT64)
        assert_equal(actual.item(0, "u").uint64(), UInt64.MAX)


def test_mojo_gpu_checked_sum_overflow_before_suppressed_consumers_and_reuse() raises:
    var frame = DataFrame([Series("v", Column[Int64]([Int64.MAX, 1]))])
    var producer = frame.lazy().select(col("v").sum().alias("total"))
    var right = DataFrame([Series("total", Column[Int64]([0]))]).lazy()
    for gpu in [False, True]:
        for optimize in [False, True]:
            with assert_raises(contains="overflow"):
                _ = _collect_sum(producer.head(0), gpu, optimize)
            if optimize:
                assert_equal(
                    _collect_sum(
                        producer.filter(lit(False)), gpu, optimize
                    ).height(),
                    0,
                )
            else:
                with assert_raises(contains="overflow"):
                    _ = _collect_sum(producer.filter(lit(False)), gpu, optimize)
            # The canonical CPU folder handles equality/Boolean operations;
            # it does not fold GT, so this predicate still observes the SUM.
            var constant_gt = producer.filter(lit(Int64(1)) > lit(Int64(2)))
            with assert_raises(contains="overflow"):
                _ = _collect_sum(constant_gt, gpu, optimize)
            with assert_raises(contains="overflow"):
                _ = _collect_sum(
                    producer.filter(col("total") < 0), gpu, optimize
                )
            with assert_raises(contains="overflow"):
                _ = _collect_sum(producer.filter(lit(True)), gpu, optimize)
            with assert_raises(contains="overflow"):
                _ = _collect_sum(
                    producer.join(right.head(0), on="total", how="inner"),
                    gpu,
                    optimize,
                )
    var unsigned = (
        DataFrame([Series("u", Column[UInt64]([UInt64.MAX, 1]))])
        .lazy()
        .select(col("u").sum())
        .head(0)
    )
    for gpu in [False, True]:
        with assert_raises(contains="overflow"):
            _ = _collect_sum(unsigned, gpu)
    var underflow = (
        DataFrame([Series("v", Column[Int64]([Int64.MIN, -1]))])
        .lazy()
        .select(col("v").sum())
        .filter(lit(False))
    )
    var hidden = (
        DataFrame(
            [Series("v", Column[Int64]([Int64.MAX, Int64.MAX], [False, False]))]
        )
        .lazy()
        .select(col("v").sum())
        .head(1)
    )
    var insufficient = frame.lazy().select(col("v").sum(3)).head(1)
    for gpu in [False, True]:
        with assert_raises(contains="overflow"):
            _ = _collect_sum(underflow, gpu)
        assert_equal(_collect_sum(hidden, gpu).item(0, "v").int64(), Int64(0))
        assert_true(_collect_sum(insufficient, gpu).item(0, "v").is_null())
    var runtime = MetalRuntime()
    with assert_raises(contains="overflow"):
        _ = producer.head(0).collect(
            accelerator=runtime, optimize=False, streaming=False
        )
    var good = _sum_source().lazy().select(col("v").sum()).head(1)
    assert_equal(
        good.collect(accelerator=runtime, optimize=False, streaming=False)
        .item(0, "v")
        .int64(),
        Int64(5),
    )


def test_mojo_gpu_checked_sum_conditional_counts_and_budget() raises:
    var frame = _sum_source()
    var counted = frame.lazy().select_exprs(
        [
            when(col("v") > 0)
            .then(lit(Int64(1)))
            .otherwise(lit(Int64(0)))
            .sum()
            .alias("positive"),
            when(col("v") < 0)
            .then(lit(Int64(1)))
            .otherwise(lit(Int64(0)))
            .sum()
            .alias("negative"),
        ]
    )
    var query = counted.select(
        (col("positive") + col("negative")).alias("total")
    )
    for gpu in [False, True]:
        assert_equal(
            _collect_sum(query, gpu).item(0, "total").int64(), Int64(3)
        )
    var measured = query.profile(
        accelerator=MetalRuntime(),
        engine="accel",
        optimize=False,
        streaming=False,
    )
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    var exact = query.collect(
        accelerator=MetalRuntime(memory_limit_bytes=peak),
        engine="accel",
        optimize=False,
        streaming=False,
    )
    assert_equal(exact.item(0, "total").int64(), Int64(3))
    with assert_raises(contains="memory budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
            optimize=False,
            streaming=False,
        )


def test_checked_sum_consumer_signatures_and_fallible_operand_guard() raises:
    var query = (
        _sum_source().lazy().select(col("v").sum().alias("total")).head(0)
    )
    assert_true(_checked_scalar_sum(query, 1))
    var dag = lower_join_dag(
        query._with_accelerator_settings(False, False, 65536), _capabilities()
    )
    var found = False
    for task in dag.tasks:
        ref node = task.query._nodes[len(task.query._nodes) - 1]
        if node.text != "metal_sum_materialization":
            continue
        found = True
        assert_true(task.terminal)
        var changed = task.query.copy()
        changed._nodes[len(changed._nodes) - 1].exprs[0] = (
            col("v").sum(2).alias("total")
        )
        with assert_raises(contains="metadata changed"):
            _ = lower_rows(changed, _capabilities(), terminal_boundary=True)
        with assert_raises(contains="invalid observed producer"):
            _ = lower_rows(task.query, _capabilities(), terminal_boundary=False)
    assert_true(found)
    # A later operand error must not preempt an earlier aggregate overflow.
    # This increment does not certify a global operand preparation phase for it.
    var fallible = (
        _sum_source()
        .lazy()
        .select_exprs(
            [
                col("v").sum().alias("first"),
                col("v").pow(lit(Int64(-1))).sum().alias("second"),
            ]
        )
        .head(0)
    )
    assert_true(not _checked_scalar_sum(fallible, 1))


def test_checked_sum_canonical_false_filter_precedes_boundary_capture() raises:
    var source = DataFrame([Series("v", Column[Int64]([Int64.MAX, 1]))])
    var query = source.lazy().select(col("v").sum()).filter(lit(False))
    var optimized = lower_join_dag(
        query._with_accelerator_settings(True, False, 65536), _capabilities()
    )
    assert_equal(len(optimized.tasks), 1)
    assert_equal(optimized.tasks[0].query._frames[0].height(), 0)
    assert_equal(optimized.tasks[0].schema.column("v").dtype(), DataType.INT64)
    var original = lower_join_dag(
        query._with_accelerator_settings(False, False, 65536), _capabilities()
    )
    var observed = False
    for task in original.tasks:
        ref node = task.query._nodes[len(task.query._nodes) - 1]
        observed |= node.text == "metal_sum_materialization" and task.terminal
    assert_true(observed)
    assert_equal(query._frames[0].height(), 2)


def test_mojo_gpu_checked_sum_scalar_when_and_numeric_mean_siblings() raises:
    # Numeric-to-Float64 casts are infallible siblings of checked SUM. The
    # scalar state leaves complete before the outer WHEN chooses its branch.
    for decimal in [False, True]:
        var columns = List[Series]()
        columns.append(Series("v", Column[Int64]([2, 4, 6])))
        columns.append(Series("amount", Column[Int64]([100, 200, 300])))
        var frame = DataFrame(columns^)
        if decimal:
            frame = frame.with_columns(
                col("amount").cast(DataType.decimal(18, 2))
            )
        var selected = (
            when(col("v") >= 4)
            .then(col("amount").cast(DataType.FLOAT64))
            .end()
            .mean()
        )
        var expression = (
            when(col("v").sum() > 10)
            .then(selected)
            .otherwise(lit(Float64(-1)))
            .alias("answer")
        )
        var query = frame.lazy().select(expression)
        var lowered = lower_aggregate_compositions(query, exact_float32=True)
        assert_true(_checked_scalar_sum(lowered, 1))
        for optimize in [False, True]:
            for gpu in [False, True]:
                assert_equal(
                    _collect_sum(query, gpu, optimize)
                    .item(0, "answer")
                    .float64(),
                    Float64(250),
                )
        var overflow = frame.with_columns(lit(Int64.MAX).alias("v"))
        var hidden = overflow.lazy().select(
            when(lit(False))
            .then(col("v").sum().cast(DataType.FLOAT64))
            .otherwise(selected)
            .alias("answer")
        )
        for optimize in [False, True]:
            for gpu in [False, True]:
                with assert_raises(contains="overflow"):
                    _ = _collect_sum(hidden, gpu, optimize)
                with assert_raises(contains="overflow"):
                    _ = _collect_sum(hidden.head(0), gpu, optimize)
