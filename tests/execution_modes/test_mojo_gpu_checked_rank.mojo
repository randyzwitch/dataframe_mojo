"""Checked eager rank inputs are observed before rank sorting and consumers."""
# contract-group: execution.backends
# contract-group: execution.apple_parity
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.metal import MetalRuntime, _capabilities
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.gpu.runtime import estimate_resources
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.lifecycle_jobs import WorkerSetting
from tests.tooling.checked_rank_cpu_proofs import ordered_source


def _rank_source() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "x",
                Column[Int64]([3, 1, 1, 9, 2], [True, True, True, False, True]),
            ),
            Series("k", Column[Int32]([0, 0, 0, 0, 1])),
        ]
    )


def _integer_words(method: String, partitioned: Bool) -> List[Int64]:
    var expected: List[Int64] = [3, 1, 2, 0, 1] if partitioned else [
        4,
        1,
        2,
        0,
        3,
    ]
    if method == "min":
        expected = [3, 1, 1, 0, 1] if partitioned else [4, 1, 1, 0, 3]
    elif method == "max":
        expected = [3, 2, 2, 0, 1] if partitioned else [4, 2, 2, 0, 3]
    elif method == "dense":
        expected = [2, 1, 1, 0, 1] if partitioned else [3, 1, 1, 0, 2]
    return expected^


# requires: metal
def test_mojo_gpu_checked_rank_literal_methods_nulls_layouts_and_partitions() raises:
    var setting = WorkerSetting(1)
    var source = _rank_source()
    for frame in [source.copy(), offset_layout(source), chunk_layout(source)]:
        for method in ["min", "max", "dense", "ordinal", "average"]:
            for partitioned in [False, True]:
                var ranked = (col("x") + lit(Int64(1))).rank(method)
                if partitioned:
                    ranked = ranked.over("k")
                var query = frame.lazy().select(ranked.alias("r"))
                var integers = _integer_words(method, partitioned)
                var average: List[Float64] = [
                    3,
                    1.5,
                    1.5,
                    0,
                    1,
                ] if partitioned else [4, 1.5, 1.5, 0, 3]
                for gpu in [False, True]:
                    var actual = query.collect(
                        accelerator=MetalRuntime(),
                        streaming=False,
                        optimize=False,
                    ) if gpu else query.collect(
                        engine="cpu", streaming=False, optimize=False
                    )
                    assert_equal(actual.height(), 5)
                    for row in range(5):
                        if row == 3:
                            assert_true(actual.item(row, "r").is_null())
                        elif method == "average":
                            assert_equal(
                                actual.item(row, "r").float64(), average[row]
                            )
                        else:
                            assert_equal(
                                actual.item(row, "r").int64(), integers[row]
                            )
        var descending = frame.lazy().select(
            (col("x") + lit(Int64(1))).rank("min", descending=True).alias("r")
        )
        var expected_descending: List[Int64] = [1, 3, 3, 0, 2]
        var filtered = frame.lazy().filter(
            (col("x") + lit(Int64(1))).rank("min") <= 2
        )
        for gpu in [False, True]:
            var actual = descending.collect(
                accelerator=MetalRuntime(), streaming=False, optimize=False
            ) if gpu else descending.collect(
                engine="cpu", streaming=False, optimize=False
            )
            for row in range(5):
                if row == 3:
                    assert_true(actual.item(row, "r").is_null())
                else:
                    assert_equal(
                        actual.item(row, "r").int64(), expected_descending[row]
                    )
            var selected = filtered.collect(
                accelerator=MetalRuntime(), streaming=False, optimize=False
            ) if gpu else filtered.collect(
                engine="cpu", streaming=False, optimize=False
            )
            assert_equal(selected.height(), 2)
            assert_equal(selected.item(0, "x").int64(), Int64(1))
            assert_equal(selected.item(1, "x").int64(), Int64(1))
    var empty = (
        source.head(0).lazy().select((col("x") + lit(Int64(1))).rank("min"))
    )
    for gpu in [False, True]:
        var actual = empty.collect(
            accelerator=MetalRuntime(), streaming=False, optimize=False
        ) if gpu else empty.collect(
            engine="cpu", streaming=False, optimize=False
        )
        assert_equal(actual.height(), 0)
    _ = setting^


# requires: metal
def test_mojo_gpu_checked_rank_operand_batch_when_errors_and_reuse() raises:
    var setting = WorkerSetting(1)
    var runtime = MetalRuntime()
    for fixture_index in range(2):
        var frame = ordered_source(
            2050 if fixture_index == 0 else 10000,
            1500 if fixture_index == 0 else 9000,
        )
        var operand = (col("x") + col("y")).pow(col("p"))
        var expected = (
            "overflow" if fixture_index == 0 else "nonnegative exponent"
        )
        for partitioned in [False, True]:
            var ranked = operand.rank("min").over(
                "k"
            ) if partitioned else operand.rank("min")
            for hidden in [False, True]:
                var expression = (
                    when(lit(False))
                    .then(ranked)
                    .otherwise(lit(Int64(7))) if hidden else ranked.copy()
                )
                for gpu in [False, True]:
                    var query = frame.lazy().select(expression.alias("r"))
                    with assert_raises(contains=expected):
                        _ = query.collect(
                            accelerator=runtime, streaming=False, optimize=False
                        ) if gpu else query.collect(
                            engine="cpu", streaming=False, optimize=False
                        )
                    with assert_raises(contains=expected):
                        _ = (
                            query.head(0)
                            .collect(
                                accelerator=runtime,
                                streaming=False,
                                optimize=False,
                            ) if gpu else query.head(0)
                            .collect(
                                engine="cpu", streaming=False, optimize=False
                            )
                        )
        var ranked_filter = frame.lazy().filter(operand.rank("min") > 0).head(0)
        for gpu in [False, True]:
            with assert_raises(contains=expected):
                _ = ranked_filter.collect(
                    accelerator=runtime, streaming=False, optimize=False
                ) if gpu else ranked_filter.collect(
                    engine="cpu", streaming=False, optimize=False
                )
        var masked = when(col("p") >= 0).then(operand).otherwise(lit(Int64(0)))
        for gpu in [False, True]:
            with assert_raises(contains="overflow"):
                _ = (
                    frame.lazy()
                    .select(masked.rank("dense"))
                    .collect(
                        accelerator=runtime, streaming=False, optimize=False
                    ) if gpu else frame.lazy()
                    .select(masked.rank("dense"))
                    .collect(engine="cpu", streaming=False, optimize=False)
                )
    for gpu in [False, True]:
        var invalid = (
            _rank_source().lazy().select((col("x") % lit(Int64(0))).rank("min"))
        )
        # Integer modulo by zero is null, so every rank remains null.
        var actual_nulls = invalid.collect(
            accelerator=runtime, streaming=False, optimize=False
        ) if gpu else invalid.collect(
            engine="cpu", streaming=False, optimize=False
        )
        assert_equal(actual_nulls.height(), 5)
        assert_equal(actual_nulls.column("x").dtype(), DataType.INT64)
        for row in range(5):
            assert_true(actual_nulls.item(row, "x").is_null())
    var bad_cast = (
        DataFrame([Series("x", Column[Int64]([Int64.MAX, 1]))])
        .lazy()
        .select(col("x").cast("int32").rank("min"))
        .head(0)
    )
    for gpu in [False, True]:
        with assert_raises(contains="cast"):
            _ = bad_cast.collect(
                accelerator=runtime, streaming=False, optimize=False
            ) if gpu else bad_cast.collect(
                engine="cpu", streaming=False, optimize=False
            )
    var good = (
        _rank_source()
        .lazy()
        .select((col("x") + lit(Int64(1))).rank("min").alias("r"))
    )
    var actual = good.collect(
        accelerator=runtime, streaming=False, optimize=False
    )
    assert_equal(actual.item(0, "r").int64(), Int64(4))
    _ = setting^


# requires: metal
def test_mojo_gpu_checked_rank_year_month_sibling_and_resident_consumers() raises:
    var setting = WorkerSetting(1)
    var source = DataFrame(
        [
            Series("year", Column[Int64]([1999, 1999, 2000, 1999, 1999])),
            Series("month", Column[Int64]([2, 1, 1, 1, 3])),
            Series("sales", Column[Float64]([4, 2, 10, 8, 6])),
            Series("k", Column[Int32]([0, 0, 0, 0, 1])),
        ]
    )
    var rank = (
        (col("year") * lit(Int64(100)) + col("month")).rank("min").over("k")
    )
    var query = source.lazy().with_columns(
        [
            col("sales").mean().over("k").alias("average"),
            rank.alias("r"),
        ]
    )
    var ranks: List[Int64] = [3, 1, 4, 1, 1]
    for gpu in [False, True]:
        var actual = query.collect(
            accelerator=MetalRuntime(), streaming=False, optimize=False
        ) if gpu else query.collect(
            engine="cpu", streaming=False, optimize=False
        )
        for row in range(5):
            assert_equal(actual.item(row, "r").int64(), ranks[row])
            assert_equal(actual.item(row, "average").float64(), Float64(6))
        var filtered = query.filter(col("r") == 1).select(col("r"))
        assert_equal(
            (
                filtered.collect(
                    accelerator=MetalRuntime(), streaming=False, optimize=False
                ) if gpu else filtered.collect(
                    engine="cpu", streaming=False, optimize=False
                )
            ).height(),
            3,
        )
    var measured = query.profile(
        accelerator=MetalRuntime(),
        engine="accel",
        streaming=False,
        optimize=False,
    )
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    _ = query.collect(
        accelerator=MetalRuntime(memory_limit_bytes=peak),
        streaming=False,
        optimize=False,
    )
    with assert_raises(contains="memory budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            streaming=False,
            optimize=False,
        )
    _ = setting^


def test_checked_rank_phase_metadata_route_and_cache_guards() raises:
    var source = _rank_source()
    var query = source.lazy().select((col("x") + lit(Int64(1))).rank("min"))
    # The certificate route remains for backends that do not decide faults
    # on the CPU; Metal itself observes the input step (ordered mode 0).
    var certified = _capabilities()
    certified.adjudicated_faults = False
    assert_equal(lower_rows(query, _capabilities()).ordered_mode, 0)
    var plan = lower_rows(query, certified)
    assert_equal(plan.ordered_mode, 10)
    assert_equal(len(plan.iana_mask_bases), len(plan.steps))
    assert_true(plan.iana_mask_bases[0] >= 0)
    assert_equal(plan.iana_mask_bases[1], -1)
    var grouped_rank = (
        source.lazy()
        .group_by("k")
        .agg(
            (col("x") + lit(Int64(1))).rank("min").cast(DataType.FLOAT64).mean()
        )
    )
    with assert_raises(contains="rank inside grouped aggregation"):
        _ = lower_rows(grouped_rank, _capabilities())
    var ordinary = certified.copy()
    ordinary.observed_rank_inputs = False
    with assert_raises(contains="checked rank inputs"):
        _ = lower_rows(query, ordinary)
    var changed = lower_rows(query, certified)
    changed.zone_words[changed.iana_mask_bases[0]] += 1
    with assert_raises(contains="geometry"):
        _ = estimate_resources(changed)
    var setting = WorkerSetting(2)
    var large = DataFrame(
        [
            Series("x", Column[Int64](List[Int64](length=4096, fill=1))),
            Series("k", Column[Int32](List[Int32](length=4096, fill=0))),
        ]
    )
    with assert_raises(contains="phase is not certified"):
        _ = lower_rows(
            large.lazy().select((col("x") + lit(Int64(1))).rank().over("k")),
            certified,
        )
    var competing = source.lazy().select_exprs(
        [
            (col("x") + lit(Int64(1))).rank().alias("r"),
            col("x").pow(lit(Int64(-1))).alias("later"),
        ]
    )
    with assert_raises(contains="phase is not certified"):
        _ = lower_rows(competing, certified)
    _ = lower_rows(competing, _capabilities())
    _ = setting^
