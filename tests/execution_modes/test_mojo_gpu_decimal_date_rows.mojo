"""Independent Date.YEAR and checked Decimal producer composition witnesses."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.lazy import LazyFrame
from dataframe.binding import bind, BoundExpr
from dataframe.accelerator import decimal_row_feed, ordered_rows
from dataframe.accelerator.constants import fold_literal_text_parsing
from dataframe.accelerator.aggregates import lower_aggregate_compositions
from dataframe.accelerator.joins import _decimal_date_producer
from dataframe.metal import MetalRuntime
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source(nullable: Bool = False, bad: Bool = False) raises -> DataFrame:
    var prices: List[Int64] = [100, 200, 300, 400, 500, 600]
    if nullable:
        prices[4] = Int64.MAX
    if bad:
        prices[5] = Int64.MAX
    return DataFrame(
        [
            Series("g", Column[Int32]([0, 1, 2, 3, 4, 5])),
            Series(
                "d",
                Column[Int64](
                    [Int64.MIN, -1, 0, 10957, 19782, Int64.MAX],
                    [True, True, True, not nullable, True, True],
                ),
            ).with_dtype(DataType.DATE),
            Series(
                "price",
                Column[Int64](
                    prices^, [True, True, True, True, not nullable, True]
                ),
            ).with_dtype(DataType.decimal(15, 2, 64)),
            Series("discount", Column[Int64]([25, 50, 0, 0, 0, 0])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series(
                "nation",
                StringColumn(
                    ["BRAZIL", "OTHER", "BRAZIL", "OTHER", "BRAZIL", "OTHER"]
                ),
            ),
        ]
    )


def _expressions() -> List[Expr]:
    return [
        col("d").dt().year().alias("y"),
        (
            col("price")
            * (lit("1").cast(DataType.decimal(15, 2)) - col("discount"))
        ).alias("volume"),
    ]


def _producer(source: DataFrame) raises -> LazyFrame:
    var left = DataFrame(
        [
            Series("g", Column[Int32]([0, 1, 2, 3, 4, 5])),
            Series("supp", StringColumn(["A", "A", "A", "A", "A", "A"])),
        ]
    )
    return left.lazy().join(source.lazy(), on="g").with_columns(_expressions())


def _grouped(source: DataFrame, ratio: Bool = False) raises -> LazyFrame:
    var producer = _producer(source)
    if ratio:
        return (
            producer.group_by(["g", "y"])
            .agg(
                (
                    when(col("nation") == "BRAZIL")
                    .then(col("volume").cast(DataType.FLOAT64))
                    .otherwise(lit(0.0))
                    .sum()
                    / col("volume").sum().cast(DataType.FLOAT64)
                ).alias("share")
            )
            .sort("g")
        )
    return (
        producer.group_by(["g", "supp", "nation", "y"])
        .agg(col("volume").sum().alias("total"))
        .sort("g")
    )


def _collect(
    query: LazyFrame, gpu: Bool, streaming: Bool, batch: Int
) raises -> DataFrame:
    if gpu:
        return query.collect(
            engine="accel",
            accelerator=MetalRuntime(),
            optimize=False,
            streaming=streaming,
            batch_size=batch,
        )
    return query.collect(
        engine="cpu", optimize=False, streaming=streaming, batch_size=batch
    )


def test_decimal_date_rows_certificate_preserves_original_prefix_and_states() raises:
    var source = _source()
    var expressions = _expressions()
    var normalized = fold_literal_text_parsing(
        source.lazy().with_columns(expressions)
    )
    var bounds = List[BoundExpr]()
    for expression in normalized._nodes[len(normalized._nodes) - 1].exprs:
        bounds.append(bind(expression, source._columns))
    assert_true(decimal_row_feed.eligible(bounds))
    assert_true(decimal_row_feed.date_year(bounds[0]))
    assert_true(not ordered_rows.eligible(bounds[0]))
    var forged = bounds[0].copy()
    forged.dtypes[0] = DataType.datetime("us")
    assert_true(not decimal_row_feed.date_year(forged))
    for ratio in [False, True]:
        var query = fold_literal_text_parsing(_grouped(source, ratio))
        # Scan, JOIN and WITH_COLUMNS occupy the original immutable DAG;
        # discover the producer rather than grant synthetic-scan provenance.
        var count = 0
        for at in range(len(query._nodes)):
            count += Int(_decimal_date_producer(query, at))
        assert_equal(count, 1)
        if ratio:
            var rewritten = lower_aggregate_compositions(
                query, exact_float32=True
            )
            var lifted = 0
            for at in range(len(rewritten._nodes)):
                lifted += Int(_decimal_date_producer(rewritten, at))
            assert_equal(lifted, 1)
    var producer = fold_literal_text_parsing(_producer(source))
    var index = len(producer._nodes) - 1
    assert_true(_decimal_date_producer(producer, index))
    assert_true(not _decimal_date_producer(producer.head(0), index))
    var dead = producer.group_by("y").agg(col("g").count())
    assert_true(not _decimal_date_producer(dead, index))
    var dropped = _grouped(source).select(["g", "y"])
    assert_true(
        not _decimal_date_producer(fold_literal_text_parsing(dropped), index)
    )
    var selected = DataFrame([Series("g", Column[Int32]([0]))])
    var joined = producer.join(selected.lazy(), on="g").select(
        ["g", "y", "volume"]
    )
    assert_true(
        _decimal_date_producer(
            joined._with_accelerator_settings(False, True, 17), index
        )
    )
    assert_true(not _decimal_date_producer(joined, index))
    var unsafe = producer.group_by(["g", "y"]).agg(
        [
            col("volume").sum(),
            col("g").cast(DataType.INT64).sum().alias("competing"),
        ]
    )
    assert_true(not _decimal_date_producer(unsafe, index))
    var prior_state = (
        source.lazy()
        .group_by(["g", "d"])
        .agg(
            [
                col("price").sum().alias("price"),
                col("discount").sum().alias("discount"),
            ]
        )
        .with_columns(_expressions())
    )
    assert_true(
        not _decimal_date_producer(prior_state, len(prior_state._nodes) - 1)
    )


def test_mojo_gpu_decimal_date_rows_literals_extremes_nulls_and_layouts() raises:
    # Literal years independently derived by binary searching Gregorian Jan1
    # day counts, not by calling or copying civil_from_days' era decomposition.
    var years: List[Int64] = [
        -25252734927764585,
        1969,
        1970,
        2000,
        2024,
        25252734927768524,
    ]
    var totals: List[Int128] = [7500, 10000, 30000, 40000, 50000, 60000]
    for nullable in [False, True]:
        var source = _source(nullable)
        var layouts: List[DataFrame] = [
            source.copy(),
            offset_layout(source),
            chunk_layout(source),
        ]
        for layout in layouts:
            var query = _grouped(layout)
            for streaming in [False, True]:
                for gpu in [False, True]:
                    var result = _collect(query, gpu, streaming, 2)
                    assert_equal(result.height(), 6)
                    assert_equal(
                        result.column("g").int32().to_list(), [0, 1, 2, 3, 4, 5]
                    )
                    assert_equal(result.column("y").dtype(), DataType.INT64)
                    assert_equal(
                        result.column("total").dtype(), DataType.decimal(38, 4)
                    )
                    for row in range(6):
                        if nullable and row == 3:
                            assert_true(result.item(row, "y").is_null())
                        else:
                            var actual_year = (
                                result.column("y").int64().value(row)
                            )
                            assert_equal(actual_year, years[row])
                        var expected = (
                            Int128(0) if nullable and row == 4 else totals[row]
                        )
                        assert_equal(
                            result.column("total")._decimal_raw(row), expected
                        )
        if not nullable:
            var shares: List[Float64] = [1, 0, 1, 0, 1, 0]
            for gpu in [False, True]:
                var result = _collect(_grouped(source, True), gpu, True, 1)
                assert_equal(result.column("share").float64().to_list(), shares)


def test_mojo_gpu_decimal_date_rows_precision_before_group_and_join() raises:
    var bad = _source(bad=True)
    var selected = DataFrame([Series("g", Column[Int32]([0]))])
    var producer = _producer(bad)
    for count in [1, 2]:
        var workers = WorkerSetting(count)
        for streaming in [False, True]:
            for batch in [1, 7]:
                for query in [
                    _grouped(bad),
                    _grouped(bad, True),
                    producer.join(selected.lazy(), on="g", how="inner").select(
                        ["g", "y", "volume"]
                    ),
                ]:
                    for gpu in [False, True]:
                        with assert_raises(contains="decimal precision"):
                            _ = _collect(query, gpu, streaming, batch)
        _ = workers^
    # Cross the real CPU streaming worker threshold, with a precision site
    # after two65536-row morsels. WorkerSetting alone on six rows would not
    # exercise parallel CPU geometry.
    var n_rows = 131073
    for count in [1, 2]:
        var workers = WorkerSetting(count)
        for bad_last in [False, True]:
            var prices = List[Int64](length=n_rows, fill=100)
            if bad_last:
                prices[n_rows - 1] = Int64.MAX
            var source = DataFrame(
                [
                    Series(
                        "d", Column[Int64](List[Int64](length=n_rows, fill=0))
                    ).with_dtype(DataType.DATE),
                    Series("price", Column[Int64](prices^)).with_dtype(
                        DataType.decimal(15, 2, 64)
                    ),
                    Series(
                        "discount",
                        Column[Int64](List[Int64](length=n_rows, fill=0)),
                    ).with_dtype(DataType.decimal(15, 2, 64)),
                ]
            )
            var query = (
                source.lazy()
                .with_columns(_expressions())
                .group_by("y")
                .agg(col("volume").sum().alias("total"))
            )
            for gpu in [False, True]:
                if bad_last:
                    with assert_raises(contains="decimal precision"):
                        _ = _collect(query, gpu, True, 65536)
                else:
                    var result = _collect(query, gpu, True, 65536)
                    assert_equal(result.height(), 1)
                    assert_equal(
                        result.column("y").int64().value(0), Int64(1970)
                    )
                    assert_equal(
                        result.column("total")._decimal_raw(0),
                        Int128(1310730000),
                    )
        _ = workers^
