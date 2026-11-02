"""Arithmetic and casts after a join or a filter are computed for the kept
rows alone, each column read through the selection where it is
(`selected_expressions`). Every result equals the eager filtered frame's:
decimal products, sums and differences at different scales, decimal
rescales up and down and to Float64, integer to decimal and Float64 casts,
Float64 and Int64 arithmetic, nullable columns and a null constant, after a
join that keeps most rows (values placed at their rows) or few (one value
per kept row), after a filter, and under a grouped aggregate. A product
that overflows only on dropped rows does not raise; one that overflows on a
kept row does."""
# contract-group: table.join
from std.testing import assert_equal, assert_raises, assert_true

from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, null
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 12_000


def frame() raises -> DataFrame:
    var ids = List[Int64]()
    var day = List[Int64]()
    var quantity = List[Int32]()
    var quantity_valid = List[Bool]()
    var count = List[Int64]()
    var price = List[String]()
    var discount = List[String]()
    var discount_valid = List[Bool]()
    var real = List[Float64]()
    for row in range(ROWS):
        ids.append(Int64(row))
        day.append(Int64((row * 7919) % 100))
        quantity.append(Int32(row % 101 - 3))
        quantity_valid.append(row % 53 != 7)
        count.append(Int64(row % 977) - 400)
        price.append(String(row % 9973) + "." + String(row % 89 + 10))
        discount.append("0." + String(row % 10) + String(row % 7))
        discount_valid.append(row % 31 != 2)
        real.append(Float64(row % 1013) / 8.0 - 20.0)
    return DataFrame(
        [
            Series("id", Column[Int64](ids^)),
            Series("day", Column[Int64](day^)),
            Series("quantity", Column[Int32](quantity^, quantity_valid)),
            Series("count", Column[Int64](count^)),
            Series("price", Column[String](price^)).cast(
                DataType.decimal(7, 2)
            ),
            Series("discount", Column[String](discount^, discount_valid)).cast(
                DataType.decimal(5, 2)
            ),
            Series("real", Column[Float64](real^)),
        ]
    )


def days(last: Int) raises -> DataFrame:
    var keys = List[Int64]()
    for d in range(last + 1):
        keys.append(Int64(d))
    return DataFrame([Series("d", Column[Int64](keys^))])


def computed() raises -> List[Expr]:
    return [
        (col("quantity").cast(DataType.decimal(19, 0)) * col("price")).alias(
            "value"
        ),
        (
            col("price")
            * (lit("1").cast(DataType.decimal(15, 2)) - col("discount"))
        ).alias("net"),
        (col("price") * col("discount")).alias("cut"),
        (col("price") + col("discount").cast(DataType.decimal(9, 4))).alias(
            "sum"
        ),
        col("price").cast(DataType.decimal(8, 1)).alias("rounded"),
        (col("price").cast(DataType.FLOAT64) * col("real")).alias("mixed"),
        (col("quantity").cast(DataType.FLOAT64) / lit(4.0) - col("real")).alias(
            "ratio"
        ),
        (col("count") * col("count") - lit(Int64(7))).alias("square"),
        (col("price") - null(DataType.decimal(7, 2))).alias("none"),
        (col("quantity").cast(DataType.INT64) + col("count")).alias("total"),
    ]


def selected(source: DataFrame, last: Int, join: Bool) raises -> DataFrame:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var plan = source.lazy()
    if join:
        plan = plan.join(days(last).lazy(), left_on=["day"], right_on=["d"])
    else:
        plan = plan.filter(col("day") <= lit(Int64(last)))
    var piped = plan.with_columns(computed()).sort("id").collect()
    var took = (
        "dataframe-path: pipeline.with_columns_selected\n" in observed.text()
    )
    _ = observed^
    _ = trace^
    assert_true(took, "the expressions were not computed through the rows")
    return piped^


def eager(source: DataFrame, last: Int) raises -> DataFrame:
    return (
        source.filter(col("day") <= lit(Int64(last)))
        .with_columns(computed())
        .sort("id")
    )


def test_kept_rows_compute_as_the_filtered_frame_does() raises:
    var source = frame()
    # Days 0 to 55 keep 56% of every morsel (values placed at their
    # rows); days 0 to 9 keep 10% (one value per kept row).
    for last in [55, 9]:
        var expected = eager(source, last)
        assert_frame(selected(source, last, True), expected)
        assert_frame(selected(source, last, False), expected)


def test_kept_rows_feed_a_grouped_aggregate() raises:
    var source = frame()
    var aggregations: List[Expr] = [
        col("value").sum().alias("value"),
        col("net").sum().alias("net"),
        col("square").sum().alias("square"),
        (col("price") * col("discount")).sum().alias("cut"),
        col("id").len().alias("rows"),
    ]
    for last in [55, 9]:
        var piped = (
            source.lazy()
            .join(days(last).lazy(), left_on=["day"], right_on=["d"])
            .with_columns(computed())
            .group_by(["quantity"])
            .agg(aggregations.copy())
            .sort("quantity")
            .collect()
        )
        var expected = (
            eager(source, last)
            .group_by(["quantity"])
            .agg(aggregations.copy())
            .sort("quantity")
        )
        assert_frame(piped, expected)


def test_only_a_kept_rows_overflow_raises() raises:
    var source = frame()
    var big = List[Int64]()
    for row in range(ROWS):
        # Past the 18 digits of a decimal(18) product on days past 90 only.
        var dropped = (row * 7919) % 100 > 90
        big.append(Int64(999_999_999) if dropped else Int64(row % 7))
    var wide = source.with_column(
        Series("big", Column[Int64](big^)).cast(DataType.decimal(9, 0, 32))
    )
    var product = (col("big") * col("big") * col("big")).alias("p")
    var fine = (
        wide.lazy()
        .join(days(90).lazy(), left_on=["day"], right_on=["d"])
        .with_columns([product.copy()])
        .collect()
    )
    assert_equal(
        fine.height(), source.filter(col("day") <= lit(Int64(90))).height()
    )
    with assert_raises():
        _ = (
            wide.lazy()
            .join(days(99).lazy(), left_on=["day"], right_on=["d"])
            .with_columns([product^])
            .collect()
        )


def test_a_projection_keeps_the_selection() raises:
    # Columns picked as they are beside computed ones, after a join that
    # keeps most rows and after one that keeps few.
    var source = frame()
    var projected: List[Expr] = [col("id"), col("price").alias("p")]
    for e in computed():
        projected.append(e.copy())
    for last in [55, 9]:
        var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
        var observed = StrategyTrace()
        var piped = (
            source.lazy()
            .join(days(last).lazy(), left_on=["day"], right_on=["d"])
            .select_exprs(projected.copy())
            .sort("id")
            .collect()
        )
        var took = "dataframe-path: pipeline.select_selected\n" in (
            observed.text()
        )
        _ = observed^
        _ = trace^
        assert_true(took, "the projection gathered its columns")
        var expected = (
            source.filter(col("day") <= lit(Int64(last)))
            .select_exprs(projected.copy())
            .sort("id")
        )
        assert_frame(piped, expected)
