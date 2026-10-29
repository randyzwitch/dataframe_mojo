"""Ordered windows preserve typed cells and restore every original row."""
# contract-group: expr.ordered_over
from std.testing import assert_raises
from dataframe import Column, DataFrame, Expr, Series, col
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import chunk_layout, offset_layout


def test_ordered_over_scatter_ties_partitions_and_lazy_key_dependencies() raises:
    var frame = DataFrame(
        [
            Series(
                "g",
                Column[Int64](
                    [0, 1, 0, 1, 0, 1], [False, True, False, True, False, True]
                ),
            ),
            Series("t", Column[Int64]([2, 2, 1, 1, 2, 0])),
            Series("z", Column[Int64]([2, 0, 0, 0, 1, 0])),
            Series("x", Column[Int64]([10, 20, 30, 40, 50, 60])),
        ]
    )
    var expressions: List[Expr] = [
        col("x").cum_sum().over("g", order_by="t").alias("sum"),
        col("x").shift().over([String("g")], order_by="t").alias("previous"),
        col("x")
        .cum_sum()
        .over("g", order_by="t", descending=True)
        .alias("descending"),
        col("x").cum_sum().over(order_by="t").alias("global"),
        col("x").first().over("g", order_by="t").alias("first"),
        col("x")
        .cum_sum()
        .over("g", order_by=[String("t"), "z"])
        .alias("multi"),
    ]
    var expected = DataFrame(
        [
            Series("sum", Column[Int64]([40, 120, 30, 100, 90, 60])),
            Series(
                "previous",
                Column[Int64](
                    [30, 40, 0, 60, 10, 0],
                    [True, True, False, True, True, False],
                ),
            ),
            Series("descending", Column[Int64]([10, 20, 90, 60, 60, 120])),
            Series("global", Column[Int64]([140, 160, 90, 130, 210, 60])),
            Series("first", Column[Int64]([30, 60, 30, 60, 30, 60])),
            Series("multi", Column[Int64]([90, 120, 30, 100, 80, 60])),
        ]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_frame(source.select_exprs(expressions, batch_size=2), expected)
        for optimize in [False, True]:
            for streaming in [False, True]:
                assert_frame(
                    source.lazy()
                    .select_exprs(expressions)
                    .collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=2,
                    ),
                    expected,
                )
        var empty = source.head(0)
        assert_frame(empty.select_exprs(expressions), expected.head(0))


def test_ordered_over_nan_null_placement_and_escaped_order_names() raises:
    var nan = Float64(0) / Float64(0)
    var order_name = String("t\x1f\x1e")
    var frame = DataFrame(
        [
            Series(
                order_name,
                Column[Float64](
                    [nan, 0, 2, 0, 2, 0], [True, True, True, False, True, True]
                ),
            ),
            Series("x", Column[Int64]([1, 2, 4, 8, 16, 32])),
        ]
    )
    var expressions: List[Expr] = [
        col("x").cum_sum().over(order_by=order_name).alias("first"),
        col("x")
        .cum_sum()
        .over(order_by=order_name, nulls_last=True)
        .alias("last"),
        col("x")
        .cum_sum()
        .over(order_by=order_name, descending=True, nulls_last=True)
        .alias("descending"),
    ]
    var expected = DataFrame(
        [
            Series("first", Column[Int64]([63, 10, 46, 8, 62, 42])),
            Series("last", Column[Int64]([55, 2, 38, 63, 54, 34])),
            Series("descending", Column[Int64]([1, 23, 5, 63, 21, 55])),
        ]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_frame(source.select_exprs(expressions), expected)
        assert_frame(
            source.lazy()
            .select_exprs(expressions)
            .collect(engine="cpu", batch_size=1),
            expected,
        )
    with assert_raises(contains="Unknown window column"):
        _ = (
            frame.lazy()
            .select(col("x").sum().over(order_by="missing"))
            .collect(engine="cpu")
        )
    with assert_raises(contains="over requires"):
        _ = frame.select(col("x").sum().over())


def test_ordered_over_sorts_categorical_labels_instead_of_dictionary_codes() raises:
    var frame = DataFrame(
        [
            Series("t", Column[String](["z", "a", "z", "b", "a", "c"])).cast(
                "categorical"
            ),
            Series("x", Column[Int64]([1, 2, 4, 8, 16, 32])),
        ]
    )
    var expected = DataFrame(
        [Series("x", Column[Int64]([59, 2, 63, 26, 18, 58]))]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        var expr = col("x").cum_sum().over(order_by="t")
        assert_frame(source.select(expr), expected)
        assert_frame(
            source.lazy().select(expr).collect(engine="cpu", batch_size=2),
            expected,
        )


def test_shift_fill_only_replaces_vacated_partition_positions() raises:
    var frame = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 0, 1, 0, 1])),
            Series("t", Column[Int64]([2, 2, 1, 1, 2, 0])),
            Series(
                "x",
                Column[Int64](
                    [10, 20, 30, 40, 50, 60],
                    [False, True, True, False, True, True],
                ),
            ),
        ]
    )
    var expressions: List[Expr] = [
        col("x").shift(1, fill_value=-1).over("g", order_by="t").alias("lag"),
        col("x").shift(-1, fill_value=-1).over("g", order_by="t").alias("lead"),
        col("x").shift(0, fill_value=-1).over("g", order_by="t").alias("same"),
        col("x")
        .shift(100, fill_value=-1)
        .over("g", order_by="t")
        .alias("outside"),
    ]
    var expected = DataFrame(
        [
            Series(
                "lag",
                Column[Int64](
                    [30, 0, -1, 60, 0, -1],
                    [True, False, True, True, False, True],
                ),
            ),
            Series(
                "lead",
                Column[Int64](
                    [50, -1, 0, 20, -1, 0],
                    [True, True, False, True, True, False],
                ),
            ),
            frame.column("x").renamed("same"),
            Series("outside", Column[Int64]([-1, -1, -1, -1, -1, -1])),
        ]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_frame(source.select_exprs(expressions), expected)
        assert_frame(
            source.lazy()
            .select_exprs(expressions)
            .collect(engine="cpu", batch_size=1),
            expected,
        )
    var text = DataFrame(
        [Series("x", Column[String](["a", "", "c"], [True, False, True]))]
    )
    assert_frame(
        text.select(col("x").shift(fill_value="empty")),
        DataFrame(
            [
                Series(
                    "x", Column[String](["empty", "a", ""], [True, True, False])
                ),
            ]
        ),
    )
