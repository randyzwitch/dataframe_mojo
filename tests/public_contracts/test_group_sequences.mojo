"""Expression row selection retains nulls, positions, and group association."""
# contract-group: expr.group_sequences
from std.testing import assert_raises
from dataframe import Column, DataFrame, Expr, ListColumn, Series, col, lit
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import chunk_layout, offset_layout


def sequence_frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 0, 1, 2, 2])),
            Series(
                "x",
                Column[Int64](
                    [1, 2, 3, 4, 5, 6], [True, False, True, True, False, True]
                ),
            ),
            Series(
                "p",
                Column[Bool](
                    [True, True, False, True, True, False],
                    [True, True, True, False, True, True],
                ),
            ),
        ]
    )


def test_filter_preserves_selected_nulls_in_lists_len_first_and_reductions() raises:
    var frame = sequence_frame()
    var filtered = col("x").filter(col("p"))
    var expressions: List[Expr] = [
        filtered.sum().alias("sum"),
        filtered.len().alias("len"),
        filtered.count().alias("count"),
        filtered.first().alias("first"),
        filtered.alias("values"),
        filtered.drop_nulls().len().alias("nonnull"),
    ]
    var expected = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 2])),
            Series("sum", Column[Int64]([1, 0, 0])),
            Series("len", Column[Int64]([1, 1, 1])),
            Series("count", Column[Int64]([1, 0, 0])),
            Series("first", Column[Int64]([1, 0, 0], [True, False, False])),
            Series(
                "values",
                ListColumn(
                    [Int64(0), 1, 2, 3],
                    Series(
                        "item", Column[Int64]([1, 0, 0], [True, False, False])
                    ),
                ),
            ),
            Series("nonnull", Column[Int64]([1, 0, 0])),
        ]
    )
    var flat = DataFrame(
        [Series("x", Column[Int64]([1, 0, 0], [True, False, False]))]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_frame(source.select(filtered), flat)
        assert_frame(
            source.group_by("g", maintain_order=True).agg(expressions), expected
        )
        for optimize in [False, True]:
            assert_frame(
                source.lazy()
                .group_by("g", maintain_order=True)
                .agg(expressions)
                .collect(engine="cpu", optimize=optimize, batch_size=1),
                expected,
            )
        assert_frame(
            source.select(col("x").filter(lit(False)).len().alias("n")),
            DataFrame([Series("n", Column[Int64]([0]))]),
        )
        assert_frame(
            source.select(col("x").filter(lit(True))), source.select(col("x"))
        )


def test_slice_gather_and_scalar_get_apply_to_each_group_before_aggregation() raises:
    var frame = sequence_frame()
    var expressions: List[Expr] = [
        col("x").head(1).alias("head"),
        col("x").tail(1).alias("tail"),
        col("x").slice(-1).sum().alias("sum"),
        col("x").get(-1).alias("get"),
        col("x").gather([1, 0, 1]).alias("gather"),
    ]
    var expected = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 2])),
            Series(
                "head",
                ListColumn(
                    [Int64(0), 1, 2, 3],
                    Series(
                        "item", Column[Int64]([1, 0, 0], [True, False, False])
                    ),
                ),
            ),
            Series(
                "tail",
                ListColumn(
                    [Int64(0), 1, 2, 3],
                    Series("item", Column[Int64]([3, 4, 6])),
                ),
            ),
            Series("sum", Column[Int64]([3, 4, 6])),
            Series("get", Column[Int64]([3, 4, 6])),
            Series(
                "gather",
                ListColumn(
                    [Int64(0), 3, 6, 9],
                    Series(
                        "item",
                        Column[Int64](
                            [3, 1, 3, 4, 0, 4, 6, 0, 6],
                            [
                                True,
                                True,
                                True,
                                True,
                                False,
                                True,
                                True,
                                False,
                                True,
                            ],
                        ),
                    ),
                ),
            ),
        ]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_frame(
            source.group_by("g", maintain_order=True).agg(expressions), expected
        )
        assert_frame(
            source.lazy()
            .group_by("g", maintain_order=True)
            .agg(expressions)
            .collect(engine="cpu", batch_size=2),
            expected,
        )
        assert_frame(
            source.select(col("x").head(0)), source.select(col("x")).head(0)
        )
        assert_frame(
            source.select(col("x").tail(0)), source.select(col("x")).head(0)
        )
        assert_frame(
            source.select(col("x").gather(List[Int]())),
            source.select(col("x")).head(0),
        )
        assert_frame(
            source.select(col("x").slice(-100, 2)),
            source.select(col("x")).head(2),
        )


def test_sequence_validation_rejects_bad_predicates_indices_and_mixed_heights() raises:
    var source = sequence_frame()
    with assert_raises(contains="Boolean predicate"):
        _ = source.select(col("x").filter(col("x")))
    with assert_raises(contains="different lengths"):
        _ = source.select_exprs(
            [col("x").filter(col("p")).alias("short"), col("x")]
        )
    with assert_raises(contains="out of bounds"):
        _ = source.select(col("x").gather([-7]))
    with assert_raises(contains="out of bounds"):
        _ = source.head(0).select(col("x").get(0))
    with assert_raises(contains="nonnegative"):
        _ = source.select(col("x").head(-1))
    with assert_raises(contains="nonnegative"):
        _ = source.select(col("x").slice(0, -2))
