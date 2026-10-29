"""Whole-expression ordering and distinct selection preserve group association."""
# contract-group: expr.group_sequences
from std.testing import assert_raises
from dataframe import Column, DataFrame, DataType, Expr, ListColumn, Series, col
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import chunk_layout, offset_layout


def ordered_group_frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 0, 1, 0, 1])),
            Series(
                "x",
                Column[Int64](
                    [3, 10, 1, 20, 3, 10],
                    [True, True, False, True, True, False],
                ),
            ),
            Series(
                "t",
                Column[Int64](
                    [2, 3, 1, 0, 2, 3], [True, True, True, False, True, True]
                ),
            ),
            Series("r", Column[Int64]([0, 1, 2, 3, 4, 5])),
            Series("label", Column[String](["a", "b", "c", "d", "e", "f"])),
        ]
    )


def assert_ordered_groups(
    source: DataFrame, expression: Expr, offsets: List[Int64], values: Series
) raises:
    var expected = DataFrame(
        [
            Series("g", Column[Int64]([0, 1])),
            Series("out", ListColumn(offsets.copy(), values.renamed("item"))),
        ]
    )
    assert_frame(
        source.group_by("g", maintain_order=True).agg(expression.alias("out")),
        expected,
    )
    for optimize in [False, True]:
        assert_frame(
            source.lazy()
            .group_by("g", maintain_order=True)
            .agg(expression.alias("out"))
            .collect(engine="cpu", optimize=optimize, batch_size=1),
            expected,
        )


def test_group_sort_unique_counts_top_bottom_and_positions_match_literal_cells() raises:
    var frame = ordered_group_frame()
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_ordered_groups(
            source,
            col("x").sort(),
            [0, 3, 6],
            Series(
                "",
                Column[Int64](
                    [0, 3, 3, 0, 10, 20], [False, True, True, False, True, True]
                ),
            ),
        )
        assert_ordered_groups(
            source,
            col("x").sort(True, True),
            [0, 3, 6],
            Series(
                "",
                Column[Int64](
                    [3, 3, 0, 20, 10, 0], [True, True, False, True, True, False]
                ),
            ),
        )
        assert_ordered_groups(
            source,
            col("x").sort_by("t", nulls_last=True),
            [0, 3, 6],
            Series(
                "",
                Column[Int64](
                    [0, 3, 3, 10, 0, 20], [False, True, True, True, False, True]
                ),
            ),
        )
        assert_ordered_groups(
            source,
            col("x").reverse(),
            [0, 3, 6],
            Series(
                "",
                Column[Int64](
                    [3, 0, 3, 0, 20, 10], [True, False, True, False, True, True]
                ),
            ),
        )
        assert_ordered_groups(
            source,
            col("x").unique(True),
            [0, 2, 5],
            Series(
                "",
                Column[Int64](
                    [3, 0, 10, 20, 0], [True, False, True, True, False]
                ),
            ),
        )
        assert_ordered_groups(
            source,
            col("x").unique_counts(),
            [0, 2, 5],
            Series("", Column[Int64]([2, 1, 1, 1, 1])),
        )
        assert_ordered_groups(
            source,
            col("x").arg_sort(),
            [0, 3, 6],
            Series("", Column[Int64]([1, 0, 2, 2, 0, 1])),
        )
        assert_ordered_groups(
            source,
            col("x").top_k(2),
            [0, 2, 4],
            Series("", Column[Int64]([3, 3, 20, 10])),
        )
        assert_ordered_groups(
            source,
            col("x").bottom_k(2),
            [0, 2, 4],
            Series("", Column[Int64]([3, 3, 10, 20])),
        )
        assert_ordered_groups(
            source,
            col("x").top_k_by("t", 2),
            [0, 2, 4],
            Series("", Column[Int64]([3, 3, 10, 0], [True, True, True, False])),
        )
        assert_ordered_groups(
            source,
            col("x").bottom_k_by("t", 2),
            [0, 2, 4],
            Series(
                "", Column[Int64]([0, 3, 10, 0], [False, True, True, False])
            ),
        )
        assert_ordered_groups(
            source,
            col("x").top_k(0),
            [0, 0, 0],
            Series("", Column[Int64](List[Int64]())),
        )
        var expected = DataFrame(
            [
                Series("g", Column[Int64]([0, 1])),
                Series("max", Column[Int64]([3, 10])),
                Series("min", Column[Int64]([0, 10], [False, True])),
                Series("distinct_sum", Column[Int64]([3, 30])),
                Series("distinct_count", Column[Int64]([1, 2])),
            ]
        )
        var expressions: List[Expr] = [
            col("x").max_by("t").alias("max"),
            col("x").min_by("t").alias("min"),
            col("x").unique().sum().alias("distinct_sum"),
            col("x").unique().count().alias("distinct_count"),
        ]
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


def test_multi_key_ordering_scalar_extrema_empty_keys_and_whole_columns() raises:
    var frame = ordered_group_frame()
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_ordered_groups(
            source,
            col("label").sort_by([String("t"), "r"]),
            [0, 3, 6],
            Series("", Column[String](["c", "a", "e", "d", "b", "f"])),
        )
        var by: List[Expr] = [col("t"), -col("r")]
        assert_ordered_groups(
            source,
            col("label").top_k_by(by, 2),
            [0, 2, 4],
            Series("", Column[String](["a", "e", "b", "f"])),
        )
        assert_frame(
            source.select(col("x").unique_counts()),
            DataFrame([Series("x", Column[Int64]([2, 1, 2, 1]))]),
        )
        assert_frame(
            source.select(col("x").arg_sort()),
            DataFrame([Series("x", Column[Int64]([2, 5, 0, 4, 1, 3]))]),
        )
        # Explicit all-null keys must yield a typed null scalar.
        var no_keys = DataFrame(
            [
                source.column("x"),
                Series(
                    "t",
                    Column[Int64](
                        [0, 0, 0, 0, 0, 0],
                        [False, False, False, False, False, False],
                    ),
                ),
            ]
        )
        assert_frame(
            no_keys.select(col("x").max_by("t")),
            DataFrame([Series("x", Column[Int64]([0], [False]))]),
        )
        assert_frame(
            no_keys.head(0).select(col("x").min_by("t")),
            DataFrame([Series("x", Column[Int64]([0], [False]))]),
        )
    with assert_raises(contains="nonnegative"):
        _ = frame.select(col("x").top_k(-1))
    with assert_raises(contains="different lengths"):
        _ = frame.select(col("x").head(1).sort_by(col("t")))


def test_ordering_nan_signed_zero_nulls_and_categorical_labels() raises:
    var nan = Float64(0) / Float64(0)
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [nan, -Float64(0), 0, 1, 9, nan],
                    [True, True, True, True, False, True],
                ),
            ),
            Series("label", Column[String](["a", "b", "c", "d", "e", "f"])),
        ]
    )
    var sorted = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [-Float64(0), 0, 1, nan, nan, 0],
                    [True, True, True, True, True, False],
                ),
            )
        ]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_frame(source.select(col("x").sort(nulls_last=True)), sorted)
        assert_frame(
            source.select(col("label").sort_by("x", nulls_last=True)),
            DataFrame(
                [
                    Series(
                        "label", Column[String](["b", "c", "d", "a", "f", "e"])
                    )
                ]
            ),
        )
        assert_frame(
            source.select(col("x").unique_counts()),
            DataFrame([Series("x", Column[Int64]([2, 2, 1, 1]))]),
        )
    var categorical = DataFrame(
        [
            Series("x", Column[String](["z", "a", "b", "a"])).cast(
                "categorical"
            ),
            Series("label", Column[String](["a", "b", "c", "d"])),
        ]
    )
    assert_frame(
        categorical.select(col("label").sort_by("x")),
        DataFrame([Series("label", Column[String](["b", "d", "c", "a"]))]),
    )


def test_expression_gather_uses_group_local_positions_and_null_indices() raises:
    var frame = ordered_group_frame()
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        var indices = col("r").arg_sort(descending=True).head(2)
        assert_ordered_groups(
            source,
            col("x").gather(indices),
            [0, 2, 4],
            Series(
                "", Column[Int64]([3, 0, 0, 20], [True, False, False, True])
            ),
        )
    var source = DataFrame(
        [
            Series("x", Column[Int64]([10, 20, 30])),
            Series("i", Column[Int64]([2, -1, 99], [True, True, False])),
        ]
    )
    var expected = DataFrame(
        [Series("x", Column[Int64]([30, 30, 0], [True, True, False]))]
    )
    for variant in [source.copy(), chunk_layout(source), offset_layout(source)]:
        assert_frame(variant.select(col("x").gather(col("i"))), expected)
        assert_frame(
            variant.lazy()
            .select(col("x").gather(col("i")))
            .collect(engine="cpu", batch_size=1),
            expected,
        )
    with assert_raises(contains="integer indices"):
        _ = source.select(col("x").gather(col("i").cast("float64")))
    with assert_raises(contains="out of bounds"):
        _ = source.select(col("x").gather(col("i") + 10))
