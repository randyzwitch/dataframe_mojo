"""IGNORE NULLS counts valid predecessors and successors within each window."""
# contract-group: expr.ordered_over
from std.testing import assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import chunk_layout, offset_layout
from tests.support.selection_types_reference import (
    assert_selection_cell,
    selection_column,
    selection_dtype,
)


def test_shift_ignore_nulls_ordered_interleaved_partitions_and_fill() raises:
    # In t order: g0=[NULL,20,NULL,10,NULL,30],
    # g1=[NULL,NULL,60,NULL,50,40], g2 contains only NULLs.
    var frame = DataFrame(
        [
            Series(
                "g",
                Column[Int64](
                    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
                ),
            ),
            Series(
                "t",
                Column[Int64](
                    [3, 3, 3, 0, 0, 0, 5, 5, 5, 1, 1, 1, 4, 4, 4, 2, 2, 2]
                ),
            ),
            Series(
                "x",
                Column[Int64](
                    [
                        10,
                        0,
                        0,
                        0,
                        0,
                        0,
                        30,
                        40,
                        0,
                        20,
                        0,
                        0,
                        0,
                        50,
                        0,
                        0,
                        60,
                        0,
                    ],
                    [
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        False,
                        True,
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                        True,
                        False,
                    ],
                ),
            ),
        ]
    )
    var expressions: List[Expr] = [
        col("x").shift(ignore_nulls=True).over("g", order_by="t").alias("lag"),
        col("x")
        .shift(-1, ignore_nulls=True)
        .over("g", order_by="t")
        .alias("lead"),
        col("x")
        .shift(2, ignore_nulls=True)
        .over("g", order_by="t")
        .alias("lag2"),
        col("x")
        .shift(-2, ignore_nulls=True)
        .over("g", order_by="t")
        .alias("lead2"),
        col("x")
        .shift(2, fill_value=-1, ignore_nulls=True)
        .over("g", order_by="t")
        .alias("filled"),
        col("x")
        .shift(0, fill_value=-1, ignore_nulls=True)
        .over("g", order_by="t")
        .alias("same"),
        col("x")
        .shift(100, fill_value=-1, ignore_nulls=True)
        .over("g", order_by="t")
        .alias("outside"),
    ]
    var expected = DataFrame(
        [
            Series(
                "lag",
                Column[Int64](
                    [
                        20,
                        60,
                        0,
                        0,
                        0,
                        0,
                        10,
                        50,
                        0,
                        0,
                        0,
                        0,
                        10,
                        60,
                        0,
                        20,
                        0,
                        0,
                    ],
                    [
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        False,
                        True,
                        False,
                        False,
                    ],
                ),
            ),
            Series(
                "lead",
                Column[Int64](
                    [
                        30,
                        50,
                        0,
                        20,
                        60,
                        0,
                        0,
                        0,
                        0,
                        10,
                        60,
                        0,
                        30,
                        40,
                        0,
                        10,
                        50,
                        0,
                    ],
                    [
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                    ],
                ),
            ),
            Series(
                "lag2",
                Column[Int64](
                    [0, 0, 0, 0, 0, 0, 20, 60, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0],
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                ),
            ),
            Series(
                "lead2",
                Column[Int64](
                    [
                        0,
                        40,
                        0,
                        10,
                        50,
                        0,
                        0,
                        0,
                        0,
                        30,
                        50,
                        0,
                        0,
                        0,
                        0,
                        30,
                        40,
                        0,
                    ],
                    [
                        False,
                        True,
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        False,
                    ],
                ),
            ),
            Series(
                "filled",
                Column[Int64](
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        20,
                        60,
                        -1,
                        -1,
                        -1,
                        -1,
                        20,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                    ]
                ),
            ),
            frame.column("x").renamed("same"),
            Series("outside", Column[Int64](List[Int64](length=18, fill=-1))),
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
        assert_frame(source.head(0).select_exprs(expressions), expected.head(0))


def test_shift_ignore_nulls_whole_column_zero_default_and_string_fill() raises:
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[String](
                    ["", "a", "", "", "b", "", "c", ""],
                    [False, True, False, False, True, False, True, False],
                ),
            )
        ]
    )
    var expressions: List[Expr] = [
        col("x").shift(ignore_nulls=True).alias("lag"),
        col("x")
        .shift(-2, fill_value="empty", ignore_nulls=True)
        .alias("lead2"),
        col("x").shift(0, fill_value="empty", ignore_nulls=True).alias("same"),
        col("x").shift(fill_value="empty").alias("default"),
        col("x")
        .shift(fill_value="empty", ignore_nulls=False)
        .alias("explicit"),
    ]
    var physical = Series(
        "default",
        Column[String](
            ["empty", "", "a", "", "", "b", "", "c"],
            [True, False, True, False, False, True, False, True],
        ),
    )
    var expected = DataFrame(
        [
            Series(
                "lag",
                Column[String](
                    ["", "", "a", "a", "a", "b", "b", "c"],
                    [False, False, True, True, True, True, True, True],
                ),
            ),
            Series(
                "lead2",
                Column[String](
                    ["b", "c", "c", "c", "empty", "empty", "empty", "empty"]
                ),
            ),
            frame.column("x").renamed("same"),
            physical.copy(),
            physical.renamed("explicit"),
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


def test_shift_ignore_nulls_preserves_logical_cells_across_layouts() raises:
    # Mixed validity is [valid,valid,NULL,valid,valid,valid,valid,NULL,valid].
    # These literal source rows count strictly earlier/later valid cells.
    var lag2: List[Int] = [-1, -1, 0, 0, 1, 3, 4, 5, 5]
    var lead2: List[Int] = [3, 4, 4, 5, 6, 8, -1, -1, -1]
    for kind in [
        0,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        22,
        25,
        28,
    ]:
        var frame = DataFrame([selection_column(kind, 9, 0, 1)])
        var expressions: List[Expr] = [
            col("x").shift(2, ignore_nulls=True).alias("lag"),
            col("x").shift(-2, ignore_nulls=True).alias("lead"),
        ]
        for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
            for lazy in [False, True]:
                var actual = (
                    source.lazy()
                    .select_exprs(expressions)
                    .collect(
                        engine="cpu", batch_size=2
                    ) if lazy else source.select_exprs(expressions)
                )
                for name in [String("lag"), "lead"]:
                    assert_true(
                        actual.column(name).dtype()
                        == (
                            DataType.STRING if kind
                            == 13 else selection_dtype(kind)
                        )
                    )
                    var rows = lag2.copy() if name == "lag" else lead2.copy()
                    for row in range(9):
                        if rows[row] < 0:
                            assert_true(actual.item(row, name).is_null())
                        else:
                            assert_selection_cell(
                                actual.item(row, name), kind, rows[row], 1
                            )


def test_first_last_ignore_nulls_ordered_all_null_partitions() raises:
    var frame = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 2, 0, 1, 0, 1, 0, 0])),
            Series("t", Column[Int64]([0, 0, 0, 1, 1, 2, 2, 3, 4])),
            Series(
                "x",
                Column[Int64](
                    [0, 0, 0, 10, 30, 0, 0, 20, 0],
                    [
                        False,
                        False,
                        False,
                        True,
                        True,
                        False,
                        False,
                        True,
                        False,
                    ],
                ),
            ),
        ]
    )
    var expressions: List[Expr] = [
        col("x")
        .first(ignore_nulls=True)
        .over("g", order_by="t")
        .alias("first"),
        col("x").last(ignore_nulls=True).over("g", order_by="t").alias("last"),
        col("x").first().over("g", order_by="t").alias("physical_first"),
        col("x")
        .last(ignore_nulls=False)
        .over("g", order_by="t")
        .alias("physical_last"),
    ]
    var valid: List[Bool] = [
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    var expected = DataFrame(
        [
            Series(
                "first",
                Column[Int64]([10, 30, 0, 10, 30, 10, 30, 10, 10], valid),
            ),
            Series(
                "last",
                Column[Int64]([20, 30, 0, 20, 30, 20, 30, 20, 20], valid),
            ),
            Series.full_null("physical_first", DataType.INT64, 9),
            Series.full_null("physical_last", DataType.INT64, 9),
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
        assert_frame(source.head(0).select_exprs(expressions), expected.head(0))
        assert_frame(
            source.select_exprs(
                [
                    col("x").first(ignore_nulls=True).alias("first"),
                    col("x").last(ignore_nulls=True).alias("last"),
                ]
            ),
            DataFrame(
                [
                    Series("first", Column[Int64]([10])),
                    Series("last", Column[Int64]([20])),
                ]
            ),
        )
        for mapping in [String("join"), "explode"]:
            var expression = (
                col("x")
                .first(ignore_nulls=True)
                .over("g", order_by="t", mapping_strategy=mapping)
                .alias("out")
            )
            var mapped_values: List[Int64] = [10, 30, 0]
            var mapped_valid: List[Bool] = [True, True, False]
            if mapping == "join":
                mapped_values = [10, 30, 0, 10, 30, 10, 30, 10, 10]
                mapped_valid = valid.copy()
            var mapped_expected = DataFrame(
                [Series("out", Column[Int64](mapped_values^, mapped_valid^))]
            )
            assert_frame(source.select(expression), mapped_expected)
            assert_frame(
                source.head(0).select(expression), mapped_expected.head(0)
            )


def test_empty_nested_filtered_window_mapping_retains_child_field_metadata() raises:
    from std.memory import ArcPointer
    from dataframe import ListColumn
    from std.testing import assert_equal

    var child = Series("item", Column[Int64]([11, 22]))
    # Packed Arrow application metadata: k -> v on the nested child field.
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    child._field_metadata = ArcPointer(metadata^)
    var original = DataFrame(
        [
            Series("x", ListColumn([0, 0, 1, 2], child.copy(), [UInt8(6)])),
            Series("g", Column[Int64]([0, 0, 0])),
            Series("t", Column[Int64]([0, 1, 2])),
        ]
    )
    for source in [
        original.copy(),
        chunk_layout(original),
        offset_layout(original),
    ]:
        for empty in [False, True]:
            var frame = source.head(0) if empty else source.copy()
            for mapping in [String("join"), "explode"]:
                var expression = (
                    col("x")
                    .drop_nulls()
                    .over("g", order_by="t", mapping_strategy=mapping)
                    .alias("out")
                )
                for lazy in [False, True]:
                    var actual = (
                        frame.lazy()
                        .select(expression)
                        .collect(
                            engine="cpu", batch_size=1
                        ) if lazy else frame.select(expression)
                    )
                    assert_equal(
                        actual.height(),
                        0 if empty else (3 if mapping == "join" else 2),
                    )
                    var output = actual.column("out")
                    assert_true(
                        output.dtype()
                        == (
                            DataType.list(
                                DataType.list(DataType.INT64)
                            ) if mapping
                            == "join" else DataType.list(DataType.INT64)
                        )
                    )
                    var shifted = (
                        output.list_column().child() if mapping
                        == "join" else output.copy()
                    )
                    var leaf = shifted.list_column().child()
                    assert_true(Bool(leaf._field_metadata))
                    assert_equal(
                        leaf._field_metadata.value()[],
                        child._field_metadata.value()[],
                    )
                    assert_true(
                        leaf._field_metadata.value().ptr()
                        == child._field_metadata.value().ptr()
                    )
                    if not empty:
                        assert_equal(
                            len(shifted), 6 if mapping == "join" else 2
                        )
                        assert_equal(
                            shifted.get(0).list().get(0).int64(), Int64(11)
                        )
                        assert_equal(
                            shifted.get(1).list().get(0).int64(), Int64(22)
                        )
