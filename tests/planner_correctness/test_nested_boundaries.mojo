"""Lazy nested shape changes retain hidden dependencies and row boundaries."""
# contract-group: planner.relational
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    ListColumn,
    Series,
    StructColumn,
    col,
)
from tests.support.coverage_generators import chunk_layout, offset_layout


def nested_boundary_source() raises -> DataFrame:
    var items = List[Series]()
    items.append(Series("item", Column[Int64]([10, 11])))
    items.append(Series("item", Column[Int64](List[Int64]())))
    items.append(Series("item", Column[Int64](List[Int64]())))
    items.append(Series("item", Column[Int64]([Int64.MAX, 13], [False, True])))
    items.append(Series("item", Column[Int64]([14])))
    items.append(Series("item", Column[Int64]([15, 16, 17])))
    items.append(Series("item", Column[Int64](List[Int64]())))
    items.append(Series("item", Column[Int64](List[Int64]())))
    items.append(Series("item", Column[Int64]([18, 19])))
    return DataFrame(
        [
            Series("id", Column[Int64]([0, 1, 2, 3, 4, 5, 6, 7, 8])),
            Series(
                "xs",
                ListColumn.from_lists(
                    items,
                    DataType.INT64,
                    [True, True, False, True, True, True, True, False, True],
                ),
            ),
            Series(
                "label",
                Column[String](
                    [
                        "zero",
                        "one",
                        "two",
                        "three",
                        "four",
                        "five",
                        "six",
                        "seven",
                        "eight",
                    ]
                ),
            ),
        ]
    )


def nested_boundary_layout(frame: DataFrame, layout: Int) raises -> DataFrame:
    return (
        frame.copy() if layout
        == 0 else offset_layout(frame) if layout
        == 1 else chunk_layout(frame)
    )


def test_lazy_explode_keeps_unprojected_list_and_filter_columns_and_preserves_slice_cardinality() raises:
    var source = nested_boundary_source()
    var all_ids: List[Int64] = [0, 0, 1, 2, 3, 3, 4, 5, 5, 5, 6, 7, 8, 8]
    var filtered_values: List[Int64] = [0, 13, 14, 15, 16, 17, 0, 0, 18, 19]
    var filtered_valid: List[Bool] = [
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    for empty in [False, True]:
        var base = source.clear() if empty else source.copy()
        for layout in range(3):
            var frame = nested_boundary_layout(base, layout)
            for optimize in [False, True]:
                for streaming in [False, True]:
                    for batch in [1, 7]:
                        var projected = (
                            frame.lazy()
                            .explode("xs")
                            .select(["id"])
                            .collect(
                                engine="cpu",
                                optimize=optimize,
                                streaming=streaming,
                                batch_size=batch,
                            )
                        )
                        assert_equal(projected.columns(), List[String](["id"]))
                        assert_equal(projected.height(), 0 if empty else 14)
                        assert_true(
                            projected.column("id").dtype() == DataType.INT64
                        )
                        for row in range(projected.height()):
                            assert_equal(
                                projected.item(row, "id").int64(), all_ids[row]
                            )
                        var filtered = (
                            frame.lazy()
                            .explode("xs")
                            .filter(col("id") > 2)
                            .select(["xs"])
                            .collect(
                                engine="cpu",
                                optimize=optimize,
                                streaming=streaming,
                                batch_size=batch,
                            )
                        )
                        assert_equal(filtered.columns(), List[String](["xs"]))
                        assert_equal(filtered.height(), 0 if empty else 10)
                        assert_true(
                            filtered.column("xs").dtype() == DataType.INT64
                        )
                        for row in range(filtered.height()):
                            var cell = filtered.item(row, "xs")
                            assert_equal(
                                cell.is_null(), not filtered_valid[row]
                            )
                            if filtered_valid[row]:
                                assert_equal(cell.int64(), filtered_values[row])
                        var sliced = (
                            frame.lazy()
                            .explode("xs")
                            .select(["id"])
                            .slice(2, 5)
                            .filter(col("id") > 2)
                            .collect(
                                engine="cpu",
                                optimize=optimize,
                                streaming=streaming,
                                batch_size=batch,
                            )
                        )
                        var slice_expected: List[Int64] = [3, 3, 4]
                        assert_equal(sliced.height(), 0 if empty else 3)
                        for row in range(sliced.height()):
                            assert_equal(
                                sliced.item(row, "id").int64(),
                                slice_expected[row],
                            )


def test_lazy_unnest_retains_struct_filter_dependencies_and_parent_nulls_after_projection() raises:
    var original = nested_boundary_source()
    var parent_bits: List[UInt8] = [UInt8(0b11101101), UInt8(0)]
    var source = DataFrame(
        [
            Series(
                "record",
                StructColumn(
                    [
                        original.column("id"),
                        original.column("label"),
                        original.column("xs"),
                    ],
                    parent_bits^,
                ),
            ),
            Series("keep", Column[Int64]([10, 11, 12, 13, 14, 15, 16, 17, 18])),
        ]
    )
    var expected: List[String] = ["three", "five", "six", "seven"]
    for empty in [False, True]:
        var base = source.clear() if empty else source.copy()
        for layout in range(3):
            var frame = nested_boundary_layout(base, layout)
            for optimize in [False, True]:
                for streaming in [False, True]:
                    for batch in [1, 7]:
                        var actual = (
                            frame.lazy()
                            .unnest("record")
                            .filter(col("id") >= 3)
                            .select(["label"])
                            .collect(
                                engine="cpu",
                                optimize=optimize,
                                streaming=streaming,
                                batch_size=batch,
                            )
                        )
                        assert_equal(actual.columns(), List[String](["label"]))
                        assert_equal(actual.height(), 0 if empty else 4)
                        assert_true(
                            actual.column("label").dtype() == DataType.STRING
                        )
                        for row in range(actual.height()):
                            assert_equal(
                                actual.item(row, "label").string(),
                                expected[row],
                            )
                        var sliced = (
                            frame.lazy()
                            .unnest("record")
                            .select(["label"])
                            .slice(1, 3)
                            .collect(
                                engine="cpu",
                                optimize=optimize,
                                streaming=streaming,
                                batch_size=batch,
                            )
                        )
                        assert_equal(sliced.height(), 0 if empty else 3)
                        if not empty:
                            assert_true(sliced.item(0, "label").is_null())
                            assert_equal(
                                sliced.item(1, "label").string(), "two"
                            )
                            assert_equal(
                                sliced.item(2, "label").string(), "three"
                            )
