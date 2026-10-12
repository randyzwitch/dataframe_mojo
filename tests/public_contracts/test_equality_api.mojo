"""Structural equality is a public result, checked against explicit truth values."""
from std.testing import assert_equal
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    ListColumn,
    StructColumn,
)
from tests.support.coverage_generators import offset_layout, chunk_layout


def _layout(frame: DataFrame, layout: Int) raises -> DataFrame:
    return (
        frame.copy() if layout
        == 0 else offset_layout(frame) if layout
        == 1 else chunk_layout(frame)
    )


def test_dataframe_and_series_equals_compare_structure_and_valid_cells() raises:
    var nan = Float64(0) / Float64(0)
    var source = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [nan, -0.0, 7.0, 999.0], [True, True, True, False]
                ),
            ),
            Series(
                "s",
                Column[String](
                    ["é", "", "a\x00b", "hidden"], [True, True, True, False]
                ),
            ),
        ]
    )
    var same = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [nan, 0.0, 7.0, -999.0], [True, True, True, False]
                ),
            ),
            Series(
                "s",
                Column[String](
                    ["é", "", "a\x00b", "different"], [True, True, True, False]
                ),
            ),
        ]
    )
    for left_layout in range(3):
        for right_layout in range(3):
            var left = _layout(source, left_layout)
            var right = _layout(same, right_layout)
            assert_equal(left.equals(right), True)
            assert_equal(left.equals(right, null_equal=False), False)
            assert_equal(
                left.head(3).equals(right.head(3), null_equal=False), True
            )
            assert_equal(left.equals(right.head(3)), False)
            assert_equal(left.equals(right.select(["s", "x"])), False)
            assert_equal(left.equals(right.rename({"x": "renamed"})), False)
            assert_equal(
                left.equals(
                    right.with_column(
                        Series(
                            "x",
                            Column[Float32](
                                [nan.cast[DType.float32](), 0.0, 7.0, 1.0],
                                [True, True, True, False],
                            ),
                        )
                    )
                ),
                False,
            )
            assert_equal(left.equals(right.take([1, 0, 2, 3])), False)
            assert_equal(
                left.equals(
                    right.with_column(
                        Series(
                            "x",
                            Column[Float64](
                                [nan, 0.0, 8.0, 1.0], [True, True, True, False]
                            ),
                        )
                    )
                ),
                False,
            )
            assert_equal(
                left.equals(
                    right.with_column(
                        Series("x", Column[Float64]([nan, 0.0, 7.0, 1.0]))
                    )
                ),
                False,
            )
            var renamed = right.column("x").renamed("other")
            assert_equal(left.column("x").equals(renamed), True)
            assert_equal(
                left.column("x").equals(renamed, check_names=True), False
            )
            assert_equal(
                left.column("x").equals(right.column("x"), null_equal=False),
                False,
            )
            assert_equal(left.clear().equals(right.clear()), True)
            assert_equal(
                left.clear().equals(right.clear(), null_equal=False), True
            )


def _nested(hidden: Int64, visible: Int64) raises -> DataFrame:
    var list = Series(
        "list",
        ListColumn(
            [0, 2, 3],
            Series("child", Column[Int64]([1, visible, hidden])),
            [UInt8(1)],
        ),
    )
    var structure = Series(
        "struct",
        StructColumn(
            [Series("field", Column[Int64]([visible, 123]))], [UInt8(1)]
        ),
    )
    return DataFrame([list^, structure^])


def test_structural_equals_reads_nested_children_and_logical_metadata() raises:
    var left = _nested(123, 2)
    var same = _nested(-456, 2)
    var changed = _nested(123, 3)
    for layout in range(3):
        var frame = _layout(left, layout)
        assert_equal(frame.equals(same), True)
        assert_equal(frame.equals(changed), False)
        assert_equal(frame.equals(same, null_equal=False), False)
        assert_equal(frame.column("list").equals(same.column("list")), True)
        assert_equal(
            frame.column("struct").equals(changed.column("struct")), False
        )
    var logical = Series("x", Column[Int64]([1, 2])).with_dtype(DataType.DATE)
    var physical = Series("x", Column[Int64]([1, 2]))
    assert_equal(logical.equals(physical), False)
    assert_equal(DataFrame([logical^]).equals(DataFrame([physical^])), False)
    var bytes: List[List[UInt8]] = [[0, 255], [1]]
    var other: List[List[UInt8]] = [[0, 254], [1]]
    assert_equal(
        Series.binary("b", bytes).equals(Series.binary("b", bytes)), True
    )
    assert_equal(
        Series.binary("b", bytes).equals(Series.binary("b", other)), False
    )


def test_structural_equals_ignores_children_hidden_by_a_null_struct_parent() raises:
    # Semantics specifies that null payloads never participate. A struct
    # field's physical validity does not make its null parent's cell valid.
    var left = DataFrame(
        [
            Series(
                "x",
                StructColumn(
                    [Series("field", Column[Int64]([123]))], [UInt8(0)]
                ),
            )
        ]
    )
    var right = DataFrame(
        [
            Series(
                "x",
                StructColumn(
                    [Series("field", Column[Int64]([-456]))], [UInt8(0)]
                ),
            )
        ]
    )
    assert_equal(left.equals(left), True)
    assert_equal(left.equals(right, null_equal=False), False)
    if not left.equals(right):
        raise Error(
            "Structural equality compared child payload hidden by a null struct parent"
        )
