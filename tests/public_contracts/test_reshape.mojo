"""Schema transforms and cell associations, rather than reshape roundtrips."""
# contract-group: expr.nested
# contract-group: table.sets_reshape
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, ListColumn, Series, concat
from tests.support.reference_model import generated_model, ReferenceTable
from tests.support.reference_relations import pivot_model
from tests.support.reference_statistics import sorted_integers, integer_quantile
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import (
    assert_reference_cell,
    reference_dtype,
)
from tests.support.reduction_contract import assert_statistic


def test_unpivot_orders_columns_then_rows_and_preserves_nulls() raises:
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            var left = generated_model(n, 3, mode)
            var right = generated_model(n, 101, 1)
            for layout in range(3):
                var frame = frame_from_model(left, 7 if layout == 1 else 0)
                frame = frame.with_column(
                    frame_from_model(right).column("i").renamed("j")
                )
                if layout == 2:
                    frame = chunk_layout(frame)
                var actual = frame.unpivot(
                    ["j", "i"],
                    ["s", "b"],
                    variable_name="source",
                    value_name="value",
                )
                assert_equal(actual.columns(), ["s", "b", "source", "value"])
                assert_equal(
                    actual.dtypes(),
                    [
                        DataType.STRING,
                        DataType.BOOL,
                        DataType.STRING,
                        DataType.INT64,
                    ],
                )
                assert_equal(actual.height(), n * 2)
                var default_on = frame.select(["j", "i"]).unpivot()
                assert_equal(default_on.columns(), ["variable", "value"])
                assert_equal(default_on.height(), n * 2)
                for column in range(2):
                    var name = "j" if column == 0 else "i"
                    for row in range(n):
                        var position = column * n + row
                        assert_reference_cell(
                            actual.item(position, "s"), left.rows[row], "s"
                        )
                        assert_reference_cell(
                            actual.item(position, "b"), left.rows[row], "b"
                        )
                        assert_equal(
                            actual.item(position, "source").string(), name
                        )
                        var expected = (
                            right.rows[row].copy() if column
                            == 0 else left.rows[row].copy()
                        )
                        assert_reference_cell(
                            actual.item(position, "value"), expected, "i"
                        )
                        assert_equal(
                            default_on.item(position, "variable").string(), name
                        )
                        assert_reference_cell(
                            default_on.item(position, "value"), expected, "i"
                        )
                with assert_raises():
                    _ = frame.unpivot(["i", "s"])
                with assert_raises():
                    _ = frame.unpivot(["i"], ["i"])
                with assert_raises():
                    _ = frame.unpivot(["i"], ["s", "s"])
                with assert_raises():
                    _ = frame.unpivot(["missing"])
                with assert_raises():
                    _ = frame.unpivot(["i"], ["s"], variable_name="s")
                with assert_raises():
                    _ = frame.unpivot(["i"], value_name="variable")
                with assert_raises():
                    _ = frame.unpivot(index=frame.columns())


def assert_pivot_cells(
    actual: DataFrame,
    model: ReferenceTable,
    index: List[String],
    function: String,
    sorted_columns: Bool,
) raises:
    var expected = pivot_model(model, index, sorted_columns)
    var names = index.copy()
    names.extend(expected.labels.copy())
    assert_equal(actual.columns(), names)
    assert_equal(actual.height(), len(expected.representatives))
    for key in index:
        assert_true(actual.column(key).dtype() == reference_dtype(key))
    var floating = function == "mean" or function == "median"
    for label in expected.labels:
        assert_true(
            actual.column(label).dtype()
            == (DataType.FLOAT64 if floating else DataType.INT64)
        )
    for row in range(len(expected.representatives)):
        for key in index:
            assert_reference_cell(
                actual.item(row, key),
                model.rows[expected.representatives[row]],
                key,
            )
        for column in range(len(expected.labels)):
            ref indices = expected.cells[row * len(expected.labels) + column]
            var cell = actual.item(row, expected.labels[column])
            if not indices:
                assert_true(cell.is_null())
                continue
            var values = model.take(indices)
            if function == "first" or function == "last" or function == "":
                var position = len(values.rows) - 1 if function == "last" else 0
                assert_reference_cell(cell, values.rows[position], "i")
            elif function == "count":
                assert_equal(cell.int64(), values.int_count())
            elif function == "len":
                assert_equal(cell.int64(), Int64(len(values.rows)))
            elif function == "sum":
                assert_equal(cell.int64(), Int64(values.int_sum()))
            else:
                assert_equal(cell.is_null(), values.int_count() == 0)
                if not values.int_count():
                    continue
                if function == "mean":
                    assert_statistic(
                        cell.float64(),
                        Float64(values.int_sum()) / Float64(values.int_count()),
                    )
                elif function == "median":
                    assert_statistic(
                        cell.float64(), integer_quantile(values, 0.5)
                    )
                else:
                    var ordered = sorted_integers(values)
                    assert_equal(
                        cell.int64(),
                        ordered[0 if function == "min" else len(ordered) - 1],
                    )


def test_pivot_associations_missing_cells_and_aggregate_policies() raises:
    var indices: List[List[String]] = [[], ["b"], ["f", "b"]]
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            var model = generated_model(n, 3, mode)
            for layout in range(3):
                var frame = frame_from_model(model, 7 if layout == 1 else 0)
                if layout == 2:
                    frame = chunk_layout(frame)
                for index in indices:
                    for sorted_columns in [False, True]:
                        var expected = pivot_model(model, index, sorted_columns)
                        var has_collision = False
                        for cell in expected.cells:
                            has_collision = has_collision or len(cell) > 1
                        if has_collision:
                            with assert_raises():
                                _ = frame.pivot(
                                    "s",
                                    index=index,
                                    values="i",
                                    sort_columns=sorted_columns,
                                )
                        else:
                            assert_pivot_cells(
                                frame.pivot(
                                    "s",
                                    index=index,
                                    values="i",
                                    sort_columns=sorted_columns,
                                ),
                                model,
                                index,
                                "",
                                sorted_columns,
                            )
                        for function in [
                            "first",
                            "last",
                            "sum",
                            "mean",
                            "min",
                            "max",
                            "count",
                            "len",
                            "median",
                        ]:
                            assert_pivot_cells(
                                frame.pivot(
                                    "s",
                                    index=index,
                                    values="i",
                                    aggregate_function=function,
                                    sort_columns=sorted_columns,
                                    batch_size=7,
                                ),
                                model,
                                index,
                                function,
                                sorted_columns,
                            )


def test_pivot_names_and_arguments_are_validated_on_empty_and_nonempty_input() raises:
    var frame = DataFrame(
        [
            Series("key", Column[Int64]([1, 1])),
            Series("label", Column[String](["x", "x"])),
            Series("value", Column[Int64]([2, 3])),
        ]
    )
    for source in [frame.copy(), frame.clear()]:
        with assert_raises():
            _ = source.pivot(
                "label",
                index=["key"],
                values="value",
                aggregate_function="unknown",
            )
        var invalid_indices: List[List[String]] = [
            ["key", "key"],
            ["label"],
            ["value"],
            ["missing"],
        ]
        for index in invalid_indices:
            with assert_raises():
                _ = source.pivot("label", index=index, values="value")
        with assert_raises():
            _ = source.pivot("missing", index=["key"], values="value")
        with assert_raises():
            _ = source.pivot("label", index=["key"], values="missing")
    var collision = frame.with_column(
        Series("label", Column[String](["key", "key"]))
    )
    with assert_raises():
        _ = collision.pivot(
            "label", index=["key"], values="value", aggregate_function="sum"
        )
    var null_label = frame.with_column(
        Series("label", Column[String](["null", "poison"], [True, False]))
    )
    with assert_raises():
        _ = null_label.pivot(
            "label", index=["key"], values="value", aggregate_function="sum"
        )


def test_explode_repeats_other_columns_and_distinguishes_null_lists_from_children() raises:
    var frame = DataFrame(
        [
            Series("id", Column[Int64]([10, 11, 12, 13])),
            Series(
                "list",
                ListColumn(
                    [0, 2, 2, 2, 3],
                    Series(
                        "child",
                        Column[Int64]([1, 2, 3], [True, False, True]),
                    ),
                    [UInt8(0b1011)],
                ),
            ),
            Series("text", Column[String](["a", "b", "c", "d"])),
        ]
    )
    var ids: List[Int64] = [10, 10, 11, 12, 13]
    var values: List[Int64] = [1, 0, 0, 0, 3]
    var validity: List[Bool] = [True, False, False, False, True]
    var texts: List[String] = ["a", "a", "b", "c", "d"]
    for layout in range(3):
        var input = (
            frame.copy() if layout
            == 0 else concat([frame.head(1), frame.copy()])
            .rechunk()
            .slice(1, 4) if layout
            == 1 else chunk_layout(frame)
        )
        var actual = input.explode("list")
        assert_equal(actual.columns(), ["id", "list", "text"])
        assert_equal(
            actual.dtypes(), [DataType.INT64, DataType.INT64, DataType.STRING]
        )
        assert_equal(actual.height(), 5)
        for row in range(5):
            assert_equal(actual.item(row, "id").int64(), ids[row])
            assert_equal(actual.item(row, "text").string(), texts[row])
            assert_equal(actual.item(row, "list").is_null(), not validity[row])
            if validity[row]:
                assert_equal(actual.item(row, "list").int64(), values[row])
        assert_equal(input.clear().explode("list").shape(), (0, 3))
        var paired = input.with_column(input.column("list").renamed("other"))
        var both = paired.explode(["list", "other"])
        assert_equal(both.height(), 5)
        for row in range(5):
            assert_equal(both.item(row, "other").is_null(), not validity[row])
            if validity[row]:
                assert_equal(both.item(row, "other").int64(), values[row])
        var invalid_columns: List[List[String]] = [
            [],
            ["list", "list"],
            ["id"],
            ["missing"],
        ]
        for columns in invalid_columns:
            with assert_raises():
                _ = input.explode(columns)
    var mismatched = frame.with_column(
        Series(
            "other",
            ListColumn([0, 1, 1, 1, 2], Series("child", Column[Int64]([1, 2]))),
        )
    )
    with assert_raises():
        _ = mismatched.explode(["list", "other"])


def test_explode_null_list_never_exposes_hidden_child_span() raises:
    var frame = DataFrame(
        [
            Series("id", Column[Int64]([7])),
            Series(
                "list",
                ListColumn(
                    [0, 2], Series("child", Column[Int64]([90, 91])), [UInt8(0)]
                ),
            ),
        ]
    )
    assert_true(frame.item(0, "list").is_null())
    var actual = frame.explode("list")
    assert_true(
        actual.height() == 1, "null-list explode must produce exactly one row"
    )
    assert_true(actual.item(0, "list").is_null())
    assert_equal(actual.item(0, "id").int64(), Int64(7))
