"""Public positional reductions retain terminal nulls and logical schemas."""
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, col
from tests.support.coverage_compare import assert_value
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.ordered_reduction_fixtures import typed_ordered_columns


def test_public_first_last_preserve_positions_nulls_and_logical_schemas() raises:
    var groups: List[Int] = [1, 0, 1, 2, 0]
    var order: List[Int] = [1, 0, 2]
    var first_rows: List[Int] = [0, 1, 3]
    var last_rows: List[Int] = [2, 4, 3]
    for mode in range(4):
        var valid = List[Bool](length=5, fill=mode != 3)
        if mode == 1:
            valid[0] = False
        if mode == 2:
            valid[4] = False
        for original in typed_ordered_columns(valid):
            if original.dtype().is_categorical():
                continue
            var raw_groups = List[Int64]()
            for group in groups:
                raw_groups.append(Int64(group))
            var input = DataFrame(
                [original.copy(), Series("g", Column[Int64](raw_groups^))]
            )
            for layout in range(3):
                var frame = (
                    input.copy() if layout
                    == 0 else offset_layout(input) if layout
                    == 1 else chunk_layout(input)
                )
                for last in [False, True]:
                    var expression = (
                        col("x").last() if last else col("x").first()
                    )
                    var global_result = frame.select(expression.alias("value"))
                    assert_equal(global_result.columns(), ["value"])
                    assert_equal(global_result.height(), 1)
                    assert_true(
                        global_result.column("value").dtype()
                        == original.dtype()
                    )
                    var row = 4 if last else 0
                    assert_equal(
                        global_result.item(0, "value").is_null(), not valid[row]
                    )
                    assert_value(
                        global_result.item(0, "value"), original.get(row)
                    )
                    var empty = frame.clear().select(expression.alias("value"))
                    assert_equal(empty.shape(), (1, 1))
                    assert_true(
                        empty.column("value").dtype() == original.dtype()
                    )
                    assert_true(empty.item().is_null())
                    var grouped = frame.group_by("g", maintain_order=True).agg(
                        [expression.alias("value")]
                    )
                    assert_equal(grouped.columns(), ["g", "value"])
                    assert_equal(grouped.height(), 3)
                    assert_true(grouped.column("g").dtype() == DataType.INT64)
                    assert_true(
                        grouped.column("value").dtype() == original.dtype()
                    )
                    for output in range(3):
                        assert_equal(
                            grouped.item(output, "g").int64(),
                            Int64(order[output]),
                        )
                        var selected = last_rows[
                            output
                        ] if last else first_rows[output]
                        assert_equal(
                            grouped.item(output, "value").is_null(),
                            not valid[selected],
                        )
                        assert_value(
                            grouped.item(output, "value"),
                            original.get(selected),
                        )


def test_public_categorical_first_last_preserve_documented_result_dtype() raises:
    # Multi-node expressions read categorical values as String. Direct ordered
    # Reducer states retain storage dtype; that is a separate kernel contract.
    var source = Series("x", Column[String](["z", "a"])).cast(
        DataType.CATEGORICAL
    )
    var frame = DataFrame([source.copy()])
    for last in [False, True]:
        var expression = col("x").last() if last else col("x").first()
        var actual = frame.select(expression)
        assert_equal(actual.height(), 1)
        assert_true(
            actual.column("x").dtype() == DataType.STRING,
            "categorical first/last expressions must return decoded String",
        )
        assert_value(actual.item(), source.get(1 if last else 0))
