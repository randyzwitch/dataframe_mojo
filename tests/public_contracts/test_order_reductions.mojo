"""Public positional reductions retain terminal nulls and logical schemas."""
# contract-group: reduce.global
# contract-group: reduce.grouped
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col
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


def test_public_lazy_reductions_without_inline_state_match_eager() raises:
    # Positions of extremes, products and moments have no fixed-size state:
    # the ungrouped sink holds them by address, each worker's rows in its
    # own state, and the finish must replay them in input order (the first
    # of many tied minima is the earliest row overall).
    var n = 300_000
    var x = List[Float64](capacity=n)
    var near_one = List[Float64](capacity=n)
    var keep = List[Int64](capacity=n)
    for i in range(n):
        x.append(Float64((i * 7919) % 1000))
        near_one.append(1.0 + Float64(i % 3 - 1) * 1e-6)
        keep.append(Int64(i % 5))
    var frame = DataFrame(
        [
            Series("x", Column[Float64](x^)),
            Series("p", Column[Float64](near_one^)),
            Series("k", Column[Int64](keep^)),
        ]
    )
    var expressions: List[Expr] = [
        col("x").arg_min().alias("arg_min"),
        col("x").arg_max().alias("arg_max"),
        col("x").skew().alias("skew"),
        col("x").kurtosis().alias("kurtosis"),
        col("p").product().alias("product"),
        col("x").sum().alias("sum"),
    ]
    for filtered in [False, True]:
        var input = frame.filter(col("k") != 2) if filtered else frame.copy()
        var expected = input.select_exprs(expressions)
        var query = (
            frame.lazy()
            .filter(col("k") != 2)
            .select_exprs(expressions) if filtered else frame.lazy()
            .select_exprs(expressions)
        )
        for streaming in [False, True]:
            var actual = query.collect(engine="cpu", streaming=streaming)
            assert_equal(actual.height(), 1)
            for name in ["arg_min", "arg_max"]:
                assert_value(actual.item(0, name), expected.item(0, name))
            for name in ["skew", "kurtosis", "product", "sum"]:
                var a = actual.item(0, name).float64()
                var b = expected.item(0, name).float64()
                assert_true(
                    abs(a - b) <= 1e-9 * max(1.0, abs(b)),
                    String(name, ": ", a, " != ", b),
                )
