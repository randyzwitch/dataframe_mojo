"""Binding precedes evaluation; nested branches evaluate only selected rows."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_nested_and_chained_branch_masks_suppress_only_unselected_overflow() raises:
    var raw = DataFrame(
        [Series("x", Column[Int64]([Int64.MIN, Int64.MAX, -2, 2]))]
    )
    var expression = (
        when(col("x") == lit(Int64.MIN))
        .then(0)
        .when(col("x") > 0)
        .then(
            when(col("x") == lit(Int64.MAX))
            .then(col("x"))
            .otherwise(lit(Int64.MAX - 2) + col("x"))
        )
        .otherwise(-col("x"))
    )
    var scalar = (
        when(col("x") < lit(Int64.MIN))
        .then(lit(Int64.MAX) + 1)
        .otherwise(col("x"))
    )
    var quotient = (
        when(col("x") == lit(Int64.MIN)).then(0).otherwise(col("x") // -1)
    )
    var expected: List[List[Int64]] = [
        [0, Int64.MAX, 2, Int64.MAX],
        [Int64.MIN, Int64.MAX, -2, 2],
        [0, -Int64.MAX, 2, -2],
    ]
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        for batch in [1, 3, 64]:
            var actual = frame.select_exprs(
                [
                    expression.copy().alias("branch"),
                    scalar.copy().alias("scalar"),
                    quotient.copy().alias("quotient"),
                ],
                batch_size=batch,
            )
            assert_equal(actual.columns(), ["branch", "scalar", "quotient"])
            assert_equal(actual.height(), 4)
            for operation in range(3):
                var name = ["branch", "scalar", "quotient"][operation]
                assert_true(actual.column(name).dtype() == DataType.INT64)
                for row in range(4):
                    assert_equal(
                        actual.item(row, name).int64(), expected[operation][row]
                    )
        with assert_raises(contains="overflow"):
            _ = frame.select(
                when(col("x") > 0).then(lit(Int64.MAX) + col("x")).otherwise(0)
            )
        assert_equal(frame.clear().select(expression.copy()).height(), 0)


def test_binding_error_precedes_overflow_and_aggregate_conditions_keep_group_shapes() raises:
    var dangerous = DataFrame([Series("x", Column[Int64]([Int64.MAX, 1]))])
    for frame in [dangerous.copy(), dangerous.clear()]:
        with assert_raises(contains="Unknown expression column"):
            _ = frame.select_exprs(
                [col("x").sum().alias("overflow"), col("missing")]
            )
        with assert_raises(contains="Unknown expression column"):
            _ = frame.select(
                when(lit(False)).then(col("missing")).otherwise(col("x"))
            )
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([2, -1, -3, 1])),
            Series("g", Column[Int64]([1, 1, 0, 0])),
        ]
    )
    var selection = (
        when(col("x").sum() > 0)
        .then(col("x").max())
        .otherwise(col("x").min())
        .alias("selected")
    )
    var grouped = frame.group_by("g", maintain_order=True).agg(
        [selection.copy()]
    )
    assert_equal(grouped.columns(), ["g", "selected"])
    assert_equal(grouped.height(), 2)
    assert_true(grouped.column("selected").dtype() == DataType.INT64)
    for row in range(2):
        assert_equal(grouped.item(row, "g").int64(), Int64(1 - row))
        assert_equal(
            grouped.item(row, "selected").int64(), Int64(2 if row == 0 else -3)
        )
    var global_result = frame.select(selection.copy())
    assert_equal(global_result.shape(), (1, 1))
    assert_equal(global_result.item().int64(), Int64(-3))
    var broadcast = frame.with_columns(selection.copy())
    assert_equal(broadcast.height(), 4)
    for row in range(4):
        assert_equal(broadcast.item(row, "selected").int64(), Int64(-3))
