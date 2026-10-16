"""Variable-height distributions have explicit projection and grouping shapes."""
# contract-group: reduce.global
# contract-group: reduce.grouped
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col, lit


def test_mode_and_value_counts_reject_row_broadcast_and_over_even_when_empty() raises:
    var raw = DataFrame(
        [
            Series(
                "x",
                Column[Int64]([3, 3, 1, 1, 9], [True, True, True, True, False]),
            ),
            Series("g", Column[Int64]([0, 1, 0, 1, 1])),
        ]
    )
    for frame in [raw.copy(), raw.clear()]:
        for distribution in [col("x").mode(), col("x").value_counts()]:
            with assert_raises(
                contains="cannot be selected with row-valued expressions"
            ):
                _ = frame.select_exprs(
                    [distribution.copy().alias("distribution"), col("g")]
                )
            with assert_raises(contains="cannot be used in over()"):
                _ = frame.select(distribution.copy().over("g"))
        var projected = frame.select_exprs(
            [col("x").mode().alias("m"), lit(Int64(7)).alias("scalar")]
        )
        assert_equal(projected.columns(), ["m", "scalar"])
        assert_equal(projected.height(), 0 if frame.height() == 0 else 2)
        assert_equal(projected.dtypes(), [DataType.INT64, DataType.INT64])
        for row in range(projected.height()):
            assert_equal(
                projected.item(row, "m").int64(), Int64(1 if row == 0 else 3)
            )
            assert_equal(projected.item(row, "scalar").int64(), Int64(7))
        var grouped = frame.group_by("g", maintain_order=True).agg(
            [col("x").mode().alias("m")]
        )
        assert_equal(grouped.height(), 0 if frame.height() == 0 else 2)
        assert_true(
            grouped.column("m").dtype() == DataType.list(DataType.INT64)
        )
        for row in range(grouped.height()):
            var values = grouped.item(row, "m").list()
            assert_equal(len(values), 2 if row == 0 else 3)
            for child in range(2):
                assert_equal(
                    values.get(child).int64(), Int64(1 if child == 0 else 3)
                )
            if row == 1:
                assert_true(values.get(2).is_null())
