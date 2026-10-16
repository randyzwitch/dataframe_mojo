"""Batch boundaries cannot reset order-dependent or partition-local state."""
# contract-group: expr.order_windows
# contract-group: execution.modes
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_float
from tests.support.window_reference import (
    WindowCells,
    cumulative,
    shifted,
    rolling,
    ranked,
    filled,
)


def test_order_windows_are_computed_over_whole_partitions_before_batch_evaluation() raises:
    var n = 65
    var payload = List[Float64]()
    var valid = List[Bool]()
    var ids = List[Int64]()
    var keys = List[Int64]()
    var key_valid = List[Bool]()
    for row in range(n):
        payload.append(Float64((row * 7) % 13 - 6))
        valid.append(row % 5 > 1)
        ids.append(Int64(row))
        keys.append(Int64(row % 3))
        key_valid.append(row % 3 != 2)
    var original = DataFrame(
        [
            Series("x", Column[Float64](payload.copy(), valid.copy())),
            Series("id", Column[Int64](ids^)),
            Series("k", Column[Int64](keys^, key_valid^)),
        ]
    )
    for layout in range(3):
        var frame = (
            original.copy() if layout
            == 0 else offset_layout(original) if layout
            == 1 else chunk_layout(original)
        )
        for partitioned in [False, True]:
            var expressions: List[Expr] = [
                col("x").cum_sum().alias("sum"),
                col("x").cum_sum(True).alias("reverse"),
                col("x").shift(2).alias("shift"),
                col("x").rolling_sum(3, 1).alias("rolling"),
                col("x").rank().alias("rank"),
                col("x").forward_fill(1).alias("fill"),
            ]
            if partitioned:
                for index in range(len(expressions)):
                    expressions[index] = expressions[index].over("k")
            expressions.append(col("id"))
            for batch in [1, 7, 64, 257]:
                var actual = frame.select_exprs(expressions, batch_size=batch)
                assert_equal(
                    actual.columns(),
                    [
                        "sum",
                        "reverse",
                        "shift",
                        "rolling",
                        "rank",
                        "fill",
                        "id",
                    ],
                )
                assert_equal(actual.height(), n)
                for row in range(n):
                    assert_equal(actual.item(row, "id").int64(), Int64(row))
                for group in range(3 if partitioned else 1):
                    var indices = List[Int]()
                    var values = List[Float64]()
                    var mask = List[Bool]()
                    for row in range(n):
                        if not partitioned or row % 3 == group:
                            indices.append(row)
                            values.append(payload[row])
                            mask.append(valid[row])
                    var cells = WindowCells(values^, mask^)
                    var expected: List[WindowCells] = [
                        cumulative(cells, "sum", False),
                        cumulative(cells, "sum", True),
                        shifted(cells, 2),
                        rolling(cells, 3, 1, "sum"),
                        ranked(cells, "average", False),
                        filled(cells, False, 1),
                    ]
                    var names: List[String] = [
                        "sum",
                        "reverse",
                        "shift",
                        "rolling",
                        "rank",
                        "fill",
                    ]
                    for column in range(len(names)):
                        assert_true(
                            actual.column(names[column]).dtype()
                            == DataType.FLOAT64
                        )
                        for index in range(len(indices)):
                            var value = actual.item(
                                indices[index], names[column]
                            )
                            assert_equal(
                                value.is_null(),
                                not expected[column].valid[index],
                            )
                            if expected[column].valid[index]:
                                assert_float(
                                    value.float64(),
                                    expected[column].values[index],
                                )
