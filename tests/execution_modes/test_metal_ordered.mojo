"""Stable native ordered reductions: scalar ordering and schema admission."""
# contract-group: execution.backends
from std.memory import bitcast
from std.math import isnan
from std.testing import assert_true, assert_false, assert_equal
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from dataframe.expr import FIRST, LAST, ARG_MIN, ARG_MAX
from dataframe_metal.ordered_plan import plan_ordered, ordered_key
from dataframe_metal.accelerator.gpu.ordered_value import (
    ordered_better,
    ordered_empty,
    ordered_ticket_row,
)


def test_metal_ordered_float_comparison_preserves_nan_zero_and_first_ties() raises:
    var values: List[UInt32] = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x3F800000,
        0xBF800000,
        0x7F800000,
        0xFF800000,
        0x7F800001,
        0x7FC12345,
        0xFFC54321,
    ]
    for a in values:
        for b in values:
            var x = bitcast[DType.float32](a)
            var y = bitcast[DType.float32](b)
            for minimum in [False, True]:
                for row in [
                    UInt32(0),
                    UInt32(1),
                    UInt32(128),
                    UInt32(Int32.MAX - 1),
                ]:
                    var old_row = UInt32(64)
                    var expected = False
                    if isnan(x) or isnan(y):
                        expected = row < old_row if isnan(x) and isnan(
                            y
                        ) else not isnan(x)
                    elif x == y:
                        expected = row < old_row
                    else:
                        expected = x < y if minimum else x > y
                    assert_equal(
                        ordered_better(
                            UInt64(a), row, UInt64(b), old_row, 2, minimum
                        ),
                        expected,
                    )


def test_metal_ordered_integer_tickets_keep_full_width_and_empty_sentinels() raises:
    var values: List[Int64] = [Int64.MIN, -1, 0, 1, Int64.MAX]
    for a in values:
        for b in values:
            for minimum in [False, True]:
                for row in [UInt32(0), UInt32(128)]:
                    assert_equal(
                        ordered_better(
                            bitcast[DType.uint64](a),
                            row,
                            bitcast[DType.uint64](b),
                            64,
                            1,
                            minimum,
                        ),
                        row < 64 if a == b else a < b if minimum else a > b,
                    )
    var narrow: List[Int32] = [Int32.MIN, -1, 0, 1, Int32.MAX]
    for a in narrow:
        for b in narrow:
            assert_equal(
                ordered_better(
                    UInt64(bitcast[DType.uint32](a)),
                    1,
                    UInt64(bitcast[DType.uint32](b)),
                    2,
                    0,
                    True,
                ),
                a <= b,
            )
    for a in [UInt64(0), UInt64(1), UInt64(255)]:
        for b in [UInt64(0), UInt64(1), UInt64(255)]:
            assert_equal(
                ordered_better(a, 1, b, 2, 3, True), not Bool(a) or Bool(b)
            )
    for op in [FIRST, LAST, ARG_MIN, ARG_MAX]:
        assert_equal(ordered_ticket_row(ordered_empty(op), op), UInt32.MAX)
        for row in [UInt32(0), UInt32(127), UInt32(Int32.MAX - 1)]:
            assert_equal(ordered_ticket_row(row + 1, op), row)


def test_metal_ordered_native_plans_global_filtered_and_tuple_grouped() raises:
    var frame = DataFrame(
        [
            Series("n", Column[Int32]([1, 2, 3])),
            Series("w", Column[Int64]([1, 2, 3])),
            Series("f", Column[Float32]([1, 2, 3])),
            Series("b", Column[Bool]([True, False, True])),
            Series("k", Column[Int32]([1, 2, 1])),
            Series("j", Column[Int64]([0, 1, 0])),
        ]
    )
    for name in ["n", "w", "f", "b"]:
        var value = col(name)
        var outputs: List[Expr] = [
            value.first().alias("first"),
            value.last().alias("last"),
            value.arg_min().alias("min"),
            value.arg_max().alias("max"),
            value.first(ignore_nulls=True).alias("nonnull_first"),
            value.last(ignore_nulls=True).alias("nonnull_last"),
        ]
        for grouped in [False, True]:
            var selected = frame.lazy().filter(col("b") | (col("n") > 0))
            var query = selected.group_by(["k", "j"]).agg(
                outputs
            ) if grouped else selected.select_exprs(outputs)
            var plan = plan_ordered(query)
            assert_true(Bool(plan))
            assert_equal(len(plan.value().fields), 2)
            assert_equal(len(plan.value().outputs), 6)
            assert_equal(plan.value().fields[0].operations, UInt32(15))
            assert_equal(plan.value().fields[1].operations, UInt32(3))
            assert_equal(Bool(plan.value().keys), grouped)
            assert_true(Bool(ordered_key(plan.value())))
    assert_false(
        Bool(plan_ordered(frame.lazy().select((col("n") + 1).first())))
    )


# requires: metal
def test_metal_ordered_public_native_values_ties_nulls_and_filtered_positions() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.ordered_reduce import execute_ordered
    from dataframe_metal.handoff import execute_timed

    _ = require_metal()
    var first_nan = bitcast[DType.float32](UInt32(0x7FC12345))
    var second_nan = bitcast[DType.float32](UInt32(0xFFC54321))
    var negative_zero = bitcast[DType.float32](UInt32(0x80000000))
    var tiny = bitcast[DType.float32](UInt32(1))
    var frame = DataFrame(
        [
            Series(
                "n",
                Column[Int32](
                    [9, 4, 4, -2, -2, 1, 8, 3],
                    [False, True, True, True, True, True, False, True],
                ),
            ),
            Series(
                "w",
                Column[Int64](
                    [
                        Int64(1 << 60) + 9,
                        Int64(1 << 60) + 4,
                        Int64(1 << 60) + 4,
                        Int64(1 << 60) - 2,
                        Int64(1 << 60) - 2,
                        Int64(1 << 60) + 1,
                        Int64(1 << 60) + 8,
                        Int64(1 << 60) + 3,
                    ],
                    [False, True, True, True, True, True, False, True],
                ),
            ),
            Series(
                "f",
                Column[Float32](
                    [
                        first_nan,
                        negative_zero,
                        0,
                        tiny,
                        -tiny,
                        second_nan,
                        first_nan,
                        7,
                    ],
                    [True, True, True, True, True, True, True, False],
                ),
            ),
            Series(
                "b",
                Column[Bool](
                    [True, False, True, False, True, False, True, True],
                    [False, True, True, True, False, True, True, False],
                ),
            ),
            Series("p", Column[Int32]([0, 1, 0, 1, 1, 0, 1, 1])),
            Series(
                "k",
                Column[Int32](
                    [2, 1, 2, 1, 2, 1, 3, 3],
                    [True, True, True, True, True, True, False, False],
                ),
            ),
            Series(
                "j",
                Column[Int64](
                    [
                        Int64(1 << 60),
                        Int64(1 << 60) + 1,
                        Int64(1 << 60),
                        Int64(1 << 60) + 1,
                        Int64(1 << 60),
                        Int64(1 << 60) + 1,
                        Int64(1 << 60) + 2,
                        Int64(1 << 60) + 2,
                    ]
                ),
            ),
        ]
    )
    var chunked = List[Series]()
    for name in frame.columns():
        var value = frame.column(name)
        chunked.append(
            Series._from_chunks(
                [value.slice(0, 1), value.slice(1, 4), value.slice(5, 3)]
            )
        )
    frame = DataFrame(chunked^)
    for name in ["n", "w", "f", "b"]:
        var value = col(name)
        var outputs: List[Expr] = [
            value.first().alias("first"),
            value.last().alias("last"),
            value.first(ignore_nulls=True).alias("nonnull_first"),
            value.last(ignore_nulls=True).alias("nonnull_last"),
            value.arg_min().alias("minimum"),
            value.arg_max().alias("maximum"),
            value.drop_nulls().arg_min().alias("nonnull_minimum"),
            value.drop_nulls().arg_max().alias("nonnull_maximum"),
        ]
        for grouped in [False, True]:
            for filtered in [False, True]:
                var selected = frame.lazy().filter(
                    col("p") > 0
                ) if filtered else frame.lazy()
                var query = selected.group_by(
                    ["k", "j"], maintain_order=True
                ).agg(outputs) if grouped else selected.select_exprs(outputs)
                var cpu = query.collect(engine="cpu")
                assert_true(execute_ordered(query)[0].equals(cpu))
                assert_true(execute_timed(query)[0].equals(cpu))
