"""Correctness coverage for the bounded direct Int64 join-id path."""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, DataType, Series


def side(
    var keys: List[Int64], var valid: List[Bool], var name: String
) raises -> DataFrame:
    var rows = List[Int64](capacity=len(keys))
    for i in range(len(keys)):
        rows.append(Int64(i))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^, valid^)),
            Series(name, Column[Int64](rows^)),
        ]
    )


def assert_rows(frame: DataFrame, left: List[Int64], right: List[Int64]) raises:
    assert_equal(frame.height(), len(left))
    ref l = frame.column("left_row").int64()
    ref r = frame.column("right_row").int64()
    for i in range(len(left)):
        assert_equal(l.is_null(i), left[i] < 0)
        assert_equal(r.is_null(i), right[i] < 0)
        if left[i] >= 0:
            assert_equal(l.value(i), left[i])
        if right[i] >= 0:
            assert_equal(r.value(i), right[i])


def assert_left_rows(frame: DataFrame, left: List[Int64]) raises:
    assert_equal(frame.height(), len(left))
    ref values = frame.column("left_row").int64()
    for i in range(len(left)):
        assert_equal(values.value(i), left[i])


def test_negative_holes_null_ids_and_every_join_order() raises:
    var left = side(
        [-3, 0, 0, 4, -3], [True, False, True, True, True], "left_row"
    )
    var right = side([4, -3, 1, 0], [True, True, True, False], "right_row")
    assert_rows(left.join(right, "k"), [0, 3, 4], [1, 0, 1])
    assert_rows(
        left.join(right, "k", "left"), [0, 1, 2, 3, 4], [1, -1, -1, 0, 1]
    )
    assert_rows(
        left.join(right, "k", "right"), [3, 0, 4, -1, -1], [0, 1, 1, 2, 3]
    )
    assert_rows(
        left.join(right, "k", "full"),
        [0, 1, 2, 3, 4, -1, -1],
        [1, -1, -1, 0, 1, 2, 3],
    )
    assert_left_rows(left.join(right, "k", "semi"), [0, 3, 4])
    assert_left_rows(left.join(right, "k", "anti"), [1, 2])


def test_all_null_int64_keys_have_no_hidden_group() raises:
    var left = side([5, -2], [False, False], "left_row")
    var right = side([8, 9], [False, False], "right_row")
    assert_rows(left.join(right, "k"), [], [])
    assert_rows(left.join(right, "k", "left"), [0, 1], [-1, -1])
    assert_rows(left.join(right, "k", "right"), [-1, -1], [0, 1])
    assert_rows(left.join(right, "k", "full"), [0, 1, -1, -1], [-1, -1, 0, 1])
    assert_left_rows(left.join(right, "k", "semi"), [])
    assert_left_rows(left.join(right, "k", "anti"), [0, 1])


def test_extreme_span_and_multi_key_join_correctness() raises:
    var left = side([Int64.MIN, 0], [True, True], "left_row")
    var right = side([Int64.MAX, 0], [True, True], "right_row")
    assert_rows(left.join(right, "k"), [1], [1])

    var left_two = left.with_column(Series("z", Column[Int64]([1, 2])))
    var right_two = right.with_column(Series("z", Column[Int64]([2, 2])))
    assert_equal(left_two.join(right_two, on=["k", "z"]).height(), 1)


def test_strided_keys_reject_off_grid_probes_and_keep_duplicates() raises:
    var left = side(
        [-120, -119, -60, 0, 60, 0],
        [True, True, True, True, True, False],
        "left_row",
    )
    var right = side(
        [-120, -60, 0, 0, 99],
        [True, True, True, True, False],
        "right_row",
    )
    assert_rows(left.join(right, "k"), [0, 2, 3, 3], [0, 1, 2, 3])
    assert_rows(
        left.join(right, "k", "left"),
        [0, 1, 2, 3, 3, 4, 5],
        [0, -1, 1, 2, 3, -1, -1],
    )
    assert_rows(
        left.join(right, "k", "right"),
        [0, 2, 3, 3, -1],
        [0, 1, 2, 3, 4],
    )
    var extremes_left = side(
        [Int64.MIN, -1, Int64.MAX],
        [True, True, True],
        "left_row",
    )
    var extremes_right = side([Int64.MIN, Int64.MAX], [True, True], "right_row")
    assert_rows(extremes_left.join(extremes_right, "k"), [0, 2], [0, 1])


def test_temporal_physical_int64_uses_the_same_dense_range() raises:
    var left = DataFrame(
        [
            Series("d", Column[Int64]([10, 12], [True, False])).with_dtype(
                DataType.DATE
            ),
            Series("left_row", Column[Int64]([0, 1])),
        ]
    )
    var right = DataFrame(
        [
            Series("d", Column[Int64]([11, 10])).with_dtype(DataType.DATE),
            Series("right_row", Column[Int64]([0, 1])),
        ]
    )
    assert_rows(left.join(right, "d"), [0], [1])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
