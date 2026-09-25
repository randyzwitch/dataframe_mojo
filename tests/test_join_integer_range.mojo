"""Correctness coverage for the bounded direct Int64 join-id path."""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, DataType, Series
from dataframe.frame import (
    _dense_right_int64_rows,
    _range_int64_membership_rows,
)
from dataframe.parallel import worker_count


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

    var interval_left = side(
        [-2, -1, 0, 1, 2, 3, 0],
        [True, True, True, True, True, True, False],
        "left_row",
    )
    var interval_right = side(
        [-1, 0, 1, 2], [True, True, True, True], "right_row"
    )
    assert_left_rows(
        interval_left.join(interval_right, "k", "semi"), [1, 2, 3, 4]
    )
    assert_left_rows(interval_left.join(interval_right, "k", "anti"), [0, 5, 6])

    var probes = List[Int64]()
    var expected_semi = List[Int]()
    var expected_anti = List[Int]()
    for row in range(140_000):
        var key = Int64(row % 400 - 200)
        probes.append(key)
        if key >= -100 and key <= 100:
            expected_semi.append(row)
        else:
            expected_anti.append(row)
    var build = List[Int64]()
    for key in range(-100, 101):
        build.append(Int64(key))
    var probe_series = Series("k", Column[Int64](probes^))
    var build_series = Series("k", Column[Int64](build^))
    var semi_rows = _range_int64_membership_rows(
        probe_series, build_series, True
    )
    var anti_rows = _range_int64_membership_rows(
        probe_series, build_series, False
    )
    assert_true(semi_rows[0])
    assert_true(anti_rows[0])
    assert_equal(semi_rows[1], expected_semi)
    assert_equal(anti_rows[1], expected_anti)


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


def test_ordered_strided_right_keys_map_rows_without_an_index() raises:
    var left = side(
        [-120, -119, -60, 0, 60, 120, 0],
        [True, True, True, True, True, True, False],
        "left_row",
    )
    var right = side([-120, -60, 0, 60], [True, True, True, True], "right_row")
    assert_rows(left.join(right, "k"), [0, 2, 3, 4], [0, 1, 2, 3])
    assert_rows(
        left.join(right, "k", "left"),
        [0, 1, 2, 3, 4, 5, 6],
        [0, -1, 1, 2, 3, -1, -1],
    )
    var extreme_left = side(
        [Int64.MIN, -1, Int64.MAX], [True, True, True], "left_row"
    )
    var extreme_right = side([Int64.MIN, Int64.MAX], [True, True], "right_row")
    assert_rows(extreme_left.join(extreme_right, "k"), [0, 2], [0, 1])


def test_ordered_equal_runs_join_without_an_index() raises:
    var left = side(
        [-120, -60, 0, 60, 120, 0],
        [True, True, True, True, True, False],
        "left_row",
    )
    var right = side(
        [-120, -120, 0, 0, 120],
        [True, True, True, True, True],
        "right_row",
    )
    assert_rows(
        left.join(right, "k"),
        [0, 0, 2, 2, 4],
        [0, 1, 2, 3, 4],
    )
    assert_rows(
        left.join(right, "k", "left"),
        [0, 0, 1, 2, 2, 3, 4, 5],
        [0, 1, -1, 2, 3, -1, 4, -1],
    )
    var one_key = side([Int64.MAX, Int64.MAX], [True, True], "right_row")
    var extreme = side([Int64.MAX, Int64.MIN], [True, True], "left_row")
    assert_rows(extreme.join(one_key, "k"), [0, 0], [0, 1])


def test_aligned_chunk_membership_matches_general_row_selection() raises:
    var key = Series(
        "k",
        Column[Int64](
            [1, 2, 3, 4, 5, 6],
            [True, True, True, True, True, False],
        ),
    )
    var row = Series("left_row", Column[Int64]([0, 1, 2, 3, 4, 5]))
    var label = Series("label", Column[String](["a", "b", "c", "d", "e", "f"]))
    var whole = DataFrame([key.copy(), row.copy(), label.copy()])
    var aligned = DataFrame(
        [
            Series._from_chunks([key.slice(0, 3), key.slice(3, 3)]),
            Series._from_chunks([row.slice(0, 3), row.slice(3, 3)]),
            Series._from_chunks([label.slice(0, 3), label.slice(3, 3)]),
        ]
    )
    var unaligned = DataFrame(
        [
            Series._from_chunks([key.slice(0, 3), key.slice(3, 3)]),
            Series._from_chunks([row.slice(0, 2), row.slice(2, 4)]),
            Series._from_chunks([label.slice(0, 3), label.slice(3, 3)]),
        ]
    )
    var holes = side([2, 4], [True, True], "right_row")
    var interval = side([2, 3, 4], [True, True, True], "right_row")
    var empty = side([0], [False], "right_row")
    var wide = side([Int64.MIN, Int64.MAX], [True, True], "right_row")
    var rights = List[DataFrame]()
    rights.append(holes.copy())
    rights.append(interval.copy())
    rights.append(empty.copy())
    rights.append(wide.copy())
    for right in rights:
        for how in [String("semi"), String("anti")]:
            var expected = whole.join(right, "k", how)
            assert_true(aligned.join(right, "k", how).equals(expected))
            assert_true(unaligned.join(right, "k", how).equals(expected))
    assert_left_rows(aligned.join(holes, "k", "semi"), [1, 3])
    assert_left_rows(aligned.join(holes, "k", "anti"), [0, 2, 4, 5])


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


def test_large_parallel_strided_probe_preserves_all_left_rows() raises:
    var rows = 2_000_003
    assert_true(worker_count(rows) > 1)
    var keys = List[Int64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for i in range(rows):
        keys.append(Int64((i % 6 - 2) * 60))
        valid.append(i % 29 != 0)
    var whole = Series("k", Column[Int64](keys^, valid^))
    var left = Series._from_chunks(
        [whole.slice(0, 770_001), whole.slice(770_001, rows - 770_001)]
    )
    var right = Series("k", Column[Int64]([-120, -60, 0, 60]))
    var result = _dense_right_int64_rows(left, right, True)
    assert_true(result[0])
    assert_equal(len(result[1]), rows)
    assert_equal(len(result[2]), rows)
    for i in range(rows):
        assert_equal(result[1][i], i)
        var expected = i % 6 if i % 29 != 0 and i % 6 < 4 else -1
        assert_equal(result[2][i], expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
