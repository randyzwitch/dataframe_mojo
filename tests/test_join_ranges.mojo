"""Join-derived bounds reject only rows proven unable to match."""
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series
from dataframe.frame import _join_range_rows


def test_bounds_handle_chunks_nulls_signed_extremes_and_temporal() raises:
    for low in [Int64.MIN + 1, Int64.MAX - 2]:
        var high = low + 1
        var values = List[Int64](length=4096, fill=0)
        var valid = List[Bool](length=4096, fill=True)
        for row in [2, 9, 2049]:
            values[row] = low
        values[4090] = high
        valid[9] = False
        var original = Series("k", Column[Int64](values^, valid^))
        var right = Series._from_chunks(
            [original.slice(0, 2037), original.slice(2037, 2059)]
        )
        var left = Series._from_chunks(
            [
                Series("k", Column[Int64]([low])),
                Series("k", Column[Int64]([high, 0], [True, False])),
            ]
        )
        var rows = _join_range_rows(left, right)
        assert_true(Bool(rows))
        assert_equal(rows.value(), [2, 2049, 4090])
        var temporal_left = Series(
            "k", left.rechunk()._data.copy(), DataType.datetime("ns")
        )
        var temporal_right = Series(
            "k", original._data.copy(), DataType.datetime("ns")
        )
        assert_equal(
            _join_range_rows(temporal_left, temporal_right).value(),
            [2, 2049, 4090],
        )
    var extremes = Series("k", Column[Int64]([Int64.MIN, Int64.MAX]))
    var probe = Series("k", Column[Int64]([-4, 0, 12]))
    assert_true(not _join_range_rows(extremes, probe))
    var nulls = Series("k", Column[Int64]([7], [False]))
    assert_equal(len(_join_range_rows(nulls, probe).value()), 0)
    var empty = Series("k", Column[Int64]([]))
    assert_equal(len(_join_range_rows(empty, probe).value()), 0)


def test_sampling_is_only_a_cost_guard() raises:
    var left = Series("k", Column[Int64]([7]))
    var values = List[Int64](length=4096, fill=7)
    for row in range(0, 4096, 16):
        values[row] = 0
    var misleading = Series("k", Column[Int64](values^))
    # Every sample rejects, but almost all actual rows match. The full scan
    # detects the poor selectivity and declines gathering; no matches are lost.
    assert_true(not _join_range_rows(left, misleading))
    var floats = Series("k", Column[Float64]([7.0]))
    assert_true(not _join_range_rows(floats, floats))


def test_range_filter_preserves_full_compound_left_join_results() raises:
    var left = DataFrame(
        [
            Series(
                "k",
                Column[Int64]([3, 1, 3, 9, 0], [True, True, True, True, False]),
            ),
            Series("s", Column[String](["ok", "ok", "ok", "ok", "ok"])),
            Series("l", Column[Int64]([0, 1, 2, 3, 4])),
        ]
    )
    var keys = List[Int64]()
    var text = List[String]()
    var ids = List[Int64]()
    var valid = List[Bool]()
    for i in range(524288):
        keys.append(Int64(100 + (i * 48271) % 524288))
        text.append("ok")
        ids.append(Int64(i))
        valid.append(i != 42)
    keys[2] = 1
    keys[7] = 3
    text[7] = "other"
    keys[11] = 1
    keys[40] = 3
    keys[42] = 3
    var right = DataFrame(
        [
            Series("k", Column[Int64](keys^, valid^)),
            Series("s", Column[String](text^)),
            Series("r", Column[Int64](ids^)),
        ]
    )
    for how in ["inner", "left"]:
        var actual = left.join(right, ["k", "s"], how=how)
        var expected_left: List[Int64] = [0, 1, 1, 2]
        var expected_right: List[Int64] = [40, 2, 11, 40]
        if how == "left":
            expected_left.append(3)
            expected_left.append(4)
        assert_equal(actual.height(), len(expected_left))
        for row in range(actual.height()):
            assert_equal(actual.item(row, "l").int64(), expected_left[row])
            if row < len(expected_right):
                assert_equal(actual.item(row, "r").int64(), expected_right[row])
            else:
                assert_true(actual.item(row, "r").is_null())
        assert_true(
            left.lazy()
            .join(right.lazy(), ["k", "s"], how=how)
            .collect()
            .equals(actual)
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
