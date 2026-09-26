"""Semi and anti joins through the direct hash index.

Keys are shaped so no bounded-range or ordered path applies (wide Int64
values, strings, two keys), which is what most real keys look like. Every
result is checked against a brute-force membership oracle, and the direct
row lists are checked against the same oracle for both key kinds.
"""
from std.collections import Dict
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
)
from dataframe.join_hash import direct_hash_semi_anti_rows

comptime ROWS = 60_000
comptime RIGHT_ROWS = 20_000


def wide(i: Int) -> Int64:
    """A key far outside any direct-address range."""
    return Int64(i) * 6364136223846793005 % 1_099_511_627_776


def left_frame() raises -> DataFrame:
    var keys = List[Int64](capacity=ROWS)
    var words = List[String](capacity=ROWS)
    var second = List[Int64](capacity=ROWS)
    var rows = List[Int64](capacity=ROWS)
    var valid = List[Bool](capacity=ROWS)
    for i in range(ROWS):
        keys.append(wide(i % 27_000))
        words.append("k" + String(i % 21_000))
        second.append(Int64(i % 3))
        rows.append(Int64(i))
        valid.append(i % 11 != 0)
    return DataFrame(
        [
            Series("k", Column[Int64](keys^, valid.copy())),
            Series("s", StringColumn(words, valid.copy())),
            Series("t", Column[Int64](second^)),
            Series("row", Column[Int64](rows^)),
        ]
    )


def right_frame() raises -> DataFrame:
    var keys = List[Int64](capacity=RIGHT_ROWS)
    var words = List[String](capacity=RIGHT_ROWS)
    var second = List[Int64](capacity=RIGHT_ROWS)
    var valid = List[Bool](capacity=RIGHT_ROWS)
    for j in range(RIGHT_ROWS):
        # Every other right key repeats, so duplicates must not repeat rows.
        keys.append(wide((j // 2) * 3 + 1))
        words.append("k" + String(j * 2 + 1))
        second.append(Int64(j % 2))
        valid.append(j % 13 != 0)
    return DataFrame(
        [
            Series("k", Column[Int64](keys^, valid.copy())),
            Series("s", StringColumn(words, valid.copy())),
            Series("t", Column[Int64](second^)),
        ]
    )


def token_of(
    cell_dtype: DataType, frame: DataFrame, name: String, i: Int
) raises -> String:
    var cell = frame.column(name).get(i)
    if cell_dtype == DataType.STRING:
        return cell.string()
    return String(cell.int64())


def expected_rows(
    left: DataFrame, right: DataFrame, keys: List[String], keep: Bool
) raises -> List[Int]:
    """Brute force: a left row is kept when the tuple of its non-null key
    values appears among the right tuples (semi) or does not (anti)."""
    var present = Dict[String, Bool]()
    for j in range(right.height()):
        var token = String()
        var ok = True
        for name in keys:
            if right.column(name).get(j).is_null():
                ok = False
            else:
                token += (
                    token_of(right.column(name).dtype(), right, name, j)
                    + "\x1f"
                )
        if ok:
            present[token] = True
    var rows = List[Int]()
    for i in range(left.height()):
        var token = String()
        var ok = True
        for name in keys:
            if left.column(name).get(i).is_null():
                ok = False
            else:
                token += (
                    token_of(left.column(name).dtype(), left, name, i) + "\x1f"
                )
        var matched = ok and token in present
        if matched == keep:
            rows.append(i)
    return rows^


def check(left: DataFrame, right: DataFrame, keys: List[String]) raises:
    for how in [String("semi"), "anti"]:
        var keep = how == "semi"
        var want = expected_rows(left, right, keys, keep)
        var got = left.join(right, keys, how)
        assert_equal(got.height(), len(want))
        assert_equal(got.width(), left.width())
        for i in range(len(want)):
            assert_equal(Int(got.column("row").get(i).int64()), want[i])
        var sources = List[Series]()
        var targets = List[Series]()
        for name in keys:
            sources.append(left.column(name))
            targets.append(right.column(name))
        var direct = direct_hash_semi_anti_rows(sources, targets, keep)
        assert_equal(len(direct), len(want))
        for i in range(len(want)):
            assert_equal(direct[i], want[i])


def test_wide_int64_keys() raises:
    check(left_frame(), right_frame(), ["k"])


def test_string_keys() raises:
    check(left_frame(), right_frame(), ["s"])


def test_two_keys() raises:
    check(left_frame(), right_frame(), ["k", "t"])


def test_no_matches_and_all_matches() raises:
    var left = left_frame()
    var none = right_frame().with_columns((col("k") + lit(Int64(1))).alias("k"))
    # Shifted wide keys share nothing with the left side.
    assert_equal(left.join(none, ["k"], "semi").height(), 0)
    assert_equal(left.join(none, ["k"], "anti").height(), ROWS)
    var all = left.select(["k"]).unique()
    var semi = left.join(all, ["k"], "semi")
    var anti = left.join(all, ["k"], "anti")
    # Null keys never match: they are the only anti rows.
    assert_equal(semi.height(), ROWS - anti.height())
    for i in range(anti.height()):
        assert_true(anti.column("k").get(i).is_null())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
