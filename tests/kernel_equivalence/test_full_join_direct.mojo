"""A full join on keys with many distinct values takes the left join's
direct hash route and then adds the right rows nothing matched, in order:
the same rows, in the same order, as a scalar model of SQL's FULL OUTER
JOIN (null keys match nothing), on two-column keys with duplicates and
unmatched rows on both sides."""
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, Series
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime LEFT = 6_000
comptime RIGHT = 5_000


def left_key(row: Int) -> Tuple[Int64, Int64]:
    return (Int64(row % 1_700), Int64(row % 3))


def right_key(row: Int) -> Tuple[Int64, Int64]:
    return (Int64((row * 7) % 2_300), Int64((row * 5) % 3))


def left_valid(row: Int) -> Bool:
    return row % 41 != 7


def right_valid(row: Int) -> Bool:
    return row % 37 != 11


def side(n: Int, left: Bool) raises -> DataFrame:
    var a = List[Int64]()
    var b = List[Int64]()
    var valid = List[Bool]()
    var ids = List[Int64]()
    for row in range(n):
        var key = left_key(row) if left else right_key(row)
        a.append(key[0])
        b.append(key[1])
        valid.append(left_valid(row) if left else right_valid(row))
        ids.append(Int64(row))
    var prefix = "l" if left else "r"
    return DataFrame(
        [
            Series(prefix + "a", Column[Int64](a^, valid.copy())),
            Series(prefix + "b", Column[Int64](b^)),
            Series(prefix + "_id", Column[Int64](ids^)),
        ]
    )


def test_full_join_on_many_keys_matches_the_model() raises:
    var left = side(LEFT, True)
    var right = side(RIGHT, False)
    # Model: left-major matches (each left row's right matches in right row
    # order, or the left row alone), then unmatched right rows in order.
    var expected_left = List[Int]()
    var expected_right = List[Int]()
    var matched = List[Bool](length=RIGHT, fill=False)
    var by_key = Dict[String, List[Int]]()
    for j in range(RIGHT):
        if not right_valid(j):
            continue
        var key = right_key(j)
        var text = String(key[0]) + "," + String(key[1])
        if text not in by_key:
            by_key[text] = List[Int]()
        by_key[text].append(j)
    for i in range(LEFT):
        var found = False
        if left_valid(i):
            var key = left_key(i)
            var text = String(key[0]) + "," + String(key[1])
            if text in by_key:
                for j in by_key[text]:
                    expected_left.append(i)
                    expected_right.append(j)
                    matched[j] = True
                    found = True
        if not found:
            expected_left.append(i)
            expected_right.append(-1)
    for j in range(RIGHT):
        if not matched[j]:
            expected_left.append(-1)
            expected_right.append(j)
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var joined = left.join(
        right, left_on=["la", "lb"], right_on=["ra", "rb"], how="full"
    )
    assert_true(
        "dataframe-path: join.full_direct\n" in observed.text(),
        "the full join did not take the direct route",
    )
    _ = observed^
    _ = trace^
    assert_equal(joined.height(), len(expected_left))
    var lids = joined.column("l_id")
    var rids = joined.column("r_id")
    for k in range(joined.height()):
        var l = lids.get(k)
        var r = rids.get(k)
        assert_equal(l.is_null(), expected_left[k] < 0)
        assert_equal(r.is_null(), expected_right[k] < 0)
        if expected_left[k] >= 0:
            assert_equal(Int(l.int64()), expected_left[k])
        if expected_right[k] >= 0:
            assert_equal(Int(r.int64()), expected_right[k])
