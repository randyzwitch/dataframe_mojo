"""Verify join hash-table builds and probes against independently enumerated
row pairs. Exercise duplicate keys, nulls, bounded probe output and
unmatched build-row scans."""
# contract-group: table.join
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, Series
from dataframe.join_type import JOIN_ANTI, JOIN_INNER, JOIN_LEFT, JOIN_SEMI
from dataframe.joins.hash_table import JoinHashTable, probe_pairs
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)
from tests.support.join_worker_setting import JoinWorkers


def test_partitioned_build_keeps_every_duplicate_null_and_row() raises:
    var setting = JoinWorkers(4)
    var domain = 600_000
    var n = 700_003
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(n):
        values.append(Int64(row % domain))
        valid.append(row % 97 != 2)
    var build = Series("k", Column[Int64](values^, valid))
    var table = JoinHashTable([build.copy()])
    assert_equal(table.radix_bits, 4)
    assert_true(table.chains_longer_than_one)
    assert_true(table.has_null)
    var probes: List[Int64] = [
        0,
        2,
        100002,
        Int64(domain - 1),
        Int64(domain),
        Int64.MAX,
        -1,
        5,
    ]
    var probe = Series("k", Column[Int64](probes.copy()))
    for outer in [False, True]:
        var pairs = probe_pairs(
            table, [probe.copy()], JOIN_LEFT if outer else JOIN_INNER, 0
        )
        var k = 0
        for i in range(len(probes)):
            var expected = List[Int]()
            var j = Int(probes[i])
            if probes[i] >= 0 and probes[i] < Int64(domain):
                while j < n:
                    if j % 97 != 2:
                        expected.append(j)
                    j += domain
            # A probe row's matches come out together, in build order.
            var actual = List[Int]()
            while k < len(pairs[0]) and pairs[0][k] == i:
                actual.append(pairs[1][k])
                k += 1
            if outer and len(expected) == 0:
                assert_equal(actual, [-1])
            else:
                assert_equal(actual, expected)
        assert_equal(k, len(pairs[0]))
    _ = setting^


def test_partitioned_key_set_finds_exactly_the_build_keys() raises:
    # Semi, anti and mark joins observe membership only, so duplicate build
    # keys must not multiply emitted probe rows.
    var setting = JoinWorkers(4)
    var domain = 300_000
    var n = 500_001
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(n):
        values.append(Int64(3 * (row % domain)))
        valid.append(row % 89 != 5)
    var build = Series("k", Column[Int64](values^, valid))
    var table = JoinHashTable([build.copy()], keys_only=True)
    assert_true(table.has_null)
    var probes = List[Int64]()
    var probe_valid = List[Bool]()
    for i in range(70_001):
        probes.append(Int64(i * 13 - 7))
        probe_valid.append(i % 31 != 4)
    var probe = Series("k", Column[Int64](probes.copy(), probe_valid))
    var semi = List[Int]()
    var anti = List[Int]()
    for i in range(len(probes)):
        var v = probes[i]
        var hit = (
            probe_valid[i] and v >= 0 and v % 3 == 0 and v < Int64(3 * domain)
        )
        if hit:
            # Value v/3 is in the build unless every row holding it is
            # null: rows v/3 and v/3 + domain (when below n).
            var r = Int(v // 3)
            var any = r % 89 != 5
            if r + domain < n:
                any = any or (r + domain) % 89 != 5
            hit = any
        if hit:
            semi.append(i)
        else:
            anti.append(i)
    assert_equal(probe_pairs(table, [probe.copy()], JOIN_SEMI, 0)[0], semi)
    assert_equal(probe_pairs(table, [probe.copy()], JOIN_ANTI, 0)[0], anti)
    _ = setting^


def test_eager_join_builds_the_smaller_side_and_keeps_order_when_asked() raises:
    # 300 left rows against 200,000 right rows: the left side's table is
    # the smaller, so the join builds on it and probes the right rows.
    var left_keys = List[Int64]()
    var left_ids = List[Int64]()
    for i in range(300):
        left_keys.append(Int64((i * 37) % 450))
        left_ids.append(Int64(i))
    var left = DataFrame(
        [
            Series("k", Column[Int64](left_keys^)),
            Series("l", Column[Int64](left_ids^)),
        ]
    )
    var right_keys = List[Int64]()
    var right_ids = List[Int64]()
    for j in range(200_000):
        right_keys.append(Int64((j * 7919) % 1_000))
        right_ids.append(Int64(j))
    var right = DataFrame(
        [
            Series("k", Column[Int64](right_keys^)),
            Series("r", Column[Int64](right_ids^)),
        ]
    )
    for how in ["inner", "left", "semi", "anti"]:
        var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
        var observed = StrategyTrace()
        var ordered = left._join_impl(
            right,
            left_on=["k"],
            right_on=["k"],
            how=0 if how
            == "inner" else (
                1 if how == "left" else (4 if how == "semi" else 5)
            ),
            keep_order=True,
        )
        assert_true(
            "dataframe-path: join.build_flipped\n" in observed.text(),
            "the smaller left side did not build",
        )
        _ = observed^
        _ = trace^
        # Every right row of a key is matched once per left row of it,
        # in left row order, then right row order.
        var expected = 0
        var last_l = Int64(-1)
        var last_r = Int64(-1)
        for row in range(ordered.height()):
            var l = ordered.item(row, "l").int64()
            assert_true(l >= last_l)
            if how == "inner" or how == "left":
                var r = ordered.item(row, "r")
                if l == last_l and not r.is_null():
                    assert_true(r.int64() > last_r)
                if not r.is_null():
                    last_r = r.int64()
                    assert_equal(
                        ordered.item(row, "k").int64(),
                        right_keys_of(Int(last_r)),
                    )
            last_l = l
        for i in range(300):
            var key = Int64((i * 37) % 450)
            var hits = 200 if key < 1_000 else 0
            if how == "inner":
                expected += hits
            elif how == "left":
                expected += max(hits, 1)
            elif how == "semi":
                expected += 1 if hits > 0 else 0
            else:
                expected += 0 if hits > 0 else 1
        assert_equal(ordered.height(), expected)


def right_keys_of(j: Int) -> Int64:
    return Int64((j * 7919) % 1_000)
