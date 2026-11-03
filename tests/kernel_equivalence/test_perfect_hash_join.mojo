"""Unique integer build keys over a small span join through slots indexed
by `key - least`: inner, left, semi and anti
joins against a scalar model, with null probe keys, probe keys outside
the span, a build whose keys repeat (which keeps the hash table), and
probes where every row matches once."""
# contract-group: table.join
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, Series, col
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 30_000


def probe() raises -> DataFrame:
    var keys = List[Int64]()
    var valid = List[Bool]()
    var ids = List[Int64]()
    for row in range(ROWS):
        keys.append(Int64((row * 7919) % 3_000) - 500)
        valid.append(row % 29 != 3)
        ids.append(Int64(row))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^, valid)),
            Series("id", Column[Int64](ids^)),
        ]
    )


def build_keys(repeat: Bool) -> List[Int64]:
    """Every third key from 0 to 1,998, each once in a scattered order;
    with `repeat`, key 300 twice."""
    var keys = List[Int64]()
    for i in range(667):
        keys.append(Int64(3 * ((i * 37) % 667)))
    if repeat:
        keys.append(Int64(300))
    return keys^


def build(repeat: Bool) raises -> DataFrame:
    var keys = build_keys(repeat)
    var payload = List[Int64]()
    for k in keys:
        payload.append(10 * k)
    return DataFrame(
        [
            Series("bk", Column[Int64](keys^)),
            Series("v", Column[Int64](payload^)),
        ]
    )


def matches(key: Int64, repeat: Bool) -> Int:
    var count = 0
    for k in build_keys(repeat):
        if k == key:
            count += 1
    return count


def test_joins_through_slots_match_the_model() raises:
    var left = probe()
    var keys = left.column("k")
    for repeat in [False, True]:
        var right = build(repeat)
        var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
        var observed = StrategyTrace()
        var inner = (
            left.lazy()
            .join(right.lazy(), left_on=["k"], right_on=["bk"])
            .collect()
        )
        var used = "dataframe-path: join.perfect_hash\n" in observed.text()
        _ = observed^
        _ = trace^
        # Slots only for unique keys; a repeated key keeps the hash table.
        assert_equal(used, not repeat)
        var semi = (
            left.lazy()
            .join(right.lazy(), left_on=["k"], right_on=["bk"], how="semi")
            .collect()
        )
        var anti = (
            left.lazy()
            .join(right.lazy(), left_on=["k"], right_on=["bk"], how="anti")
            .collect()
        )
        var expected_inner = 0
        var expected_semi = 0
        for row in range(ROWS):
            var cell = keys.get(row)
            var m = 0 if cell.is_null() else matches(cell.int64(), repeat)
            expected_inner += m
            if m > 0:
                expected_semi += 1
        assert_equal(inner.height(), expected_inner)
        assert_equal(semi.height(), expected_semi)
        assert_equal(anti.height(), ROWS - expected_semi)
        # Payloads line up with their keys.
        for r in range(0, inner.height(), 997):
            assert_equal(
                inner.item(r, "v").int64(), 10 * inner.item(r, "k").int64()
            )


def test_every_row_matching_keeps_its_order() raises:
    """A probe whose every key is in the build, once each: the join keeps
    every row, in order, with its payload."""
    var keys = List[Int64]()
    var ids = List[Int64]()
    for row in range(ROWS):
        keys.append(Int64((row * 13) % 667 * 3))
        ids.append(Int64(row))
    var left = DataFrame(
        [Series("k", Column[Int64](keys^)), Series("id", Column[Int64](ids^))]
    )
    var right = build(False)
    var joined = (
        left.lazy().join(right.lazy(), left_on=["k"], right_on=["bk"]).collect()
    )
    assert_equal(joined.height(), ROWS)
    var sorted = joined.sort(["id"])
    for r in range(0, ROWS, 1_013):
        assert_equal(sorted.item(r, "id").int64(), Int64(r))
        assert_equal(
            sorted.item(r, "v").int64(), 10 * sorted.item(r, "k").int64()
        )


def test_keys_spanning_the_whole_int64_range_join_in_every_kind() raises:
    # Random 64-bit ids span more than 2**63 (Int64.MIN and Int64.MAX
    # among them): a signed `most - least` overflows, which once sized
    # the direct slots from a negative span.
    var probe_keys = List[Int64]()
    var build_keys = List[Int64]()
    var x = UInt64(0x9E3779B97F4A7C15)
    for row in range(300):
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        probe_keys.append(Int64(Int(x)))
        if row % 2 == 0:
            build_keys.append(Int64(Int(x)))
        else:
            build_keys.append(Int64(Int(~x)))
    probe_keys[0] = Int64.MIN
    build_keys[0] = Int64.MIN
    build_keys[1] = Int64.MAX
    var ids = List[Int64]()
    for row in range(300):
        ids.append(Int64(row))
    var left = DataFrame(
        [
            Series("k", Column[Int64](probe_keys.copy())),
            Series("id", Column[Int64](ids.copy())),
        ]
    )
    var right = DataFrame(
        [
            Series("bk", Column[Int64](build_keys.copy())),
            Series("v", Column[Int64](ids.copy())),
        ]
    )
    var expected: List[Int] = [150, 300, 300, 450, 150, 150]
    var kinds: List[String] = ["inner", "left", "right", "full", "semi", "anti"]
    for k in range(len(kinds)):
        var how = kinds[k]
        for lazy in [False, True]:
            var out = (
                left.lazy()
                .join(right.lazy(), left_on=["k"], right_on=["bk"], how=how)
                .collect() if lazy else left.join(
                    right, left_on=["k"], right_on=["bk"], how=how
                )
            )
            assert_equal(out.height(), expected[k])
            if how != "inner":
                continue
            for r in range(out.height()):
                var id = Int(out.item(r, "id").int64())
                var v = Int(out.item(r, "v").int64())
                assert_equal(out.item(r, "k").int64(), build_keys[v])
                assert_equal(probe_keys[id], build_keys[v])
