"""Unique integer build keys over a small span join through slots indexed
by `key - least`: inner,
left, semi and anti joins against a scalar model, with null probe keys, probe keys outside
the span, a build whose keys repeat (which keeps the hash table), and
probes where every row matches once. Builds over many vectors gather
their columns (nulls, inlined and longer strings, Booleans) and fill
their slots on several finalize tasks at once, and a repeated key in any
of their rows keeps the hash table."""
# contract-group: table.join
from std.testing import assert_equal, assert_true

from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    Series,
    StringColumn,
    col,
)
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


def test_builds_over_many_vectors_join_through_slots() raises:
    """A build of 20,000 unique keys (several vectors, so several finalize
    tasks fill the slots and gather the payload columns at once) joins
    through slots with every payload on its key; the same build with one
    key repeated in its last rows turns back into a hash join with the
    same answers plus the repeat's extra match."""
    comptime BUILD = 20_000
    var probe_keys = List[Int64]()
    var ids = List[Int64]()
    for row in range(ROWS):
        probe_keys.append(Int64((row * 4099) % 62_000) - 1_000)
        ids.append(Int64(row))
    var left = DataFrame(
        [
            Series("k", Column[Int64](probe_keys.copy())),
            Series("id", Column[Int64](ids^)),
        ]
    )
    for repeat in [False, True]:
        var keys = List[Int64]()
        for i in range(BUILD):
            keys.append(Int64(3 * ((i * 7919) % BUILD)))
        if repeat:
            keys.append(Int64(300))
        var values = List[Int64]()
        var texts = List[String]()
        for k in keys:
            values.append(10 * k)
            texts.append(String("key ", k))
        var right = DataFrame(
            [
                Series("bk", Column[Int64](keys^)),
                Series("v", Column[Int64](values^)),
                Series("s", StringColumn(texts^)),
            ]
        )
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
        assert_equal(used, not repeat)
        var expected = 0
        for row in range(ROWS):
            var key = probe_keys[row]
            if key >= 0 and key < 3 * BUILD and key % 3 == 0:
                expected += 1
                if repeat and key == 300:
                    expected += 1
        assert_equal(inner.height(), expected)
        for r in range(inner.height()):
            var key = inner.item(r, "k").int64()
            assert_equal(inner.item(r, "v").int64(), 10 * key)
            assert_equal(inner.item(r, "s").string(), String("key ", key))


comptime MANY = 20_000


def many_keys(repeat_at: Int, repeat_of: Int) -> List[Int64]:
    """MANY unique keys (every third from 0) in a scattered order; with
    `repeat_at` >= 0, the key at row `repeat_of` again at row `repeat_at`
    (rows after it shift down one)."""
    var keys = List[Int64]()
    for i in range(MANY):
        keys.append(Int64(3 * ((i * 7919) % MANY)))
    if repeat_at >= 0:
        keys.insert(repeat_at, keys[repeat_of])
    return keys^


def text_of(key: Int64) -> String:
    # Short keys inline in their row; the others live in the heap.
    if key % 2 == 0:
        return String("k", key)
    return String("a longer string for key ", key)


def many_build(var keys: List[Int64]) raises -> DataFrame:
    """Payloads of every kind, each null on some keys of its own."""
    var values = List[Int64]()
    var values_valid = List[Bool]()
    var reals = List[Float64]()
    var texts = List[String]()
    var texts_valid = List[Bool]()
    var flags = List[Bool]()
    var flags_valid = List[Bool]()
    for k in keys:
        values.append(10 * k)
        values_valid.append(k % 7 != 1)
        reals.append(Float64(k) / 4)
        texts.append(text_of(k))
        texts_valid.append(k % 11 != 2)
        flags.append(k % 5 == 0)
        flags_valid.append(k % 13 != 4)
    return DataFrame(
        [
            Series("bk", Column[Int64](keys^)),
            Series("v", Column[Int64](values^, values_valid)),
            Series("f", Column[Float64](reals^)),
            Series("s", StringColumn(texts^, texts_valid)),
            Series("b", BoolColumn(flags^, flags_valid)),
        ]
    )


def probe_key(row: Int) -> Int64:
    """Each key twice (so the probe side is never a perfect hash build),
    a third of them in the build's range."""
    return Int64(((row % 15_000) * 4099) % 62_000 - 1_000)


def many_probe() raises -> DataFrame:
    """Wider rows than the build's, so the build side stays the build."""
    var keys = List[Int64]()
    var columns = List[Series]()
    for row in range(ROWS):
        keys.append(probe_key(row))
    columns.append(Series("k", Column[Int64](keys^)))
    for c in range(8):
        var ids = List[Int64]()
        for row in range(ROWS):
            ids.append(Int64(row + c))
        columns.append(Series(String("id", c), Column[Int64](ids^)))
    return DataFrame(columns^)


def check_payloads(joined: DataFrame, matched_only: Bool) raises:
    """Every row's payloads are its key's, nulls where the build's are,
    and all null on a row without a match."""
    for r in range(joined.height()):
        var key = joined.item(r, "k").int64()
        var hit = key >= 0 and key < 3 * MANY and key % 3 == 0
        if matched_only:
            assert_true(hit)
        var v = joined.item(r, "v")
        var s = joined.item(r, "s")
        var b = joined.item(r, "b")
        var f = joined.item(r, "f")
        if not hit:
            assert_true(v.is_null() and s.is_null() and b.is_null())
            assert_true(f.is_null())
            continue
        assert_equal(v.is_null(), key % 7 == 1)
        if not v.is_null():
            assert_equal(v.int64(), 10 * key)
        assert_equal(f.float64(), Float64(key) / 4)
        assert_equal(s.is_null(), key % 11 == 2)
        if not s.is_null():
            assert_equal(s.string(), text_of(key))
        assert_equal(b.is_null(), key % 13 == 4)
        if not b.is_null():
            assert_equal(b.bool(), key % 5 == 0)


def test_payloads_of_every_kind_gather_over_many_vectors() raises:
    """A build over many vectors gathers each payload on the finalize task
    that reads its rows: nullable integers, floats, inlined and heap
    strings with nulls, and Booleans with nulls all land on their keys,
    for an inner and a left join, lazily and eagerly."""
    var left = many_probe()
    var right = many_build(many_keys(-1, 0))
    for how in ["inner", "left"]:
        for lazy in [False, True]:
            var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
            var observed = StrategyTrace()
            var out = (
                left.lazy()
                .join(right.lazy(), left_on=["k"], right_on=["bk"], how=how)
                .collect() if lazy else left.join(
                    right, left_on=["k"], right_on=["bk"], how=how
                )
            )
            var used = "dataframe-path: join.perfect_hash\n" in observed.text()
            _ = observed^
            _ = trace^
            if how == "inner" and lazy:
                assert_true(used)
            var expected = 0
            for row in range(ROWS):
                var key = Int(probe_key(row))
                if key >= 0 and key < 3 * MANY and key % 3 == 0:
                    expected += 1
                elif how == "left":
                    expected += 1
            assert_equal(out.height(), expected)
            check_payloads(out, how == "inner")


def test_a_repeated_key_anywhere_keeps_the_hash_table() raises:
    """A key repeated in neighbouring rows, at the build's two ends (rows
    different finalize tasks read), or for the least and the greatest key
    (slots different tasks fill) is found, and the join runs through the
    hash table with the repeat's extra matches."""
    var left = many_probe()
    # (row the repeat goes to, row whose key it repeats)
    var cases: List[Tuple[Int, Int]] = [
        (1, 0),
        (MANY, 0),
        (MANY // 2, 1),
        (MANY // 3, MANY - 1),
    ]
    var keys0 = many_keys(-1, 0)
    # The rows holding the least (0) and the greatest key.
    var least_row = 0
    var greatest_row = 0
    for i in range(MANY):
        if keys0[i] == 0:
            least_row = i
        if keys0[i] == Int64(3 * (MANY - 1)):
            greatest_row = i
    cases.append((MANY, least_row))
    cases.append((0, greatest_row))
    for c in cases:
        var keys = many_keys(c[0], c[1])
        var repeated = keys[c[0]]
        var right = many_build(keys^)
        var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
        var observed = StrategyTrace()
        var out = (
            left.lazy()
            .join(right.lazy(), left_on=["k"], right_on=["bk"])
            .collect()
        )
        var used = "dataframe-path: join.perfect_hash\n" in observed.text()
        _ = observed^
        _ = trace^
        assert_true(not used)
        var expected = 0
        for row in range(ROWS):
            var key = probe_key(row)
            if key >= 0 and key < 3 * MANY and key % 3 == 0:
                expected += 2 if key == repeated else 1
        assert_equal(out.height(), expected)
        check_payloads(out, True)
