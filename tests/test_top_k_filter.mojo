"""`rank("ordinal").over(keys) <= k` filters keep each partition's first k
rows without ranking every row (top_k.mojo). The kept rows must be exactly
those whose ordinal rank is at most k: ascending and descending, ties by
row order, NaN ranked as `rank` ranks it, null values (null rank) dropped,
a null partition key as its own partition, and `<` as well as `<=`.
"""
from std.collections import Dict
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Series, col, lit
from dataframe.parallel import worker_count


def set_threads(n: Int):
    var name = String("DATAFRAME_THREADS")
    var value = String(n)
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


comptime ROWS = 196_608


def frame() raises -> DataFrame:
    var keys = List[Int64](capacity=ROWS)
    var key_valid = List[Bool](capacity=ROWS)
    var tags = List[Int64](capacity=ROWS)
    var values = List[Float64](capacity=ROWS)
    var valid = List[Bool](capacity=ROWS)
    var nan = Float64(0) / Float64(0)
    for i in range(ROWS):
        keys.append(Int64((i * 7919) % 2000))
        key_valid.append(i % 211 != 5)
        tags.append(Int64(i % 3))
        # Few distinct values, so ties are common.
        var v = Float64((i * 104729) % 37)
        if i % 97 == 11:
            v = nan
        values.append(v)
        valid.append(i % 53 != 7)
    var row = List[Int64](capacity=ROWS)
    for i in range(ROWS):
        row.append(Int64(i))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^, key_valid^)),
            Series("t", Column[Int64](tags^)),
            Series("v", Column[Float64](values^, valid^)),
            Series("row", Column[Int64](row^)),
        ]
    )


def reference(
    data: DataFrame, keys: List[String], k: Int, descending: Bool
) raises -> List[Int]:
    """Rows whose ordinal rank in their partition is at most k."""
    var members = Dict[String, List[Int]]()
    var order = List[String]()
    for i in range(data.height()):
        var label = String()
        for name in keys:
            var cell = data.column(name).get(i)
            label += ("null" if cell.is_null() else String(cell.int64())) + "|"
        if label not in members:
            members[label] = List[Int]()
            order.append(label)
        if not data.column("v").get(i).is_null():
            members[label].append(i)
    var kept = List[Bool](length=data.height(), fill=False)
    for label in order:
        var rows = members[label].copy()
        # Insertion sort by (value as rank orders it, row): NaN above every
        # number, ties by row.
        for a in range(1, len(rows)):
            var j = a
            while j > 0:
                var x = data.column("v").get(rows[j - 1]).float64()
                var y = data.column("v").get(rows[j]).float64()
                var xn = x != x
                var yn = y != y
                var after: Bool
                if descending:
                    after = (not xn and yn) or (not xn and not yn and x < y)
                else:
                    after = (xn and not yn) or (not xn and not yn and x > y)
                if not after:
                    break
                var t = rows[j - 1]
                rows[j - 1] = rows[j]
                rows[j] = t
                j -= 1
        for r in range(min(k, len(rows))):
            kept[rows[r]] = True
    var out = List[Int]()
    for i in range(len(kept)):
        if kept[i]:
            out.append(i)
    return out^


def check(data: DataFrame, keys: List[String], k: Int, descending: Bool) raises:
    var ranked = col("v").rank("ordinal", descending=descending).over(keys)
    var got = data.filter(ranked.copy() <= lit(Int64(k)))
    var want = reference(data, keys, k, descending)
    assert_equal(got.height(), len(want))
    for i in range(len(want)):
        assert_equal(Int(got.item(i, "row").int64()), want[i])
    var strict = data.filter(ranked < lit(Int64(k + 1)))
    assert_equal(strict.height(), len(want))


def test_top_k_matches_ordinal_rank() raises:
    set_threads(8)
    var data = frame()
    assert_true(worker_count(data.height()) > 1)
    for k in [0, 1, 2, 40]:
        check(data, ["k"], k, True)
    check(data, ["k"], 2, False)
    check(data, ["k", "t"], 3, True)


def test_one_worker() raises:
    set_threads(1)
    check(frame().slice(0, 10_000), ["k"], 2, True)
    set_threads(8)


def test_large_k_singletons_and_extreme_limits() raises:
    set_threads(8)
    var rows = 131_072
    var ids = List[Int64](capacity=rows)
    var values = List[Float64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for i in range(rows):
        ids.append(Int64(i))
        values.append(Float64(i % 29))
        valid.append(i % 19 != 0)
    var data = DataFrame(
        [
            Series("k", Column[Int64](ids^)),
            Series("v", Column[Float64](values^, valid^)),
        ]
    )
    assert_true(worker_count(rows) > 1)
    var ranked = col("v").rank("ordinal").over("k")
    var expected = data.filter(col("v").is_not_null())
    for k in [Int64(1000), Int64.MAX]:
        assert_true(data.filter(ranked.copy() <= lit(k)).equals(expected))
        assert_true(data.filter(ranked.copy() < lit(k)).equals(expected))
    for k in [Int64.MIN, Int64(-1), Int64(0)]:
        assert_equal(data.filter(ranked.copy() <= lit(k)).height(), 0)
        assert_equal(data.filter(ranked.copy() < lit(k)).height(), 0)


def test_heap_matches_full_ranking_on_skewed_groups() raises:
    set_threads(8)
    var rows = 131_073
    var keys = List[Int64](capacity=rows)
    var values = List[Float64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    var ids = List[Int64](capacity=rows)
    for i in range(rows):
        keys.append(Int64(0 if i % 3 != 0 else i % 17))
        var v = Float64((rows - i) % 997)
        if i % 97 == 0:
            v = Float64(0) / Float64(0)
        elif i % 101 == 0:
            v = -Float64(0)
        elif i % 103 == 0:
            v = Float64(0) / Float64(1)
        values.append(v)
        valid.append(i % 13 != 0 and i % 17 != 16)
        ids.append(Int64(i))
    var value = Series("v", Column[Float64](values^, valid^))
    var data = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series._from_chunks(
                [value.slice(0, 65535), value.slice(65535, rows - 65535)]
            ),
            Series("row", Column[Int64](ids^)),
        ]
    )
    assert_true(worker_count(rows) > 1)
    for descending in [False, True]:
        var ranked = col("v").rank("ordinal", descending=descending).over("k")
        # Materializing rank prevents the top-k pattern from matching.
        var ordinary = data.with_columns(ranked.copy().alias("r"))
        for k in [1, 2, 31, 1000, 70_000, rows]:
            var want = ordinary.filter(col("r") <= lit(Int64(k))).column("row")
            var got = data.filter(ranked.copy() <= lit(Int64(k))).column("row")
            assert_true(got.equals(want))
            var strict = data.filter(ranked.copy() < lit(Int64(k + 1))).column(
                "row"
            )
            assert_true(strict.equals(want))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
