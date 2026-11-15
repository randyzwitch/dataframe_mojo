"""Window partitions against a scalar model: string keys short, long, empty,
Unicode and null, with equal strings read at the end of their byte buffer
(by bytes) and inside it (by masked word loads), an integer key with nulls,
a mean over a partition and a rank within one, and a rank over one string
key on several workers, where the gathered key is in several pieces."""
# contract-group: expr.order_windows
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, DataType, Series, col, lit
from tests.support.group_strategy_reference import ScopedEnvironment

comptime ROWS = 4_000
comptime PARTS = 97


def part_of(row: Int) -> Int:
    # The last rows repeat partitions of earlier ones, so a key string read
    # at the end of the buffer equals one read inside it.
    return (row * 31) % PARTS


def text_of(p: Int) -> String:
    var kind = p % 5
    if kind == 0:
        return ""
    if kind == 1:
        return "ab" + String(p)
    if kind == 2:
        return "a longer partition key 東京 " + String(p)
    if kind == 3:
        return String(p) + "é"
    return "exactly12by" + String(p % 10)


def text_valid(p: Int) -> Bool:
    return p % 23 != 7


def number_valid(p: Int) -> Bool:
    # Null only where the text identifies the partition.
    return not (p % 19 == 4 and text_valid(p))


def frame() raises -> DataFrame:
    var texts = List[String]()
    var tv = List[Bool]()
    var numbers = List[Int64]()
    var nv = List[Bool]()
    var values = List[Int64]()
    for row in range(ROWS):
        var p = part_of(row)
        texts.append(text_of(p) if text_valid(p) else String("hidden"))
        tv.append(text_valid(p))
        numbers.append(Int64(p % 3))
        nv.append(number_valid(p))
        values.append(Int64((row * 7) % 101))
    return DataFrame(
        [
            Series("t", Column[String](texts^, tv)),
            Series("n", Column[Int64](numbers^, nv)),
            Series("v", Column[Int64](values^)),
        ]
    )


def key_of(p: Int) -> String:
    var text = String("null") if not text_valid(p) else "t:" + text_of(p)
    return text + "|" + (String(p % 3) if number_valid(p) else "null")


def test_mean_and_rank_over_string_partitions_match_scalar_values() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "1")
    var data = frame()
    var out = data.with_columns(
        [
            col("v").mean().over(["t", "n"]).alias("mean"),
            col("v").rank(method="min").over(["t", "n"]).alias("rank"),
        ]
    )
    var row_keys = List[String](capacity=ROWS)
    var totals = Dict[String, Int]()
    var counts = Dict[String, Int]()
    var members = Dict[String, List[Int]]()
    for row in range(ROWS):
        var key = key_of(part_of(row))
        var value = (row * 7) % 101
        row_keys.append(key)
        totals[key] = totals.get(key, 0) + value
        counts[key] = counts.get(key, 0) + 1
        if key not in members:
            members[key] = List[Int]()
        members[key].append(value)
    for row in range(ROWS):
        ref key = row_keys[row]
        var mean = out.item(row, "mean").float64()
        assert_true(
            abs(mean - Float64(totals[key]) / Float64(counts[key])) < 1e-9
        )
        # Rank "min": one plus the partition's values below this row's.
        var value = (row * 7) % 101
        var below = 0
        for other in members[key]:
            if other < value:
                below += 1
        var rank = out.item(row, "rank")
        var r = Int(rank.int64()) if rank.dtype() == DataType.INT64 else Int(
            rank.uint32()
        )
        assert_equal(r, below + 1)
    _ = workers^


comptime WIDE_ROWS = 300_000
comptime WIDE_PARTS = 37
comptime WIDE_VALUES = 1_000


def wide_part(row: Int) -> Int:
    return (row * 13) % WIDE_PARTS


def wide_value(row: Int) -> Int:
    return (row * 7919) % WIDE_VALUES


def wide_key(row: Int) -> String:
    var p = wide_part(row)
    return "t:" + text_of(p) if text_valid(p) else String("null")


def test_rank_over_one_string_key_with_nulls_on_many_workers() raises:
    # Enough rows that gathering the key splits it into several pieces, one
    # per worker, and partitions are numbered by tasks whose row ranges
    # cross those pieces: every key comparison must read the right piece.
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var texts = List[String](capacity=WIDE_ROWS)
    var valid = List[Bool](capacity=WIDE_ROWS)
    var values = List[Int64](capacity=WIDE_ROWS)
    for row in range(WIDE_ROWS):
        var p = wide_part(row)
        texts.append(text_of(p))
        valid.append(text_valid(p))
        values.append(Int64(wide_value(row)))
    var data = DataFrame(
        [
            Series("t", Column[String](texts^, valid)),
            Series("v", Column[Int64](values^)),
        ]
    )
    # Rank "min", descending: one plus the partition's values above the
    # row's. Partitions are the key strings (several parts share one), and
    # nulls form one partition.
    var above = Dict[String, List[Int]]()
    for row in range(WIDE_ROWS):
        var key = wide_key(row)
        if key not in above:
            above[key] = List[Int](length=WIDE_VALUES + 1, fill=0)
        above[key][wide_value(row)] += 1
    var keys = List[String]()
    for key in above.keys():
        keys.append(key)
    for key in keys:
        ref counts = above[key]
        var total = 0
        for v in range(WIDE_VALUES, -1, -1):
            var here = counts[v]
            counts[v] = total
            total += here
    var expression = (
        col("v").rank(method="min", descending=True).over("t").alias("rank")
    )
    var eager = data.with_columns([expression.copy()])
    var lazy = data.lazy().with_columns([expression.copy()]).collect()
    for result in [eager.copy(), lazy.copy()]:
        var ranks = result.column("rank").cast(DataType.INT64)
        for row in range(WIDE_ROWS):
            assert_equal(
                Int(ranks[row].int64()),
                above[wide_key(row)][wide_value(row)] + 1,
            )
    _ = workers^
