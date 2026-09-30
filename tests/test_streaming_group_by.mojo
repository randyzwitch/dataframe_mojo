"""Streaming group-by (#326): a lazy plan over an in-memory frame streams
only when its keys repeat, and file scans keep streaming with per-group
state merged across batches. Whichever path runs, results match the eager
group-by, including `n_unique` over groups that hold one distinct value
(kept inline) and groups whose values spill into sets, with nulls, NaN and
-0.0.
"""
from std.testing import TestSuite, assert_true

from dataframe import (
    Column,
    DataFrame,
    Series,
    StringColumn,
    Expr,
    col,
    read_csv,
    scan_csv,
    write_csv,
)

comptime CSV_PATH = "/tmp/dataframe_mojo_streaming_group_by.csv"


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def frame(rows: Int, groups: Int, seed: UInt64) raises -> DataFrame:
    var rng = Lcg(seed)
    var keys = List[Int64](capacity=rows)
    var ints = List[Int64](capacity=rows)
    var int_valid = List[Bool](capacity=rows)
    var floats = List[Float64](capacity=rows)
    var words = List[String](capacity=rows)
    var nan = Float64(0) / Float64(0)
    for i in range(rows):
        var k = rng.next(groups)
        keys.append(Int64(k))
        # Even groups hold one value, odd groups several.
        ints.append(Int64(k) if k % 2 == 0 else Int64(rng.next(40)))
        int_valid.append(rng.next(17) != 0)
        var pick = rng.next(12)
        floats.append(
            nan if pick == 0 else (-0.0 if pick == 1 else Float64(pick % 4))
        )
        words.append("w" + String(rng.next(30)))
        _ = i
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("i", Column[Int64](ints^, int_valid^)),
            Series("f", Column[Float64](floats^)),
            Series("s", StringColumn(words)),
        ]
    )


def aggregations() -> List[Expr]:
    return [
        col("i").n_unique().alias("i_unique"),
        col("f").n_unique().alias("f_unique"),
        col("s").n_unique().alias("s_unique"),
        col("i").sum().alias("i_sum"),
        col("f").len().alias("rows"),
    ]


def check(data: DataFrame, label: String) raises:
    var eager = data.group_by("k", maintain_order=True).agg(aggregations())
    var lazy = (
        data.lazy()
        .group_by("k", maintain_order=True)
        .agg(aggregations())
        .collect(batch_size=4096)
    )
    assert_true(lazy.equals(eager), label + " lazy frame")
    # The scan is checked against the eager group-by of the same file, so
    # that CSV's rendering of values is not part of the comparison.
    write_csv(data, CSV_PATH)
    var reread = read_csv(CSV_PATH)
    var expected = reread.group_by("k", maintain_order=True).agg(aggregations())
    var streamed = (
        scan_csv(CSV_PATH)
        .group_by("k", maintain_order=True)
        .agg(aggregations())
        .collect(batch_size=4096)
    )
    assert_true(streamed.equals(expected), label + " streamed scan")


def test_few_and_many_groups() raises:
    # Keys that repeat stream from the frame; mostly distinct keys run
    # eagerly; the CSV scan always streams.
    check(frame(50_000, 20, 3), "20 groups")
    check(frame(50_000, 30_000, 5), "30,000 groups")
    check(frame(50_000, 50_000, 7), "mostly distinct")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
