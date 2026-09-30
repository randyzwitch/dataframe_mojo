"""Sorting by packed keys (#331) against the general rank-and-merge sort.

Sizes cross the parallel threshold, so the bucket pass, dynamically claimed
bucket sorts, heavy buckets sorted in merged runs, and tie settling for long
strings all run. Every case checks that the packed path applies (or, for
keys it cannot pack, that it declines) and that it returns exactly the row
order of the general path.
"""
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from dataframe import Column, DataFrame, Series, col
from dataframe.packed_sort import packed_arg_sort
from dataframe.series import sort_indices


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def general_order(
    frame: DataFrame,
    by: List[String],
    descending: List[Bool],
    nulls_last: List[Bool],
) raises -> List[Int]:
    return sort_indices(frame._sort_ranks(by, descending, nulls_last))


def check(
    frame: DataFrame, by: List[String], expect_packed: Bool = True
) raises:
    """Every direction and null placement of `by`, packed against general."""
    var columns = List[Series]()
    for name in by:
        columns.append(frame.column(name))
    var n = len(by)
    for pattern in range(4):
        var descending = List[Bool]()
        var nulls_last = List[Bool]()
        for k in range(n):
            # Alternate per key so mixed directions are covered.
            descending.append(((pattern + k) & 1) == 1)
            nulls_last.append(((pattern >> 1) & 1) == 1)
        var packed = packed_arg_sort(columns, descending, nulls_last)
        var label = String(by[0]) + " pattern " + String(pattern)
        if not expect_packed:
            assert_false(Bool(packed), label)
            continue
        assert_true(Bool(packed), label)
        assert_equal(
            packed.value(),
            general_order(frame, by, descending, nulls_last),
            label,
        )
        # The public entry point takes the same path.
        assert_equal(
            frame.arg_sort(by, descending=descending, nulls_last=nulls_last),
            packed.value(),
            label,
        )


def mixed_frame(rows: Int, seed: UInt64) raises -> DataFrame:
    var rng = Lcg(seed)
    var nan = Float64(0) / Float64(0)
    var small = List[Int64]()
    var small_valid = List[Bool]()
    var wide = List[Int64]()
    var unsigned = List[UInt64]()
    var floats = List[Float64]()
    var float_valid = List[Bool]()
    var short = List[String]()
    var short_valid = List[Bool]()
    var long = List[String]()
    var long_valid = List[Bool]()
    var flags = List[Bool]()
    var words: List[String] = [
        "",
        "a",
        "a\0",
        "ab",
        "b",
        "é",
        "zz",
        "a b",
    ]
    for i in range(rows):
        small.append(Int64(rng.next(7)) - 3)
        small_valid.append(rng.next(9) != 0)
        # Both ends of Int64, so the range needs every bit.
        var w = rng.next(4)
        wide.append(
            Int64.MIN if w
            == 0 else (Int64.MAX if w == 1 else Int64(rng.next(1000)) - 500)
        )
        var u = rng.next(3)
        unsigned.append(
            UInt64.MAX if u
            == 0 else (UInt64(1) << 63 if u == 1 else UInt64(rng.next(100)))
        )
        var pick = rng.next(10)
        floats.append(
            nan if pick
            == 0 else (
                -0.0 if pick
                == 1 else (
                    0.0 if pick == 2 else Float64(rng.next(2000)) / 7 - 140
                )
            )
        )
        float_valid.append(rng.next(8) != 0)
        short.append(words[rng.next(len(words))])
        short_valid.append(rng.next(6) != 0)
        # A long shared prefix, then values that differ only past the
        # bytes a key holds, so ties need settling by the strings.
        var tail = String(rng.next(50))
        long.append(
            "https://example.org/path/"
            + ("section-" if rng.next(2) == 0 else "sect")
            + tail
            + ("/" * rng.next(3))
        )
        long_valid.append(rng.next(7) != 0)
        flags.append(rng.next(2) == 0)
        _ = i
    return DataFrame(
        [
            Series("small", Column[Int64](small^, small_valid^)),
            Series("wide", Column[Int64](wide^)),
            Series("unsigned", Column[UInt64](unsigned^)),
            Series("float", Column[Float64](floats^, float_valid^)),
            Series("short", Column[String](short^, short_valid^)),
            Series("long", Column[String](long^, long_valid^)),
            Series("flag", Column[Bool](flags^)),
        ]
    )


def test_single_keys_serial_and_parallel() raises:
    for rows in [0, 1, 7, 1000, 70_000]:
        var frame = mixed_frame(rows, UInt64(rows) + 3)
        var names: List[String] = [
            "small",
            "wide",
            "unsigned",
            "float",
            "short",
            "long",
            "flag",
        ]
        for name in names:
            check(frame, [name])


def test_multiple_keys() raises:
    var frame = mixed_frame(60_000, 11)
    check(frame, ["small", "float"])
    check(frame, ["short", "small", "flag"])
    check(frame, ["flag", "short", "long"])
    check(frame, ["small", "wide"])


def test_keys_that_do_not_pack_fall_back() raises:
    var frame = mixed_frame(5_000, 5)
    # A long string settles ties only as the last key.
    check(frame, ["long", "small"], expect_packed=False)
    # Two full-width keys and the row index exceed 128 bits.
    check(frame, ["wide", "unsigned"], expect_packed=False)
    # The general path still sorts them.
    var order = frame.arg_sort(["long", "small"])
    assert_equal(
        order,
        general_order(frame, ["long", "small"], [False, False], [True, True]),
    )


def test_skewed_values_fill_heavy_buckets() raises:
    # Most rows share one value and the rest spread widely, so one bucket
    # holds most of the column and is sorted in parallel runs.
    var rng = Lcg(17)
    var values = List[Int64]()
    var floats = List[Float64]()
    for _ in range(200_000):
        var common = rng.next(10) != 0
        values.append(Int64(42) if common else Int64(rng.next(1 << 30)))
        floats.append(
            1.0
            + Float64(rng.next(1000))
            * 1e-12 if common else Float64(rng.next(1 << 20))
        )
    var frame = DataFrame(
        [
            Series("v", Column[Int64](values^)),
            Series("f", Column[Float64](floats^)),
        ]
    )
    check(frame, ["v"])
    check(frame, ["f"])
    check(frame, ["v", "f"])


def test_chunked_and_temporal_columns() raises:
    var frame = mixed_frame(40_000, 23)
    var stacked = frame.vstack(frame.slice(100, 30_000))
    assert_true(stacked.column("small").is_chunked())
    check(stacked, ["small"])
    check(stacked, ["short", "float"])
    var dated = stacked.with_columns(
        [col("small").cast("datetime[ms]").alias("when")]
    )
    check(dated, ["when"])


def test_series_argsort_and_sort() raises:
    var frame = mixed_frame(30_000, 29)
    var column = frame.column("float")
    for d in range(2):
        for last in range(2):
            assert_equal(
                column.argsort(d == 1, last == 1),
                sort_indices([column._sort_ranks(d == 1, last == 1)]),
            )
    var sorted = frame.sort(["short", "small"], descending=True)
    var expected = frame.take(
        general_order(frame, ["short", "small"], [True, True], [True, True])
    )
    assert_true(sorted.equals(expected))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
