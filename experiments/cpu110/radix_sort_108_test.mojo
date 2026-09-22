"""Correctness checks for the isolated stable Int64/temporal radix candidate."""
from std.testing import TestSuite, assert_true

from dataframe import Column, DataFrame, DataType, Series, StringColumn


def check_order(frame: DataFrame, by: List[String]) raises:
    assert_true(
        frame.sort(by).equals(frame.take(frame.arg_sort(by))),
        "ascending order",
    )
    assert_true(
        frame.sort(by, descending=True).equals(
            frame.take(frame.arg_sort(by, descending=True))
        ),
        "descending order",
    )


def test_signed_extremes_ties_and_temporal_direction() raises:
    var values: List[Int64] = [
        Int64.MIN,
        -5,
        -5,
        -1,
        0,
        0,
        4,
        Int64.MAX,
        Int64.MIN,
        Int64.MAX,
    ]
    var row = List[Int64](capacity=len(values))
    for i in range(len(values)):
        row.append(Int64(i))
    var ints = DataFrame(
        [
            Series("k", Column[Int64](values.copy())),
            Series("row", Column[Int64](row.copy())),
        ]
    )
    check_order(ints, ["k"])
    var dates = DataFrame(
        [
            Series("d", Column[Int64](values^)).with_dtype(DataType.DATE),
            Series("row", Column[Int64](row^)),
        ]
    )
    check_order(dates, ["d"])


def test_nullable_string_and_multikey_fallbacks_match_merge() raises:
    var a: List[Int64] = [2, 1, 2, 1, 3, 1]
    var av: List[Bool] = [True, True, False, True, True, True]
    var b: List[String] = ["z", "a", "b", "a", "c", "a"]
    var bv: List[Bool] = [True, True, True, False, True, True]
    var row: List[Int64] = [0, 1, 2, 3, 4, 5]
    var frame = DataFrame(
        [
            Series("a", Column[Int64](a^, av^)),
            Series("b", StringColumn(b^, bv^)),
            Series("row", Column[Int64](row^)),
        ]
    )
    check_order(frame, ["a"])
    check_order(frame, ["b"])
    check_order(frame, ["a", "b"])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
