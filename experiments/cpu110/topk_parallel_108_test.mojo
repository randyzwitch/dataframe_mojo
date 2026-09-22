"""Correctness checks for the isolated parallel `smallest_indices` candidate.

Build with `-I /tmp/df_topk_experiment`, whose dataframe/series.mojo routes
eligible selection through per-range stable top-k. This source deliberately
uses public DataFrame operations so rank encoding and gather participate.
"""
from std.testing import TestSuite, assert_true

from dataframe import Column, DataFrame, Series, StringColumn


def frame() raises -> DataFrame:
    var a: List[Int64] = [3, 3, 1, 1, 2, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    var a_valid: List[Bool] = [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    var b: List[String] = [
        "c",
        "a",
        "b",
        "z",
        "a",
        "b",
        "a",
        "c",
        "x",
        "b",
        "a",
        "c",
        "a",
        "b",
        "c",
        "z",
    ]
    var b_valid: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    var row = List[Int64](capacity=len(a))
    for i in range(len(a)):
        row.append(Int64(i))
    return DataFrame(
        [
            Series("a", Column[Int64](a^, a_valid^)),
            Series("b", StringColumn(b^, b_valid^)),
            Series("row", Column[Int64](row^)),
        ]
    )


def check(frame: DataFrame, k: Int, by: List[String]) raises:
    assert_true(
        frame.top_k(k, by).equals(frame.sort(by, descending=True).head(k)),
        "top-k stable order",
    )
    assert_true(
        frame.bottom_k(k, by).equals(frame.sort(by).head(k)),
        "bottom-k stable order",
    )


def test_stable_ties_nulls_directions_and_multikey() raises:
    var input = frame()
    check(input, 0, ["a"])
    check(input, 1, ["a"])
    check(input, 7, ["a"])
    check(input, 7, ["b"])
    check(input, 7, ["a", "b"])


def test_parallel_path_keeps_multikey_ties_nulls_and_gather_order() raises:
    # worker_count reaches two at 131072 rows, so this exercises the actual
    # range jobs rather than only the serial fallback used by the tiny matrix.
    comptime rows = 131_072
    var a = List[Int64](capacity=rows)
    var av = List[Bool](capacity=rows)
    var b = List[Int64](capacity=rows)
    var bv = List[Bool](capacity=rows)
    var payload = List[Int64](capacity=rows)
    for row in range(rows):
        a.append(Int64(row % 17))
        av.append(row % 113 != 0)
        b.append(Int64((row // 17) % 11))
        bv.append(row % 127 != 0)
        payload.append(Int64(row))
    var input = DataFrame(
        [
            Series("a", Column[Int64](a^, av^)),
            Series("b", Column[Int64](b^, bv^)),
            Series("row", Column[Int64](payload^)),
        ]
    )
    check(input, 31, ["a", "b"])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
