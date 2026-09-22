"""Whole-pipeline stable sort harness for the isolated #108 radix candidate.

Usage: radix_pipeline_108 ROWS int64|string

Build with `-I /tmp/df_radix_experiment`. `merge_sort_ns` explicitly forces
the existing rank-plus-merge path; `candidate_sort_ns` calls DataFrame.sort,
which takes stable radix only for a single non-null physical Int64 key. Both
paths include rank construction and final gather. String input deliberately
uses the merge fallback, verifying that the eligibility gate preserves it.
"""
from std.sys import argv
from std.time import monotonic

from dataframe import Column, DataFrame, Series, StringColumn
from dataframe.series import sort_indices


comptime REPETITIONS = 4


def int_frame(rows: Int) raises -> DataFrame:
    var values = List[Int64](capacity=rows)
    var payload = List[Int64](capacity=rows)
    var state = UInt64(0x9E3779B97F4A7C15)
    for row in range(rows):
        state = state * 6364136223846793005 + 1442695040888963407
        values.append(Int64(state))
        payload.append(Int64(row))
    return DataFrame(
        [
            Series("key", Column[Int64](values^)),
            Series("payload", Column[Int64](payload^)),
        ]
    )


def string_frame(rows: Int) raises -> DataFrame:
    var values = List[String](capacity=rows)
    var payload = List[Int64](capacity=rows)
    var state = UInt64(0xD1B54A32D192ED03)
    for row in range(rows):
        state = state * 6364136223846793005 + 1442695040888963407
        values.append("key_" + String(Int((state >> 23) % 250_003)))
        payload.append(Int64(row))
    return DataFrame(
        [
            Series("key", StringColumn(values^)),
            Series("payload", Column[Int64](payload^)),
        ]
    )


def merge_sort(frame: DataFrame) raises -> DataFrame:
    return frame.take(sort_indices(frame._sort_ranks(["key"], [False], [True])))


def best_merge(frame: DataFrame) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var started = monotonic()
        var result = merge_sort(frame)
        best = min(best, monotonic() - started)
        if result.height() != frame.height():
            raise Error("merge sort height changed")
    return best


def best_candidate(frame: DataFrame) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var started = monotonic()
        var result = frame.sort("key")
        best = min(best, monotonic() - started)
        if result.height() != frame.height():
            raise Error("candidate sort height changed")
    return best


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("usage: radix_pipeline_108 ROWS int64|string")
    var rows = Int(String(args[1]))
    var kind = String(args[2])
    if rows < 0 or (kind != "int64" and kind != "string"):
        raise Error("ROWS must be nonnegative and kind must be int64 or string")
    var frame = int_frame(rows) if kind == "int64" else string_frame(rows)
    if not frame.sort("key").equals(merge_sort(frame)):
        raise Error("candidate differs from stable merge sort")
    print("kind", kind, sep=",")
    print("rows", rows, sep=",")
    print("merge_sort_ns", best_merge(frame), sep=",")
    print("candidate_sort_ns", best_candidate(frame), sep=",")
