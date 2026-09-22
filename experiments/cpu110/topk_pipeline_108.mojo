"""Whole-pipeline #108 top-k harness: rank encoding, selection and gather.

Usage: topk_pipeline_108 ROWS int64|string

Build this source with `-I /tmp/df_topk_experiment`. It reports serial full
sort, the clone's private serial heap selection, and `DataFrame.top_k`; the
last is the parallel candidate. Both Int64 and String keys include nulls and
repeated values, and the result is checked against a stable full-sort head
before measurement.
"""
from std.sys import argv
from std.time import monotonic

from dataframe import Column, DataFrame, Series, StringColumn
from dataframe.series import _smallest_indices_serial


comptime K = 100
comptime REPETITIONS = 4


def int_frame(rows: Int) raises -> DataFrame:
    var values = List[Int64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    var payload = List[Int64](capacity=rows)
    var state = UInt64(0xA24BAED4963EE407)
    for row in range(rows):
        state = state * 6364136223846793005 + 1442695040888963407
        values.append(Int64((state >> 23) % 20_003) - 10_001)
        valid.append(row % 31 != 0)
        payload.append(Int64(row))
    return DataFrame(
        [
            Series("key", Column[Int64](values^, valid^)),
            Series("payload", Column[Int64](payload^)),
        ]
    )


def string_frame(rows: Int) raises -> DataFrame:
    var values = List[String](capacity=rows)
    var valid = List[Bool](capacity=rows)
    var payload = List[Int64](capacity=rows)
    var state = UInt64(0x8CB92BA72F3D8DD7)
    for row in range(rows):
        state = state * 6364136223846793005 + 1442695040888963407
        # Equal padded strings make stable row ties observable.
        var code = Int((state >> 23) % 20_003)
        values.append("key_" + String(code))
        valid.append(row % 31 != 0)
        payload.append(Int64(row))
    return DataFrame(
        [
            Series("key", StringColumn(values^, valid^)),
            Series("payload", Column[Int64](payload^)),
        ]
    )


def serial_top(frame: DataFrame) raises -> DataFrame:
    var ranks = frame._sort_ranks(["key"], [True], [True])
    return frame.take(_smallest_indices_serial(ranks, K))


def best_full(frame: DataFrame) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var started = monotonic()
        var result = frame.sort("key", descending=True).head(K)
        best = min(best, monotonic() - started)
        if result.height() != K:
            raise Error("full sort returned the wrong height")
    return best


def best_serial_top(frame: DataFrame) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var started = monotonic()
        var result = serial_top(frame)
        best = min(best, monotonic() - started)
        if result.height() != K:
            raise Error("serial top-k returned the wrong height")
    return best


def best_candidate_top(frame: DataFrame) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var started = monotonic()
        var result = frame.top_k(K, "key")
        best = min(best, monotonic() - started)
        if result.height() != K:
            raise Error("candidate top-k returned the wrong height")
    return best


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("usage: topk_pipeline_108 ROWS int64|string")
    var rows = Int(String(args[1]))
    if rows < K:
        raise Error("ROWS must be at least K")
    var kind = String(args[2])
    var frame = int_frame(rows) if kind == "int64" else string_frame(rows)
    if kind != "int64" and kind != "string":
        raise Error("kind must be int64 or string")
    var expected = frame.sort("key", descending=True).head(K)
    if not frame.top_k(K, "key").equals(expected):
        raise Error("candidate top-k differs from stable full sort")
    if not serial_top(frame).equals(expected):
        raise Error("serial heap differs from stable full sort")
    print("kind", kind, sep=",")
    print("rows", rows, sep=",")
    print("k", K, sep=",")
    print("full_sort_ns", best_full(frame), sep=",")
    print("serial_topk_ns", best_serial_top(frame), sep=",")
    print("candidate_topk_ns", best_candidate_top(frame), sep=",")
