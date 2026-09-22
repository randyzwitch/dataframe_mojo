"""Phase-reporting synthetic benchmark for raw-key partitioned #105 joins.

Usage: raw_partitioned_join_bench int64|string ROWS REPETITIONS WORKERS
The right side is a unique dimension and the left side repeats those keys, so
the workload avoids a deliberately quadratic many-to-many result.
"""
from std.sys import argv

from dataframe import Column, DataFrame, Series
from raw_partitioned_join_lib import (
    RawPartitionedJoin,
    raw_partitioned_int64_inner,
    raw_partitioned_string_inner,
)


comptime RIGHT_KEYS = 100_003


def int_inputs(rows: Int) raises -> Tuple[DataFrame, DataFrame]:
    var left_keys = List[Int64](capacity=rows)
    var left_valid = List[Bool](capacity=rows)
    var right_keys = List[Int64](capacity=RIGHT_KEYS)
    for key in range(RIGHT_KEYS):
        right_keys.append(Int64(key))
    for row in range(rows):
        left_keys.append(Int64(row % RIGHT_KEYS))
        left_valid.append(row % 127 != 0)
    return (
        DataFrame([Series("k", Column[Int64](left_keys^, left_valid^))]),
        DataFrame([Series("k", Column[Int64](right_keys^))]),
    )


def string_inputs(rows: Int) raises -> Tuple[DataFrame, DataFrame]:
    var left_keys = List[String](capacity=rows)
    var left_valid = List[Bool](capacity=rows)
    var right_keys = List[String](capacity=RIGHT_KEYS)
    for key in range(RIGHT_KEYS):
        right_keys.append("key_" + String(key))
    for row in range(rows):
        left_keys.append("key_" + String(row % RIGHT_KEYS))
        left_valid.append(row % 127 != 0)
    return (
        DataFrame([Series("k", Column[String](left_keys^, left_valid^))]),
        DataFrame([Series("k", Column[String](right_keys^))]),
    )


def report(kind: String, rows: Int, result: RawPartitionedJoin):
    print(
        "kind,rows,output_rows,partition_ns,build_probe_ns,order_ns,gather_ns"
    )
    print(
        kind,
        ",",
        rows,
        ",",
        len(result.left_rows),
        ",",
        result.partition_ns,
        ",",
        result.build_probe_ns,
        ",",
        result.order_ns,
        ",",
        result.gather_ns,
        sep="",
    )


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error(
            "usage: raw_partitioned_join_bench int64|string ROWS REPETITIONS WORKERS"
        )
    var kind = String(args[1])
    var rows = Int(String(args[2]))
    var repetitions = Int(String(args[3]))
    var workers = Int(String(args[4]))
    if rows <= 0 or repetitions <= 0 or workers <= 0:
        raise Error("rows, repetitions, and workers must be positive")
    if kind == "int64":
        var inputs = int_inputs(rows)
        var best = raw_partitioned_int64_inner(
            inputs[0], inputs[1], 0, 0, workers
        )
        for _ in range(1, repetitions):
            var result = raw_partitioned_int64_inner(
                inputs[0], inputs[1], 0, 0, workers
            )
            if (
                result.partition_ns
                + result.build_probe_ns
                + result.order_ns
                + result.gather_ns
                < best.partition_ns
                + best.build_probe_ns
                + best.order_ns
                + best.gather_ns
            ):
                best = result^
        report(kind, rows, best)
        return
    if kind == "string":
        var inputs = string_inputs(rows)
        var best = raw_partitioned_string_inner(
            inputs[0], inputs[1], 0, 0, workers
        )
        for _ in range(1, repetitions):
            var result = raw_partitioned_string_inner(
                inputs[0], inputs[1], 0, 0, workers
            )
            if (
                result.partition_ns
                + result.build_probe_ns
                + result.order_ns
                + result.gather_ns
                < best.partition_ns
                + best.build_probe_ns
                + best.order_ns
                + best.gather_ns
            ):
                best = result^
        report(kind, rows, best)
        return
    raise Error("kind must be int64 or string")
