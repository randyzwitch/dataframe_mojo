"""End-to-end inner join timing for the isolated count-prefix-fill branch.

Usage: join_inner_end_to_end DATA_DIR ROWS REPETITIONS
CSV loading is setup. Timed work is exactly `left.join(right, "jk")`.
"""
from std.sys import argv
from std.time import monotonic

from dataframe import CsvField, CsvSchema, DataFrame, read_csv


def left_schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("key_low"),
            CsvField.int64("key_high"),
            CsvField.int64("key_skew"),
            CsvField.string("key_str"),
            CsvField.int64("jk"),
            CsvField.float64("x"),
            CsvField.float64("y"),
            CsvField.int64("n"),
        ]
    )


def right_schema() raises -> CsvSchema:
    return CsvSchema([CsvField.int64("jk"), CsvField.float64("r")])


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error("usage: join_inner_end_to_end DATA_DIR ROWS REPETITIONS")
    var directory = String(args[1])
    var rows = String(args[2])
    var repetitions = Int(String(args[3]))
    var left = read_csv(directory + "/left_" + rows + ".csv", left_schema())
    var right = read_csv(directory + "/right_" + rows + ".csv", right_schema())
    var warm = left.join(right, "jk")
    var best = Int.MAX
    for _ in range(repetitions):
        var start = monotonic()
        var result = left.join(right, "jk")
        best = min(best, monotonic() - start)
        if result.height() != warm.height():
            raise Error("join output height changed")
    print("workload,rows,output_rows,best_ns")
    print(
        "inner_join_count_prefix_fill,",
        rows,
        ",",
        warm.height(),
        ",",
        best,
        sep="",
    )
