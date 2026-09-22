"""Contiguous 10-column CSV ingestion baseline for the streaming prototype."""
from std.sys import argv
from std.time import monotonic

from dataframe import CsvField, CsvSchema, read_csv

comptime ITERATIONS = 8


def schema() raises -> CsvSchema:
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
            CsvField.int64("extra_a"),
            CsvField.float64("extra_b"),
        ]
    )


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: bench_csv_10col CSV_PATH")
    var path = String(args[1])
    var warm = read_csv(path, schema())
    var rows = warm.height()
    for iteration in range(ITERATIONS):
        var started = monotonic()
        var frame = read_csv(path, schema())
        var elapsed = monotonic() - started
        if frame.height() != rows:
            raise Error("CSV row count changed between iterations")
        if iteration > 0:
            print("iteration,", iteration, sep="")
            print("read_ns,", elapsed, sep="")
    print("rows,", rows, sep="")
