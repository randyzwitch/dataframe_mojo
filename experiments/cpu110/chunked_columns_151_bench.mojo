"""Measure deferred CSV reassembly in the isolated ChunkedFrame prototype.

Usage: bench_chunked_csv CSV_PATH chunked|contiguous read|sum|row|rechunk.
The harness retains the same contiguous reference frame in both layouts. It
runs one warmup and seven reported iterations; chunked output is materialized
once and checked cell-for-cell against that reference before timing.
"""
from std.sys import argv
from std.time import monotonic

from dataframe import CsvField, CsvSchema, DataFrame, read_csv, read_csv_chunked


comptime ITERATIONS = 8
comptime X_COLUMN = 5


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
        ]
    )


def same_float(left: Float64, right: Float64) -> Bool:
    return left - right <= 1e-6 and right - left <= 1e-6


def consume_contiguous(frame: DataFrame, mode: String, rows: Int) raises -> Int:
    var started = monotonic()
    if mode == "sum":
        _ = frame._columns[X_COLUMN].sum()
    elif mode == "row":
        _ = frame._columns[X_COLUMN].get(0)
        _ = frame._columns[X_COLUMN].get(rows // 2)
        _ = frame._columns[X_COLUMN].get(rows - 1)
    elif mode == "rechunk":
        _ = frame.copy()
    return monotonic() - started


def run_contiguous(path: String, mode: String, reference: DataFrame) raises:
    var rows = reference.height()
    var expected_sum = reference._columns[X_COLUMN].sum().float64()
    var expected_first = reference._columns[X_COLUMN].get(0).float64()
    var expected_middle = reference._columns[X_COLUMN].get(rows // 2).float64()
    var expected_last = reference._columns[X_COLUMN].get(rows - 1).float64()
    for iteration in range(ITERATIONS):
        var started = monotonic()
        var frame = read_csv(path, schema())
        var read_elapsed = monotonic() - started
        if iteration == 0 and not frame.equals(reference):
            raise Error("contiguous warmup differs from reference")
        var consume_elapsed = consume_contiguous(frame, mode, rows)
        if mode == "sum" and not same_float(
            frame._columns[X_COLUMN].sum().float64(), expected_sum
        ):
            raise Error("contiguous sum differs from reference")
        if mode == "row" and (
            frame._columns[X_COLUMN].get(0).float64() != expected_first
            or frame._columns[X_COLUMN].get(rows // 2).float64()
            != expected_middle
            or frame._columns[X_COLUMN].get(rows - 1).float64() != expected_last
        ):
            raise Error("contiguous row access differs from reference")
        if iteration > 0:
            print("iteration,", iteration, sep="")
            print("read_ns,", read_elapsed, sep="")
            print("consume_ns,", consume_elapsed, sep="")
    print("rows,", rows, sep="")
    print("chunks,1")


def run_chunked(path: String, mode: String, reference: DataFrame) raises:
    var rows = reference.height()
    var expected_sum = reference._columns[X_COLUMN].sum().float64()
    var expected_first = reference._columns[X_COLUMN].get(0).float64()
    var expected_middle = reference._columns[X_COLUMN].get(rows // 2).float64()
    var expected_last = reference._columns[X_COLUMN].get(rows - 1).float64()
    var chunks = 0
    for iteration in range(ITERATIONS):
        var started = monotonic()
        var frame = read_csv_chunked(path, schema())
        var read_elapsed = monotonic() - started
        if iteration == 0:
            chunks = frame.chunk_count()
        elif frame.chunk_count() != chunks:
            raise Error("chunk layout changed between iterations")
        var consume_started = monotonic()
        if mode == "sum":
            _ = frame.sum_float64_at(X_COLUMN)
        elif mode == "row":
            _ = frame.item_at(0, X_COLUMN)
            _ = frame.item_at(rows // 2, X_COLUMN)
            _ = frame.item_at(rows - 1, X_COLUMN)
        elif mode == "rechunk":
            _ = frame.rechunk()
        var consume_elapsed = monotonic() - consume_started
        if mode == "sum" and not same_float(
            frame.sum_float64_at(X_COLUMN).float64(), expected_sum
        ):
            raise Error("chunked sum differs from reference")
        if mode == "row" and (
            frame.item_at(0, X_COLUMN).float64() != expected_first
            or frame.item_at(rows // 2, X_COLUMN).float64() != expected_middle
            or frame.item_at(rows - 1, X_COLUMN).float64() != expected_last
        ):
            raise Error("chunked row access differs from reference")
        if iteration > 0:
            print("iteration,", iteration, sep="")
            print("read_ns,", read_elapsed, sep="")
            print("consume_ns,", consume_elapsed, sep="")
    print("rows,", rows, sep="")
    print("chunks,", chunks, sep="")


def main() raises:
    var args = argv()
    if len(args) != 4 and len(args) != 5:
        raise Error(
            "usage: bench_chunked_csv CSV_PATH chunked|contiguous read|sum|row|rechunk [--rss]"
        )
    var path = String(args[1])
    var layout = String(args[2])
    var mode = String(args[3])
    var rss_only = len(args) == 5
    if rss_only and String(args[4]) != "--rss":
        raise Error("unknown option")
    if layout != "chunked" and layout != "contiguous":
        raise Error("layout must be chunked or contiguous")
    if mode != "read" and mode != "sum" and mode != "row" and mode != "rechunk":
        raise Error("mode must be read, sum, row, or rechunk")

    # Retained in both modes, so /usr/bin/time compares the same lifetime.
    var reference = read_csv(path, schema())
    if not rss_only:
        var candidate = read_csv_chunked(path, schema())
        var materialized = candidate.rechunk()
        if not materialized.equals(reference):
            raise Error(
                "chunked CSV materialization differs from contiguous read"
            )
    if layout == "chunked":
        run_chunked(path, mode, reference)
    else:
        run_contiguous(path, mode, reference)
