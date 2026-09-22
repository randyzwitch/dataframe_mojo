"""Compare the bounded #109 CSV pipeline with the equivalent eager chain.

Usage: bench_streaming_109_10col CSV_PATH eager|stream|parallel [BATCH_ROWS]. Add
`BATCH_ROWS --rss EXPECTED_SUM` for a streaming-only RSS run: no eager frame
is constructed, and final sums use documented floating-reassociation tolerance.
"""
from std.sys import argv
from std.time import monotonic

from dataframe import (
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    Series,
    col,
    lit,
    read_csv,
    stream_csv_filter_select_sum,
    stream_csv_parallel_filter_select_sum,
)


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


def same_float(left: Float64, right: Float64) -> Bool:
    # Float reductions may reassociate partitions; this bound is much tighter
    # than the error expected for the generated benchmark's magnitudes.
    var delta = left - right
    if delta < 0:
        delta = -delta
    var scale = max(
        left if left >= 0 else -left, right if right >= 0 else -right
    )
    return delta <= 1e-6 + 1e-12 * scale


def eager(path: String) raises -> DataFrame:
    var input = read_csv(path, schema())
    var selected = input.filter(col("x") > lit(Float64(0))).select_exprs(
        [(col("x") * lit(Float64(2))).alias("double"), col("y")]
    )
    var result = selected.column("double").sum()
    var values = List[Float64]([0.0 if result.is_null() else result.float64()])
    var valid = List[Bool]([not result.is_null()])
    return DataFrame([Series("sum", Column[Float64](values^, valid))])


def main() raises:
    var args = argv()
    if len(args) < 3 or len(args) > 6:
        raise Error(
            "usage: bench_streaming_109_10col CSV_PATH eager|stream|parallel [BATCH_ROWS] [--rss EXPECTED_SUM]"
        )
    var path = String(args[1])
    var mode = String(args[2])
    if mode != "eager" and mode != "stream" and mode != "parallel":
        raise Error("mode must be eager, stream, or parallel")
    var batch_rows = 8192 if len(args) == 3 else Int(String(args[3]))
    var window_bytes = (32 << 20) if mode == "parallel" and len(
        args
    ) == 3 else batch_rows
    var rss_only = len(args) == 6
    if rss_only and String(args[4]) != "--rss":
        raise Error("--rss requires EXPECTED_SUM")
    if len(args) == 5:
        raise Error("--rss requires EXPECTED_SUM")

    var expected_sum = Float64(0)
    if rss_only:
        expected_sum = Float64(String(args[5]))
    else:
        var expected = eager(path)
        expected_sum = expected.item().float64()
        var checked = stream_csv_filter_select_sum(
            path, schema(), batch_rows=batch_rows
        )
        if not same_float(checked.item().float64(), expected_sum):
            raise Error("stream pipeline differs from eager chain")

    var repetitions = 1 if rss_only else ITERATIONS
    for iteration in range(repetitions):
        var started = monotonic()
        var result = eager(path) if mode == "eager" else (
            stream_csv_filter_select_sum(
                path, schema(), batch_rows=batch_rows
            ) if mode
            == "stream" else stream_csv_parallel_filter_select_sum(
                path, schema(), window_bytes=window_bytes
            )
        )
        var elapsed = monotonic() - started
        if not same_float(result.item().float64(), expected_sum):
            raise Error("pipeline result changed between iterations")
        if iteration > 0 or rss_only:
            print("iteration,", iteration, sep="")
            print("pipeline_ns,", elapsed, sep="")
    print("sum,", expected_sum, sep="")
    print("batch_rows,", batch_rows, sep="")
    if mode == "parallel":
        print("window_bytes,", window_bytes, sep="")
