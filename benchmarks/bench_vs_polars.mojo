"""The dataframe_mojo side of the Polars head-to-head (see scripts/bench_polars.py).

Usage: bench_vs_polars DATA_DIR ROWS REPETITIONS

Reads DATA_DIR/left_ROWS.csv and DATA_DIR/right_ROWS.csv, written by the
Python driver so both engines read the same bytes, runs each workload once
to warm up and then REPETITIONS timed runs, and prints one line per workload:

    workload<TAB>best_ns<TAB>height<TAB>value

`height` and `value` (an order-insensitive column total) let the driver check
that both engines computed the same thing before comparing their times.
Input construction and the check values are untimed. Thread count comes from
DATAFRAME_THREADS, which the driver pins equal to POLARS_MAX_THREADS.
"""
from std.sys import argv
from std.time import monotonic

from dataframe import (
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Expr,
    col,
    lit,
    read_csv,
)
from dataframe.parallel import worker_count


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


def total(frame: DataFrame, name: String) raises -> Float64:
    """Order-insensitive sum of one column, nulls skipped. A Bool column
    counts its true rows, as Polars' sum does."""
    var expr = col(name)
    if frame.column(name).dtype() == DataType.BOOL:
        expr = expr.cast(DataType.INT64)
    var cell = frame.select(expr.sum()).item()
    if cell.is_null():
        return 0
    if cell.dtype() == DataType.INT64:
        return Float64(cell.int64())
    return cell.float64()


def report(workload: String, best: Int, height: Int, value: Float64):
    print(workload, best, height, value, sep="\t")


def timed_csv(path: String, repetitions: Int) raises -> DataFrame:
    var warm = read_csv(path, left_schema())
    var best = Int.MAX
    for _ in range(repetitions):
        var start = monotonic()
        var frame = read_csv(path, left_schema())
        best = min(best, monotonic() - start)
        if frame.height() != warm.height():
            raise Error("csv height changed between runs")
    report("csv_read", best, warm.height(), total(warm, "x"))
    return warm^


def timed_with_columns(
    frame: DataFrame, workload: String, expr: Expr, repetitions: Int
) raises:
    var warm = frame.with_columns(expr.alias("out"))
    var best = Int.MAX
    for _ in range(repetitions):
        var start = monotonic()
        var result = frame.with_columns(expr.alias("out"))
        best = min(best, monotonic() - start)
        if result.height() != warm.height():
            raise Error(workload + " height changed between runs")
    report(workload, best, warm.height(), total(warm, "out"))


def timed_filter(frame: DataFrame, repetitions: Int) raises:
    var predicate = col("x") > lit(Float64(0))
    var warm = frame.filter(predicate)
    var best = Int.MAX
    for _ in range(repetitions):
        var start = monotonic()
        var result = frame.filter(predicate)
        best = min(best, monotonic() - start)
        if result.height() != warm.height():
            raise Error("filter height changed between runs")
    report("filter", best, warm.height(), total(warm, "y"))


def timed_global_sum(frame: DataFrame, repetitions: Int) raises:
    var warm = frame.select(col("x").sum())
    var best = Int.MAX
    for _ in range(repetitions):
        var start = monotonic()
        var result = frame.select(col("x").sum())
        best = min(best, monotonic() - start)
        if result.height() != 1:
            raise Error("global sum must produce one row")
    report("global_sum", best, 1, warm.item().float64())


def timed_grouped(
    frame: DataFrame, workload: String, key: String, repetitions: Int
) raises:
    var exprs: List[Expr] = [
        col("x").sum().alias("s"),
        col("n").count().alias("c"),
    ]
    var warm = frame.group_by(key).agg(exprs)
    var best = Int.MAX
    for _ in range(repetitions):
        var start = monotonic()
        var result = frame.group_by(key).agg(exprs)
        best = min(best, monotonic() - start)
        if result.height() != warm.height():
            raise Error(workload + " group count changed between runs")
    report(workload, best, warm.height(), total(warm, "s") + total(warm, "c"))


def timed_join(left: DataFrame, right: DataFrame, repetitions: Int) raises:
    var warm = left.join(right, on="jk", how="inner")
    var best = Int.MAX
    for _ in range(repetitions):
        var start = monotonic()
        var result = left.join(right, on="jk", how="inner")
        best = min(best, monotonic() - start)
        if result.height() != warm.height():
            raise Error("join height changed between runs")
    report("join_inner", best, warm.height(), total(warm, "r"))


def timed_sort(frame: DataFrame, repetitions: Int) raises:
    var by: List[String] = ["key_low", "x"]
    var warm = frame.sort(by)
    var best = Int.MAX
    for _ in range(repetitions):
        var start = monotonic()
        var result = frame.sort(by)
        best = min(best, monotonic() - start)
        if result.height() != warm.height():
            raise Error("sort height changed between runs")
    # The value is order-sensitive on purpose: both engines must agree on
    # the row order, not only the row set.
    var order_check = warm.head(1000).select(col("x").sum()).item()
    var value = 0.0 if order_check.is_null() else order_check.float64()
    report("sort_multi", best, warm.height(), value)


def main() raises:
    var args = argv()
    if len(args) != 4 and len(args) != 5:
        raise Error(
            "usage: bench_vs_polars DATA_DIR ROWS REPETITIONS [--csv-only]"
        )
    var csv_only = len(args) == 5
    if csv_only and String(args[4]) != "--csv-only":
        raise Error("unknown benchmark option")
    var data_dir = String(args[1])
    var rows = String(args[2])
    var repetitions = Int(String(args[3]))
    print("# threads=", worker_count(1 << 40), sep="")

    var left = timed_csv(data_dir + "/left_" + rows + ".csv", repetitions)
    if csv_only:
        return
    var right = read_csv(data_dir + "/right_" + rows + ".csv", right_schema())

    timed_with_columns(
        left,
        "arithmetic_chain",
        (col("x") + lit(Float64(3)))
        * (col("y") - lit(Float64(2)))
        / lit(Float64(4)),
        repetitions,
    )
    timed_with_columns(
        left, "nullable_compare", col("x") > col("y"), repetitions
    )
    timed_filter(left, repetitions)
    timed_global_sum(left, repetitions)
    timed_grouped(left, "grouped_low", "key_low", repetitions)
    timed_grouped(left, "grouped_high", "key_high", repetitions)
    timed_grouped(left, "grouped_skew", "key_skew", repetitions)
    timed_grouped(left, "grouped_str", "key_str", repetitions)
    timed_join(left, right, repetitions)
    timed_sort(left, repetitions)
