"""Parquet format/Arrow-import yardstick, NOT a native Mojo Parquet reader.

Usage: parquet_arrow_experiment FILE.parquet THREADS
Includes pyarrow read, table rechunking, Arrow export and Mojo's copying
import. Reports read/import separately plus the best whole operation.
"""
from std.python import Python
from std.sys import argv
from std.time import monotonic
from dataframe import ArrowArray, ArrowSchema, import_arrow, col
from dataframe.arrow import _leak, _reclaim


def main() raises:
    var args = argv()
    var path = String(args[1])
    var threads = Int(String(args[2]))
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var pc = Python.import_module("pyarrow.compute")
    pa.set_cpu_count(threads)
    var best = Int.MAX
    var best_read = 0
    var best_import = 0
    var expected_rows = 0
    var expected_sum = Float64(0)
    for iteration in range(8):
        var start = monotonic()
        var table = pq.read_table(path, use_threads=threads > 1)
        var read_elapsed = monotonic() - start
        var batch = table.combine_chunks().to_batches()[0]
        var array = _leak(ArrowArray())
        var schema = _leak(ArrowSchema())
        batch._export_to_c(array, schema)
        var frame = import_arrow(array, schema)
        _ = _reclaim[ArrowArray](array)
        _ = _reclaim[ArrowSchema](schema)
        var elapsed = monotonic() - start
        var total = frame.select(col("x").sum()).item().float64()
        if iteration == 0:
            expected_rows = frame.height()
            expected_sum = Float64(String(pc.sum(table.column("x")).as_py()))
        if frame.height() != expected_rows or abs(total - expected_sum) > 1e-6:
            raise Error("Arrow bridge row/sum mismatch")
        if iteration > 0 and elapsed < best:
            best = elapsed
            best_read = read_elapsed
            best_import = elapsed - read_elapsed
    print(
        "total_ns",
        best,
        "read_ns",
        best_read,
        "rechunk_and_import_ns",
        best_import,
        "rows",
        expected_rows,
        sep=",",
    )
