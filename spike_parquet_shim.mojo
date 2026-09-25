"""Spike: read Parquet through the dfparquet shim (a minimal static Arrow
build behind three C symbols) and import via the C Data Interface.

Usage: spike_parquet_shim LIBRARY.so FILE.parquet [REPETITIONS]
"""
from std.ffi import OwnedDLHandle
from std.memory import Pointer
from std.sys import argv
from std.time import monotonic

from dataframe import DataFrame, col
from dataframe.arrow import ArrowArray, ArrowSchema, _read_c_string, import_arrow


def read_parquet(
    lib: OwnedDLHandle, path: String, mut read_ns: Int, mut import_ns: Int
) raises -> DataFrame:
    var c_path = List[UInt8](capacity=path.byte_length() + 1)
    c_path.extend(path.as_bytes())
    c_path.append(0)
    var array = ArrowArray()
    var schema = ArrowSchema()
    var error = 0
    var start = monotonic()
    var status = lib.get_function[Int32]("dfq_read_parquet")(
        c_path.unsafe_ptr(),
        Int32(1),
        Int(0),
        Int32(0),
        Pointer(to=array),
        Pointer(to=schema),
        Pointer(to=error),
    )
    read_ns += monotonic() - start
    if status != 0:
        var text = _read_c_string(error)
        lib.get_function[NoneType]("dfq_free")(error)
        raise Error("dfq_read_parquet: " + text)
    start = monotonic()
    var frame = import_arrow(array, schema)
    import_ns += monotonic() - start
    return frame^


def main() raises:
    var args = argv()
    if len(args) < 3 or len(args) > 4:
        raise Error("usage: spike_parquet_shim LIBRARY.so FILE.parquet [REPS]")
    var lib = OwnedDLHandle(String(args[1]))
    print("arrow", _read_c_string(lib.get_function[Int]("dfq_arrow_version")()))
    var path = String(args[2])
    var repetitions = Int(String(args[3])) if len(args) == 4 else 0
    var read_ns = 0
    var import_ns = 0
    var frame = read_parquet(lib, path, read_ns, import_ns)
    if repetitions == 0:
        print(frame)
        for name in frame.columns():
            print(name, frame.column(name).dtype(), sep="\t")
        return
    print(frame.height(), "rows,", frame.width(), "columns")
    print(
        "x_sum", frame.select(col("x").sum()).item(),
        "x_nulls", frame.select(col("x").null_count()).item(),
        "n_sum", frame.select(col("n").sum()).item(),
        "jk_sum", frame.select(col("jk").sum()).item(),
        "str_bytes",
        frame.select(col("key_str").str().len_bytes().sum()).item(),
    )
    var best = Int.MAX
    var best_read = 0
    var best_import = 0
    for _ in range(repetitions):
        var r = 0
        var i = 0
        var start = monotonic()
        var again = read_parquet(lib, path, r, i)
        var total = monotonic() - start
        if again.height() != frame.height():
            raise Error("row count changed between reads")
        if total < best:
            best = total
            best_read = r
            best_import = i
    print(
        "best of", repetitions, ":", Float64(best) / 1e6, "ms",
        "(arrow read", Float64(best_read) / 1e6,
        "import", Float64(best_import) / 1e6, ")",
    )
