"""Shared worker protocol for the Mojo side of the external suites.

Each suite runner is invoked as

    RUNNER SUITE QUERIES REPS name=path [name=path ...]

where QUERIES is a comma-separated list. It loads every named Parquet table
into memory once (untimed), then for each query runs it once to warm up and
times REPS runs, each producing a materialized DataFrame. It prints the same
lines as benchmarks/suites/engines.py, each tagged with its query:

    time<TAB>QUERY<TAB>NANOSECONDS                  one per timed run
    summary<TAB>QUERY<TAB>HEIGHT<TAB>V1,V2<TAB>N1,N2 see engines.py `summary`
    unsupported<TAB>QUERY<TAB>REASON
    failed<TAB>QUERY<TAB>MESSAGE

A query reports unsupported by raising an error that starts with
"unsupported:"; any other error fails that query alone.
"""
from std.collections import Dict
from std.io import FileDescriptor
from std.sys import argv
from std.time import monotonic

from dataframe import DataFrame, DataType, Expr, col, lit, read_parquet


def unsupported(reason: String) -> Error:
    return Error("unsupported: " + reason)


def date_lit(text: String) -> Expr:
    """A Date literal from ISO text."""
    return lit(text).str().to_date()


def load_tables() raises -> Dict[String, DataFrame]:
    var args = argv()
    var tables = Dict[String, DataFrame]()
    for i in range(4, len(args)):
        var arg = String(args[i])
        var parts = arg.split("=", 1)
        if len(parts) != 2:
            raise Error("expected name=path, found " + arg)
        tables[String(parts[0])] = read_parquet(String(parts[1]))
    return tables^


def summary_value(frame: DataFrame, name: String) raises -> Float64:
    """One number per column, matching engines.py's `summary`."""
    var dtype = frame.column(name).dtype()
    var expr: Expr
    if dtype == DataType.STRING:
        expr = col(name).str().len_bytes().sum()
    elif dtype.is_date():
        expr = col(name).cast(DataType.INT64).sum()
    elif dtype.is_datetime():
        # Whole seconds, flooring like Polars' dt.epoch("s").
        expr = (col(name).cast(DataType.INT64) // lit(dtype.per_second())).sum()
    elif dtype == DataType.BOOL or dtype.is_numeric():
        expr = col(name).cast(DataType.FLOAT64).sum()
    else:
        expr = col(name).is_not_null().cast(DataType.INT64).sum()
    var cell = frame.select(expr).item()
    if cell.is_null():
        return 0.0
    if cell.dtype() == DataType.INT64:
        return Float64(cell.int64())
    return cell.float64()


def run[
    query: def(String, Dict[String, DataFrame]) raises thin -> DataFrame
]() raises:
    """Load once, then warm up, time and summarize each query (module docs)."""
    var args = argv()
    if len(args) < 5:
        raise Error("usage: RUNNER SUITE QUERIES REPS name=path ...")
    var names = String(args[2]).split(",")
    var reps = Int(String(args[3]))
    var tables = load_tables()
    for part in names:
        var name = String(part)
        # Lets a trace (DATAFRAME_TRACE_PATHS) attribute paths to queries.
        print("dataframe-query:", name, file=FileDescriptor(2))
        try:
            var result = query(name, tables)
            for _ in range(reps):
                var start = monotonic()
                result = query(name, tables)
                print("time\t" + name + "\t" + String(monotonic() - start))
            var values = String()
            var columns = String()
            for i in range(result.width()):
                if i > 0:
                    values += ","
                    columns += ","
                values += String(summary_value(result, result.columns()[i]))
                columns += result.columns()[i]
            print(
                "summary\t"
                + name
                + "\t"
                + String(result.height())
                + "\t"
                + values
                + "\t"
                + columns
            )
        except e:
            var message = String(e)
            if message.startswith("unsupported:"):
                print("unsupported\t" + name + "\t" + String(message[byte=13:]))
            else:
                print("failed\t" + name + "\t" + message.replace("\n", " "))
