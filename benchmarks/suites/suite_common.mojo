"""Shared worker protocol for the Mojo side of the external suites.

Each suite runner is invoked as

    RUNNER SUITE QUERY REPS name=path [name=path ...]

It loads every named Parquet table into memory (untimed), runs the query
once to warm up, then times REPS runs of the full query, each producing a
materialized DataFrame. It prints the same lines as benchmarks/suites/
engines.py: `time<TAB>ns` per run and
`summary<TAB>height<TAB>v1,v2,...<TAB>name1,name2,...`,
or `unsupported<TAB>reason` when this library cannot express the query.
A query signals the latter by raising an error that starts with
"unsupported:".
"""
from std.collections import Dict
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
    """Load, warm up, time and summarize one query (see module docs)."""
    var args = argv()
    if len(args) < 5:
        raise Error("usage: RUNNER SUITE QUERY REPS name=path ...")
    var name = String(args[2])
    var reps = Int(String(args[3]))
    var tables = load_tables()
    var result: DataFrame
    try:
        result = query(name, tables)
    except e:
        var message = String(e)
        if message.startswith("unsupported:"):
            print("unsupported\t" + String(message[byte=13:]))
            return
        raise e^
    for _ in range(reps):
        var start = monotonic()
        result = query(name, tables)
        print("time\t" + String(monotonic() - start))
    var values = String()
    var names = String()
    for i in range(result.width()):
        if i > 0:
            values += ","
            names += ","
        values += String(summary_value(result, result.columns()[i]))
        names += result.columns()[i]
    print("summary\t" + String(result.height()) + "\t" + values + "\t" + names)
