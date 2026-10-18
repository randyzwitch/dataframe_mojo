"""Complete-collection samples; answer summaries and table loading are untimed.

A fresh process handles one query, engine and round. 'cold' is its first
query invocation after loading; it does not claim cold filesystem caches.
"""
from std.collections import Dict
from std.sys import argv
from std.time import perf_counter_ns
from dataframe import DataFrame, read_parquet
from suite_common import summary_value


def _clean(value: String) -> String:
    return value.replace("\n", " ").replace("\t", " ")


def _invoke[
    query: def(String, Dict[String, DataFrame]) raises thin -> DataFrame
](
    name: String, tables: Dict[String, DataFrame], phase: String, index: Int
) raises:
    var started = perf_counter_ns()
    var result = query(name, tables)
    var elapsed = perf_counter_ns() - started
    print("sample\t" + phase + "\t" + String(index) + "\t" + String(elapsed))
    try:
        var values = String()
        var columns = String()
        for i in range(result.width()):
            if i > 0:
                values += ","
                columns += ","
            var column = result.columns()[i]
            values += String(summary_value(result, column))
            columns += column
        print(
            "answer\t"
            + phase
            + "\t"
            + String(index)
            + "\t"
            + String(result.height())
            + "\t"
            + values
            + "\t"
            + columns
        )
    except error:
        raise Error("untimed answer summary: " + String(error))
    # Release the completed answer before the next invocation starts.
    _ = result^


def run[
    query: def(String, Dict[String, DataFrame]) raises thin -> DataFrame
]() raises:
    var args = argv()
    var name = String(args[2])
    var repetitions = Int(String(args[4]))
    var warmups = Int(String(args[5]))
    var tables = Dict[String, DataFrame]()
    try:
        for i in range(6, len(args)):
            var argument = String(args[i]).split("=", 1)
            if len(argument) != 2:
                raise Error("expected table=path")
            tables[String(argument[0])] = read_parquet(String(argument[1]))
    except error:
        print("outcome\tinput_failed\t" + _clean(String(error)))
        return
    try:
        _invoke[query](name, tables, "cold", 0)
        for i in range(warmups):
            _invoke[query](name, tables, "warmup", i)
        for i in range(repetitions):
            _invoke[query](name, tables, "warm", i)
    except error:
        var reason = _clean(String(error))
        var status = "answer_failed" if reason.startswith(
            "untimed answer summary:"
        ) else (
            "unsupported" if "unsupported" in reason
            or "do not support" in reason else "execution_failed"
        )
        print("outcome\t" + status + "\t" + reason)
        return
    print("outcome\tok\t")
