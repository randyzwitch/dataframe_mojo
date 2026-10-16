"""Check supplementary input answers against independent oracle Parquet.

Imports the existing forced workers staged by bench_metal_suites.py. No query
expression or fallback is implemented here. CPU and Metal run separately.
Usage: WITNESS ID SUITE QUERY cpu|metal EXPECTED name=path [name=path ...]
"""
from std.collections import Dict
from std.sys import argv
from dataframe import DataFrame, read_parquet
from cpu_tpcds import query as cpu_tpcds
from metal_tpcds import query as metal_tpcds
from cpu_clickbench import query as cpu_clickbench
from metal_clickbench import query as metal_clickbench


def main() raises:
    var args = argv()
    if len(args) < 7:
        raise Error(
            "usage: WITNESS ID SUITE QUERY cpu|metal EXPECTED name=path ..."
        )
    var identity = String(args[1])
    var suite = String(args[2])
    var query = String(args[3])
    var engine = String(args[4])
    if engine != "cpu" and engine != "metal":
        raise Error("Witness engine must be forced cpu or metal")
    var expected = read_parquet(String(args[5]))
    var tables = Dict[String, DataFrame]()
    for index in range(6, len(args)):
        var argument = String(args[index])
        var parts = argument.split("=", 1)
        if len(parts) != 2:
            raise Error("Witness input must be name=path")
        tables[String(parts[0])] = read_parquet(String(parts[1]))
    var actual: DataFrame
    if suite == "tpcds":
        actual = cpu_tpcds(query, tables) if engine == "cpu" else metal_tpcds(
            query, tables
        )
    elif suite == "clickbench":
        actual = cpu_clickbench(
            query, tables
        ) if engine == "cpu" else metal_clickbench(query, tables)
    else:
        raise Error("Unknown supplementary witness suite")
    if expected.height() == 0 or not actual.equals(expected):
        raise Error(
            "Supplementary witness differs from independent typed ordered oracle"
        )
    print(
        "witness_oracle\t"
        + identity
        + "\t"
        + engine
        + "\t"
        + String(actual.height())
        + "\tTrue"
    )
