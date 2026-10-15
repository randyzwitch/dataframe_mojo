"""Export the benchmark plans; invoked by query_diagrams.py, not timed."""
from std.sys import argv

from dataframe import LazyFrame
from suite_common import load_tables
from pdsh import plan as pdsh_plan
from tpcds import plan as tpcds_plan
from clickbench_plan import query as clickbench_plan


def main() raises:
    var args = argv()
    var suite = String(args[1])
    var tables = load_tables()
    for name in String(args[2]).split(","):
        var q = String(name)
        print("QUERY\t" + q)
        try:
            var plan: LazyFrame
            if suite == "pdsh":
                plan = pdsh_plan(q, tables)
            elif suite == "tpcds":
                plan = tpcds_plan(q, tables)
            else:
                plan = clickbench_plan(q, tables)
            print("ORIGINAL")
            print(plan.explain(optimize=False))
            print("OPTIMIZED")
            print(plan.explain())
        except e:
            print("ERROR\t" + String(e).replace("\n", " "))
        print("END_QUERY")
