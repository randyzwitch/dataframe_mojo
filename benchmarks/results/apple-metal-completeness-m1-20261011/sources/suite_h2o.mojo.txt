from std.sys import argv
from metal_suite_common import run
from cpu_h2o import groupby as cpu_groupby, join as cpu_join
from metal_h2o import groupby as metal_groupby, join as metal_join
def main() raises:
    if String(argv()[1]) == "h2o_groupby":
        run[cpu_groupby, metal_groupby]()
    else:
        run[cpu_join, metal_join]()
