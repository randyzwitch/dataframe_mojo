"""Opt-in tracing of specialized execution paths, for benchmark coverage.

With DATAFRAME_TRACE_PATHS set to any value, each specialized join,
group-by, filter and sort path prints `dataframe-path: NAME` to stderr when
it produces a result. `scripts/bench_suites.py --trace` collects these per
query and reports paths that only one benchmark query exercises, the sign of
a fast path fitted to a benchmark rather than to a data property (see
docs/benchmarks.md). Unset, each call costs one environment lookup; calls
sit at operator level, never inside row loops.
"""
from std.io import FileDescriptor
from std.os import getenv


def trace_path(name: StaticString):
    if getenv("DATAFRAME_TRACE_PATHS"):
        print("dataframe-path:", name, file=FileDescriptor(2))
