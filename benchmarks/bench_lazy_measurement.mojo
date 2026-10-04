"""Focused development benchmark for measured-input reuse.

Build with mojo build -O3 -I . benchmarks/bench_lazy_measurement.mojo.
Run with DATAFRAME_THREADS=8 and PLANNER_STREAMING=0 or 1.
The workload is identical for the baseline and changed engine; correctness
is checked against unoptimized collection outside the timed interval.
This supplements the existing suites without changing their workloads.
"""
from dataframe import Column, DataFrame, Series, col
from std.os import getenv
from std.time import monotonic
from std.testing import assert_true


def main() raises:
    var n = 1_000_000
    var keys = List[Int64](capacity=n)
    var a = List[String](capacity=n)
    var b = List[String](capacity=n)
    for i in range(n):
        keys.append(Int64(i))
        a.append("KEEP" if i % 10 == 0 else "DROP")
        b.append("KEEP" if i % 5 == 0 else "DROP")
    var base = DataFrame(
        [
            Series("a", Column[Int64](keys.copy())),
            Series("b", Column[Int64](keys.copy())),
        ]
    )
    var left = DataFrame(
        [
            Series("a", Column[Int64](keys.copy())),
            Series("tag_a", Column[String](a^)),
        ]
    )
    var right = DataFrame(
        [Series("b", Column[Int64](keys^)), Series("tag_b", Column[String](b^))]
    )
    var query = (
        base.lazy()
        .join(
            left.lazy().filter(
                col("tag_a").str().to_lowercase().str().contains("keep")
            ),
            "a",
        )
        .join(
            right.lazy().filter(
                col("tag_b").str().to_lowercase().str().contains("keep")
            ),
            "b",
        )
    )
    var streaming = getenv("PLANNER_STREAMING") == "1"
    var expected = query.collect(optimize=False, streaming=False)
    var best = Float64(1e30)
    for rep in range(4):
        var start = monotonic()
        var result = query.collect(streaming=streaming)
        var ms = Float64(monotonic() - start) / 1e6
        if rep > 0:
            best = min(best, ms)
        assert_true(result.equals(expected))
    print(
        "unchanged join order",
        "rows",
        n,
        "streaming",
        streaming,
        "best_ms",
        best,
        "correct",
        True,
    )
