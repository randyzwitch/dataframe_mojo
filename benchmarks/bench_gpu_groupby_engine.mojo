"""Focused end-to-end grouped reducer gate, buildable against either engine.

Times complete collect(), including preparation, upload, kernels and result
publication. Source creation and CPU verification are outside Metal timing.
"""
from std.testing import assert_true
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, Series, col
from dataframe.metal import MetalRuntime


def main() raises:
    for scenario in range(3):
        var rows = 4099 if scenario == 0 else 1_000_003
        var cardinality = (
            17 if scenario == 0 else 101 if scenario == 1 else 100_003
        )
        var keys = List[Int64](capacity=rows)
        var values = List[Int64](capacity=rows)
        var valid = List[Bool](capacity=rows)
        for row in range(rows):
            keys.append(Int64((row * 7919) % cardinality))
            values.append(Int64(row % 1009 - 504))
            valid.append(row % 19 != 0)
        var frame = DataFrame(
            [
                Series("key", Column[Int64](keys^)),
                Series("value", Column[Int64](values^, valid^)),
            ]
        )
        var query = (
            frame.lazy()
            .group_by("key", maintain_order=True)
            .agg(
                [
                    col("value").min().alias("min"),
                    col("value").max().alias("max"),
                    col("value").sum().alias("sum"),
                    col("value").count().alias("count"),
                ]
            )
        )
        var expected = query.collect(engine="cpu")
        for round in range(4):
            var began = Int(perf_counter_ns())
            var actual = query.collect(
                accelerator=MetalRuntime(), engine="accel"
            )
            var elapsed = Int(perf_counter_ns()) - began
            assert_true(actual.equals(expected))
            print(
                "rows",
                rows,
                "groups",
                cardinality,
                "round",
                round,
                "collect_ns",
                elapsed,
            )
