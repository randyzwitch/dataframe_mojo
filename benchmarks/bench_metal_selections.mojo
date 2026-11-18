"""Compare CPU, materialized Metal handoff, and retained-selection handoff.

Run on Apple silicon with the Metal provider. All three paths use the same
query and independently computed answers. Arguments: rows, repetitions.
The materialized path uses the CPU scheduler's existing ordinary-column
provider ports; the explicit runtime also installs the selection ports.
"""
from std.sys import argv
from std.time import perf_counter_ns
from std.testing import assert_true
from dataframe import Column, DataFrame, Series, col, lit
from dataframe_metal.metal import MetalRuntime


def run(rows: Int, repetitions: Int) raises:
    var keys = List[Int64](capacity=rows)
    var values = List[Int64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for row in range(rows):
        keys.append(Int64(row % 1024))
        values.append(Int64(row % 33 - 16))
        valid.append(row % 17 != 3)
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^, valid)),
        ]
    )
    var build_keys = List[Int64]()
    var payload = List[Int64]()
    for key in range(1024):
        if key % 5 == 2:
            continue
        for copy in range(2 if key % 3 == 0 else 1):
            build_keys.append(Int64(key))
            payload.append(Int64(key % 11 + copy))
    var dim = DataFrame(
        [
            Series("k", Column[Int64](build_keys^)),
            Series("w", Column[Int64](payload^)),
        ]
    )
    var plan = (
        frame.lazy()
        .filter((col("k") % lit(Int64(7))) != lit(Int64(1)))
        .join(dim.lazy(), on="k")
        .select_exprs(
            [
                col("v").sum().alias("v_sum"),
                col("v").count().alias("v_count"),
                col("w").sum().alias("w_sum"),
            ]
        )
    )
    var v_sum = Int64(0)
    var v_count = Int64(0)
    var w_sum = Int64(0)
    for row in range(rows):
        var key = row % 1024
        if key % 7 == 1 or key % 5 == 2:
            continue
        for copy in range(2 if key % 3 == 0 else 1):
            if row % 17 != 3:
                v_sum += Int64(row % 33 - 16)
                v_count += 1
            w_sum += Int64(key % 11 + copy)
    var expected = DataFrame(
        [
            Series("v_sum", Column[Int64]([v_sum])),
            Series("v_count", Column[Int64]([v_count])),
            Series("w_sum", Column[Int64]([w_sum])),
        ]
    )
    print("PLAN", plan.explain(engine="cpu"))
    var runtime = MetalRuntime()
    var cpu = plan.profile(engine="cpu")
    assert_true(cpu[0].equals(expected), "CPU differs from scalar oracle")
    print("CPU_PROFILE", cpu[1])
    var ordinary = plan._with_handoff().profile(engine="cpu")
    assert_true(
        ordinary[0].equals(expected),
        "ordinary handoff differs from scalar oracle",
    )
    print("MATERIALIZED_PROFILE", ordinary[1])
    var gpu = plan.profile(accelerator=runtime)
    assert_true(gpu[0].equals(expected), "Metal differs from scalar oracle")
    print("METAL_PROFILE", gpu[1])
    var selected = False
    for row in range(gpu[1].height()):
        selected |= (
            gpu[1].item(row, "algorithm").string() == "handoff selections"
        )
    assert_true(
        selected, "benchmark did not exercise the native selection handoff"
    )
    for repeat in range(repetitions):
        for side in range(3):
            var mode = (side + repeat) % 3
            var started = Int(perf_counter_ns())
            var result: DataFrame
            if mode == 0:
                result = plan.collect(engine="cpu")
            elif mode == 1:
                result = plan._with_handoff().collect(engine="cpu")
            else:
                result = plan.collect(accelerator=runtime)
            var elapsed = Int(perf_counter_ns()) - started
            assert_true(result.equals(expected), "timed answer differs")
            var label = "cpu" if mode == 0 else (
                "materialized" if mode == 1 else "selections"
            )
            print("SAMPLE", label, rows, repeat, elapsed)


def main() raises:
    var args = argv()
    run(
        Int(args[1]) if len(args) > 1 else 1_000_000,
        Int(args[2]) if len(args) > 2 else 3,
    )
